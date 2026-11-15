"""
continuity_engine.py — LUNAR EDGE Phase 3

Mission Continuity Envelope Engine.

Performs deterministic sensitivity and robustness analysis for lunar
landing sites.

Evaluates how mission opportunities and constraints degrade or hold when:
1. Spatial Landing Dispersion: Touchdown occurs off-target.
2. Temporal Launch Slippage: Landing date shifts from the nominal date.

Determines whether critical mission requirements still succeed or fail,
providing mission planners with a quantitative Robustness Score.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import List, Tuple, Optional

import numpy as np

from app.utils.date_utils import (
    parse_and_normalize_utc,
    normalize_to_iso_utc,
)
from app.engines.geometry_engine import GeometryEngine
from app.engines.horizon_engine import compute_horizon_profile
from app.engines.terrain_engine import TerrainEngine
from app.engines.visibility_engine import compute_terrain_visibility_time_series
from app.models.schemas import (
    LandingSite,
    MissionRequirements,
    PerturbedEvaluation,
    ContinuityEnvelopeResult,
    ContinuityMetrics,
    VisibilityTimeSeriesResponse,
)

logger = logging.getLogger(__name__)


class ContinuityEngine:
    """
    Engine for calculating the Mission Continuity Envelope.

    Evaluates sensitivity of mission requirements to:
    - Landing location dispersion.
    - Launch/landing date slippage.
    """

    def __init__(
        self,
        terrain_engine: Optional[TerrainEngine] = None,
        geometry_engine: Optional[GeometryEngine] = None,
    ):
        self.terrain_engine = terrain_engine or TerrainEngine()
        self.geometry_engine = geometry_engine or GeometryEngine()

    # ------------------------------------------------------------------
    # CONTINUITY METRICS
    # ------------------------------------------------------------------

    @staticmethod
    def compute_continuity_metrics(
        vis_response: VisibilityTimeSeriesResponse,
    ) -> ContinuityMetrics:
        """
        Calculates comprehensive continuity and window metrics
        from a visibility time series.
        """

        summary = vis_response.summary
        samples = vis_response.samples

        total_mission_hours = summary.duration_days * 24.0
        dt_hours = summary.step_minutes / 60.0
        uncertainty_hours = dt_hours

        d2n = 0
        n2d = 0
        comms_acq = 0
        comms_lost = 0
        simul_acq = 0
        simul_lost = 0

        for i in range(len(samples) - 1):
            s_curr = samples[i]
            s_next = samples[i + 1]

            # Sun illumination transitions
            if s_curr.is_sunlit and not s_next.is_sunlit:
                d2n += 1
            elif not s_curr.is_sunlit and s_next.is_sunlit:
                n2d += 1

            # Earth communication transitions
            if not s_curr.is_earth_visible and s_next.is_earth_visible:
                comms_acq += 1
            elif s_curr.is_earth_visible and not s_next.is_earth_visible:
                comms_lost += 1

            # Simultaneous power + communications transitions
            if (
                not s_curr.is_simultaneous_operating
                and s_next.is_simultaneous_operating
            ):
                simul_acq += 1
            elif (
                s_curr.is_simultaneous_operating
                and not s_next.is_simultaneous_operating
            ):
                simul_lost += 1

        sun_pct = round(
            (summary.sunlit_hours / max(total_mission_hours, 1e-6)) * 100.0,
            2,
        )

        earth_pct = round(
            (summary.earth_visible_hours / max(total_mission_hours, 1e-6)) * 100.0,
            2,
        )

        simul_pct = round(
            (summary.simultaneous_hours / max(total_mission_hours, 1e-6)) * 100.0,
            2,
        )

        explanation = (
            f"Mission duration: {summary.duration_days:.1f} days "
            f"({total_mission_hours:.1f} h) sampled at "
            f"{summary.step_minutes:.1f}-min resolution "
            f"(temporal quantization uncertainty +/-"
            f"{uncertainty_hours:.2f} h). "
            f"Total sunlight received: {summary.sunlit_hours:.1f} h "
            f"({sun_pct}%), with the longest unbroken sunlit window "
            f"lasting {summary.longest_sunlit_window_hours:.1f} h "
            f"across {d2n} sunset transitions. "
            f"Longest continuous darkness period is "
            f"{summary.longest_dark_period_hours:.1f} h. "
            f"Direct Earth line-of-sight is available "
            f"for {summary.earth_visible_hours:.1f} h "
            f"({earth_pct}%), with a maximum continuous comms window "
            f"of {summary.longest_earth_visible_window_hours:.1f} h "
            f"and blackout period of "
            f"{summary.longest_earth_blackout_hours:.1f} h "
            f"across {comms_lost} occultation events. "
            f"Simultaneous power and comms coincide for "
            f"{summary.simultaneous_hours:.1f} h "
            f"({simul_pct}%), yielding a maximum continuous "
            f"operating window of "
            f"{summary.longest_simultaneous_window_hours:.1f} h."
        )

        return ContinuityMetrics(
            site_name=summary.site_name,
            start_utc=summary.start_utc,
            duration_days=summary.duration_days,
            step_minutes=summary.step_minutes,
            total_mission_hours=round(total_mission_hours, 2),
            temporal_uncertainty_hours=round(uncertainty_hours, 3),
            total_sunlight_hours=summary.sunlit_hours,
            sunlight_availability_pct=sun_pct,
            longest_continuous_illumination_hours=(
                summary.longest_sunlit_window_hours
            ),
            longest_continuous_darkness_hours=(
                summary.longest_dark_period_hours
            ),
            total_earth_visibility_hours=summary.earth_visible_hours,
            earth_visibility_pct=earth_pct,
            longest_continuous_earth_visibility_hours=(
                summary.longest_earth_visible_window_hours
            ),
            longest_continuous_earth_blackout_hours=(
                summary.longest_earth_blackout_hours
            ),
            total_simultaneous_hours=summary.simultaneous_hours,
            simultaneous_availability_pct=simul_pct,
            longest_continuous_simultaneous_hours=(
                summary.longest_simultaneous_window_hours
            ),
            day_to_night_transitions=d2n,
            night_to_day_transitions=n2d,
            comms_acquired_transitions=comms_acq,
            comms_lost_transitions=comms_lost,
            simultaneous_acquired_transitions=simul_acq,
            simultaneous_lost_transitions=simul_lost,
            explanation=explanation,
        )

    # ------------------------------------------------------------------
    # DEM-REFRESHED LANDING SITE
    # ------------------------------------------------------------------

    def _refresh_site_from_dem(
        self,
        site: LandingSite,
    ) -> Tuple[LandingSite, List[str]]:
        """
        Refreshes elevation, slope and aspect using terrain-engine DEM data.

        Original user-supplied values are retained when DEM values cannot
        be evaluated. Such failures are returned separately so they can
        be reported in the evaluation.
        """

        reasons: List[str] = []

        elevation, elevation_status, _ = (
            self.terrain_engine.get_elevation_safe(
                site.latitude_deg,
                site.longitude_deg,
            )
        )

        if elevation_status == "EVALUATED" and elevation is not None:
            elevation_m = float(elevation)
        else:
            elevation_m = site.elevation_m
            reasons.append(
                "DEM elevation could not be evaluated at the landing site."
            )

        slope, aspect, slope_status, _ = (
            self.terrain_engine.compute_point_slope_aspect_safe(
                site.latitude_deg,
                site.longitude_deg,
            )
        )

        if slope_status == "EVALUATED" and slope is not None:
            slope_deg = float(slope)
            aspect_deg = (
                float(aspect)
                if aspect is not None
                else site.aspect_deg
            )
        else:
            slope_deg = site.slope_deg
            aspect_deg = site.aspect_deg
            reasons.append(
                "DEM slope/aspect could not be evaluated at the landing site."
            )

        refreshed_site = LandingSite(
            name=site.name,
            latitude_deg=site.latitude_deg,
            longitude_deg=site.longitude_deg,
            elevation_m=elevation_m,
            slope_deg=slope_deg,
            aspect_deg=aspect_deg,
        )

        return refreshed_site, reasons

    # ------------------------------------------------------------------
    # MAIN CONTINUITY ENVELOPE
    # ------------------------------------------------------------------

    def evaluate_continuity_envelope(
        self,
        site: LandingSite,
        start_utc: str,
        duration_days: float = 14.0,
        step_minutes: float = 30.0,
        max_slope_deg: float = 15.0,
        horizon_radius_km: float = 20.0,
        requirements: Optional[MissionRequirements] = None,
        spatial_radii_m: Optional[List[float]] = None,
        temporal_offsets_days: Optional[List[float]] = None,
    ) -> ContinuityEnvelopeResult:
        """
        Runs the full Mission Continuity Envelope analysis.

        Args:
            site: Target landing site.
            start_utc: Nominal landing epoch ISO string.
            duration_days: Mission duration in days.
            step_minutes: Geometry calculation time step.
            max_slope_deg: Maximum acceptable landing-site slope.
            horizon_radius_km: Terrain horizon extraction radius.
            requirements: Mission power, battery and communication limits.
            spatial_radii_m: Landing dispersion distances.
            temporal_offsets_days: Date-slippage offsets.
        """

        reqs = requirements or MissionRequirements()

        radii = (
            spatial_radii_m
            if spatial_radii_m is not None
            else [100.0, 250.0]
        )

        slippages = (
            temporal_offsets_days
            if temporal_offsets_days is not None
            else [-2.0, -1.0, 1.0, 2.0]
        )

        # --------------------------------------------------------------
        # 1. Refresh baseline site from DEM
        # --------------------------------------------------------------

        baseline_site, baseline_dem_reasons = (
            self._refresh_site_from_dem(site)
        )

        # --------------------------------------------------------------
        # 2. Baseline evaluation
        # --------------------------------------------------------------

        baseline_vis, baseline_passes, baseline_reasons = (
            self._evaluate_single_configuration(
                site=baseline_site,
                start_utc=start_utc,
                duration_days=duration_days,
                step_minutes=step_minutes,
                max_slope_deg=max_slope_deg,
                horizon_radius_km=horizon_radius_km,
                reqs=reqs,
            )
        )

        baseline_reasons = (
            baseline_dem_reasons + baseline_reasons
        )

        baseline_passes = (
            baseline_passes and len(baseline_dem_reasons) == 0
        )

        baseline_summary = baseline_vis.summary

        baseline_metrics = self.compute_continuity_metrics(
            baseline_vis
        )

        perturbations: List[PerturbedEvaluation] = []
        spatial_results: List[bool] = []
        temporal_results: List[bool] = []

        # --------------------------------------------------------------
        # 3. Spatial perturbations
        # --------------------------------------------------------------

        obs_x, obs_y = (
            self.terrain_engine.selenographic_to_polar_stereo(
                baseline_site.latitude_deg,
                baseline_site.longitude_deg,
            )
        )

        r_obs = float(np.sqrt(obs_x**2 + obs_y**2))

        if r_obs < 1e-3:
            u_north = np.array([0.0, 1.0])
            u_east = np.array([1.0, 0.0])
        else:
            lon_rad = float(np.arctan2(obs_x, obs_y))

            u_north = np.array(
                [np.sin(lon_rad), np.cos(lon_rad)]
            )

            u_east = np.array(
                [np.cos(lon_rad), -np.sin(lon_rad)]
            )

        for radius in radii:

            directions = [
                (
                    "North",
                    float(radius * u_north[0]),
                    float(radius * u_north[1]),
                ),
                (
                    "East",
                    float(radius * u_east[0]),
                    float(radius * u_east[1]),
                ),
                (
                    "South",
                    float(-radius * u_north[0]),
                    float(-radius * u_north[1]),
                ),
                (
                    "West",
                    float(-radius * u_east[0]),
                    float(-radius * u_east[1]),
                ),
            ]

            for direction_name, dx, dy in directions:

                pert_x = obs_x + dx
                pert_y = obs_y + dy

                pert_lat, pert_lon = (
                    self.terrain_engine.polar_stereo_to_selenographic(
                        pert_x,
                        pert_y,
                    )
                )

                # Query actual DEM elevation
                elev_val, elev_status, _ = (
                    self.terrain_engine.get_elevation_safe(
                        pert_lat,
                        pert_lon,
                    )
                )

                spatial_dem_reasons: List[str] = []

                if elev_status == "EVALUATED" and elev_val is not None:
                    pert_elev = float(elev_val)
                else:
                    pert_elev = baseline_site.elevation_m
                    spatial_dem_reasons.append(
                        f"DEM elevation unavailable at "
                        f"{direction_name} {int(radius)}m perturbation."
                    )

                # Query actual DEM slope and aspect
                pt_slope, pt_aspect, sa_status, _ = (
                    self.terrain_engine.compute_point_slope_aspect_safe(
                        pert_lat,
                        pert_lon,
                    )
                )

                if sa_status == "EVALUATED" and pt_slope is not None:
                    pert_slope = float(pt_slope)

                    pert_aspect = (
                        float(pt_aspect)
                        if pt_aspect is not None
                        else baseline_site.aspect_deg
                    )
                else:
                    pert_slope = baseline_site.slope_deg
                    pert_aspect = baseline_site.aspect_deg

                    spatial_dem_reasons.append(
                        f"DEM slope/aspect unavailable at "
                        f"{direction_name} {int(radius)}m perturbation."
                    )

                pert_site = LandingSite(
                    name=(
                        f"{baseline_site.name} "
                        f"({direction_name} {int(radius)}m)"
                    ),
                    latitude_deg=pert_lat,
                    longitude_deg=pert_lon,
                    elevation_m=pert_elev,
                    slope_deg=pert_slope,
                    aspect_deg=pert_aspect,
                )

                pert_vis, passes, reasons = (
                    self._evaluate_single_configuration(
                        site=pert_site,
                        start_utc=start_utc,
                        duration_days=duration_days,
                        step_minutes=step_minutes,
                        max_slope_deg=max_slope_deg,
                        horizon_radius_km=horizon_radius_km,
                        reqs=reqs,
                    )
                )

                reasons = spatial_dem_reasons + reasons

                passes = passes and len(spatial_dem_reasons) == 0

                summary = pert_vis.summary

                spatial_results.append(passes)

                perturbations.append(
                    PerturbedEvaluation(
                        displacement_dx_m=round(dx, 1),
                        displacement_dy_m=round(dy, 1),
                        displacement_distance_m=round(radius, 1),
                        date_slippage_days=0.0,
                        evaluated_lat_deg=round(pert_lat, 6),
                        evaluated_lon_deg=round(pert_lon, 6),
                        evaluated_start_utc=start_utc,
                        sunlit_hours=summary.sunlit_hours,
                        sunlit_fraction=summary.sunlit_fraction,
                        longest_sunlit_window_hours=(
                            summary.longest_sunlit_window_hours
                        ),
                        longest_dark_period_hours=(
                            summary.longest_dark_period_hours
                        ),
                        longest_comms_window_hours=(
                            summary.longest_earth_visible_window_hours
                        ),
                        longest_simultaneous_hours=(
                            summary.longest_simultaneous_window_hours
                        ),
                        meets_requirements=passes,
                        failure_reasons=reasons,
                    )
                )

        # --------------------------------------------------------------
        # 4. Temporal perturbations
        # --------------------------------------------------------------

        base_dt = parse_and_normalize_utc(start_utc)

        for slip in slippages:

            pert_dt = base_dt + timedelta(days=slip)
            pert_utc = normalize_to_iso_utc(pert_dt)

            pert_vis, passes, reasons = (
                self._evaluate_single_configuration(
                    site=baseline_site,
                    start_utc=pert_utc,
                    duration_days=duration_days,
                    step_minutes=step_minutes,
                    max_slope_deg=max_slope_deg,
                    horizon_radius_km=horizon_radius_km,
                    reqs=reqs,
                )
            )

            # A DEM failure at the nominal site also affects confidence
            # in the temporal perturbation evaluations.
            reasons = baseline_dem_reasons + reasons

            passes = (
                passes and len(baseline_dem_reasons) == 0
            )

            summary = pert_vis.summary

            temporal_results.append(passes)

            perturbations.append(
                PerturbedEvaluation(
                    displacement_dx_m=0.0,
                    displacement_dy_m=0.0,
                    displacement_distance_m=0.0,
                    date_slippage_days=round(slip, 1),
                    evaluated_lat_deg=round(
                        baseline_site.latitude_deg,
                        6,
                    ),
                    evaluated_lon_deg=round(
                        baseline_site.longitude_deg,
                        6,
                    ),
                    evaluated_start_utc=pert_utc,
                    sunlit_hours=summary.sunlit_hours,
                    sunlit_fraction=summary.sunlit_fraction,
                    longest_sunlit_window_hours=(
                        summary.longest_sunlit_window_hours
                    ),
                    longest_dark_period_hours=(
                        summary.longest_dark_period_hours
                    ),
                    longest_comms_window_hours=(
                        summary.longest_earth_visible_window_hours
                    ),
                    longest_simultaneous_hours=(
                        summary.longest_simultaneous_window_hours
                    ),
                    meets_requirements=passes,
                    failure_reasons=reasons,
                )
            )

        # --------------------------------------------------------------
        # 5. Success rates and robustness score
        # --------------------------------------------------------------

        spatial_rate = (
            float(np.mean(spatial_results))
            if spatial_results
            else 1.0
        )

        temporal_rate = (
            float(np.mean(temporal_results))
            if temporal_results
            else 1.0
        )

        overall_score = (
            0.5 * spatial_rate
            + 0.5 * temporal_rate
        )

        return ContinuityEnvelopeResult(
            baseline_site=baseline_site,
            baseline_summary=baseline_summary,
            requirements=reqs,
            baseline_meets_requirements=baseline_passes,
            baseline_failure_reasons=baseline_reasons,
            perturbations=perturbations,
            spatial_success_rate=round(spatial_rate, 4),
            temporal_success_rate=round(temporal_rate, 4),
            overall_robustness_score=round(overall_score, 4),
            baseline_continuity_metrics=baseline_metrics,
        )

    # ------------------------------------------------------------------
    # SINGLE CONFIGURATION EVALUATION
    # ------------------------------------------------------------------

    def _evaluate_single_configuration(
        self,
        site: LandingSite,
        start_utc: str,
        duration_days: float,
        step_minutes: float,
        max_slope_deg: float,
        horizon_radius_km: float,
        reqs: MissionRequirements,
    ) -> Tuple[VisibilityTimeSeriesResponse, bool, List[str]]:
        """
        Evaluates geometry, terrain horizon and visibility for one
        specific mission configuration.

        Uses the same terrain horizon settings as the master mission
        evaluation.
        """

        # --------------------------------------------------------------
        # A. Ephemeris geometry time series
        # --------------------------------------------------------------

        geo_samples, _ = (
            self.geometry_engine.compute_mission_window_time_series(
                site=site,
                start_utc=start_utc,
                duration_days=duration_days,
                step_minutes=step_minutes,
            )
        )

        # --------------------------------------------------------------
        # B. Terrain horizon profile
        # --------------------------------------------------------------

        horizon_available = True

        try:
            patch = self.terrain_engine.extract_patch(
                lat_deg=site.latitude_deg,
                lon_deg=site.longitude_deg,
                radius_km=horizon_radius_km,
            )

            horizon = compute_horizon_profile(
                patch=patch,
                n_azimuths=360,
                n_range_steps=250,
            )

        except Exception as exc:
            horizon_available = False

            logger.exception(
                "DEM horizon extraction failed for %s: %s",
                site.name,
                exc,
            )

            # Fallback allows the visibility response to be constructed,
            # but the evaluation will explicitly fail below.
            horizon = np.zeros(360, dtype=np.float64)

        # --------------------------------------------------------------
        # C. Terrain-aware visibility
        # --------------------------------------------------------------

        vis_resp = compute_terrain_visibility_time_series(
            geometry_samples=geo_samples,
            horizon_profile=horizon,
            site_name=site.name,
        )

        summary = vis_resp.summary

        # --------------------------------------------------------------
        # D. Test mission requirements
        # --------------------------------------------------------------

        reasons: List[str] = []

        # Terrain horizon availability
        if not horizon_available:
            reasons.append(
                "Terrain horizon could not be evaluated from DEM data; "
                "visibility results are not considered reliable."
            )

        # Landing slope requirement
        if site.slope_deg > max_slope_deg:
            reasons.append(
                f"Landing-site slope ({site.slope_deg:.2f}°) "
                f"> maximum allowed slope ({max_slope_deg:.2f}°)."
            )

        # Continuous sunlight requirement
        if (
            summary.longest_sunlit_window_hours
            < reqs.min_continuous_sunlight_hours
        ):
            reasons.append(
                f"Continuous sunlight window "
                f"({summary.longest_sunlit_window_hours:.1f}h) "
                f"< required "
                f"({reqs.min_continuous_sunlight_hours:.1f}h)"
            )

        # Maximum darkness / battery requirement
        if (
            summary.longest_dark_period_hours
            > reqs.max_tolerable_dark_hours
        ):
            reasons.append(
                f"Continuous darkness period "
                f"({summary.longest_dark_period_hours:.1f}h) "
                f"> battery limit "
                f"({reqs.max_tolerable_dark_hours:.1f}h)"
            )

        # Earth communications requirement
        if (
            summary.longest_earth_visible_window_hours
            < reqs.min_continuous_earth_comms_hours
        ):
            reasons.append(
                f"Continuous Earth comms window "
                f"({summary.longest_earth_visible_window_hours:.1f}h) "
                f"< required "
                f"({reqs.min_continuous_earth_comms_hours:.1f}h)"
            )

        # Simultaneous power + communications requirement
        if (
            summary.longest_simultaneous_window_hours
            < reqs.min_simultaneous_operating_hours
        ):
            reasons.append(
                f"Continuous simultaneous window "
                f"({summary.longest_simultaneous_window_hours:.1f}h) "
                f"< required "
                f"({reqs.min_simultaneous_operating_hours:.1f}h)"
            )

        meets_requirements = len(reasons) == 0

        return vis_resp, meets_requirements, reasons