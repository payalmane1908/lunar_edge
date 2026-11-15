"""
mission_engine.py — LUNAR EDGE Phase 3

Master mission planning and decision support engine.
Coordinates geometry, terrain, visibility, and continuity engines to evaluate
candidate landing locations across the lunar south pole.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from app.engines.geometry_engine import GeometryEngine, CANDIDATE_LANDING_SITES
from app.engines.terrain_engine import TerrainEngine
from app.engines.horizon_engine import compute_horizon_profile
from app.engines.visibility_engine import compute_terrain_visibility_time_series
from app.engines.continuity_engine import ContinuityEngine
from app.models.schemas import (
    LandingSite,
    HorizonProfile,
    VisibilityTimeSeriesResponse,
    ContinuityEnvelopeResult,
    MissionRequirements,
    ConstraintEvaluation,
    MissionEvaluationResponse,
    ContinuityMetrics,
    VisibilitySummary,
    UncertaintyDistributionConfig,
    ProbabilisticResilienceResponse,
    CounterfactualRecoveryResponse,
)

logger = logging.getLogger(__name__)


class MissionEngine:
    """
    Main orchestration engine for LUNAR EDGE mission analysis.
    """

    def __init__(
        self,
        terrain_engine: Optional[TerrainEngine] = None,
        geometry_engine: Optional[GeometryEngine] = None,
        continuity_engine: Optional[ContinuityEngine] = None,
    ):
        self.terrain_engine = terrain_engine or TerrainEngine()
        self.geometry_engine = geometry_engine or GeometryEngine()
        self.continuity_engine = continuity_engine or ContinuityEngine(
            terrain_engine=self.terrain_engine,
            geometry_engine=self.geometry_engine,
        )
        self._resilience_engine = None
        self._recovery_engine = None

    @property
    def resilience_engine(self):
        if self._resilience_engine is None:
            from app.engines.resilience_engine import ResilienceEngine
            self._resilience_engine = ResilienceEngine(
                terrain_engine=self.terrain_engine,
                geometry_engine=self.geometry_engine,
            )
        return self._resilience_engine

    @property
    def recovery_engine(self):
        if self._recovery_engine is None:
            from app.engines.recovery_engine import CounterfactualRecoveryEngine
            self._recovery_engine = CounterfactualRecoveryEngine(mission_engine=self)
        return self._recovery_engine

    def initialize(self):
        """Pre-load DEM metadata and ensure kernels are available."""
        try:
            self.terrain_engine.load()
            logger.info("MissionEngine: TerrainEngine loaded successfully.")
        except Exception as e:
            logger.warning("MissionEngine: DEM not loaded yet (%s). Will lazy-load on demand.", e)

    def get_candidate_sites(self) -> Dict[str, LandingSite]:
        """Return the dictionary of Artemis III candidate sites with LOLA DEM ground-truth elevations."""
        try:
            sites = {}
            for k, s in CANDIDATE_LANDING_SITES.items():
                dem_elev = self.terrain_engine.get_elevation(s.latitude_deg, s.longitude_deg)
                sites[k] = s.model_copy(update={"elevation_m": round(float(dem_elev), 1)})
            return sites
        except Exception:
            return CANDIDATE_LANDING_SITES

    def compute_horizon(
        self,
        site: LandingSite,
        radius_km: float = 20.0,
        n_azimuths: int = 360,
    ) -> HorizonProfile:
        """
        Computes the 360° terrain horizon obstruction profile for a landing site.
        """
        patch = self.terrain_engine.extract_patch(
            lat_deg=site.latitude_deg,
            lon_deg=site.longitude_deg,
            radius_km=radius_km,
        )
        horizon = compute_horizon_profile(patch=patch, n_azimuths=n_azimuths)

        azimuths = [float(i * (360.0 / n_azimuths)) for i in range(n_azimuths)]
        elevations = [float(h) for h in horizon]

        updated_site = site.model_copy(update={"elevation_m": round(patch.observer_elevation_m, 2)})

        return HorizonProfile(
            site=updated_site,
            azimuths_deg=azimuths,
            horizon_elevations_deg=elevations,
            min_horizon_deg=round(float(np.min(horizon)), 3),
            max_horizon_deg=round(float(np.max(horizon)), 3),
            mean_horizon_deg=round(float(np.mean(horizon)), 3),
            observer_elevation_m=round(patch.observer_elevation_m, 2),
            radius_km=radius_km,
        )

    def evaluate_mission_visibility(
        self,
        site: LandingSite,
        start_utc: str,
        duration_days: float = 14.0,
        step_minutes: float = 15.0,
        horizon_radius_km: float = 20.0,
    ) -> VisibilityTimeSeriesResponse:
        """
        Computes full terrain-aware visibility time series (Sun, Earth, intervals).
        """
        # 1. Local terrain horizon and observer elevation from DEM
        try:
            patch = self.terrain_engine.extract_patch(
                lat_deg=site.latitude_deg,
                lon_deg=site.longitude_deg,
                radius_km=horizon_radius_km,
            )
            horizon = compute_horizon_profile(patch=patch, n_azimuths=360)
            eval_site = site.model_copy(update={"elevation_m": round(patch.observer_elevation_m, 2)})
        except Exception as e:
            logger.warning("DEM horizon extraction unavailable, using spherical horizon (0°): %s", e)
            horizon = np.zeros(360, dtype=np.float64)
            eval_site = site

        # 2. Ephemeris & topocentric geometry using true DEM observer position
        geo_samples, _ = self.geometry_engine.compute_mission_window_time_series(
            site=eval_site,
            start_utc=start_utc,
            duration_days=duration_days,
            step_minutes=step_minutes,
        )

        # 3. Terrain-aware visibility
        return compute_terrain_visibility_time_series(
            geometry_samples=geo_samples,
            horizon_profile=horizon,
            site_name=eval_site.name,
        )

    def evaluate_continuity_envelope(
        self,
        site: LandingSite,
        start_utc: str,
        duration_days: float = 14.0,
        step_minutes: float = 30.0,
        max_slope_deg: float = 15.0,
        horizon_radius_km: float = 20.0,
        requirements: Optional[MissionRequirements] = None,
    ) -> ContinuityEnvelopeResult:
        """
        Runs the Mission Continuity Envelope robustness analysis.

        Uses configurable slope constraints and terrain horizon radius
        to maintain consistency with master mission evaluation.
        """

        return self.continuity_engine.evaluate_continuity_envelope(
            site=site,
            start_utc=start_utc,
            duration_days=duration_days,
            step_minutes=step_minutes,
            max_slope_deg=max_slope_deg,
            horizon_radius_km=horizon_radius_km,
            requirements=requirements,
        )

    def evaluate_candidate_mission(
        self,
        site: LandingSite,
        start_utc: str,
        duration_days: float = 14.0,
        step_minutes: float = 15.0,
        max_slope_deg: float = 15.0,
        requirements: Optional[MissionRequirements] = None,
        horizon_radius_km: float = 20.0,
    ) -> MissionEvaluationResponse:
        """
        Master scientific evaluation of a candidate landing site against mission constraints.
        Integrates DEM topography, SPICE ephemerides, ray-cast horizon, and continuity metrics.
        Strictly enforces that composite scores never override failed mission constraints.
        """
        reqs = requirements or MissionRequirements()
        total_mission_hours = duration_days * 24.0

        # 1. Geographic DEM Coverage Boundary Check (-80°S polar DEM limit)
        if site.latitude_deg > -80.0:
            outside_constraints = [
                ConstraintEvaluation(
                    name="DEM Coverage",
                    description="South polar DEM coverage limit (-80.0°S)",
                    operator="<=",
                    threshold=-80.0,
                    actual_value=site.latitude_deg,
                    unit="deg",
                    margin=round(-80.0 - site.latitude_deg, 4),
                    status="OUTSIDE_COVERAGE",
                    is_binding=True,
                )
            ]
            empty_summary = VisibilitySummary(
                site_name=site.name,
                start_utc=start_utc,
                duration_days=duration_days,
                step_minutes=step_minutes,
                total_samples=0,
                sunlit_hours=0.0,
                partial_hours=0.0,
                dark_hours=0.0,
                sunlit_fraction=0.0,
                longest_sunlit_window_hours=0.0,
                longest_dark_period_hours=0.0,
                earth_visible_hours=0.0,
                earth_visible_fraction=0.0,
                longest_earth_visible_window_hours=0.0,
                longest_earth_blackout_hours=0.0,
                simultaneous_hours=0.0,
                simultaneous_fraction=0.0,
                longest_simultaneous_window_hours=0.0,
                intervals=[],
            )
            empty_continuity = ContinuityMetrics(
                site_name=site.name,
                start_utc=start_utc,
                duration_days=duration_days,
                step_minutes=step_minutes,
                total_mission_hours=total_mission_hours,
                temporal_uncertainty_hours=round(step_minutes / 60.0, 3),
                total_sunlight_hours=0.0,
                sunlight_availability_pct=0.0,
                longest_continuous_illumination_hours=0.0,
                longest_continuous_darkness_hours=0.0,
                total_earth_visibility_hours=0.0,
                earth_visibility_pct=0.0,
                longest_continuous_earth_visibility_hours=0.0,
                longest_continuous_earth_blackout_hours=0.0,
                total_simultaneous_hours=0.0,
                simultaneous_availability_pct=0.0,
                longest_continuous_simultaneous_hours=0.0,
                day_to_night_transitions=0,
                night_to_day_transitions=0,
                comms_acquired_transitions=0,
                comms_lost_transitions=0,
                simultaneous_acquired_transitions=0,
                simultaneous_lost_transitions=0,
                explanation=f"Site latitude {site.latitude_deg:.4f}° is north of -80°S LOLA South Polar DEM boundary.",
            )
            return MissionEvaluationResponse(
                site=site,
                start_utc=start_utc,
                duration_days=duration_days,
                step_minutes=step_minutes,
                overall_classification="OUTSIDE_COVERAGE",
                is_feasible=False,
                composite_score=0.0,
                constraints=outside_constraints,
                binding_constraints=outside_constraints,
                continuity_metrics=empty_continuity,
                visibility_summary=empty_summary,
                identified_risks=[
                    f"Candidate location ({site.latitude_deg:.4f}°N, {site.longitude_deg:.4f}°E) lies outside the LOLA South Polar 80m DEM."
                ],
                recommendation="Candidate cannot be screened with south polar high-resolution DEM. Relocate within 80°S-90°S.",
                data_completeness="0% LOLA South Pole 80m DEM coverage.",
                provenance={
                    "dem_dataset": "LOLA South Polar DEM 80m/pixel (coverage: -80° to -90°)",
                    "spice_kernels": "DE440s, moon_pa_de440_200108, naif0012",
                },
            )

        # 2. DEM Ground-Truth Elevation, Point Slope, and Aspect
        elev_val, elev_status, _ = self.terrain_engine.get_elevation_safe(site.latitude_deg, site.longitude_deg)
        actual_elev = float(elev_val) if elev_status == "EVALUATED" and elev_val is not None else site.elevation_m

        pt_slope, pt_aspect, sa_status, _ = self.terrain_engine.compute_point_slope_aspect_safe(
            site.latitude_deg, site.longitude_deg
        )
        actual_slope = float(pt_slope) if sa_status == "EVALUATED" and pt_slope is not None else site.slope_deg
        actual_aspect = float(pt_aspect) if sa_status == "EVALUATED" and pt_aspect is not None else site.aspect_deg

        eval_site = site.model_copy(
            update={
                "elevation_m": round(actual_elev, 2),
                "slope_deg": round(actual_slope, 2),
                "aspect_deg": round(actual_aspect, 2),
            }
        )

        # 3. Horizon Profile & Terrain Visibility
        try:
            patch = self.terrain_engine.extract_patch(
                lat_deg=eval_site.latitude_deg,
                lon_deg=eval_site.longitude_deg,
                radius_km=horizon_radius_km,
            )
            horizon = compute_horizon_profile(patch=patch, n_azimuths=360, n_range_steps=250)
            obs_elev = patch.observer_elevation_m
            eval_site = eval_site.model_copy(update={"elevation_m": round(obs_elev, 2)})
        except Exception as e:
            logger.warning("DEM horizon extraction fallback: %s", e)
            horizon = np.zeros(360, dtype=np.float64)

        # Ephemeris geometry
        geo_samples, geo_summary = self.geometry_engine.compute_mission_window_time_series(
            site=eval_site,
            start_utc=start_utc,
            duration_days=duration_days,
            step_minutes=step_minutes,
        )

        # Visibility time-series
        vis_resp = compute_terrain_visibility_time_series(
            geometry_samples=geo_samples,
            horizon_profile=horizon,
            site_name=eval_site.name,
        )
        vis_summary = vis_resp.summary

        # Continuity Metrics
        cont_metrics = self.continuity_engine.compute_continuity_metrics(vis_resp)

        # 4. Rigorous Constraint Evaluation
        constraints: List[ConstraintEvaluation] = []

        # Constraint 1: Terrain Slope (Lander Stability)
        slope_margin = round(max_slope_deg - actual_slope, 2)
        slope_pass = slope_margin >= 0.0
        constraints.append(
            ConstraintEvaluation(
                name="Terrain Slope",
                description="Local terrain slope for lander stability and tip-over prevention",
                operator="<=",
                threshold=max_slope_deg,
                actual_value=round(actual_slope, 2),
                unit="deg",
                margin=slope_margin,
                status="EVALUATED_PASS" if slope_pass else "EVALUATED_FAIL",
                is_binding=False,
            )
        )

        # Constraint 2: Minimum Continuous Sunlight (Solar Power)
        sun_margin = round(cont_metrics.longest_continuous_illumination_hours - reqs.min_continuous_sunlight_hours, 2)
        sun_pass = sun_margin >= 0.0
        constraints.append(
            ConstraintEvaluation(
                name="Continuous Sunlight",
                description="Unbroken direct solar illumination window for prime mission surface ops",
                operator=">=",
                threshold=reqs.min_continuous_sunlight_hours,
                actual_value=cont_metrics.longest_continuous_illumination_hours,
                unit="hours",
                margin=sun_margin,
                status="EVALUATED_PASS" if sun_pass else "EVALUATED_FAIL",
                is_binding=False,
            )
        )

        # Constraint 3: Maximum Tolerable Darkness (Battery Survival)
        dark_margin = round(reqs.max_tolerable_dark_hours - cont_metrics.longest_continuous_darkness_hours, 2)
        dark_pass = dark_margin >= 0.0
        constraints.append(
            ConstraintEvaluation(
                name="Battery Darkness Limit",
                description="Maximum tolerable unbroken darkness period survived on internal battery",
                operator="<=",
                threshold=reqs.max_tolerable_dark_hours,
                actual_value=cont_metrics.longest_continuous_darkness_hours,
                unit="hours",
                margin=dark_margin,
                status="EVALUATED_PASS" if dark_pass else "EVALUATED_FAIL",
                is_binding=False,
            )
        )

        # Constraint 4: Direct Earth Communications Window
        comms_margin = round(
            cont_metrics.longest_continuous_earth_visibility_hours - reqs.min_continuous_earth_comms_hours, 2
        )
        comms_pass = comms_margin >= 0.0
        constraints.append(
            ConstraintEvaluation(
                name="Earth Communications",
                description="Minimum continuous direct-to-Earth line-of-sight communications window",
                operator=">=",
                threshold=reqs.min_continuous_earth_comms_hours,
                actual_value=cont_metrics.longest_continuous_earth_visibility_hours,
                unit="hours",
                margin=comms_margin,
                status="EVALUATED_PASS" if comms_pass else "EVALUATED_FAIL",
                is_binding=False,
            )
        )

        # Constraint 5: Simultaneous Power & Communications Window
        simul_margin = round(
            cont_metrics.longest_continuous_simultaneous_hours - reqs.min_simultaneous_operating_hours, 2
        )
        simul_pass = simul_margin >= 0.0
        constraints.append(
            ConstraintEvaluation(
                name="Simultaneous Power & Comms",
                description="Window where both solar arrays are illuminated and Earth LOS is unobstructed",
                operator=">=",
                threshold=reqs.min_simultaneous_operating_hours,
                actual_value=cont_metrics.longest_continuous_simultaneous_hours,
                unit="hours",
                margin=simul_margin,
                status="EVALUATED_PASS" if simul_pass else "EVALUATED_FAIL",
                is_binding=False,
            )
        )

        # 5. Binding Constraints Identification
        failed_constraints = [c for c in constraints if c.status == "EVALUATED_FAIL"]
        if failed_constraints:
            for c in failed_constraints:
                c.is_binding = True
            binding = failed_constraints
            is_feasible = False
        else:
            # All passed; find the one with the smallest normalized safety margin
            min_c = min(constraints, key=lambda c: c.margin / (c.threshold + 1e-4) if c.margin is not None else 1e9)
            min_c.is_binding = True
            binding = [min_c]
            is_feasible = True

        # 6. Overall Classification
        # Rule: Scores never override failed constraints.
        if is_feasible:
            classification = "FEASIBLE"
        else:
            # Check if conditional: slope passed and deficits are mild (margin deficits <= 25% of threshold)
            max_deficit_ratio = max(
                [-c.margin / (c.threshold + 1e-4) for c in failed_constraints if c.margin is not None and c.margin < 0],
                default=1.0,
            )
            if slope_pass and max_deficit_ratio <= 0.25:
                classification = "CONDITIONAL"
            else:
                classification = "INFEASIBLE"

        # 7. Normalized Composite Score (0 - 100)
        # Note: Composite score reflects operational suitability, but never overrides failure!
        slope_score = max(0.0, min(100.0, (1.0 - (actual_slope / max(max_slope_deg, 1.0))) * 100.0))
        sun_score = min(100.0, (cont_metrics.sunlight_availability_pct))
        earth_score = min(100.0, (cont_metrics.earth_visibility_pct))
        simul_score = min(100.0, (cont_metrics.simultaneous_availability_pct))
        raw_composite = 0.25 * slope_score + 0.35 * sun_score + 0.25 * earth_score + 0.15 * simul_score
        
        # Penalize if failed constraints exist
        if not is_feasible:
            composite_score = round(raw_composite * 0.5, 1)  # Penalized to prevent misleading optimism
        else:
            composite_score = round(raw_composite, 1)

        # 8. Identified Operational Risks
        risks: List[str] = []
        if geo_summary.earth_min_elevation_deg < 3.0:
            risks.append(
                f"Earth geometric elevation drops to {geo_summary.earth_min_elevation_deg:.1f}° "
                f"(below 3.0° safety mask), creating high vulnerability to terrain occultation during lunar libration cycles."
            )
        if actual_slope > 10.0:
            risks.append(
                f"Local terrain slope ({actual_slope:.1f}°) approaches lander tilt limit ({max_slope_deg:.1f}°), "
                f"requiring hazard detection and terrain-relative navigation (TRN) divert capability."
            )
        if cont_metrics.longest_continuous_darkness_hours > 0.8 * reqs.max_tolerable_dark_hours:
            risks.append(
                f"Longest continuous darkness period ({cont_metrics.longest_continuous_darkness_hours:.1f}h) "
                f"consumes >80% of battery endurance ({reqs.max_tolerable_dark_hours:.1f}h)."
            )
        if cont_metrics.comms_lost_transitions > 0:
            risks.append(
                f"Local topography produces {cont_metrics.comms_lost_transitions} Earth line-of-sight occultation events."
            )
        if not risks:
            risks.append("No critical topographic or geometric flight risks identified within evaluated mission envelope.")

        # 9. Recommendation
        if classification == "FEASIBLE":
            recommendation = (
                f"Site '{site.name}' satisfies all {len(constraints)} mission constraints with positive margins. "
                f"Primary binding constraint is '{binding[0].name}' (margin: +{binding[0].margin:.1f} {binding[0].unit}). "
                f"Recommended for detailed flight trajectory and precision landing envelope assessment."
            )
        elif classification == "CONDITIONAL":
            recommendation = (
                f"Site '{site.name}' is CONDITIONAL due to mild constraint deficits: "
                + ", ".join([f"{c.name} ({c.margin:.1f} {c.unit})" for c in failed_constraints])
                + ". Consider Phase 10 counterfactual recovery (e.g. launch date shift or local terrace relocation)."
            )
        else:
            recommendation = (
                f"Site '{site.name}' is INFEASIBLE under current mission requirements. "
                f"Critical violations: "
                + ", ".join([f"{c.name} ({c.margin:.1f} {c.unit})" for c in failed_constraints])
                + ". Requires operational requirement relaxation or alternative landing site."
            )

        data_completeness = (
            "LOLA South Polar DEM (80m/pixel resolution) + JPL SPICE DE440s / moon_pa_de440_200108. "
            "Suitable for regional landing site screening and horizon obstruction modeling. "
            "Not certified for sub-80m meter-scale boulder or micro-crater touchdown hazard clearance."
        )

        provenance = {
            "dem_dataset": "LOLA South Polar DEM 80m/pixel (coverage: -80° to -90°S)",
            "spice_ephemerides": "DE440s, moon_pa_de440_200108, naif0012",
            "time_step_minutes": step_minutes,
            "horizon_radius_km": horizon_radius_km,
            "horizon_azimuth_rays": 360,
        }

        return MissionEvaluationResponse(
            site=eval_site,
            start_utc=start_utc,
            duration_days=duration_days,
            step_minutes=step_minutes,
            overall_classification=classification,
            is_feasible=is_feasible,
            composite_score=composite_score,
            constraints=constraints,
            binding_constraints=binding,
            continuity_metrics=cont_metrics,
            visibility_summary=vis_summary,
            identified_risks=risks,
            recommendation=recommendation,
            data_completeness=data_completeness,
            provenance=provenance,
        )

    def evaluate_probabilistic_resilience(
        self,
        site: LandingSite,
        start_utc: str,
        duration_days: float = 14.0,
        step_minutes: float = 30.0,
        max_slope_deg: float = 15.0,
        requirements: Optional[MissionRequirements] = None,
        uncertainty_config: Optional[UncertaintyDistributionConfig] = None,
    ) -> ProbabilisticResilienceResponse:
        """
        Runs Monte Carlo probabilistic mission resilience and uncertainty propagation.
        """
        return self.resilience_engine.evaluate_resilience(
            site=site,
            start_utc=start_utc,
            duration_days=duration_days,
            step_minutes=step_minutes,
            max_slope_deg=max_slope_deg,
            requirements=requirements,
            uncertainty_config=uncertainty_config,
        )

    def plan_counterfactual_recovery(
        self,
        site: LandingSite,
        start_utc: str,
        duration_days: float = 14.0,
        step_minutes: float = 30.0,
        max_slope_deg: float = 15.0,
        requirements: Optional[MissionRequirements] = None,
    ) -> CounterfactualRecoveryResponse:
        """
        Executes counterfactual recovery analysis for infeasible/conditional candidate sites.
        """
        return self.recovery_engine.plan_recovery(
            site=site,
            start_utc=start_utc,
            duration_days=duration_days,
            step_minutes=step_minutes,
            max_slope_deg=max_slope_deg,
            requirements=requirements,
        )
