"""
LUNAR EDGE — SPICE Ephemeris & Topocentric Geometry Service

Provides high-fidelity, DE440-consistent celestial geometry for observers on the lunar surface.

Scientific Foundation:
  - Ephemeris: JPL DE440s (1849–2150 CE) via de440s.bsp
  - Lunar Frame: MOON_ME (Mean Earth/Polar Axis) defined via moon_de440_250416.tf
    and associated via moon_assoc_me.tf
  - Dynamic Lunar Orientation: moon_pa_de440_200625.bpc (libration-accurate)
  - Physical Constants: IAU triaxial radii via pck00011.tpc
  - Aberration Correction: 'CN+S' (Converged Newtonian light-time + stellar aberration)
  - Reference Horizon: Spherical lunar reference surface (IAU Moon 2000, R=1,737.4 km)
  - Azimuth Convention: 0° = North, 90° = East, 180° = South, 270° = West (clockwise)
  - Elevation Convention: Degrees positive above the local horizontal tangent plane
  - South Pole Singularity: At lat = -90°, North is defined along the Prime Meridian
    (+X_MOON_ME, toward mean sub-Earth direction).
"""

import threading
from pathlib import Path
from typing import Optional, Union
import numpy as np
import spiceypy as spice

from app.utils.coordinate_utils import (
    MOON_RADIUS_METERS,
    MOON_RADIUS_KM,
    selenographic_to_cartesian_moon_me,
    compute_topocentric_basis,
    vector_to_azel,
    compute_solar_incidence,
)
from app.utils.date_utils import normalize_to_spice_utc

# Kernel directory relative to this file
KERNELS_DIR = Path(__file__).parent.parent.parent / "data" / "kernels"

# Ordered list of DE440-consistent kernels required for lunar surface operations
REQUIRED_KERNELS = [
    "naif0012.tls",               # Leap seconds (UTC <-> ET)
    "de440s.bsp",                 # JPL DE440s ephemeris (Sun, Earth, Moon positions)
    "pck00011.tpc",               # IAU planetary constants & radii
    "moon_pa_de440_200625.bpc",   # Lunar dynamic orientation (MOON_PA frame)
    "moon_de440_250416.tf",       # Lunar frame kernel (DE440 frames)
    "moon_assoc_me.tf",           # MOON_ME alias association
    "earth_latest_high_prec.bpc", # High-accuracy Earth orientation (EOP)
]

# Physical radii in kilometres for angular diameter calculations
SUN_RADIUS_KM = 696_000.0
EARTH_EQUATORIAL_RADIUS_KM = 6_378.1366


class SPICEService:
    """
    Thread-safe SPICE Ephemeris and Topocentric Geometry Service.
    Loads kernels once into CSPICE memory pool and caches state.
    Uses re-entrant lock (RLock) to prevent deadlocks during instantiation.
    """
    _instance: Optional["SPICEService"] = None
    _lock = threading.RLock()
    _kernels_loaded = False

    def __new__(cls) -> "SPICEService":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def load_kernels(self, force_reload: bool = False) -> None:
        """Load all required kernels into the SPICE pool."""
        with self._lock:
            if self._kernels_loaded and not force_reload:
                return

            # Verify every required kernel exists
            missing = [k for k in REQUIRED_KERNELS if not (KERNELS_DIR / k).exists()]
            if missing:
                raise FileNotFoundError(
                    f"Required SPICE kernels missing from {KERNELS_DIR}: {missing}. "
                    "Run the kernel acquisition script first."
                )

            # Clear previous pool if reloading
            if self._kernels_loaded and force_reload:
                spice.kclear()

            # Load in documented order
            for kernel_name in REQUIRED_KERNELS:
                kernel_path = str(KERNELS_DIR / kernel_name)
                spice.furnsh(kernel_path)

            self._kernels_loaded = True

    def utc_to_et(self, utc_str: str) -> float:
        """Convert ISO UTC string (e.g. '2026-01-15T12:00:00Z') to SPICE Ephemeris Time (ET)."""
        with self._lock:
            self.load_kernels()
            normalized_utc = normalize_to_spice_utc(utc_str)
            try:
                return float(spice.str2et(normalized_utc))
            except Exception as e:
                raise ValueError(f"Invalid UTC timestamp string '{utc_str}': {e}") from e

    def et_to_utc(self, et: float, fmt: str = "ISOC") -> str:
        """Convert SPICE Ephemeris Time (ET) to ISO UTC string."""
        with self._lock:
            self.load_kernels()
            return str(spice.et2utc(et, fmt, 3))

    def compute_body_topocentric(
        self,
        body: str,
        lat_deg: float,
        lon_deg: float,
        elevation_m: float,
        et: float,
        abcorr: str = "CN+S",
    ) -> dict:
        """
        Compute topocentric azimuth, elevation, distance, and apparent angular radius
        of a celestial body ('SUN' or 'EARTH') as seen from a lunar surface location.

        Returns dict with:
          - azimuth_deg: Azimuth in degrees [0, 360), clockwise from North
          - elevation_deg: Geometric elevation in degrees [-90, +90], positive above local horizon
          - distance_km: Geometric observer-to-body distance in kilometres
          - angular_radius_deg: Apparent angular semi-diameter in degrees
          - angular_diameter_deg: Apparent angular diameter in degrees
          - light_time_s: One-way light time from target to observer in seconds
          - topocentric_vector_km: [East, North, Up] vector in local frame
        """
        with self._lock:
            self.load_kernels()

            # 1. Observer position in MOON_ME frame (km)
            obs_pos_km = selenographic_to_cartesian_moon_me(lat_deg, lon_deg, elevation_m)

            # 2. Topocentric orthonormal basis vectors in MOON_ME coordinates
            up, north, east = compute_topocentric_basis(lat_deg, lon_deg)

            # 3. Target position relative to Moon center in MOON_ME frame (km)
            body_id = body.upper()
            target_state, lt = spice.spkpos(body_id, et, "MOON_ME", abcorr, "MOON")

            # 4. Observer-to-target relative vector in MOON_ME (km)
            rel_vec = target_state - obs_pos_km

            # 5. Project onto topocentric horizon frame using centralized utility
            azimuth_deg, elevation_deg, distance_km = vector_to_azel(rel_vec, up, north, east)

            # 6. Apparent angular radius calculation: sin(theta) = Body_Radius / Distance
            if body_id == "SUN":
                body_radius_km = SUN_RADIUS_KM
            elif body_id == "EARTH":
                body_radius_km = EARTH_EQUATORIAL_RADIUS_KM
            else:
                radii = spice.bodvrd(body_id, "RADII", 3)[1]
                body_radius_km = float(np.mean(radii))

            sin_ang_rad = min(1.0, body_radius_km / max(1e-6, distance_km))
            angular_radius_deg = float(np.degrees(np.arcsin(sin_ang_rad)))

            # Sub-body point coordinates on Moon (selenographic lat/lon)
            try:
                sub_pt, _, _ = spice.subpnt("Near point: ellipsoid", "MOON", et, "MOON_ME", abcorr, body_id)
                _, sub_lon_rad, sub_lat_rad = spice.reclat(sub_pt)
                sub_lat_deg = float(np.degrees(sub_lat_rad))
                sub_lon_deg = float(np.degrees(sub_lon_rad))
            except Exception:
                sub_lat_deg = None
                sub_lon_deg = None

            return {
                "body": body_id,
                "azimuth_deg": azimuth_deg,
                "elevation_deg": elevation_deg,
                "distance_km": distance_km,
                "angular_radius_deg": angular_radius_deg,
                "angular_diameter_deg": angular_radius_deg * 2.0,
                "light_time_s": float(lt),
                "sub_lat_deg": sub_lat_deg,
                "sub_lon_deg": sub_lon_deg,
                "topocentric_vector_km": {
                    "east": float(np.dot(rel_vec, east)),
                    "north": float(np.dot(rel_vec, north)),
                    "up": float(np.dot(rel_vec, up)),
                },
            }

    def compute_subpnt(self, body: str, et: float) -> tuple[float, float]:
        """
        Compute the sub-body selenographic latitude and longitude on the Moon in degrees.
        Uses MOON_ME reference frame and IAU Moon triaxial ellipsoid.
        """
        with self._lock:
            self.load_kernels()
            body_id = body.upper()
            pt, _, _ = spice.subpnt("Near point: ellipsoid", "MOON", et, "MOON_ME", "CN+S", body_id)
            _, lon_rad, lat_rad = spice.reclat(pt)
            return float(np.degrees(lat_rad)), float(np.degrees(lon_rad))

    def compute_earth_libration(self, et: float) -> dict:
        """
        Compute lunar optical and physical libration angles as seen from Earth.
        Returns sub-Earth latitude (latitude libration) and longitude (longitude libration).
        """
        sub_lat, sub_lon = self.compute_subpnt("EARTH", et)
        return {
            "et": et,
            "utc": self.et_to_utc(et),
            "libration_lat_deg": sub_lat,
            "libration_lon_deg": sub_lon,
            "sub_earth_point": {
                "latitude_deg": sub_lat,
                "longitude_deg": sub_lon,
            },
        }

    def compute_solar_and_earth_geometry(
        self,
        lat_deg: float,
        lon_deg: float,
        elevation_m: float,
        et: float,
        slope_deg: float = 0.0,
        aspect_deg: float = 0.0,
    ) -> dict:
        """
        Compute simultaneous Sun and Earth geometry at an epoch,
        including solar incidence angle on local terrain slope.
        """
        sun = self.compute_body_topocentric("SUN", lat_deg, lon_deg, elevation_m, et)
        earth = self.compute_body_topocentric("EARTH", lat_deg, lon_deg, elevation_m, et)

        solar_incidence = compute_solar_incidence(
            sun["elevation_deg"],
            sun["azimuth_deg"],
            slope_deg,
            aspect_deg,
        )

        return {
            "et": et,
            "utc": self.et_to_utc(et),
            "sun": sun,
            "earth": earth,
            "solar_incidence": solar_incidence,
        }

    def compute_time_series(
        self,
        lat_deg: float,
        lon_deg: float,
        elevation_m: float,
        start_utc: str,
        duration_days: float,
        step_minutes: float = 15.0,
        slope_deg: float = 0.0,
        aspect_deg: float = 0.0,
    ) -> list[dict]:
        """
        Compute Sun and Earth geometry time-series over a mission window.
        Optimized by pre-computing static observer position and basis vectors once.
        """
        with self._lock:
            self.load_kernels()

            normalized_start = normalize_to_spice_utc(start_utc)
            start_et = float(spice.str2et(normalized_start))
            total_seconds = duration_days * 86400.0
            step_seconds = step_minutes * 60.0
            n_steps = int(np.ceil(total_seconds / step_seconds)) + 1

            et_array = np.linspace(start_et, start_et + total_seconds, n_steps)

            # Precompute static observer position and basis vectors
            obs_pos_km = selenographic_to_cartesian_moon_me(lat_deg, lon_deg, elevation_m)
            up, north, east = compute_topocentric_basis(lat_deg, lon_deg)

            records = []
            for et in et_array:
                # 1. Sun relative vector
                sun_state, _ = spice.spkpos("SUN", et, "MOON_ME", "CN+S", "MOON")
                v_sun = sun_state - obs_pos_km
                az_sun, el_sun, d_sun = vector_to_azel(v_sun, up, north, east)
                ang_rad_sun = float(np.degrees(np.arcsin(min(1.0, SUN_RADIUS_KM / max(1e-6, d_sun)))))

                # 2. Earth relative vector
                earth_state, _ = spice.spkpos("EARTH", et, "MOON_ME", "CN+S", "MOON")
                v_earth = earth_state - obs_pos_km
                az_earth, el_earth, d_earth = vector_to_azel(v_earth, up, north, east)
                ang_rad_earth = float(np.degrees(np.arcsin(min(1.0, EARTH_EQUATORIAL_RADIUS_KM / max(1e-6, d_earth)))))

                # 3. Solar incidence
                solar_inc = compute_solar_incidence(el_sun, az_sun, slope_deg, aspect_deg)

                records.append({
                    "et": float(et),
                    "utc": str(spice.et2utc(et, "ISOC", 3)),
                    "sun_azimuth_deg": az_sun,
                    "sun_elevation_deg": el_sun,
                    "sun_distance_km": d_sun,
                    "sun_angular_radius_deg": ang_rad_sun,
                    "sun_angular_diameter_deg": ang_rad_sun * 2.0,
                    "earth_azimuth_deg": az_earth,
                    "earth_elevation_deg": el_earth,
                    "earth_distance_km": d_earth,
                    "earth_angular_radius_deg": ang_rad_earth,
                    "earth_angular_diameter_deg": ang_rad_earth * 2.0,
                    "solar_incidence_angle_deg": solar_inc["incidence_angle_deg"],
                    "solar_cosine_incidence": solar_inc["cosine_incidence"],
                    # IMPORTANT SCIENTIFIC DISTINCTION:
                    # Geometric visibility (above reference sphere horizon) is NOT terrain visibility.
                    # Terrain obstruction will be evaluated by Module C (Visibility Engine).
                    "is_sun_above_geometric_horizon": el_sun > 0.0,
                    "is_sun_disc_partially_above_geometric_horizon": el_sun > -ang_rad_sun,
                    "is_earth_above_geometric_horizon": el_earth > 0.0,
                    "is_earth_disc_partially_above_geometric_horizon": el_earth > -ang_rad_earth,
                })

            return records


# Global singleton instance
spice_service = SPICEService()
