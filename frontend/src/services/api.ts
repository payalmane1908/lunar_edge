/**
 * LUNAR EDGE — Backend API Service Client
 * Direct REST communication with FastAPI backend.
 */

import {
  LandingSite,
  EpochGeometry,
  TimeSeriesSample,
  GeometrySummary,
  HorizonProfile,
  VisibilityTimeSeriesResponse,
  ContinuityEnvelopeResult,
  MissionRequirements,
  SystemHealth,
  MissionEvaluationResponse,
  ProbabilisticResilienceResponse,
  CounterfactualRecoveryResponse,
  UncertaintyDistributionConfig,
} from '../types/api';

const API_BASE = `${(import.meta.env.VITE_API_BASE_URL || '').replace(/\/+$/, '')}/api`;

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let errorDetail = 'Unknown API error';
    try {
      const errJson = await res.json();
      errorDetail = errJson.detail || errJson.message || JSON.stringify(errJson);
    } catch {
      errorDetail = `HTTP ${res.status}: ${res.statusText}`;
    }
    throw new Error(errorDetail);
  }
  return res.json();
}

export const api = {
  async getHealth(): Promise<SystemHealth> {
    const res = await fetch(`${API_BASE}/health`);
    return handleResponse<SystemHealth>(res);
  },

  async getCandidateSites(): Promise<Record<string, LandingSite>> {
    const res = await fetch(`${API_BASE}/sites`);
    return handleResponse<Record<string, LandingSite>>(res);
  },

  async getEpochGeometry(site: LandingSite, utc: string): Promise<EpochGeometry> {
    const res = await fetch(`${API_BASE}/geometry/epoch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ site, utc }),
    });
    return handleResponse<EpochGeometry>(res);
  },

  async getGeometryTimeSeries(
    site: LandingSite,
    start_utc: string,
    duration_days: number = 14.0,
    step_minutes: number = 15.0
  ): Promise<{ summary: GeometrySummary; samples: TimeSeriesSample[] }> {
    const res = await fetch(`${API_BASE}/geometry/timeseries`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        site,
        start_utc,
        duration_days,
        step_minutes,
      }),
    });
    return handleResponse<{ summary: GeometrySummary; samples: TimeSeriesSample[] }>(res);
  },

  async getElevation(
    latitude_deg: number,
    longitude_deg: number
  ): Promise<{ latitude_deg: number; longitude_deg: number; elevation_m: number; polar_stereo_x_m: number; polar_stereo_y_m: number }> {
    const res = await fetch(`${API_BASE}/terrain/elevation`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ latitude_deg, longitude_deg }),
    });
    return handleResponse(res);
  },

  async getHorizon(
    site: LandingSite,
    radius_km: number = 20.0,
    n_azimuths: number = 360
  ): Promise<HorizonProfile> {
    const res = await fetch(`${API_BASE}/terrain/horizon`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        site,
        radius_km,
        n_azimuths,
      }),
    });
    return handleResponse<HorizonProfile>(res);
  },

  async evaluateVisibility(
    site: LandingSite,
    start_utc: string,
    duration_days: number = 14.0,
    step_minutes: number = 15.0,
    horizon_radius_km: number = 20.0
  ): Promise<VisibilityTimeSeriesResponse> {
    const res = await fetch(`${API_BASE}/visibility/evaluate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        site,
        start_utc,
        duration_days,
        step_minutes,
        horizon_radius_km,
      }),
    });
    return handleResponse<VisibilityTimeSeriesResponse>(res);
  },

  async evaluateContinuityEnvelope(
    site: LandingSite,
    start_utc: string,
    duration_days: number = 14.0,
    step_minutes: number = 30.0,
    requirements?: MissionRequirements
  ): Promise<ContinuityEnvelopeResult> {
    const res = await fetch(`${API_BASE}/continuity/envelope`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        site,
        start_utc,
        duration_days,
        step_minutes,
        requirements,
      }),
    });
    return handleResponse<ContinuityEnvelopeResult>(res);
  },

  async discoverOpportunities(
    req: {
      mission_name: string;
      target_region: string;
      bounds?: { lat_min: number; lat_max: number; lon_min: number; lon_max: number };
      start_utc: string;
      duration_days: number;
      step_minutes?: number;
      requirements: MissionRequirements;
      max_slope_deg?: number;
    }
  ): Promise<import('../types/api').OpportunityDiscoveryResponse> {
    const res = await fetch(`${API_BASE}/discovery/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    });
    return handleResponse<import('../types/api').OpportunityDiscoveryResponse>(res);
  },

  async evaluateMission(
    site: LandingSite,
    start_utc: string,
    duration_days: number = 14.0,
    step_minutes: number = 15.0,
    max_slope_deg: number = 15.0,
    requirements?: MissionRequirements,
    horizon_radius_km: number = 20.0
  ): Promise<MissionEvaluationResponse> {
    const res = await fetch(`${API_BASE}/mission/evaluate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        site,
        start_utc,
        duration_days,
        step_minutes,
        max_slope_deg,
        requirements,
        horizon_radius_km,
      }),
    });
    return handleResponse<MissionEvaluationResponse>(res);
  },

  async evaluateResilience(
    site: LandingSite,
    start_utc: string,
    duration_days: number = 14.0,
    step_minutes: number = 30.0,
    max_slope_deg: number = 15.0,
    requirements?: MissionRequirements,
    uncertainty_config?: UncertaintyDistributionConfig
  ): Promise<ProbabilisticResilienceResponse> {
    const res = await fetch(`${API_BASE}/mission/resilience`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        site,
        start_utc,
        duration_days,
        step_minutes,
        max_slope_deg,
        requirements,
        uncertainty_config,
      }),
    });
    return handleResponse<ProbabilisticResilienceResponse>(res);
  },

  async planRecovery(
    site: LandingSite,
    start_utc: string,
    duration_days: number = 14.0,
    step_minutes: number = 30.0,
    max_slope_deg: number = 15.0,
    requirements?: MissionRequirements
  ): Promise<CounterfactualRecoveryResponse> {
    const res = await fetch(`${API_BASE}/mission/recovery`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        site,
        start_utc,
        duration_days,
        step_minutes,
        max_slope_deg,
        requirements,
      }),
    });
    return handleResponse<CounterfactualRecoveryResponse>(res);
  },
};
