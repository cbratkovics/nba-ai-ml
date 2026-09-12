/**
 * Data layer: every figure on the site comes from a published JSON file.
 *
 * Files are fetched server-side from the public Hugging Face repos (the `resolve`
 * endpoint redirects to a CDN and only allows browser CORS from huggingface.co) with
 * ISR revalidation of one hour. There is no API server. A missing file (404) is a
 * normal state before the season starts and is returned as `null`, never faked.
 */

export const DATASET_REPO = 'cbratkovics/nba-game-logs'
export const MODEL_REPO = 'cbratkovics/nba-stat-predictor'
/** Model revision pinned in nba/config.py; the site shows the report published with it. */
export const MODEL_REVISION = 'fb427de136e1d6c4b591ae30cf30488f44935182'
export const REPLAY_SEASON = '2025-26'
export const REVALIDATE_SECONDS = 3600

export const DATASET_BASE = `https://huggingface.co/datasets/${DATASET_REPO}/resolve/main`
export const MODEL_BASE = `https://huggingface.co/${MODEL_REPO}/resolve/${MODEL_REVISION}`

export const LINKS = {
  dataset: `https://huggingface.co/datasets/${DATASET_REPO}`,
  model: `https://huggingface.co/${MODEL_REPO}`,
  github: 'https://github.com/cbratkovics/nba-ai-ml',
  reconciliation: 'https://github.com/cbratkovics/nba-ai-ml/blob/master/docs/reconciliation.md',
}

/** Paths read from the dataset repo. */
export const DATASET_PATHS = {
  latest: 'predictions/latest.json',
  rolling: 'predictions/rolling_metrics.json',
  replaySummary: `replay/${REPLAY_SEASON}/replay.json`,
  replayDaily: `replay/${REPLAY_SEASON}/daily_mae.json`,
  replaySample: (date: string) => `replay/${REPLAY_SEASON}/sample_${date}.json`,
}
/** Path read from the model repo. */
export const MODEL_PATHS = { metrics: 'metrics.json' }

export type Target = 'pts' | 'reb' | 'ast'
export const TARGETS: Target[] = ['pts', 'reb', 'ast']
export const TARGET_LABEL: Record<Target, string> = {
  pts: 'Points',
  reb: 'Rebounds',
  ast: 'Assists',
}

// ---------- shapes of the published files ----------

export interface SlatePrediction {
  player_id: number
  player_name: string
  team: string
  opponent: string
  home: boolean
  pred_pts: number
  pred_reb: number
  pred_ast: number
}

export interface LatestSlate {
  date: string
  model_revision: string
  dataset_revision: string
  generated_at: string
  n_games: number
  n_players: number
  predictions: SlatePrediction[]
}

export interface RollingLine {
  date: string
  n_predicted: number
  n_with_actuals: number
  n_missing_actuals: number
  mae: Record<Target, number | null>
  mae_restricted: Record<Target, number | null>
  n_restricted: number
  rolling_30d: { days: number; n: number; mae: Record<Target, number | null> }
}

export interface MetricValues {
  mae: number
  rmse: number
  r2: number
  n: number
}

export interface MetricsReport {
  generated_at: string
  git_sha: string
  dataset: { repo: string; version: string }
  split: {
    train_seasons: string[]
    holdout_season: string
    train_dates: { start: string; end: string }
    holdout_dates: { start: string; end: string }
    n_train_rows: number
    n_holdout_rows: number
    min_minutes: number
  }
  features: string[]
  metrics: Record<Target, Record<'model' | 'baseline_last10' | 'baseline_season', MetricValues>>
}

export interface ReplaySummary {
  season: string
  generated_at: string
  model_revision: string
  dataset_revision: string
  reference: { model_mae: Record<Target, number>; n: number }
  n_dates: number
  n_predicted: number
  n_with_actuals: number
  n_missing_actuals: number
  n_restricted: number
  mae_restricted: Record<Target, number>
  mae_unrestricted: Record<Target, number>
  unpredicted_actual_rows: { all: number; minutes_ge_min: number }
  diff_vs_metrics_json: Record<Target, number>
  tolerance: number
  passed: boolean
}

export interface ReplayDay {
  date: string
  n: number
  model: Record<Target, number>
  baseline_last10: Record<Target, number>
}

export interface ReplayDaily {
  season: string
  population: string
  n_dates: number
  first_date: string | null
  last_date: string | null
  sample_date: string | null
  sample_file: string | null
  days: ReplayDay[]
}

export interface SampleRow {
  game_id: string
  player_id: number
  player_name: string
  team: string
  opponent: string
  home: boolean
  pred_pts: number
  pred_reb: number
  pred_ast: number
  actual_pts: number | null
  actual_reb: number | null
  actual_ast: number | null
  minutes: number | null
  has_actual: boolean
}

export interface ReplaySample {
  date: string
  model_revision: string
  dataset_revision: string
  n_games: number
  n_players: number
  n_with_actuals: number
  rows: SampleRow[]
}

// ---------- fetching ----------

export interface Fetched<T> {
  data: T | null
  /** 200 when present, 404 when the file is not published yet. */
  status: number
  url: string
}

async function fetchText(url: string): Promise<{ text: string | null; status: number }> {
  const res = await fetch(url, { next: { revalidate: REVALIDATE_SECONDS } })
  if (res.status === 404) return { text: null, status: 404 }
  if (!res.ok) throw new Error(`fetch ${url} failed with ${res.status}`)
  return { text: await res.text(), status: res.status }
}

export async function fetchJson<T>(url: string): Promise<Fetched<T>> {
  const { text, status } = await fetchText(url)
  return { data: text === null ? null : (JSON.parse(text) as T), status, url }
}

/** rolling_metrics.json is JSON Lines: one object per line. */
export async function fetchJsonLines<T>(url: string): Promise<Fetched<T[]>> {
  const { text, status } = await fetchText(url)
  if (text === null) return { data: null, status, url }
  const lines = text
    .split('\n')
    .map((l) => l.trim())
    .filter(Boolean)
    .map((l) => JSON.parse(l) as T)
  return { data: lines, status, url }
}

export const getLatestSlate = () => fetchJson<LatestSlate>(`${DATASET_BASE}/${DATASET_PATHS.latest}`)
export const getRollingMetrics = () =>
  fetchJsonLines<RollingLine>(`${DATASET_BASE}/${DATASET_PATHS.rolling}`)
export const getReplaySummary = () =>
  fetchJson<ReplaySummary>(`${DATASET_BASE}/${DATASET_PATHS.replaySummary}`)
export const getReplayDaily = () => fetchJson<ReplayDaily>(`${DATASET_BASE}/${DATASET_PATHS.replayDaily}`)
export const getReplaySample = (date: string) =>
  fetchJson<ReplaySample>(`${DATASET_BASE}/${DATASET_PATHS.replaySample(date)}`)
export const getMetricsReport = () => fetchJson<MetricsReport>(`${MODEL_BASE}/${MODEL_PATHS.metrics}`)
