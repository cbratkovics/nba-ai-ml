-- The replay rows of mart_policy_metrics must reproduce reports/policy_<season>.json per
-- (population, target): the counts exactly, the rates and band coverages within
-- var('tol_policy') (the mart recomputes the same rows with the same thresholds, so the
-- tolerance is float noise; ADR-0015). Every published row must have a computed twin. Any
-- row returned is a disagreement. Empty until the artifact exists.
with published as (
    select * from {{ ref('brz_policy_report') }}
),

computed as (
    select *
    from {{ ref('mart_policy_metrics') }}
    where run_kind = 'replay' and season = '{{ var("holdout_season") }}'
)

select
    p.population,
    p.target,
    p.n as published_n,
    c.n as computed_n,
    p.threshold as published_threshold,
    c.threshold as computed_threshold,
    p.n_called as published_n_called,
    c.n_called as computed_n_called,
    p.n_hit as published_n_hit,
    c.n_hit as computed_n_hit,
    p.hit_rate as published_hit_rate,
    c.hit_rate as computed_hit_rate,
    p.season_mean_same_rows_hit_rate as published_season_mean_hit_rate,
    c.season_mean_same_rows_hit_rate as computed_season_mean_hit_rate,
    p.band_coverage_80 as published_band_coverage_80,
    c.band_coverage_80 as computed_band_coverage_80
from published as p
left join computed as c
    on p.population = c.population and p.target = c.target and p.model_revision = c.model_revision
where
    c.n is null
    or p.n <> c.n
    or p.threshold <> c.threshold
    or p.n_called <> c.n_called
    or p.n_resolved <> c.n_resolved
    or p.n_push <> c.n_push
    or p.n_hit <> c.n_hit
    or p.season_mean_same_rows_n <> c.season_mean_same_rows_n
    or p.season_mean_same_rows_n_hit <> c.season_mean_same_rows_n_hit
    or p.season_mean_same_rows_n_miss <> c.season_mean_same_rows_n_miss
    or p.season_mean_same_rows_n_tie <> c.season_mean_same_rows_n_tie
    or p.season_mean_same_rows_n_missing <> c.season_mean_same_rows_n_missing
    or abs(p.coverage - c.coverage) > {{ var('tol_policy') }}
    or abs(p.hit_rate - c.hit_rate) > {{ var('tol_policy') }}
    or abs(p.season_mean_same_rows_hit_rate - c.season_mean_same_rows_hit_rate) > {{ var('tol_policy') }}
    or abs(p.coin_flip_half_width_95 - c.coin_flip_half_width_95) > {{ var('tol_policy') }}
    or abs(p.band_coverage_50 - c.band_coverage_50) > {{ var('tol_policy') }}
    or abs(p.band_coverage_80 - c.band_coverage_80) > {{ var('tol_policy') }}
    or p.model_beats_both <> c.model_beats_both
