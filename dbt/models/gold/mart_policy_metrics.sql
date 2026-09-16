-- Grain: one row per (season, run_kind, model_revision, population, target): the decision
-- policy's numbers at the chosen threshold, recomputed from fct_decision_policy over rows
-- with a box score. The replay rows reconcile to reports/policy_<season>.json
-- (assert_policy_metrics_reconcile_to_policy_report); nightly rows are the same statistics
-- on the live season. Both causal baselines are reported next to the model: the coin flip
-- (0.5, with its 95% half width at n_resolved) and the season-mean sign on the same rows.
with resolved as (
    select * from {{ ref('fct_decision_policy') }} where has_actual
),

agg as (
    select
        season,
        run_kind,
        model_revision,
        population,
        target,
        max(threshold) as threshold,
        count(*) as n,
        count(*) filter (where decision in ('over', 'under')) as n_called,
        count(*) filter (where outcome in ('hit', 'miss')) as n_resolved,
        count(*) filter (where outcome = 'push') as n_push,
        count(*) filter (where outcome = 'hit') as n_hit,
        count(*) filter (where outcome = 'miss') as n_miss,
        count(*) filter (where outcome in ('hit', 'miss') and season_mean_outcome in ('hit', 'miss')) as season_mean_same_rows_n,
        count(*) filter (where outcome in ('hit', 'miss') and season_mean_outcome = 'hit') as season_mean_same_rows_hit,
        avg(case when within_band_50 then 1.0 else 0.0 end) filter (where within_band_50 is not null) as band_coverage_50,
        avg(case when within_band_80 then 1.0 else 0.0 end) filter (where within_band_80 is not null) as band_coverage_80
    from resolved
    group by season, run_kind, model_revision, population, target
),

rates as (
    select
        *,
        n_called / n as coverage,
        case when n_resolved > 0 then n_hit / n_resolved end as hit_rate,
        n_hit - n_miss as net_correct,
        case when n_resolved > 0 then 1.96 * sqrt(0.25 / n_resolved) end as coin_flip_half_width_95,
        case when season_mean_same_rows_n > 0 then season_mean_same_rows_hit / season_mean_same_rows_n end as season_mean_same_rows_hit_rate
    from agg
)

select
    cast(season as varchar) as season,
    cast(run_kind as varchar) as run_kind,
    cast(model_revision as varchar) as model_revision,
    cast(population as varchar) as population,
    cast(target as varchar) as target,
    cast(threshold as double) as threshold,
    cast(n as integer) as n,
    cast(n_called as integer) as n_called,
    cast(coverage as double) as coverage,
    cast(n_resolved as integer) as n_resolved,
    cast(n_push as integer) as n_push,
    cast(n_hit as integer) as n_hit,
    cast(hit_rate as double) as hit_rate,
    cast(net_correct as integer) as net_correct,
    cast(0.5 as double) as coin_flip_hit_rate,
    cast(coin_flip_half_width_95 as double) as coin_flip_half_width_95,
    cast(season_mean_same_rows_n as integer) as season_mean_same_rows_n,
    cast(season_mean_same_rows_hit_rate as double) as season_mean_same_rows_hit_rate,
    cast(
        hit_rate is not null and season_mean_same_rows_hit_rate is not null
        and hit_rate - 0.5 > coin_flip_half_width_95 and hit_rate > season_mean_same_rows_hit_rate
        as boolean
    ) as model_beats_both,
    cast(band_coverage_50 as double) as band_coverage_50,
    cast(band_coverage_80 as double) as band_coverage_80
from rates
