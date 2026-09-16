-- Grain: one row per (season, run_kind, model_revision, population, target, threshold): the
-- coverage curve, i.e. the policy's coverage and hit rate at every grid threshold, with the
-- season-mean sign on the same rows and as a policy with its own threshold. Thresholds are
-- the artifact's grid (brz_policy_curve); the replay rows reconcile point for point to
-- reports/policy_<season>.json (assert_policy_sweep_reconciles_to_policy_report).
with resolved as (
    select
        season,
        run_kind,
        model_revision,
        population,
        target,
        edge,
        baseline_season - baseline_last10 as season_edge,
        sign(actual - baseline_last10) as side
    from {{ ref('fct_decision_policy') }}
    where has_actual
),

grid as (
    select distinct target, threshold from {{ ref('brz_policy_curve') }}
),

scored as (
    select
        r.season,
        r.run_kind,
        r.model_revision,
        r.population,
        r.target,
        g.threshold,
        abs(r.edge) > g.threshold as called,
        abs(r.edge) > g.threshold and r.side <> 0 as resolved,
        abs(r.edge) > g.threshold and r.side <> 0 and sign(r.edge) = r.side as hit,
        abs(r.edge) > g.threshold and r.side = 0 as push,
        abs(r.edge) > g.threshold and r.side <> 0 and r.season_edge is not null and r.season_edge <> 0 as sm_same,
        abs(r.edge) > g.threshold and r.side <> 0 and r.season_edge is not null and sign(r.season_edge) = r.side as sm_same_hit,
        r.season_edge is not null and abs(r.season_edge) > g.threshold as sm_called,
        r.season_edge is not null and abs(r.season_edge) > g.threshold and r.side <> 0 as sm_resolved,
        r.season_edge is not null and abs(r.season_edge) > g.threshold and r.side <> 0 and sign(r.season_edge) = r.side as sm_hit
    from resolved as r
    inner join grid as g on r.target = g.target
),

agg as (
    select
        season,
        run_kind,
        model_revision,
        population,
        target,
        threshold,
        count(*) as n,
        count(*) filter (where called) as n_called,
        count(*) filter (where resolved) as n_resolved,
        count(*) filter (where push) as n_push,
        count(*) filter (where hit) as n_hit,
        count(*) filter (where sm_same) as season_mean_same_rows_n,
        count(*) filter (where sm_same_hit) as season_mean_same_rows_hit,
        count(*) filter (where sm_called) as season_mean_own_n_called,
        count(*) filter (where sm_resolved) as season_mean_own_resolved,
        count(*) filter (where sm_hit) as season_mean_own_hit
    from scored
    group by season, run_kind, model_revision, population, target, threshold
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
    cast(n_called / n as double) as coverage,
    cast(n_resolved as integer) as n_resolved,
    cast(n_push as integer) as n_push,
    cast(n_hit as integer) as n_hit,
    cast(case when n_resolved > 0 then n_hit / n_resolved end as double) as hit_rate,
    cast(n_hit - (n_resolved - n_hit) as integer) as net_correct,
    cast(season_mean_same_rows_n as integer) as season_mean_same_rows_n,
    cast(case when season_mean_same_rows_n > 0 then season_mean_same_rows_hit / season_mean_same_rows_n end as double) as season_mean_same_rows_hit_rate,
    cast(season_mean_own_n_called as integer) as season_mean_own_n_called,
    cast(case when season_mean_own_resolved > 0 then season_mean_own_hit / season_mean_own_resolved end as double) as season_mean_own_hit_rate
from agg
