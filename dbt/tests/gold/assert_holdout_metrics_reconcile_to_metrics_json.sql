-- The training-population rows must agree with reports/metrics.json within
-- var('tol_metrics_json') for the model MAE and the last-10 baseline MAE. The populations are
-- not identical: metrics.json scores every holdout row with both baselines (22,244), the replay
-- only the rows the roster rule slated (22,075; the 169 others are post-trade debuts and
-- players absent from their team's previous ten games), so exact equality is impossible and
-- the observed gap (+0.0021 pts, +0.0008 reb, +0.0010 ast on the model) is recorded in
-- ADR-0011. n is checked to within 1%.
with computed as (
    select season, population, target, n, model_mae, baseline_last10_mae
    from {{ ref('mart_holdout_metrics') }}
    where run_kind = 'replay' and population = 'min10'
),

published as (
    select
        season,
        target,
        max(case when predictor = 'model' then n end) as n,
        max(case when predictor = 'model' then mae end) as model_mae,
        max(case when predictor = 'baseline_last10' then mae end) as baseline_last10_mae
    from {{ ref('brz_metrics') }}
    group by season, target
)

select
    p.season,
    p.target,
    p.n as published_n,
    c.n as computed_n,
    p.model_mae as published_model_mae,
    c.model_mae as computed_model_mae,
    p.baseline_last10_mae as published_baseline_mae,
    c.baseline_last10_mae as computed_baseline_mae
from published as p
left join computed as c on p.season = c.season and p.target = c.target
where
    c.n is null
    or abs(p.n - c.n) > 0.01 * p.n
    or abs(p.model_mae - c.model_mae) > {{ var('tol_metrics_json') }}
    or abs(p.baseline_last10_mae - c.baseline_last10_mae) > {{ var('tol_metrics_json') }}
