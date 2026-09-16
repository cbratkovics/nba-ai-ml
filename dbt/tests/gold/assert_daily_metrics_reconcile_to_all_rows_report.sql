-- Day by day, the replay's all-rows daily MAE (model and last-10 baseline) and row count in
-- the warehouse must equal reports/replay_all_rows_<season>.json within var('tol_replay').
with computed as (
    select season, game_date, target, n, model_mae, baseline_last10_mae
    from {{ ref('mart_daily_metrics') }}
    where run_kind = 'replay' and population = 'all'
)

select
    p.season,
    p.game_date,
    p.target,
    p.n as published_n,
    c.n as computed_n,
    p.model_mae as published_model_mae,
    c.model_mae as computed_model_mae,
    p.baseline_last10_mae as published_baseline_mae,
    c.baseline_last10_mae as computed_baseline_mae
from {{ ref('brz_replay_all_rows') }} as p
left join computed as c on p.season = c.season and p.game_date = c.game_date and p.target = c.target
where
    c.n is null
    or p.n <> c.n
    or abs(p.model_mae - c.model_mae) > {{ var('tol_replay') }}
    or abs(p.baseline_last10_mae - c.baseline_last10_mae) > {{ var('tol_replay') }}
