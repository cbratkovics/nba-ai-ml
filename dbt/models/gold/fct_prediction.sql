-- Grain: one row per (player_id, game_id, model_revision): every prediction with its box
-- score when one exists, the absolute errors of the model and of the last-10 baseline, and
-- the population the row belongs to (min10 = the writer's training-population flag, else
-- all; ADR-0009). Errors are null until the box score exists.
with predictions as (
    select * from {{ ref('slv_predictions') }}
),

residuals as (
    select * from {{ ref('slv_residuals') }}
)

select
    cast(p.player_id as bigint) as player_id,
    cast(p.game_id as varchar) as game_id,
    cast(p.model_revision as varchar) as model_revision,
    cast(p.feature_version as varchar) as feature_version,
    cast(p.dataset_revision as varchar) as dataset_revision,
    cast(p.run_kind as varchar) as run_kind,
    cast(p.game_date as date) as game_date,
    cast(p.season as varchar) as season,
    cast(p.player_name as varchar) as player_name,
    cast(p.team as varchar) as team,
    cast(p.opponent as varchar) as opponent,
    cast(p.home as boolean) as home,
    cast(p.pred_pts as double) as pred_pts,
    cast(p.pred_reb as double) as pred_reb,
    cast(p.pred_ast as double) as pred_ast,
    cast(p.pts_mean_last10 as double) as baseline_last10_pts,
    cast(p.reb_mean_last10 as double) as baseline_last10_reb,
    cast(p.ast_mean_last10 as double) as baseline_last10_ast,
    cast(r.actual_pts as double) as actual_pts,
    cast(r.actual_reb as double) as actual_reb,
    cast(r.actual_ast as double) as actual_ast,
    cast(r.minutes as double) as minutes,
    cast(coalesce(r.has_actual, false) as boolean) as has_actual,
    cast(
        p.pts_mean_last10 is not null and p.reb_mean_last10 is not null and p.ast_mean_last10 is not null as boolean
    ) as baseline_defined,
    cast(coalesce(r.in_metrics_population, false) as boolean) as in_metrics_population,
    cast(case when coalesce(r.in_metrics_population, false) then 'min10' else 'all' end as varchar) as population,
    cast(abs(r.actual_pts - p.pred_pts) as double) as abs_error_pts,
    cast(abs(r.actual_reb - p.pred_reb) as double) as abs_error_reb,
    cast(abs(r.actual_ast - p.pred_ast) as double) as abs_error_ast,
    cast(abs(r.actual_pts - p.pts_mean_last10) as double) as baseline_abs_error_pts,
    cast(abs(r.actual_reb - p.reb_mean_last10) as double) as baseline_abs_error_reb,
    cast(abs(r.actual_ast - p.ast_mean_last10) as double) as baseline_abs_error_ast
from predictions as p
left join residuals as r
    on p.player_id = r.player_id and p.game_id = r.game_id and p.model_revision = r.model_revision
