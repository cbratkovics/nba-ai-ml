-- Grain: one row per (player_id, game_id, model_revision): a prediction joined to its box
-- score by the pipeline (nightly residual files, run_kind = nightly) or by the replay
-- (run_kind = replay). has_actual is false when the player did not play or the game is not
-- ingested yet; in_metrics_population is the writer's own training-population flag.
-- Incremental by date with the same lookback as slv_game_logs.
{{
    config(
        materialized='incremental',
        incremental_strategy='delete+insert',
        unique_key=['player_id', 'game_id', 'model_revision'],
        on_schema_change='fail'
    )
}}

with unioned as (
    select 'nightly' as run_kind, * from {{ ref('brz_residuals') }}
    union all
    select 'replay' as run_kind, * from {{ ref('brz_replay_residuals') }}
)

select
    player_id,
    game_id,
    model_revision,
    cast('{{ var('feature_version') }}' as varchar) as feature_version,
    run_kind,
    date as game_date,
    {{ season_of('date') }} as season,
    player_name,
    team,
    opponent,
    home,
    pred_pts,
    pred_reb,
    pred_ast,
    actual_pts,
    actual_reb,
    actual_ast,
    resid_pts,
    resid_reb,
    resid_ast,
    minutes,
    pts_mean_last10,
    reb_mean_last10,
    ast_mean_last10,
    has_actual,
    game_ingested,
    in_metrics_population,
    dataset_revision,
    source_file
from unioned
{% if is_incremental() %}
where date >= (
    select coalesce(max(t.game_date), date '1900-01-01') - interval ({{ var('lookback_days') }}) day
    from {{ this }} as t
)
{% endif %}
