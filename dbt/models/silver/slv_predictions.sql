-- Grain: one row per (player_id, game_id, model_revision): every prediction the pipeline made,
-- from the nightly slate files (run_kind = nightly) and from the holdout-season replay
-- (run_kind = replay, whose prediction columns live in the replay residual files).
-- Incremental by date with the same lookback as slv_game_logs; a nightly date's file replaces
-- its rows. The unique test on the grain is what the writers only imply through the roster
-- rule (one team per player per date).
{{
    config(
        materialized='incremental',
        incremental_strategy='delete+insert',
        unique_key=['player_id', 'game_id', 'model_revision'],
        on_schema_change='fail'
    )
}}

with nightly as (
    select
        'nightly' as run_kind,
        date as game_date,
        game_id,
        player_id,
        player_name,
        team,
        opponent,
        home,
        pred_pts,
        pred_reb,
        pred_ast,
        pts_mean_last10,
        reb_mean_last10,
        ast_mean_last10,
        pts_mean_season,
        reb_mean_season,
        ast_mean_season,
        games_played_season,
        minutes_mean_last10,
        input_staleness_days,
        tip_off_utc,
        event_id,
        schedule_source,
        cast(generated_at as timestamp) as prediction_generated_at,
        model_revision,
        dataset_revision,
        source_file
    from {{ ref('brz_predictions') }}
),

replay as (
    select
        'replay' as run_kind,
        date as game_date,
        game_id,
        player_id,
        player_name,
        team,
        opponent,
        home,
        pred_pts,
        pred_reb,
        pred_ast,
        pts_mean_last10,
        reb_mean_last10,
        ast_mean_last10,
        cast(null as double) as pts_mean_season,
        cast(null as double) as reb_mean_season,
        cast(null as double) as ast_mean_season,
        cast(null as integer) as games_played_season,
        cast(null as double) as minutes_mean_last10,
        cast(null as double) as input_staleness_days,
        cast(null as timestamp) as tip_off_utc,
        cast(null as varchar) as event_id,
        cast(null as varchar) as schedule_source,
        cast(null as timestamp) as prediction_generated_at,
        model_revision,
        dataset_revision,
        source_file
    from {{ ref('brz_replay_residuals') }}
),

unioned as (
    select * from nightly
    union all
    select * from replay
)

select
    player_id,
    game_id,
    model_revision,
    cast('{{ var('feature_version') }}' as varchar) as feature_version,
    run_kind,
    game_date,
    {{ season_of('game_date') }} as season,
    player_name,
    team,
    opponent,
    home,
    pred_pts,
    pred_reb,
    pred_ast,
    pts_mean_last10,
    reb_mean_last10,
    ast_mean_last10,
    pts_mean_season,
    reb_mean_season,
    ast_mean_season,
    games_played_season,
    minutes_mean_last10,
    input_staleness_days,
    tip_off_utc,
    event_id,
    schedule_source,
    prediction_generated_at,
    dataset_revision,
    source_file
from unioned
{% if is_incremental() %}
where game_date >= (
        select coalesce(max(t.game_date), date '1900-01-01') - interval ({{ var('lookback_days') }}) day
        from {{ this }} as t
    )
{% endif %}
