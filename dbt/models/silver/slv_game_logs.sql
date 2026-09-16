-- Grain: one row per (player_id, game_id); the canonical game-log schema, typed, one row per
-- player who played (the ingest drops DNP rows before publishing).
--
-- Materialisation: incremental (delete+insert on the grain key). The largest table in the
-- warehouse and one whose transformation is local to its own grain, so appending by game
-- date is safe. History-dependent columns (prior games, the population flag) live in gold
-- (fct_player_game), never here, so an incremental run needs no history.
--
-- Lookback: the Kaggle dump restates recent box scores and the daily ingest re-reads the
-- newest `DAILY_LOOKBACK_DAYS` (7) days, so an incremental run reprocesses every row whose
-- game date is within var('lookback_days') (14 = 7 + 7) of the newest date already loaded,
-- and delete+insert replaces those rows. A restatement older than the lookback is only
-- picked up by a full refresh. mart_restatement_lag records the lag the ingest actually
-- observed and tests/silver/assert_lookback_covers_restatement_lag.sql fails when the
-- lookback no longer covers it (ADR-0004: recalibrated after 30 in-season daily reports).
--
-- Full-refresh policy: `dbt build --full-refresh` (warehouse.yml weekly dispatch) at the
-- start of a season, after any change to this model's SQL or columns (on_schema_change =
-- fail), or after a restatement older than the lookback. tests/test_dbt_incremental.py
-- proves a full refresh and an incremental run over the same input produce identical rows.
--
-- Deduplication rule: the publisher enforces uniqueness on (player_id, game_id); if two
-- files ever carry the same key, the row with the higher minutes wins, ties by source_file.
{{
    config(
        materialized='incremental',
        incremental_strategy='delete+insert',
        unique_key=['player_id', 'game_id'],
        on_schema_change='fail'
    )
}}

with in_scope as (
    select *
    from {{ ref('brz_game_logs') }}
    {% if is_incremental() %}
    where game_date >= (
        select coalesce(max(t.game_date), date '1900-01-01') - interval ({{ var('lookback_days') }}) day
        from {{ this }} as t
    )
    {% endif %}
),

ranked as (
    select
        *,
        row_number() over (
            partition by player_id, game_id
            order by minutes desc nulls last, source_file asc
        ) as dedup_rank
    from in_scope
)

select
    player_id,
    game_id,
    game_date,
    season,
    player_name,
    team,
    opponent,
    home,
    minutes,
    pts,
    reb,
    ast,
    fgm,
    fga,
    fg3m,
    fg3a,
    ftm,
    fta,
    oreb,
    dreb,
    stl,
    blk,
    tov,
    pf,
    plus_minus,
    source,
    dataset_revision,
    source_file
from ranked
where dedup_rank = 1
