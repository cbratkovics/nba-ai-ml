-- Grain: one row per (game_date, source, book, market, player_name_raw, capture_id): every line
-- captured or typed, with the matched player. A book can list several points for a player
-- (alternate lines come as separate markets and are not captured); if a snapshot ever carried
-- two rows for the same key, the later book update wins. Never exported (ADR-0031).
{{ config(meta={'export': false}) }}

with ranked as (
    select
        *,
        row_number() over (
            partition by game_date, source, book, market, player_name_raw, capture_id
            order by line_last_update_utc desc nulls last, source_file asc
        ) as dedup_rank
    from {{ ref('brz_lines') }}
)

select
    capture_id,
    game_date,
    {{ season_of('game_date') }} as season,
    source,
    book,
    market,
    player_name_raw,
    player_id,
    match_method,
    team,
    event_id,
    home_team,
    away_team,
    commence_time_utc,
    snapshot_at_utc,
    line_last_update_utc,
    line,
    over_price,
    under_price,
    source_file
from ranked
where dedup_rank = 1
