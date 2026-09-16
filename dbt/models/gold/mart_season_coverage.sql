-- Grain: one row per season: games and rows in the warehouse against the season contract
-- (var('full_season_games') unless the season_exceptions seed declares otherwise) and the
-- players-per-game range. tests/gold/assert_season_game_counts.sql fails when a season
-- misses its declared count.
with games as (
    select
        season,
        count(*) as games,
        min(n_players) as players_per_game_min,
        max(n_players) as players_per_game_max,
        count(*) filter (where is_cup_final) as cup_finals
    from {{ ref('dim_game') }}
    group by season
),

rows_per_season as (
    select
        season,
        count(*) as player_games,
        count(distinct player_id) as players,
        min(game_date) as first_game_date,
        max(game_date) as last_game_date
    from {{ ref('fct_player_game') }}
    group by season
),

exceptions as (
    select
        season,
        games_declared,
        reason
    from {{ ref('season_exceptions') }}
),

missing as (
    select
        season,
        count(*) as missing_games_listed
    from {{ ref('known_missing_games') }}
    group by season
)

select
    cast(g.season as varchar) as season,
    cast(g.games as integer) as games,
    cast({{ var('full_season_games') }} as integer) as games_expected,
    cast(coalesce(e.games_declared, {{ var('full_season_games') }}) as integer) as games_declared,
    cast(g.games = coalesce(e.games_declared, {{ var('full_season_games') }}) as boolean) as meets_contract,
    cast(coalesce(m.missing_games_listed, 0) as integer) as missing_games_listed,
    cast(e.reason as varchar) as exception_reason,
    cast(g.cup_finals as integer) as cup_finals,
    cast(g.players_per_game_min as integer) as players_per_game_min,
    cast(g.players_per_game_max as integer) as players_per_game_max,
    cast(r.player_games as integer) as player_games,
    cast(r.players as integer) as players,
    cast(r.first_game_date as date) as first_game_date,
    cast(r.last_game_date as date) as last_game_date
from games as g
inner join rows_per_season as r on g.season = r.season
left join exceptions as e on g.season = e.season
left join missing as m on g.season = m.season
