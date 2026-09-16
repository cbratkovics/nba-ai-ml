-- Grain: one row per game_id, from the players who logged minutes in it.
with per_game as (
    select
        game_id,
        min(game_date) as game_date,
        min(season) as season,
        max(case when home then team end) as home_team,
        max(case when not home then team end) as away_team,
        count(*) as n_players,
        count(*) filter (where home) as n_home_players,
        count(*) filter (where not home) as n_away_players,
        sum(pts) filter (where home) as home_pts,
        sum(pts) filter (where not home) as away_pts
    from {{ ref('slv_game_logs') }}
    group by game_id
)

select
    cast(game_id as varchar) as game_id,
    cast(game_date as date) as game_date,
    cast(season as varchar) as season,
    cast(home_team as varchar) as home_team,
    cast(away_team as varchar) as away_team,
    cast(n_players as integer) as n_players,
    cast(n_home_players as integer) as n_home_players,
    cast(n_away_players as integer) as n_away_players,
    cast(home_pts as integer) as home_pts,
    cast(away_pts as integer) as away_pts,
    cast(game_id like '006%' as boolean) as is_cup_final
from per_game
