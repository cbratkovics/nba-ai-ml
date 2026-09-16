-- Grain: one row per (team, season): abbreviation as stored for that season (TeamHistories
-- of the Kaggle dump; San Antonio is SAN), with coverage counts.
select
    cast(team as varchar) as team,
    cast(season as varchar) as season,
    cast(count(distinct game_id) as integer) as games,
    cast(count(distinct player_id) as integer) as players,
    cast(count(*) as integer) as player_games,
    cast(min(game_date) as date) as first_game_date,
    cast(max(game_date) as date) as last_game_date
from {{ ref('slv_game_logs') }}
group by team, season
