-- Season contract: every season holds var('full_season_games') distinct games, unless the
-- season_exceptions seed declares a documented shortfall (2024-25 = 1,223). Seasons still in
-- progress (fewer than 40 game dates) are exempt.
select
    season,
    games,
    games_declared,
    exception_reason
from {{ ref('mart_season_coverage') }}
where
    not meets_contract
    and date_diff('day', first_game_date, last_game_date) >= 150
