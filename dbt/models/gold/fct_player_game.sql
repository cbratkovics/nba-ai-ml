-- Grain: one row per (player_id, game_id): the box score plus the history counts that decide
-- which population the row belongs to.
--
-- population (the cohort of this warehouse, ADR-0009):
--   min10 — the training population: at least var('min_minutes') minutes and both baselines
--           defined, i.e. at least one earlier game in the season (season-to-date mean) and so
--           at least one earlier game overall (last-10 mean). Rows the model was trained and
--           evaluated on (reports/metrics.json).
--   all   — every other row with a box score. Every metric mart reports both; on `all` the
--           last-10 baseline is the better predictor (README, "All rows").
-- tests/gold/assert_population_flag_matches_replay.sql proves this flag equals the
-- in_metrics_population flag the replay residuals carry, row for row.
with history as (
    select
        *,
        count(*) over (
            partition by player_id order by game_date, game_id
            rows between unbounded preceding and 1 preceding
        ) as prior_games,
        count(*) over (
            partition by player_id, season order by game_date, game_id
            rows between unbounded preceding and 1 preceding
        ) as prior_games_season
    from {{ ref('slv_game_logs') }}
)

select
    cast(player_id as bigint) as player_id,
    cast(game_id as varchar) as game_id,
    cast(game_date as date) as game_date,
    cast(season as varchar) as season,
    cast(player_name as varchar) as player_name,
    cast(team as varchar) as team,
    cast(opponent as varchar) as opponent,
    cast(home as boolean) as home,
    cast(minutes as double) as minutes,
    cast(pts as integer) as pts,
    cast(reb as integer) as reb,
    cast(ast as integer) as ast,
    cast(fgm as integer) as fgm,
    cast(fga as integer) as fga,
    cast(fg3m as integer) as fg3m,
    cast(fg3a as integer) as fg3a,
    cast(ftm as integer) as ftm,
    cast(fta as integer) as fta,
    cast(oreb as integer) as oreb,
    cast(dreb as integer) as dreb,
    cast(stl as integer) as stl,
    cast(blk as integer) as blk,
    cast(tov as integer) as tov,
    cast(pf as integer) as pf,
    cast(plus_minus as integer) as plus_minus,
    cast(source as varchar) as source,
    cast(dataset_revision as varchar) as dataset_revision,
    cast(coalesce(prior_games, 0) as integer) as prior_games,
    cast(coalesce(prior_games_season, 0) as integer) as prior_games_season,
    cast(
        case
            when minutes >= {{ var('min_minutes') }} and coalesce(prior_games_season, 0) >= 1 then 'min10'
            else 'all'
        end as varchar
    ) as population
from history
