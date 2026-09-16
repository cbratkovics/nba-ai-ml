-- Grain: one row per (player_id, stint_no): the runs of consecutive games a player played for
-- one team, derived from the game logs, so the team "as of" any date can be answered for the
-- whole history (the snapshot only covers changes since its first build).
--
-- As-of rule (the slate's roster rule, nba/predict/slate.py pending_rows): on date D a player
-- belongs to the team of their most recent game strictly before D. So a stint is effective
-- for dates > effective_from_date (its first game) up to and including effective_to_date
-- (the next stint's first game date); the current stint has effective_to_date null.
-- Usage: where D > effective_from_date and (effective_to_date is null or D <= effective_to_date).
-- tests/gold/assert_replay_roster_rule_reproduces_from_asof.sql proves the replay's slate
-- assignments reproduce from this table.
with ordered as (
    select
        player_id,
        player_name,
        team,
        game_id,
        game_date,
        lag(team) over (partition by player_id order by game_date, game_id) as prev_team
    from {{ ref('slv_game_logs') }}
),

flagged as (
    select
        *,
        case when prev_team is null or prev_team <> team then 1 else 0 end as stint_start
    from ordered
),

numbered as (
    select
        *,
        sum(stint_start) over (
            partition by player_id order by game_date, game_id rows unbounded preceding
        ) as stint_no
    from flagged
),

stints as (
    select
        player_id,
        stint_no,
        team,
        min(game_date) as first_game_date,
        min(game_id) filter (where stint_start = 1) as first_game_id,
        max(game_date) as last_game_date,
        count(*) as games,
        max(player_name) as player_name
    from numbered
    group by player_id, stint_no, team
)

select
    cast(player_id as bigint) as player_id,
    cast(stint_no as integer) as stint_no,
    cast(team as varchar) as team,
    cast(player_name as varchar) as player_name,
    cast(first_game_date as date) as first_game_date,
    cast(first_game_id as varchar) as first_game_id,
    cast(last_game_date as date) as last_game_date,
    cast(games as integer) as games,
    cast(first_game_date as date) as effective_from_date,
    cast(lead(first_game_date) over (partition by player_id order by stint_no) as date) as effective_to_date,
    cast(lead(first_game_date) over (partition by player_id order by stint_no) is null as boolean) as is_current
from stints
