-- Grain: one row per (player_id, game_id): the 23 asof_v1 features of nba/features/asof.py
-- recomputed in SQL over slv_game_logs (ADR-0032). Internal layer: not exported, no contract
-- (nothing outside the warehouse reads it; the parity test below is its contract), and not on
-- the serving path: the slate still calls the Python module, which stays the one feature
-- implementation the published model was trained on.
--
-- Every feature uses the player's games strictly before this one, in the module's order
-- (player_id, game_date, game_id): `rows between N preceding and 1 preceding` is pandas'
-- shift(1) then rolling(N, min_periods=1); an empty frame averages to null, as the module's
-- first game is NaN. Rolling windows run across seasons; the season mean and games played
-- restart per (player_id, season); the opponent mean runs across seasons per
-- (player_id, opponent). days_rest is a double with null on the first game (pandas
-- diff().dt.days); back_to_back is 0 there (NaN == 1 is false). home is the integer the
-- module emits.
--
-- tests/intermediate/assert_int_features_match_feature_module.sql compares all 23 columns row
-- for row with the module's own output over the same files (brz_features_asof_v1) and fails
-- on any difference above var('tol_feature_parity'), any null on one side only, or any row on
-- one side only. reports/feature_parity_asof_v1.json records the largest difference observed
-- per feature; the tolerance is set from it.
with ordered as (
    select
        player_id,
        game_id,
        game_date,
        season,
        team,
        opponent,
        home,
        minutes,
        pts,
        reb,
        ast
    from {{ ref('slv_game_logs') }}
),

features as (
    select
        player_id,
        game_id,
        game_date,
        season,
        team,
        opponent,
        {% for stat in ['pts', 'reb', 'ast', 'minutes'] %}
        {% for w in [5, 10, 20] %}
        avg({{ stat }}) over (
            partition by player_id order by game_date, game_id
            rows between {{ w }} preceding and 1 preceding
        ) as {{ stat }}_mean_last{{ w }},
        {% endfor %}
        avg({{ stat }}) over (
            partition by player_id, season order by game_date, game_id
            rows between unbounded preceding and 1 preceding
        ) as {{ stat }}_mean_season,
        {% endfor %}
        count(*) over (
            partition by player_id, season order by game_date, game_id
            rows between unbounded preceding and 1 preceding
        ) as games_played_season,
        game_date - lag(game_date) over (partition by player_id order by game_date, game_id) as days_rest,
        {% for stat in ['pts', 'reb', 'ast'] %}
        avg({{ stat }}) over (
            partition by player_id, opponent order by game_date, game_id
            rows between unbounded preceding and 1 preceding
        ) as {{ stat }}_mean_vs_opp,
        {% endfor %}
        home
    from ordered
)

select
    cast(player_id as bigint) as player_id,
    cast(game_id as varchar) as game_id,
    cast(game_date as date) as game_date,
    cast(season as varchar) as season,
    cast(team as varchar) as team,
    cast(opponent as varchar) as opponent,
    {% for stat in ['pts', 'reb', 'ast', 'minutes'] %}
    {% for w in [5, 10, 20] %}
    cast({{ stat }}_mean_last{{ w }} as double) as {{ stat }}_mean_last{{ w }},
    {% endfor %}
    {% endfor %}
    {% for stat in ['pts', 'reb', 'ast', 'minutes'] %}
    cast({{ stat }}_mean_season as double) as {{ stat }}_mean_season,
    {% endfor %}
    cast(coalesce(games_played_season, 0) as integer) as games_played_season,
    cast(days_rest as double) as days_rest,
    cast(coalesce(days_rest = 1, false) as integer) as back_to_back,
    cast(home as integer) as home,
    {% for stat in ['pts', 'reb', 'ast'] %}
    cast({{ stat }}_mean_vs_opp as double) as {{ stat }}_mean_vs_opp,
    {% endfor %}
    cast('{{ var("feature_version") }}' as varchar) as feature_version
from features
