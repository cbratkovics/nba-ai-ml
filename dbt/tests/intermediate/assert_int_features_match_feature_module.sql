-- The feature parity proof (ADR-0032): int_player_game_features (SQL over slv_game_logs) equals
-- the Python feature module's output over the same game-log files (brz_features_asof_v1,
-- written by nba/warehouse/features.py) on all 23 features, row for row. A row fails when
--   * it exists on one side only (the file is missing, stale, or the grains differ),
--   * any feature is null on one side only,
--   * an integer feature differs at all, or a double feature differs by more than
--     var('tol_feature_parity') (set from the largest difference observed over 130,414 rows,
--     reports/feature_parity_asof_v1.json: 2.1e-14 on the minutes means, 0 everywhere else).
-- Returns the key, the side(s) present and the first feature that disagreed, so the failure
-- names the formula to look at.
with sql_side as (
    select * from {{ ref('int_player_game_features') }}
),

module_side as (
    select * from {{ ref('brz_features_asof_v1') }}
),

compared as (
    select
        coalesce(s.player_id, m.player_id) as player_id,
        coalesce(s.game_id, m.game_id) as game_id,
        s.player_id is not null as in_sql,
        m.player_id is not null as in_module,
        case
            {% for name in feature_names() %}
            when (s.{{ name }} is null) <> (m.{{ name }} is null) then '{{ name }}: null on one side'
            {% if name in integer_feature_names() %}
            when s.{{ name }} <> m.{{ name }} then '{{ name }}: integer differs'
            {% else %}
            when abs(s.{{ name }} - m.{{ name }}) > {{ var('tol_feature_parity') }}
                then '{{ name }}: differs by more than the tolerance'
            {% endif %}
            {% endfor %}
        end as first_disagreement
    from sql_side as s
    full outer join module_side as m
        on s.player_id = m.player_id and s.game_id = m.game_id
)

select
    player_id,
    game_id,
    in_sql,
    in_module,
    coalesce(first_disagreement, 'row on one side only') as reason
from compared
where not in_sql or not in_module or first_disagreement is not null
