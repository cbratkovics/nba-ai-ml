-- Grain: one row per (season, run_kind, model_revision, game_date, population, target): the
-- daily MAE of the model and of the last-10 baseline, both populations. For the replay the
-- `all` rows reproduce reports/replay_all_rows_<season>.json day by day
-- (tests/gold/assert_daily_metrics_reconcile_to_all_rows_report.sql).
with scored as (
    select *
    from {{ ref('fct_prediction') }}
    where has_actual and baseline_defined
),

both_populations as (
    select 'all' as population, * exclude (population) from scored
    union all
    select 'min10' as population, * exclude (population) from scored where in_metrics_population
)

{% for t in var('targets') %}
select
    cast(season as varchar) as season,
    cast(run_kind as varchar) as run_kind,
    cast(model_revision as varchar) as model_revision,
    cast(max(feature_version) as varchar) as feature_version,
    cast(game_date as date) as game_date,
    cast(population as varchar) as population,
    cast('{{ t }}' as varchar) as target,
    cast(count(*) as integer) as n,
    cast(avg(abs_error_{{ t }}) as double) as model_mae,
    cast(avg(baseline_abs_error_{{ t }}) as double) as baseline_last10_mae
from both_populations
group by season, run_kind, model_revision, game_date, population
{{ "union all" if not loop.last }}
{% endfor %}
