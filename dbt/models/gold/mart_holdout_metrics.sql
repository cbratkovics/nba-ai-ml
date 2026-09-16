-- Grain: one row per (season, run_kind, model_revision, population, target): MAE of the model
-- and of the last-10 baseline recomputed from fct_prediction over rows with a box score and a
-- defined baseline. Both populations are reported (ADR-0009): all = every such row, min10 =
-- the training population. tests/gold/assert_holdout_metrics_reconcile_* prove the replay rows
-- reproduce reports/replay_<season>.json exactly and reports/metrics.json within the stated
-- tolerance (ADR-0011).
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
    cast(max(dataset_revision) as varchar) as dataset_revision,
    cast(population as varchar) as population,
    cast('{{ t }}' as varchar) as target,
    cast(count(*) as integer) as n,
    cast(avg(abs_error_{{ t }}) as double) as model_mae,
    cast(sqrt(avg(abs_error_{{ t }} * abs_error_{{ t }})) as double) as model_rmse,
    cast(avg(baseline_abs_error_{{ t }}) as double) as baseline_last10_mae,
    cast(avg(abs_error_{{ t }}) - avg(baseline_abs_error_{{ t }}) as double) as model_minus_baseline_mae
from both_populations
group by season, run_kind, model_revision, population
{{ "union all" if not loop.last }}
{% endfor %}
