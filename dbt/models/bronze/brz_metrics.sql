-- reports/metrics.json unpivoted: one row per (target, predictor) with the published holdout
-- metrics on the training population, plus the split and identity fields on every row.
{% set predictors = ['model', 'baseline_last10', 'baseline_season'] %}
{% set selects = [] %}
{% for t in var('targets') %}
{% for p in predictors %}
{% do selects.append(
"select
    cast(filename as varchar) as source_file,
    cast(git_sha as varchar) as model_commit,
    cast(dataset.\"version\" as varchar) as dataset_version,
    cast(split.holdout_season as varchar) as season,
    cast(split.min_minutes as double) as min_minutes,
    cast(split.n_train_rows as integer) as n_train_rows,
    cast(split.n_holdout_rows as integer) as n_holdout_rows,
    cast('" ~ t ~ "' as varchar) as target,
    cast('" ~ p ~ "' as varchar) as predictor,
    cast(metrics." ~ t ~ "." ~ p ~ ".n as integer) as n,
    cast(metrics." ~ t ~ "." ~ p ~ ".mae as double) as mae,
    cast(metrics." ~ t ~ "." ~ p ~ ".rmse as double) as rmse,
    cast(metrics." ~ t ~ "." ~ p ~ ".r2 as double) as r2
from report") %}
{% endfor %}
{% endfor %}
with report as (
    select * from {{ source('repo_reports', 'metrics_json') }}
)

{{ selects | join('\nunion all\n') }}
