-- reports/replay_<season>.json summary: one row per target with the restricted (training
-- population) and unrestricted (all rows with a box score) MAE and the row counts.
with report as (
    select * from {{ source('repo_reports', 'replay_report') }}
)

{% for t in var('targets') %}
select
    cast(filename as varchar) as source_file,
    cast(season as varchar) as season,
    cast(git_sha as varchar) as replay_commit,
    cast(model_revision as varchar) as model_revision,
    cast(dataset_revision as varchar) as dataset_revision,
    cast(reference.metrics_json_git_sha as varchar) as model_commit,
    cast('{{ t }}' as varchar) as target,
    cast(n_dates as integer) as n_dates,
    cast(n_predicted as integer) as n_predicted,
    cast(n_with_actuals as integer) as n_with_actuals,
    cast(n_restricted as integer) as n_restricted,
    cast(mae_restricted.{{ t }} as double) as mae_restricted,
    cast(mae_unrestricted.{{ t }} as double) as mae_unrestricted,
    cast(reference.model_mae.{{ t }} as double) as metrics_json_mae,
    cast(diff_vs_metrics_json.{{ t }} as double) as diff_vs_metrics_json,
    cast(tolerance as double) as tolerance,
    cast(passed as boolean) as passed
from report
{{ "union all" if not loop.last }}
{% endfor %}
