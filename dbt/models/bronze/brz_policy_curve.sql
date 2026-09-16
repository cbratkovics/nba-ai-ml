-- The coverage curves of reports/policy_<season>.json: one row per (population, target,
-- threshold) with the model policy's coverage and hit rate at that threshold and both
-- baselines on the same rows. Optional like brz_policy_report.
{% set populations = ['min10', 'all'] %}
{% set selects = [] %}
{% for p in populations %}
{% for t in var('targets') %}
{% set b = 'populations.' ~ p ~ '.targets.' ~ t %}
{% do selects.append(
"select
    cast(filename as varchar) as source_file,
    cast(season as varchar) as season,
    cast(model_revision as varchar) as model_revision,
    cast('" ~ p ~ "' as varchar) as population,
    cast('" ~ t ~ "' as varchar) as target,
    cast(c.threshold as double) as threshold,
    cast(c.n_called as integer) as n_called,
    cast(c.coverage as double) as coverage,
    cast(c.n_resolved as integer) as n_resolved,
    cast(c.n_push as integer) as n_push,
    cast(c.n_hit as integer) as n_hit,
    cast(c.hit_rate as double) as hit_rate,
    cast(c.net_correct as integer) as net_correct,
    cast(c.season_mean_same_rows_n as integer) as season_mean_same_rows_n,
    cast(c.season_mean_same_rows_hit_rate as double) as season_mean_same_rows_hit_rate,
    cast(c.season_mean_own_n_called as integer) as season_mean_own_n_called,
    cast(c.season_mean_own_hit_rate as double) as season_mean_own_hit_rate
from report, unnest(" ~ b ~ ".coverage_curve) as u(c)") %}
{% endfor %}
{% endfor %}
{% if files_exist(var('reports_root') ~ '/policy_' ~ var('holdout_season') ~ '.json') %}
with report as (
    select * from {{ source('repo_reports', 'policy_report') }}
)

{{ selects | join('\nunion all\n') }}
{% else %}
{{ empty_typed_relation(policy_curve_columns()) }}
{% endif %}
