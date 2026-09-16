-- reports/replay_all_rows_<season>.json unnested: one row per (date, target) with the model
-- and last-10 baseline MAE on every replayed row with a box score and a defined baseline.
with report as (
    select * from {{ source('repo_reports', 'replay_all_rows') }}
),

days as (
    select
        r.filename,
        r.season,
        r.population,
        d.*
    from report as r, unnest(r.days) as t (d)
)

{% for t in var('targets') %}
select
    cast(filename as varchar) as source_file,
    cast(season as varchar) as season,
    cast(population as varchar) as population_label,
    cast(date as date) as game_date,
    cast('{{ t }}' as varchar) as target,
    cast(n as integer) as n,
    cast(model.{{ t }} as double) as model_mae,
    cast(baseline_last10.{{ t }} as double) as baseline_last10_mae
from days
{{ "union all" if not loop.last }}
{% endfor %}
