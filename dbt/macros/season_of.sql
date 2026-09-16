{#- Season label of a game date, the rule the package uses (nba.ingest.kaggle_dump.season_from_date):
    the start year is the calendar year when the month is October or later, else the year before. -#}
{% macro season_of(date_expr) -%}
    (
        cast(case when month({{ date_expr }}) >= 10 then year({{ date_expr }}) else year({{ date_expr }}) - 1 end as varchar)
        || '-' || right('0' || cast((case when month({{ date_expr }}) >= 10 then year({{ date_expr }}) else year({{ date_expr }}) - 1 end + 1) % 100 as varchar), 2)
    )
{%- endmacro %}
