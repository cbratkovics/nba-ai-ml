{#- True when at least one file matches the glob (DuckDB glob()). Optional product families
    (nightly predictions, residuals, daily reports) do not exist before the first in-season run;
    their bronze models select an empty, typed relation instead of failing with "No files
    found". At parse time (execute = false) the source is referenced so lineage stays intact. -#}
{% macro files_exist(pattern) %}
    {% if not execute %}
        {{ return(true) }}
    {% endif %}
    {% set r = run_query("select count(*) as n from glob('" ~ pattern ~ "')") %}
    {{ return(r.columns[0].values()[0] > 0) }}
{% endmacro %}

{#- `select <typed nulls> where false` for a list of [name, type] pairs. -#}
{% macro empty_typed_relation(columns) -%}
select
{% for name, type in columns -%}
    cast(null as {{ type }}) as {{ name }}{{ "," if not loop.last }}
{% endfor -%}
where false
{%- endmacro %}
