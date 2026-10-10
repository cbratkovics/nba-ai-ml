-- Typed copy of the private lines files lines/<date>.parquet and lines/<date>_manual.parquet
-- (nba/lines/capture.py LINE_COLUMNS; ADR-0031). Optional family: empty (typed) without the
-- private repo (CI) or before the first capture. Never exported.
{{ config(meta={'export': false}) }}
{% if files_exist(var('warehouse_root') ~ '/lines/*.parquet') %}
select
    {{ typed_select(line_columns(), 'l') }}
from {{ source('lines_files', 'lines') }} as l
{% else %}
{{ empty_typed_relation(line_columns()) }}
{% endif %}
