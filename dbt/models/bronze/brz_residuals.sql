-- Typed copy of the nightly residual files residuals/<date>.parquet (nba/predict/residuals.py
-- RESIDUAL_COLUMNS). Optional family: empty (typed) until the first in-season run.
{% if files_exist(var('warehouse_root') ~ '/residuals/*.parquet') %}
select
    {{ typed_select(residual_columns(), 'r') }}
from {{ source('warehouse_files', 'residuals') }} as r
{% else %}
{{ empty_typed_relation(residual_columns()) }}
{% endif %}
