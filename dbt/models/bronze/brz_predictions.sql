-- Typed copy of the nightly slate files predictions/<date>.parquet (nba/predict/slate.py
-- OUTPUT_COLUMNS). Optional family: empty (typed) until the first in-season run.
{% if files_exist(var('warehouse_root') ~ '/predictions/*.parquet') %}
select
    {{ typed_select(prediction_columns(), 'p') }}
from {{ source('warehouse_files', 'predictions') }} as p
{% else %}
{{ empty_typed_relation(prediction_columns()) }}
{% endif %}
