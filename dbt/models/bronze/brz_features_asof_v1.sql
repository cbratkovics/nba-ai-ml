-- Typed copy of features/asof_v1.parquet: the Python feature module's own output over the
-- loaded game logs (nba/warehouse/features.py, written by the loader). The reference side of
-- the feature parity proof (ADR-0032); the warehouse never serves it. Optional family: empty
-- (typed) when the loader step was skipped, in which case the parity test fails on every row
-- of slv_game_logs rather than passing on nothing.
{{ config(meta={'export': false}) }}
{% if files_exist(var('warehouse_root') ~ '/features/asof_v1.parquet') %}
select
    {{ typed_select(feature_columns(), 'f') }}
from {{ source('warehouse_files', 'features_asof_v1') }} as f
{% else %}
{{ empty_typed_relation(feature_columns()) }}
{% endif %}
