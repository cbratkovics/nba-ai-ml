-- Typed copy of the holdout-season replay residuals, replay/<season>/residuals/<date>.parquet:
-- one file per replayed date in the nightly residual layout, written by nba/predict/replay.py
-- (the run whose summary is reports/replay_<season>.json).
select
    {{ typed_select(residual_columns(), 'r') }}
from {{ source('warehouse_files', 'replay_residuals') }} as r
