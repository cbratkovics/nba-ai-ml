-- The incremental lookback must cover every restatement the ingest has observed: the maximum
-- restatement_lag_days across all daily reports may not exceed var('lookback_days').
-- Passes trivially while no restatement has been observed (ADR-0004, ADR-0008).
select
    report_date,
    max_lag_days,
    lookback_days
from {{ ref('mart_restatement_lag') }}
where not within_lookback
