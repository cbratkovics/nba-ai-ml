-- The incremental lookback should cover every restatement the ingest has observed: the maximum
-- restatement_lag_days across all daily reports may not exceed var('lookback_days').
-- Passes trivially while no restatement has been observed (ADR-0004, ADR-0008).
-- Severity warn (ADR-0026): a lag beyond the lookback is an operational threshold to act on
-- (raise lookback_days, full refresh; docs/RUNBOOK.md), not a wrong published number, so it
-- must not block the nightly build; the workflow opens a `warehouse-warn` issue instead.
{{ config(severity='warn') }}

select
    report_date,
    max_lag_days,
    lookback_days
from {{ ref('mart_restatement_lag') }}
where not within_lookback
