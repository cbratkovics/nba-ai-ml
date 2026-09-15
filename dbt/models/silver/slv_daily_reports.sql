-- Grain: one row per nightly ingest run date: the reconciliation counts of the daily ingest
-- against the stored game logs.
select
    report_date,
    generated_at,
    dataset_revision_before,
    dataset_revision_after,
    stored_rows,
    stored_max_game_date,
    window_start_exclusive,
    window_rows_in_dump,
    window_rows_after_rules,
    dnp_dropped_in_window,
    n_new,
    n_changed,
    n_unchanged,
    len(changed_examples) as n_changed_examples,
    schedule_file,
    pushed,
    source_file
from {{ ref('brz_daily_reports') }}
