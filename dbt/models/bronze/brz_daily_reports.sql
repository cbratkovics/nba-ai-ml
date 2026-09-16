-- One row per nightly ingest report (daily_reports/<date>.json); the changed-row examples stay
-- a JSON list here and are unnested in slv_restatements. Optional family: empty until the
-- first nightly run has pushed a report.
{% if files_exist(var('warehouse_root') ~ '/daily_reports/*.json') %}
select
    cast(filename as varchar) as source_file,
    cast(date as date) as report_date,
    cast(generated_at as timestamp) as generated_at,
    cast(dataset_revision_before as varchar) as dataset_revision_before,
    cast(dataset_revision_after as varchar) as dataset_revision_after,
    cast(stored_rows as bigint) as stored_rows,
    cast(stored_max_game_date as date) as stored_max_game_date,
    cast(window_start_exclusive as date) as window_start_exclusive,
    cast(window_rows_in_dump as bigint) as window_rows_in_dump,
    cast(window_rows_after_rules as bigint) as window_rows_after_rules,
    cast(dnp_dropped_in_window as bigint) as dnp_dropped_in_window,
    cast(counts."new" as integer) as n_new,
    cast(counts.changed as integer) as n_changed,
    cast(counts.unchanged as integer) as n_unchanged,
    cast(changed_examples as json[]) as changed_examples,
    cast(schedule_file as varchar) as schedule_file,
    cast(pushed as boolean) as pushed
from {{ source('warehouse_files', 'daily_reports') }}
{% else %}
{{ empty_typed_relation([
    ['source_file', 'varchar'], ['report_date', 'date'], ['generated_at', 'timestamp'],
    ['dataset_revision_before', 'varchar'], ['dataset_revision_after', 'varchar'], ['stored_rows', 'bigint'],
    ['stored_max_game_date', 'date'], ['window_start_exclusive', 'date'], ['window_rows_in_dump', 'bigint'],
    ['window_rows_after_rules', 'bigint'], ['dnp_dropped_in_window', 'bigint'], ['n_new', 'integer'],
    ['n_changed', 'integer'], ['n_unchanged', 'integer'], ['changed_examples', 'json[]'],
    ['schedule_file', 'varchar'], ['pushed', 'boolean'],
]) }}
{% endif %}
