-- One row per capture record captures/<date>.json (nba/lines/capture.py Capture; ADR-0031):
-- the decision to capture, written before any priced call, and its outcome. Optional family.
{{ config(meta={'export': false}) }}
{% if files_exist(var('warehouse_root') ~ '/captures/*.json') %}
select
    cast(filename as varchar) as source_file,
    cast(capture_id as varchar) as capture_id,
    cast("date" as date) as game_date,
    cast(trigger as varchar) as trigger,
    cast(policy_version as varchar) as policy_version,
    cast(snapshot_at_utc as timestamp) as snapshot_at_utc,
    cast(decided_before_fetch as boolean) as decided_before_fetch,
    cast(status as varchar) as status,
    cast(n_events as integer) as n_events,
    cast(n_in_scope as integer) as n_in_scope,
    cast(n_out_of_scope as integer) as n_out_of_scope,
    cast(planned_cost as integer) as planned_cost,
    cast(charged as integer) as charged,
    cast(n_calls_ok as integer) as n_calls_ok,
    cast(n_calls_failed as integer) as n_calls_failed,
    cast(n_lines as integer) as n_lines,
    cast(n_players as integer) as n_players,
    cast(n_unmatched as integer) as n_unmatched
from {{ source('lines_files', 'captures') }}
{% else %}
{{ empty_typed_relation([
    ['source_file', 'varchar'], ['capture_id', 'varchar'], ['game_date', 'date'], ['trigger', 'varchar'],
    ['policy_version', 'varchar'], ['snapshot_at_utc', 'timestamp'], ['decided_before_fetch', 'boolean'],
    ['status', 'varchar'], ['n_events', 'integer'], ['n_in_scope', 'integer'], ['n_out_of_scope', 'integer'],
    ['planned_cost', 'integer'], ['charged', 'integer'], ['n_calls_ok', 'integer'], ['n_calls_failed', 'integer'],
    ['n_lines', 'integer'], ['n_players', 'integer'], ['n_unmatched', 'integer'],
]) }}
{% endif %}
