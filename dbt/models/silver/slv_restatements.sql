-- Grain: one row per changed-row example in a daily ingest report (up to 20 per report):
-- a stored box score the source restated, with restatement_lag_days = run date - game date
-- (written by the ingest from 2026-09-15; derived here for older reports). Empty until the
-- source restates a row in-season.
with examples as (
    select
        r.report_date,
        r.source_file,
        t.example
    from {{ ref('brz_daily_reports') }} as r, unnest(r.changed_examples) as t (example)
)

select
    cast(report_date as date) as report_date,
    cast(json_extract_string(example, '$.player_id') as bigint) as player_id,
    cast(json_extract_string(example, '$.game_id') as varchar) as game_id,
    cast(json_extract_string(example, '$.game_date') as date) as game_date,
    cast(json_extract_string(example, '$.player_name') as varchar) as player_name,
    cast(
        coalesce(
            cast(json_extract_string(example, '$.restatement_lag_days') as integer),
            date_diff('day', cast(json_extract_string(example, '$.game_date') as date), report_date)
        ) as integer
    ) as restatement_lag_days,
    cast(json_extract(example, '$.fields') as json) as changed_fields,
    cast(source_file as varchar) as source_file
from examples
