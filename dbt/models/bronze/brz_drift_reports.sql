-- Typed copy of the nightly drift reports drift/<date>.json (nba/drift, ADR-0016 to
-- ADR-0018), unnested to one row per (date, feature) with the run-level verdict on every
-- row. Optional family: empty (typed) until the first nightly run writes one.
{% if files_exist(var('warehouse_root') ~ '/drift/*.json') %}
with report as (
    select * from {{ source('warehouse_files', 'drift_reports') }}
)

select
    cast(filename as varchar) as source_file,
    cast("date" as date) as run_date,
    cast(season as varchar) as season,
    cast(position as varchar) as position,
    cast(reference_mode as varchar) as reference_mode,
    cast(feature_version as varchar) as feature_version,
    cast(model_revision as varchar) as model_revision,
    cast("window".n_rows as integer) as n_rows,
    cast(thresholds.psi as double) as psi_threshold,
    cast(thresholds.min_features as integer) as min_features,
    cast(thresholds.calibrated as boolean) as calibrated,
    cast(status as varchar) as status,
    cast(n_flagged as integer) as n_flagged,
    cast(slate_status as varchar) as slate_status,
    cast(no_schedule_streak as integer) as no_schedule_streak,
    cast(no_schedule_warn as boolean) as no_schedule_warn,
    cast(f.feature as varchar) as feature,
    cast(f.psi as double) as psi,
    cast(f.flagged as boolean) as flagged
from report, unnest(cast(features as struct(feature varchar, psi double, flagged boolean)[])) as u (f)
union all
-- a run without PSI (insufficient window, or the off-season) keeps its verdict row
select
    cast(filename as varchar) as source_file,
    cast("date" as date) as run_date,
    cast(season as varchar) as season,
    cast(position as varchar) as position,
    cast(reference_mode as varchar) as reference_mode,
    cast(feature_version as varchar) as feature_version,
    cast(model_revision as varchar) as model_revision,
    cast("window".n_rows as integer) as n_rows,
    cast(thresholds.psi as double) as psi_threshold,
    cast(thresholds.min_features as integer) as min_features,
    cast(thresholds.calibrated as boolean) as calibrated,
    cast(status as varchar) as status,
    cast(n_flagged as integer) as n_flagged,
    cast(slate_status as varchar) as slate_status,
    cast(no_schedule_streak as integer) as no_schedule_streak,
    cast(no_schedule_warn as boolean) as no_schedule_warn,
    cast(null as varchar) as feature,
    cast(null as double) as psi,
    cast(null as boolean) as flagged
from report
where len(features) = 0
{% else %}
{{ empty_typed_relation(drift_report_columns()) }}
{% endif %}
