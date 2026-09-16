-- Grain: one row per (run_date, model_revision, feature): the nightly drift check's PSI per
-- model feature against the day-aligned reference (ADR-0017) with the run's verdict on every
-- row (ADR-0018): status ok / warn / hold / insufficient, the threshold that applied and
-- whether it was calibrated, the flagged count, the slate outcome and the no-schedule streak.
-- Every row is a committed drift/<date>.json product; nothing is recomputed here. A run
-- without PSI (insufficient window, off-season) is one row with a null feature.
select
    cast(run_date as date) as run_date,
    cast(season as varchar) as season,
    cast(model_revision as varchar) as model_revision,
    cast(feature_version as varchar) as feature_version,
    cast(feature as varchar) as feature,
    cast(psi as double) as psi,
    cast(flagged as boolean) as flagged,
    cast(status as varchar) as status,
    cast(position as varchar) as position,
    cast(reference_mode as varchar) as reference_mode,
    cast(n_rows as integer) as n_rows,
    cast(psi_threshold as double) as psi_threshold,
    cast(min_features as integer) as min_features,
    cast(calibrated as boolean) as calibrated,
    cast(n_flagged as integer) as n_flagged,
    cast(slate_status as varchar) as slate_status,
    cast(no_schedule_streak as integer) as no_schedule_streak,
    cast(no_schedule_warn as boolean) as no_schedule_warn,
    cast(source_file as varchar) as source_file
from {{ ref('brz_drift_reports') }}
