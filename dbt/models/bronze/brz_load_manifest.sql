-- One row: which dataset-repo revision the loader pulled and how many files per family.
select
    cast(filename as varchar) as source_file,
    cast(loaded_at as timestamp) as loaded_at,
    cast(dataset_repo as varchar) as dataset_repo,
    cast(dataset_revision as varchar) as dataset_revision,
    cast(replay_season as varchar) as replay_season,
    cast(model_revision as varchar) as model_revision,
    cast(model_commit as varchar) as model_commit,
    cast(families.game_logs as integer) as n_game_log_files,
    cast(families.predictions as integer) as n_prediction_files,
    cast(families.residuals as integer) as n_residual_files,
    cast(families.daily_reports as integer) as n_daily_report_files
from {{ source('warehouse_files', 'load_manifest') }}
