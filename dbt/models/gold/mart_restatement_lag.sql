-- Grain: one row per nightly ingest run date: how many stored rows the source restated and how
-- far back (days between the run date and the game date), against the silver lookback.
-- tests/silver/assert_lookback_covers_restatement_lag.sql fails when the observed maximum
-- exceeds var('lookback_days'). Passes trivially while no restatement has been observed; that
-- is the point of logging it (ADR-0008).
with lags as (
    select
        report_date,
        count(*) as n_examples,
        max(restatement_lag_days) as max_lag_days,
        quantile_cont(restatement_lag_days, 0.5) as p50_lag_days,
        quantile_cont(restatement_lag_days, 0.9) as p90_lag_days
    from {{ ref('slv_restatements') }}
    group by report_date
)

select
    cast(r.report_date as date) as report_date,
    cast(r.n_changed as integer) as n_changed,
    cast(r.n_new as integer) as n_new,
    cast(coalesce(l.n_examples, 0) as integer) as n_examples,
    cast(l.max_lag_days as integer) as max_lag_days,
    cast(l.p50_lag_days as double) as p50_lag_days,
    cast(l.p90_lag_days as double) as p90_lag_days,
    cast({{ var('lookback_days') }} as integer) as lookback_days,
    cast(coalesce(l.max_lag_days, 0) <= {{ var('lookback_days') }} as boolean) as within_lookback,
    cast(r.stored_max_game_date as date) as stored_max_game_date,
    cast(r.window_start_exclusive as date) as window_start_exclusive
from {{ ref('slv_daily_reports') }} as r
left join lags as l on r.report_date = l.report_date
