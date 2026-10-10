-- The timestamp rule (ADR-0030): a prediction must be stamped before the line it is scored
-- against was captured, and the capture before tip-off. A prediction later than its line
-- means the pipeline ran out of order (the nightly must precede the snapshot) and no bet may
-- rest on it: fct_paper_bet already refuses such rows, and this test fails the build so the
-- ordering is fixed rather than silently excluded. Lines read after tip-off (a late manual
-- row) are excluded and counted in mart_ledger_coverage, not failed here.
select
    game_date,
    source,
    book,
    market,
    player_id,
    prediction_generated_at,
    snapshot_at_utc,
    tip_off_utc
from {{ ref('fct_paper_bet') }}
where has_prediction and not prediction_before_line
