-- Grain: one row per season: how much of the season the ledger covers and what is still open,
-- the numbers every ledger page must show (ADR-0031): captured nights against game nights,
-- bets per captured night, lines whose player did not match, bets still pending, and the
-- lines excluded by the timestamp rules. Nothing here is a rate on a rate.
with nights as (
    select season, count(distinct game_date) as game_nights
    from {{ ref('fct_prediction') }}
    where run_kind = 'nightly'
    group by season
),

captures as (
    select
        {{ season_of('game_date') }} as season,
        count(*) filter (where status = 'captured') as nights_captured,
        count(*) filter (where status = 'refused') as nights_refused,
        count(*) filter (where status = 'no_games') as nights_no_games,
        sum(n_out_of_scope) as games_out_of_scope,
        sum(n_in_scope) as games_in_scope,
        sum(charged) as credits_charged,
        sum(n_unmatched) as lines_unmatched
    from {{ ref('brz_captures') }}
    group by season
),

lines as (
    select
        season,
        count(*) as lines_total,
        count(*) filter (where player_id is null) as lines_unmatched_player,
        count(*) filter (where source = 'manual') as lines_manual
    from {{ ref('slv_lines') }}
    group by season
),

bets as (
    select
        season,
        count(*) filter (where has_prediction) as lines_with_prediction,
        count(*) filter (where has_prediction and not eligible) as lines_ineligible,
        count(*) filter (where has_prediction and not prediction_before_line) as lines_prediction_after_line,
        count(*) filter (where has_prediction and not line_before_tipoff) as lines_after_tipoff,
        count(*) filter (where is_bet) as bets_total,
        count(*) filter (where is_bet and outcome = 'pending') as bets_pending,
        count(*) filter (where is_bet and outcome in ('won', 'lost')) as bets_settled,
        count(*) filter (where is_bet and outcome = 'void') as bets_void,
        count(distinct game_date) filter (where is_bet) as nights_with_bets
    from {{ ref('fct_paper_bet') }}
    group by season
)

select
    cast(n.season as varchar) as season,
    cast(n.game_nights as integer) as game_nights,
    cast(coalesce(c.nights_captured, 0) as integer) as nights_captured,
    cast(coalesce(c.nights_refused, 0) as integer) as nights_refused,
    cast(coalesce(c.nights_no_games, 0) as integer) as nights_no_games,
    cast(coalesce(c.games_in_scope, 0) as integer) as games_in_scope,
    cast(coalesce(c.games_out_of_scope, 0) as integer) as games_out_of_scope,
    cast(coalesce(c.credits_charged, 0) as integer) as credits_charged,
    cast(coalesce(l.lines_total, 0) as integer) as lines_total,
    cast(coalesce(l.lines_manual, 0) as integer) as lines_manual,
    cast(coalesce(l.lines_unmatched_player, 0) as integer) as lines_unmatched_player,
    cast(coalesce(b.lines_with_prediction, 0) as integer) as lines_with_prediction,
    cast(coalesce(b.lines_ineligible, 0) as integer) as lines_ineligible,
    cast(coalesce(b.lines_prediction_after_line, 0) as integer) as lines_prediction_after_line,
    cast(coalesce(b.lines_after_tipoff, 0) as integer) as lines_after_tipoff,
    cast(coalesce(b.bets_total, 0) as integer) as bets_total,
    cast(coalesce(b.bets_settled, 0) as integer) as bets_settled,
    cast(coalesce(b.bets_pending, 0) as integer) as bets_pending,
    cast(coalesce(b.bets_void, 0) as integer) as bets_void,
    cast(coalesce(b.nights_with_bets, 0) as integer) as nights_with_bets,
    cast(
        case when coalesce(c.nights_captured, 0) > 0 then coalesce(b.bets_total, 0) / c.nights_captured end as double
    ) as bets_per_captured_night
from nights as n
left join captures as c on n.season = c.season
left join lines as l on n.season = l.season
left join bets as b on n.season = b.season
