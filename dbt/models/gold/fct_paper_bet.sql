-- Grain: one row per (game_date, source, book, market, player_id): every matched line with the
-- frozen model's prediction for that player and night, the pre-registered rule applied
-- (ADR-0030, nba/ledger/rule.py: the dbt vars mirror it and a test pins the hash), and the
-- outcome once the box score exists. Settlement is computed here, never stored elsewhere, so
-- it is idempotent and backfills itself when a late box score arrives (source-stall design,
-- ADR-0031). Raw lines and prices stay in this table, which is never exported; only the
-- aggregates in mart_ledger* leave the warehouse.
--
--   eligible   has a prediction, prior game this season, last-10 minutes floor, staleness cap,
--              prediction stamped before the snapshot, snapshot before tip-off
--   side       over / under when the edge against the line reaches the market's threshold
--   is_bet     eligible and side is over or under; a bet is flat one unit at the captured price
--   outcome    pending / won / lost / push / void, exactly nba.ledger.rule.settle
{{ config(meta={'export': false}) }}

with lines as (
    select *
    from {{ ref('slv_lines') }}
    where player_id is not null
),

predictions as (
    select *
    from {{ ref('fct_prediction') }}
    where run_kind = 'nightly'
),

joined as (
    select
        l.*,
        p.game_id,
        p.box_score_game_id,
        p.model_revision,
        p.feature_version,
        p.player_name,
        coalesce(l.commence_time_utc, p.tip_off_utc) as tip_off_utc,
        p.prediction_generated_at,
        p.games_played_season,
        p.minutes_mean_last10,
        p.input_staleness_days,
        case l.market
            when 'player_points' then p.pred_pts
            when 'player_rebounds' then p.pred_reb
            when 'player_assists' then p.pred_ast
        end as prediction,
        case l.market
            when 'player_points' then p.baseline_last10_pts
            when 'player_rebounds' then p.baseline_last10_reb
            when 'player_assists' then p.baseline_last10_ast
        end as baseline_last10,
        case l.market
            when 'player_points' then {{ var('ledger_edge_threshold_pts') }}
            when 'player_rebounds' then {{ var('ledger_edge_threshold_reb') }}
            when 'player_assists' then {{ var('ledger_edge_threshold_ast') }}
        end as threshold
    from lines as l
    left join predictions as p on l.player_id = p.player_id and l.game_date = p.game_date
),

decided as (
    select
        *,
        prediction - line as edge,
        prediction is not null as has_prediction,
        coalesce(games_played_season >= {{ var('ledger_min_prior_games_season') }}, false) as prior_game_ok,
        coalesce(minutes_mean_last10 >= {{ var('ledger_min_minutes_mean_last10') }}, false) as minutes_ok,
        coalesce(input_staleness_days <= {{ var('ledger_max_input_staleness_days') }}, false) as staleness_ok,
        coalesce(prediction_generated_at < snapshot_at_utc, false) as prediction_before_line,
        coalesce(snapshot_at_utc < tip_off_utc, false) as line_before_tipoff,
        case
            when prediction is null or threshold is null then 'pass'
            when prediction - line >= threshold then 'over'
            when line - prediction >= threshold then 'under'
            else 'pass'
        end as side
    from joined
),

eligible as (
    select
        *,
        has_prediction and prior_game_ok and minutes_ok and staleness_ok
        and prediction_before_line and line_before_tipoff as eligible,
        case when side = 'over' then over_price when side = 'under' then under_price end as price
    from decided
),

settled as (
    select
        e.*,
        case e.market
            when 'player_points' then g.pts
            when 'player_rebounds' then g.reb
            when 'player_assists' then g.ast
        end as actual,
        d.game_id is not null as game_ingested,
        g.player_id is not null as played
    from eligible as e
    left join {{ ref('dim_game') }} as d on e.box_score_game_id = d.game_id
    left join {{ ref('fct_player_game') }} as g
        on e.player_id = g.player_id and e.box_score_game_id = g.game_id
),

outcomes as (
    select
        *,
        eligible and side in ('over', 'under') as is_bet,
        case
            when not (eligible and side in ('over', 'under')) then null
            when not game_ingested then 'pending'
            when not played or actual is null then 'void'
            when actual = line then 'push'
            when (actual > line) = (side = 'over') then 'won'
            else 'lost'
        end as outcome
    from settled
)

select
    cast(game_date as date) as game_date,
    cast(season as varchar) as season,
    cast(source as varchar) as source,
    cast(book as varchar) as book,
    cast(market as varchar) as market,
    cast(player_id as bigint) as player_id,
    cast(coalesce(player_name, player_name_raw) as varchar) as player_name,
    cast(player_name_raw as varchar) as player_name_raw,
    cast(team as varchar) as team,
    cast(capture_id as varchar) as capture_id,
    cast(event_id as varchar) as event_id,
    cast(game_id as varchar) as game_id,
    cast(box_score_game_id as varchar) as box_score_game_id,
    cast(model_revision as varchar) as model_revision,
    cast(feature_version as varchar) as feature_version,
    cast(line as double) as line,
    cast(over_price as integer) as over_price,
    cast(under_price as integer) as under_price,
    cast(snapshot_at_utc as timestamp) as snapshot_at_utc,
    cast(line_last_update_utc as timestamp) as line_last_update_utc,
    cast(prediction_generated_at as timestamp) as prediction_generated_at,
    cast(tip_off_utc as timestamp) as tip_off_utc,
    cast(prediction as double) as prediction,
    cast(baseline_last10 as double) as baseline_last10,
    cast(edge as double) as edge,
    cast(threshold as double) as threshold,
    cast(games_played_season as integer) as games_played_season,
    cast(minutes_mean_last10 as double) as minutes_mean_last10,
    cast(input_staleness_days as double) as input_staleness_days,
    cast(has_prediction as boolean) as has_prediction,
    cast(prior_game_ok as boolean) as prior_game_ok,
    cast(minutes_ok as boolean) as minutes_ok,
    cast(staleness_ok as boolean) as staleness_ok,
    cast(prediction_before_line as boolean) as prediction_before_line,
    cast(line_before_tipoff as boolean) as line_before_tipoff,
    cast(eligible as boolean) as eligible,
    cast(side as varchar) as side,
    cast(is_bet as boolean) as is_bet,
    cast(price as integer) as price,
    cast(
        case
            when price > 0 then 100.0 / (price + 100)
            when price < 0 then -price / (-price + 100.0)
        end as double
    ) as implied_probability,
    cast({{ var('ledger_unit') }} as double) as units,
    cast(actual as double) as actual,
    cast(game_ingested as boolean) as game_ingested,
    cast(played as boolean) as played,
    cast(outcome as varchar) as outcome,
    cast(
        case
            when outcome = 'won' and price > 0 then {{ var('ledger_unit') }} * price / 100.0
            when outcome = 'won' and price < 0 then {{ var('ledger_unit') }} * 100.0 / -price
            when outcome = 'lost' then -{{ var('ledger_unit') }}
            when outcome in ('push', 'void', 'pending') then 0.0
        end as double
    ) as profit_units,
    cast('{{ var("ledger_rule_version") }}' as varchar) as rule_version,
    cast('{{ var("ledger_rule_hash") }}' as varchar) as rule_hash
from outcomes
