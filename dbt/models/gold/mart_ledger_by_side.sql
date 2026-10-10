-- Grain: one row per (season, source_group, market, side): the ledger's hit rate, n and ROI for
-- overs and unders separately (ADR-0030 amendment 3). The model is a squared-error regressor
-- (a conditional mean) and a book's line sits near the median, so on a right-skewed stat like
-- points the rule may lean to overs; this mart shows whether it does and how each side fares.
-- Ratios of sums; nothing is adjusted for.
with bets as (
    select * from {{ ref('fct_paper_bet') }} where is_bet
),

grouped as (
    {% for grp in ['api', 'manual', 'all'] %}
    select
        season,
        '{{ grp }}' as source_group,
        market,
        side,
        count(*) as n_bets,
        count(*) filter (where outcome in ('won', 'lost')) as n_settled,
        count(*) filter (where outcome = 'won') as n_won,
        count(*) filter (where outcome = 'pending') as n_pending,
        sum(implied_probability) filter (where outcome in ('won', 'lost')) as sum_implied,
        sum(units) filter (where outcome in ('won', 'lost')) as units_staked,
        sum(profit_units) filter (where outcome in ('won', 'lost')) as units_profit,
        stddev_samp(profit_units) filter (where outcome in ('won', 'lost')) as profit_sd
    from bets
    {% if grp != 'all' %}where source = '{{ grp }}'{% endif %}
    group by season, market, side
    {{ "union all" if not loop.last }}
    {% endfor %}
)

select
    cast(season as varchar) as season,
    cast(source_group as varchar) as source_group,
    cast(source_group = 'api' as boolean) as is_headline,
    cast(market as varchar) as market,
    cast(side as varchar) as side,
    cast(n_bets as integer) as n_bets,
    cast(n_settled as integer) as n_settled,
    cast(n_won as integer) as n_won,
    cast(n_pending as integer) as n_pending,
    cast(case when n_settled > 0 then n_won / n_settled end as double) as hit_rate,
    cast(
        case
            when n_settled > 0
                then 1.96 * sqrt((n_won / n_settled) * (1 - n_won / n_settled) / n_settled)
        end as double
    ) as hit_rate_half_width_95,
    cast(case when n_settled > 0 then sum_implied / n_settled end as double) as breakeven_hit_rate,
    cast(units_staked as double) as units_staked,
    cast(units_profit as double) as units_profit,
    cast(case when units_staked > 0 then units_profit / units_staked end as double) as roi_units,
    cast(
        case when n_settled > 1 then 1.96 * profit_sd / sqrt(n_settled) end as double
    ) as roi_half_width_95,
    cast('out_of_sample' as varchar) as evaluation
from grouped
