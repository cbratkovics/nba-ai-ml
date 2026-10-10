-- Grain: one row per (season, source_group, book, market): the paper ledger's record, as ratios
-- of sums over fct_paper_bet (ADR-0030). source_group is api, manual or all (the brief: API
-- and hand-entered lines reported separately as well as together). Every row carries its
-- sample size and 95% half-widths; docs/METRICS.md says how many bets the numbers need.
--
--   hit_rate              won / (won + lost); pushes and voids are counted, not scored
--   breakeven_hit_rate    mean implied probability of the prices taken on settled bets
--   roi_units             profit / units staked on settled (won + lost) bets
with bets as (
    select * from {{ ref('fct_paper_bet') }} where is_bet
),

grouped as (
    {% for grp in ['api', 'manual', 'all'] %}
    select
        season,
        '{{ grp }}' as source_group,
        {% if grp == 'all' %}'all'{% else %}book{% endif %} as book_group,
        market,
        count(*) as n_bets,
        count(*) filter (where outcome in ('won', 'lost')) as n_settled,
        count(*) filter (where outcome = 'won') as n_won,
        count(*) filter (where outcome = 'lost') as n_lost,
        count(*) filter (where outcome = 'push') as n_push,
        count(*) filter (where outcome = 'void') as n_void,
        count(*) filter (where outcome = 'pending') as n_pending,
        sum(implied_probability) filter (where outcome in ('won', 'lost')) as sum_implied,
        sum(units) filter (where outcome in ('won', 'lost')) as units_staked,
        sum(profit_units) filter (where outcome in ('won', 'lost')) as units_profit,
        stddev_samp(profit_units) filter (where outcome in ('won', 'lost')) as profit_sd,
        count(distinct game_date) as nights_with_bets
    from bets
    {% if grp != 'all' %}where source = '{{ grp }}'{% endif %}
    group by season, {% if grp == 'all' %}market{% else %}book, market{% endif %}
    {{ "union all" if not loop.last }}
    {% endfor %}
)

select
    cast(season as varchar) as season,
    cast(source_group as varchar) as source_group,
    cast(book_group as varchar) as book,
    cast(market as varchar) as market,
    cast(n_bets as integer) as n_bets,
    cast(n_settled as integer) as n_settled,
    cast(n_won as integer) as n_won,
    cast(n_lost as integer) as n_lost,
    cast(n_push as integer) as n_push,
    cast(n_void as integer) as n_void,
    cast(n_pending as integer) as n_pending,
    cast(nights_with_bets as integer) as nights_with_bets,
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
