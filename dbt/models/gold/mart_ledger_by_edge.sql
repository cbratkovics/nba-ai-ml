-- Grain: one row per (season, source_group, market, edge_bucket): the ledger's hit rate and
-- ROI by how far the prediction sat from the line, ratios of sums over fct_paper_bet. The
-- buckets start at the rule's threshold; a bet below it cannot exist (ADR-0030).
with bets as (
    select
        *,
        case
            when abs(edge) < 2.5 then '[threshold, 2.5)'
            when abs(edge) < 3.5 then '[2.5, 3.5)'
            when abs(edge) < 5 then '[3.5, 5)'
            else '[5, inf)'
        end as edge_bucket
    from {{ ref('fct_paper_bet') }}
    where is_bet
),

grouped as (
    {% for grp in ['api', 'manual', 'all'] %}
    select
        season,
        '{{ grp }}' as source_group,
        market,
        edge_bucket,
        count(*) as n_bets,
        count(*) filter (where outcome in ('won', 'lost')) as n_settled,
        count(*) filter (where outcome = 'won') as n_won,
        count(*) filter (where outcome = 'pending') as n_pending,
        sum(units) filter (where outcome in ('won', 'lost')) as units_staked,
        sum(profit_units) filter (where outcome in ('won', 'lost')) as units_profit
    from bets
    {% if grp != 'all' %}where source = '{{ grp }}'{% endif %}
    group by season, market, edge_bucket
    {{ "union all" if not loop.last }}
    {% endfor %}
)

select
    cast(season as varchar) as season,
    cast(source_group as varchar) as source_group,
    cast(market as varchar) as market,
    cast(edge_bucket as varchar) as edge_bucket,
    cast(n_bets as integer) as n_bets,
    cast(n_settled as integer) as n_settled,
    cast(n_won as integer) as n_won,
    cast(n_pending as integer) as n_pending,
    cast(case when n_settled > 0 then n_won / n_settled end as double) as hit_rate,
    cast(units_staked as double) as units_staked,
    cast(units_profit as double) as units_profit,
    cast(case when units_staked > 0 then units_profit / units_staked end as double) as roi_units,
    cast('out_of_sample' as varchar) as evaluation
from grouped
