-- The replay rows in the warehouse must reproduce reports/replay_<season>.json: on the
-- training population (min10) the model MAE and n equal mae_restricted / n_restricted, and on
-- all rows they equal mae_unrestricted / n_with_actuals, within var('tol_replay') (float noise;
-- the mart recomputes the same rows). Any row returned is a disagreement.
with computed as (
    select season, model_revision, population, target, n, model_mae
    from {{ ref('mart_holdout_metrics') }}
    where run_kind = 'replay'
),

published as (
    select
        season,
        model_revision,
        target,
        'min10' as population,
        n_restricted as n,
        mae_restricted as mae
    from {{ ref('brz_replay_report') }}
    union all
    select
        season,
        model_revision,
        target,
        'all' as population,
        n_with_actuals as n,
        mae_unrestricted as mae
    from {{ ref('brz_replay_report') }}
)

select
    p.season,
    p.population,
    p.target,
    p.n as published_n,
    c.n as computed_n,
    p.mae as published_mae,
    c.model_mae as computed_mae
from published as p
left join computed as c
    on p.season = c.season and p.model_revision = c.model_revision
    and p.population = c.population and p.target = c.target
where
    c.n is null
    or p.n <> c.n
    or abs(p.mae - c.model_mae) > {{ var('tol_replay') }}
