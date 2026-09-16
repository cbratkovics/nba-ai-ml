-- Every point of the artifact's coverage curves must be reproduced by mart_policy_sweep on
-- the replay rows: counts exactly, rates within var('tol_policy'). Any row returned is a
-- disagreement. Empty until the artifact exists.
with published as (
    select * from {{ ref('brz_policy_curve') }}
),

computed as (
    select *
    from {{ ref('mart_policy_sweep') }}
    where run_kind = 'replay' and season = '{{ var("holdout_season") }}'
)

select
    p.population,
    p.target,
    p.threshold,
    p.n_called as published_n_called,
    c.n_called as computed_n_called,
    p.n_hit as published_n_hit,
    c.n_hit as computed_n_hit,
    p.hit_rate as published_hit_rate,
    c.hit_rate as computed_hit_rate,
    p.season_mean_own_n_called as published_season_mean_own_n_called,
    c.season_mean_own_n_called as computed_season_mean_own_n_called
from published as p
left join computed as c
    on p.population = c.population and p.target = c.target and p.threshold = c.threshold
        and p.model_revision = c.model_revision
where
    c.n_called is null
    or p.n_called <> c.n_called
    or p.n_resolved <> c.n_resolved
    or p.n_push <> c.n_push
    or p.n_hit <> c.n_hit
    or p.net_correct <> c.net_correct
    or p.season_mean_same_rows_n <> c.season_mean_same_rows_n
    or p.season_mean_own_n_called <> c.season_mean_own_n_called
    or abs(p.coverage - c.coverage) > {{ var('tol_policy') }}
    or coalesce(abs(p.hit_rate - c.hit_rate) > {{ var('tol_policy') }}, p.hit_rate is not null or c.hit_rate is not null)
    or coalesce(abs(p.season_mean_same_rows_hit_rate - c.season_mean_same_rows_hit_rate) > {{ var('tol_policy') }}, p.season_mean_same_rows_hit_rate is not null or c.season_mean_same_rows_hit_rate is not null)
    or coalesce(abs(p.season_mean_own_hit_rate - c.season_mean_own_hit_rate) > {{ var('tol_policy') }}, p.season_mean_own_hit_rate is not null or c.season_mean_own_hit_rate is not null)
