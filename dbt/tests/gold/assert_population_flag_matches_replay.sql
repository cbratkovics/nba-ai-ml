-- fct_player_game.population must equal the training-population flag the replay residuals
-- carry (in_metrics_population), row for row over every replayed row with a box score: the
-- warehouse's history counts reproduce the feature module's "both baselines defined" rule.
select
    r.player_id,
    r.game_id,
    r.in_metrics_population,
    g.population,
    g.minutes,
    g.prior_games_season
from {{ ref('slv_residuals') }} as r
inner join {{ ref('fct_player_game') }} as g on r.player_id = g.player_id and r.game_id = g.game_id
where
    r.run_kind = 'replay'
    and r.has_actual
    and r.in_metrics_population <> (g.population = 'min10')
