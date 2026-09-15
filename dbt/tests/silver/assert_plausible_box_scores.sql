-- Plausibility, warn not fail: a single box score above var('plausible_max_pts') points,
-- var('plausible_max_reb') rebounds or var('plausible_max_ast') assists is reported unless the
-- known_stat_exceptions seed lists it with a verification (docs/reconciliation.md finding 9).
{{ config(severity='warn') }}

select
    g.player_id,
    g.game_id,
    g.game_date,
    g.player_name,
    g.pts,
    g.reb,
    g.ast
from {{ ref('slv_game_logs') }} as g
left join {{ ref('known_stat_exceptions') }} as k on g.game_id = k.game_id and g.player_id = k.player_id
where
    k.game_id is null
    and (
        g.pts > {{ var('plausible_max_pts') }}
        or g.reb > {{ var('plausible_max_reb') }}
        or g.ast > {{ var('plausible_max_ast') }}
    )
