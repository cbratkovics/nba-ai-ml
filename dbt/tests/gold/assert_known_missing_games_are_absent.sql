-- The seeded missing games really are absent (a game that appears would mean the seed and the
-- season exception are stale).
select k.game_id
from {{ ref('known_missing_games') }} as k
inner join {{ ref('dim_game') }} as g on k.game_id = g.game_id
