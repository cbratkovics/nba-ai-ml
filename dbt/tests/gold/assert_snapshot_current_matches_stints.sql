-- The snapshot's current version (team of the most recent game at the last build) must agree
-- with the current stint derived from the game logs for every snapshotted player.
select
    c.player_id,
    c.team as snapshot_team,
    s.team as stint_team
from {{ ref('dim_player_current') }} as c
left join {{ ref('dim_player_asof') }} as s on c.player_id = s.player_id and s.is_current
where s.team is null or s.team <> c.team
