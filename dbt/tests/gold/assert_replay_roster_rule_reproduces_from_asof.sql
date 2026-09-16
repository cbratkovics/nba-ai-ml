-- The replay's trade-window rule (a player is slated only for the team of their most recent
-- game before the date) must reproduce from dim_player_asof: for every replayed prediction the
-- stint effective on its date names the same team.
select
    p.player_id,
    p.game_id,
    p.game_date,
    p.team as predicted_team,
    a.team as asof_team
from {{ ref('fct_prediction') }} as p
left join {{ ref('dim_player_asof') }} as a
    on
        p.player_id = a.player_id
        and p.game_date > a.effective_from_date
        and (a.effective_to_date is null or p.game_date <= a.effective_to_date)
where
    p.run_kind = 'replay'
    and (a.team is null or a.team <> p.team)
