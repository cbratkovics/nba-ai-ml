{#- SCD Type 2 history of each player's current team, derived from the latest game in the
    game logs: `check` strategy on team, so a new version is captured on the first build after
    a player's newest game is with a different team. as_of_game_date / as_of_game_id say which
    game the version was derived from. History starts at the first build; earlier team changes
    are answered by dim_player_asof, which derives stints from the game logs themselves. -#}
{% snapshot snp_player_team %}

{{
    config(
        schema='snapshots',
        unique_key='player_id',
        strategy='check',
        check_cols=['team'],
        hard_deletes='ignore'
    )
}}

with latest as (
    select
        player_id,
        player_name,
        team,
        game_date as as_of_game_date,
        game_id as as_of_game_id,
        row_number() over (partition by player_id order by game_date desc, game_id desc) as rn
    from {{ ref('slv_game_logs') }}
)

select
    player_id,
    player_name,
    team,
    as_of_game_date,
    as_of_game_id
from latest
where rn = 1

{% endsnapshot %}
