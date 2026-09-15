-- Grain: one row per player_id, the current version of the snp_player_team snapshot (SCD2):
-- the team of the player's most recent game as of the last build. Not exported.
{{ config(materialized='view', meta={'export': false}) }}

select
    cast(player_id as bigint) as player_id,
    cast(player_name as varchar) as player_name,
    cast(team as varchar) as team,
    cast(as_of_game_date as date) as as_of_game_date,
    cast(as_of_game_id as varchar) as as_of_game_id,
    cast(dbt_valid_from as timestamp) as version_valid_from_utc
from {{ ref('snp_player_team') }}
where dbt_valid_to is null
