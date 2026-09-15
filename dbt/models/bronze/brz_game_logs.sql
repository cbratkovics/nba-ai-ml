-- Typed one-to-one copy of the per-season game-log parquet files; every row carries the
-- file it came from and the dataset-repo revision the loader pulled.
select
    cast(g.filename as varchar) as source_file,
    cast(m.dataset_revision as varchar) as dataset_revision,
    cast(g.game_id as varchar) as game_id,
    cast(g.game_date as date) as game_date,
    cast(g.season as varchar) as season,
    cast(g.player_id as bigint) as player_id,
    cast(g.player_name as varchar) as player_name,
    cast(g.team as varchar) as team,
    cast(g.opponent as varchar) as opponent,
    cast(g.home as boolean) as home,
    cast(g.minutes as double) as minutes,
    cast(g.pts as integer) as pts,
    cast(g.reb as integer) as reb,
    cast(g.ast as integer) as ast,
    cast(g.fgm as integer) as fgm,
    cast(g.fga as integer) as fga,
    cast(g.fg3m as integer) as fg3m,
    cast(g.fg3a as integer) as fg3a,
    cast(g.ftm as integer) as ftm,
    cast(g.fta as integer) as fta,
    cast(g.oreb as integer) as oreb,
    cast(g.dreb as integer) as dreb,
    cast(g.stl as integer) as stl,
    cast(g.blk as integer) as blk,
    cast(g.tov as integer) as tov,
    cast(g.pf as integer) as pf,
    cast(g.plus_minus as integer) as plus_minus,
    cast(g.source as varchar) as source
from {{ source('warehouse_files', 'game_logs') }} as g
cross join {{ ref('brz_load_manifest') }} as m
