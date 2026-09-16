-- reports/policy_<season>.json, the committed policy evaluation (ADR-0015), unpivoted to one
-- row per (population, target): the chosen threshold, the in-sample hit rates of the model
-- and of both causal baselines at that threshold, and the band quantiles. Optional: empty
-- (typed) until the artifact is written, so the first build of a fresh warehouse succeeds
-- and the policy marts are then filled by the second build (nba/decisions/evaluate.py).
{% set populations = ['min10', 'all'] %}
{% set selects = [] %}
{% for p in populations %}
{% for t in var('targets') %}
{% set b = 'populations.' ~ p ~ '.targets.' ~ t %}
{% do selects.append(
"select
    cast(filename as varchar) as source_file,
    cast(season as varchar) as season,
    cast(git_sha as varchar) as policy_commit,
    cast(model_revision as varchar) as model_revision,
    cast(model_commit as varchar) as model_commit,
    cast(in_sample as boolean) as in_sample,
    cast(min_coverage as double) as min_coverage,
    cast('" ~ p ~ "' as varchar) as population,
    cast('" ~ t ~ "' as varchar) as target,
    cast(" ~ b ~ ".n as integer) as n,
    cast(" ~ b ~ ".threshold as double) as threshold,
    cast(" ~ b ~ ".n_called as integer) as n_called,
    cast(" ~ b ~ ".coverage as double) as coverage,
    cast(" ~ b ~ ".n_resolved as integer) as n_resolved,
    cast(" ~ b ~ ".n_push as integer) as n_push,
    cast(" ~ b ~ ".n_hit as integer) as n_hit,
    cast(" ~ b ~ ".hit_rate as double) as hit_rate,
    cast(" ~ b ~ ".baselines.coin_flip.hit_rate as double) as coin_flip_hit_rate,
    cast(" ~ b ~ ".baselines.coin_flip.half_width_95 as double) as coin_flip_half_width_95,
    cast(" ~ b ~ ".baselines.season_mean_sign.n as integer) as season_mean_same_rows_n,
    cast(" ~ b ~ ".baselines.season_mean_sign.n_hit as integer) as season_mean_same_rows_n_hit,
    cast(" ~ b ~ ".baselines.season_mean_sign.n_miss as integer) as season_mean_same_rows_n_miss,
    cast(" ~ b ~ ".baselines.season_mean_sign.n_tie as integer) as season_mean_same_rows_n_tie,
    cast(" ~ b ~ ".baselines.season_mean_sign.n_missing as integer) as season_mean_same_rows_n_missing,
    cast(" ~ b ~ ".baselines.season_mean_sign.hit_rate as double) as season_mean_same_rows_hit_rate,
    cast(" ~ b ~ ".model_beats_both as boolean) as model_beats_both,
    cast(" ~ b ~ ".verdict as varchar) as verdict,
    cast(" ~ b ~ ".bands.quantiles.q10 as double) as band_q10,
    cast(" ~ b ~ ".bands.quantiles.q25 as double) as band_q25,
    cast(" ~ b ~ ".bands.quantiles.q75 as double) as band_q75,
    cast(" ~ b ~ ".bands.quantiles.q90 as double) as band_q90,
    cast(" ~ b ~ ".bands.coverage_50 as double) as band_coverage_50,
    cast(" ~ b ~ ".bands.coverage_80 as double) as band_coverage_80
from report") %}
{% endfor %}
{% endfor %}
{% if files_exist(var('reports_root') ~ '/policy_' ~ var('holdout_season') ~ '.json') %}
with report as (
    select * from {{ source('repo_reports', 'policy_report') }}
)

{{ selects | join('\nunion all\n') }}
{% else %}
{{ empty_typed_relation(policy_report_columns()) }}
{% endif %}
