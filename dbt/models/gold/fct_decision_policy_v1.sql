-- Grain: one row per (player_id, game_id, model_revision, population, target): the decision
-- policy (ADR-0001, ADR-0015) applied to every prediction with a last-10 baseline.
--
--   edge              prediction - baseline_last10 (both from the slate row, scored before tip-off)
--   decision          over if edge > threshold, under if edge < -threshold, else no_call; null
--                     until the policy artifact exists (brz_policy_report)
--   outcome           hit / miss / push once the box score exists (push: actual = baseline)
--   season_mean_*     the second causal baseline: the sign of season_mean - baseline_last10,
--                     where season_mean is the player's season-to-date mean before the game
--                     (fct_player_game.<stat>_mean_season_prior); null before the box score
--   band_*            prediction + residual quantiles of the population (50% = q25..q75,
--                     80% = q10..q90), in-sample on the holdout season (ADR-0006)
--
-- population is a column, not a filter (ADR-0009): every row appears under `all`, and rows
-- in the training population appear again under `min10` with that population's threshold
-- and bands. Thresholds come from reports/policy_<season>.json through brz_policy_report;
-- mart_policy_metrics and mart_policy_sweep recompute the artifact's numbers from these rows
-- and the singular tests assert_policy_* fail on any disagreement. Versioned (v1) because the
-- site and the exported parquet read it; the alias keeps the plain name.
{{ config(alias='fct_decision_policy') }}

with predictions as (
    select * from {{ ref('fct_prediction') }} where baseline_defined
),

both_populations as (
    select 'all' as population, * exclude (population) from predictions
    union all
    select 'min10' as population, * exclude (population) from predictions where in_metrics_population
),

history as (
    select player_id, game_id, pts_mean_season_prior, reb_mean_season_prior, ast_mean_season_prior
    from {{ ref('fct_player_game') }}
),

thresholds as (
    select population, target, threshold, band_q10, band_q25, band_q75, band_q90
    from {{ ref('brz_policy_report') }}
),

{% for t in var('targets') %}
edges_{{ t }} as (
    select
        p.player_id,
        p.game_id,
        p.model_revision,
        p.feature_version,
        p.dataset_revision,
        p.run_kind,
        p.game_date,
        p.season,
        p.player_name,
        p.team,
        p.opponent,
        p.home,
        p.population,
        '{{ t }}' as target,
        p.pred_{{ t }} as prediction,
        p.baseline_last10_{{ t }} as baseline_last10,
        h.{{ t }}_mean_season_prior as baseline_season,
        p.pred_{{ t }} - p.baseline_last10_{{ t }} as edge,
        th.threshold,
        p.pred_{{ t }} + th.band_q25 as band_low_50,
        p.pred_{{ t }} + th.band_q75 as band_high_50,
        p.pred_{{ t }} + th.band_q10 as band_low_80,
        p.pred_{{ t }} + th.band_q90 as band_high_80,
        p.actual_{{ t }} as actual,
        p.has_actual
    from both_populations as p
    left join history as h on p.player_id = h.player_id and p.game_id = h.game_id
    left join thresholds as th on p.population = th.population and th.target = '{{ t }}'
),

decided_{{ t }} as (
    select
        *,
        case
            when threshold is null then null
            when edge > threshold then 'over'
            when edge < -threshold then 'under'
            else 'no_call'
        end as decision,
        case
            when baseline_season is null then 'no_call'
            when baseline_season - baseline_last10 > 0 then 'over'
            when baseline_season - baseline_last10 < 0 then 'under'
            else 'no_call'
        end as season_mean_decision
    from edges_{{ t }}
),

final_{{ t }} as (
    select
        *,
        case
            when decision not in ('over', 'under') or not has_actual then null
            when actual = baseline_last10 then 'push'
            when (decision = 'over' and actual > baseline_last10)
                or (decision = 'under' and actual < baseline_last10) then 'hit'
            else 'miss'
        end as outcome,
        case
            when season_mean_decision not in ('over', 'under') or not has_actual then null
            when actual = baseline_last10 then 'push'
            when (season_mean_decision = 'over' and actual > baseline_last10)
                or (season_mean_decision = 'under' and actual < baseline_last10) then 'hit'
            else 'miss'
        end as season_mean_outcome,
        case when has_actual and threshold is not null then actual between band_low_50 and band_high_50 end as within_band_50,
        case when has_actual and threshold is not null then actual between band_low_80 and band_high_80 end as within_band_80
    from decided_{{ t }}
),
{% endfor %}

unioned as (
    {% for t in var('targets') %}
    select * from final_{{ t }}
    {{ "union all" if not loop.last }}
    {% endfor %}
)

select
    cast(player_id as bigint) as player_id,
    cast(game_id as varchar) as game_id,
    cast(model_revision as varchar) as model_revision,
    cast(feature_version as varchar) as feature_version,
    cast(dataset_revision as varchar) as dataset_revision,
    cast(run_kind as varchar) as run_kind,
    cast(game_date as date) as game_date,
    cast(season as varchar) as season,
    cast(player_name as varchar) as player_name,
    cast(team as varchar) as team,
    cast(opponent as varchar) as opponent,
    cast(home as boolean) as home,
    cast(population as varchar) as population,
    cast(target as varchar) as target,
    cast(prediction as double) as prediction,
    cast(baseline_last10 as double) as baseline_last10,
    cast(baseline_season as double) as baseline_season,
    cast(edge as double) as edge,
    cast(threshold as double) as threshold,
    cast(decision as varchar) as decision,
    cast(band_low_50 as double) as band_low_50,
    cast(band_high_50 as double) as band_high_50,
    cast(band_low_80 as double) as band_low_80,
    cast(band_high_80 as double) as band_high_80,
    cast(actual as double) as actual,
    cast(has_actual as boolean) as has_actual,
    cast(outcome as varchar) as outcome,
    cast(season_mean_decision as varchar) as season_mean_decision,
    cast(season_mean_outcome as varchar) as season_mean_outcome,
    cast(within_band_50 as boolean) as within_band_50,
    cast(within_band_80 as boolean) as within_band_80
from unioned
