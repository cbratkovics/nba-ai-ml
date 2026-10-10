import Link from 'next/link'
import CoverageCurveChart from '@/components/CoverageCurveChart'
import DecisionsTable from '@/components/DecisionsTable'
import {
  POLICY_SUMMARY,
  POPULATIONS,
  POPULATION_LABEL,
  REPLAY_SEASON,
  TARGETS,
  TARGET_LABEL,
  getGoldManifest,
  getGoldMart,
  getLatestDecisions,
  getReplayDaily,
  goldSourceNote,
  modelIdentity,
  pinnedRows,
  type GoldExportManifestRow,
  type GoldPolicyMetric,
  type GoldPolicySweepPoint,
  type PolicyTarget,
  type Population,
  type Target,
} from '@/lib/data'

export const revalidate = 3600

const pct = (v: number | null | undefined, digits = 1) => (v === null || v === undefined ? '–' : `${(v * 100).toFixed(digits)}%`)

interface PolicyView {
  source: 'gold' | 'artifact'
  targets: Record<Target, PolicyTarget>
  note: string
}

/**
 * The replay evaluation of the policy for one population. From the warehouse's
 * mart_policy_metrics (hit rates, coverage, both baselines, band coverage) and
 * mart_policy_sweep (the coverage curve) when the gold export is published; two dbt tests
 * hold those rows to the committed artifact, which is the fallback. The thresholds, the band
 * quantiles and the verdict sentence are policy parameters and text from the artifact in
 * both cases, since the marts apply them rather than recompute them.
 */
function policyView(
  population: Population,
  metrics: GoldPolicyMetric[],
  sweep: GoldPolicySweepPoint[],
  manifest: GoldExportManifestRow[] | null,
): PolicyView {
  const artifact = POLICY_SUMMARY.populations[population].targets
  const rows = metrics.filter((m) => m.population === population)
  const complete = TARGETS.every((t) => rows.some((m) => m.target === t))
  if (!complete) {
    return {
      source: 'artifact',
      targets: artifact,
      note: `Source: the committed artifact ${POLICY_SUMMARY.source_file} (the gold exports mart_policy_metrics.json and mart_policy_sweep.json were not available at render time).`,
    }
  }
  const targets = {} as Record<Target, PolicyTarget>
  for (const t of TARGETS) {
    const m = rows.find((x) => x.target === t)!
    const a = artifact[t]
    const curve = sweep
      .filter((p) => p.population === population && p.target === t)
      .sort((x, y) => x.threshold - y.threshold)
      .map((p) => ({
        threshold: p.threshold,
        coverage: p.coverage,
        n_resolved: p.n_resolved,
        hit_rate: p.hit_rate,
        season_mean_same_rows_hit_rate: p.season_mean_same_rows_hit_rate,
        season_mean_own_hit_rate: p.season_mean_own_hit_rate,
      }))
    targets[t] = {
      n: m.n,
      threshold: m.threshold,
      n_called: m.n_called,
      coverage: m.coverage,
      n_resolved: m.n_resolved,
      n_push: m.n_push,
      n_hit: m.n_hit,
      hit_rate: m.hit_rate,
      baselines: {
        coin_flip: { hit_rate: m.coin_flip_hit_rate, n: m.n_resolved, half_width_95: m.coin_flip_half_width_95 },
        season_mean_sign: {
          n: m.season_mean_same_rows_n,
          n_hit: m.season_mean_same_rows_n_hit,
          n_miss: m.season_mean_same_rows_n_miss,
          n_tie: m.season_mean_same_rows_n_tie,
          n_missing: m.season_mean_same_rows_n_missing,
          hit_rate: m.season_mean_same_rows_hit_rate,
          same_rows: true,
          abstentions_scored_as: a.baselines.season_mean_sign.abstentions_scored_as,
          own_threshold_n_called: a.baselines.season_mean_sign.own_threshold_n_called,
          own_threshold_hit_rate: a.baselines.season_mean_sign.own_threshold_hit_rate,
        },
      },
      model_beats_both: m.model_beats_both,
      verdict: a.verdict,
      bands: { quantiles: a.bands.quantiles, coverage_50: m.band_coverage_50, coverage_80: m.band_coverage_80 },
      coverage_curve: curve.length > 0 ? curve : a.coverage_curve,
    }
  }
  return {
    source: 'gold',
    targets,
    note: `Source: ${goldSourceNote('mart_policy_metrics', manifest)} and mart_policy_sweep.json, ${REPLAY_SEASON} replay rows; the dbt tests assert_policy_metrics_reconcile_to_policy_report and assert_policy_sweep_reconciles_to_policy_report hold these rows to the committed artifact ${POLICY_SUMMARY.source_file} (counts exact, rates within 1e-6), which supplies the thresholds, the band quantiles and the verdict text.`,
  }
}

function PolicyTable({ targets }: { targets: Record<Target, PolicyTarget> }) {
  return (
    <div className="mt-4 overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-white/10 text-left text-xs uppercase tracking-wide text-text-secondary">
            <th className="py-2 pr-4">Target</th>
            <th className="py-2 pr-4 text-right">Threshold</th>
            <th className="py-2 pr-4 text-right">Coverage</th>
            <th className="py-2 pr-4 text-right">Resolved calls</th>
            <th className="py-2 pr-4 text-right">Model hit rate</th>
            <th className="py-2 pr-4 text-right">Season-mean sign, same rows</th>
            <th className="py-2 pr-4 text-right">Coin flip 95% band</th>
            <th className="py-2 pr-4">Beats both</th>
          </tr>
        </thead>
        <tbody>
          {TARGETS.map((t) => {
            const tb: PolicyTarget = targets[t]
            const half = tb.baselines.coin_flip.half_width_95
            return (
              <tr key={t} className="border-b border-white/5 text-text-primary">
                <td className="py-2 pr-4">{TARGET_LABEL[t]}</td>
                <td className="py-2 pr-4 text-right tabular-nums">{tb.threshold.toFixed(2)}</td>
                <td className="py-2 pr-4 text-right tabular-nums">
                  {pct(tb.coverage)} ({tb.n_called.toLocaleString()} of {tb.n.toLocaleString()})
                </td>
                <td className="py-2 pr-4 text-right tabular-nums">
                  {tb.n_resolved.toLocaleString()} ({tb.n_push} pushes)
                </td>
                <td className="py-2 pr-4 text-right tabular-nums">{pct(tb.hit_rate)}</td>
                <td className="py-2 pr-4 text-right tabular-nums">
                  {pct(tb.baselines.season_mean_sign.hit_rate)} ({tb.baselines.season_mean_sign.n_tie + tb.baselines.season_mean_sign.n_missing}{' '}
                  abstentions)
                </td>
                <td className="py-2 pr-4 text-right tabular-nums">
                  {half === null ? '–' : `50% ± ${(half * 100).toFixed(1)}`}
                </td>
                <td className={`py-2 pr-4 ${tb.model_beats_both ? 'text-success' : 'text-danger'}`}>
                  {tb.model_beats_both ? 'yes' : 'no'}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function BandsTable({ targets }: { targets: Record<Target, PolicyTarget> }) {
  return (
    <div className="mt-4 overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-white/10 text-left text-xs uppercase tracking-wide text-text-secondary">
            <th className="py-2 pr-4">Target</th>
            <th className="py-2 pr-4 text-right">q10</th>
            <th className="py-2 pr-4 text-right">q25</th>
            <th className="py-2 pr-4 text-right">q75</th>
            <th className="py-2 pr-4 text-right">q90</th>
            <th className="py-2 pr-4 text-right">Inside 50% band</th>
            <th className="py-2 pr-4 text-right">Inside 80% band</th>
          </tr>
        </thead>
        <tbody>
          {TARGETS.map((t) => {
            const b = targets[t].bands
            return (
              <tr key={t} className="border-b border-white/5 text-text-primary">
                <td className="py-2 pr-4">{TARGET_LABEL[t]}</td>
                {(['q10', 'q25', 'q75', 'q90'] as const).map((q) => (
                  <td key={q} className="py-2 pr-4 text-right tabular-nums">
                    {b.quantiles[q] > 0 ? '+' : ''}
                    {b.quantiles[q].toFixed(2)}
                  </td>
                ))}
                <td className="py-2 pr-4 text-right tabular-nums">{pct(b.coverage_50)}</td>
                <td className="py-2 pr-4 text-right tabular-nums">{pct(b.coverage_80)}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

export default async function DecisionsPage() {
  const [latest, daily, goldMetrics, goldSweep, manifest] = await Promise.all([
    getLatestDecisions(),
    getReplayDaily(),
    getGoldMart<GoldPolicyMetric>('mart_policy_metrics'),
    getGoldMart<GoldPolicySweepPoint>('mart_policy_sweep'),
    getGoldManifest(),
  ])
  const d = latest.data
  const s = POLICY_SUMMARY
  const metrics = pinnedRows(goldMetrics.data, REPLAY_SEASON)
  const sweep = pinnedRows(goldSweep.data, REPLAY_SEASON)
  const views = Object.fromEntries(
    POPULATIONS.map((p) => [p, policyView(p, metrics, sweep, manifest.data)]),
  ) as Record<Population, PolicyView>
  const losing = POPULATIONS.filter((p) => !TARGETS.every((t) => views[p].targets[t].model_beats_both))

  return (
    <div className="space-y-8">
      <section className="glass-card p-8">
        <h1 className="text-2xl font-bold text-text-primary">Decisions</h1>
        <p className="mt-4 max-w-3xl text-text-secondary">
          A line-free directional call per player and stat: <span className="text-text-primary">over</span> when the
          model&apos;s prediction exceeds the player&apos;s last-10-game mean by more than a threshold,{' '}
          <span className="text-text-primary">under</span> when it falls short by more, otherwise no call. There are no
          sportsbook lines on this page (the{' '}
          <Link href="/ledger" className="text-secondary hover:underline">
            paper ledger
          </Link>{' '}
          is where real lines enter). A call resolves against the same last-10 mean once the box score exists. Thresholds
          and the bands around each prediction were chosen per population on the {s.season} replay rows, so the hit rates
          below are in-sample policy-fit results, not out-of-sample validation or demonstrated betting performance; the
          full coverage curve is shown so any other threshold can be read off.
        </p>
      </section>

      <section className="glass-card p-8">
        <h2 className="text-xl font-semibold text-text-primary">Today&apos;s calls</h2>
        {d ? (
          <>
            <p className="mt-2 text-sm text-text-secondary">
              {d.date}: {d.n_games} games, {d.n_players} players, model {modelIdentity(d.model_revision)}. Calls under the{' '}
              {POPULATION_LABEL.min10.toLowerCase()} policy (thresholds pts {d.policy.populations.min10.targets.pts.threshold}, reb{' '}
              {d.policy.populations.min10.targets.reb.threshold}, ast {d.policy.populations.min10.targets.ast.threshold}):{' '}
              {TARGETS.map((t) => `${d.n_calls.min10[t]} ${TARGET_LABEL[t].toLowerCase()}`).join(', ')}. Whether a slated
              player ends up in the training population (10+ minutes) is only known after the game; the all-rows policy,
              where the model does not beat the season-mean sign, is in the file too.
            </p>
            <div className="mt-6">
              <DecisionsTable rows={d.rows} population="min10" />
            </div>
            <p className="mt-3 text-xs text-text-secondary">
              Source: decisions/latest.json in the dataset repo, written by the nightly job from the slate and the committed
              policy ({d.policy.report}).
            </p>
          </>
        ) : (
          <div className="mt-4 space-y-2 text-text-secondary">
            <p className="text-text-primary">No slate today, so no calls.</p>
            <p>
              The nightly job writes decisions only on days with a slate, and the 2026-27 season starts in October. The last
              replayed date was {daily.data?.last_date ?? 'not published'}
              {daily.data ? ` (${daily.data.season} replay)` : ''}. Until then this page shows the replay evaluation of the
              policy.
            </p>
            <p className="text-xs">
              Checked {latest.url} (HTTP {latest.status}).
            </p>
          </div>
        )}
      </section>

      {POPULATIONS.map((p) => {
        const block = s.populations[p]
        const view = views[p]
        const beatsEverywhere = TARGETS.every((t) => view.targets[t].model_beats_both)
        return (
          <section key={p} className="glass-card p-8">
            <h2 className="text-xl font-semibold text-text-primary">
              Replay evaluation, {POPULATION_LABEL[p].toLowerCase()} ({view.targets.pts.n.toLocaleString()} rows)
            </h2>
            <p className="mt-2 text-sm text-text-secondary">
              Population: {block.description}. Two causal baselines are scored on the rows the model calls: a coin flip
              (50% by definition, with its 95% band at that many calls) and the sign of the player&apos;s season-to-date mean
              minus the last-10 mean, which is also known before tip-off. Both are scored on exactly the rows the model
              resolved, one n per comparison; where the season-mean sign has no side (a tie with the last-10 mean, or no
              season mean on a season debut) it is scored as a coin flip, 0.5, and those abstentions are counted. Threshold
              rule: the largest grid value that still calls at least {(s.min_coverage * 100).toFixed(0)}% of the rows.
            </p>
            {!beatsEverywhere && (
              <p className="mt-3 text-sm text-danger">
                {TARGETS.filter((t) => !view.targets[t].model_beats_both)
                  .map((t) => view.targets[t].verdict)
                  .join(' ')}
              </p>
            )}
            {beatsEverywhere && (
              <p className="mt-3 text-sm text-success">{TARGETS.map((t) => view.targets[t].verdict).join(' ')}</p>
            )}
            <PolicyTable targets={view.targets} />
            <h3 className="mt-6 text-sm font-semibold uppercase tracking-wide text-text-secondary">Coverage curve</h3>
            <div className="mt-2">
              <CoverageCurveChart targets={view.targets} />
            </div>
            <h3 className="mt-6 text-sm font-semibold uppercase tracking-wide text-text-secondary">
              Bands: residual quantiles (actual minus prediction)
            </h3>
            <BandsTable targets={view.targets} />
            <p className="mt-3 text-xs text-text-secondary">{view.note}</p>
          </section>
        )
      })}

      <section className="glass-card p-8">
        <h2 className="text-xl font-semibold text-text-primary">Provenance</h2>
        <p className="mt-2 text-sm text-text-secondary">
          {s.in_sample_note} Artifact {s.source_file} (sha256 {s.source_sha256.slice(0, 12)}), written{' '}
          {s.generated_at.slice(0, 10)} at code commit {s.git_sha.slice(0, 7)} for model{' '}
          {modelIdentity(s.model_revision, s.model_commit)}. The warehouse&apos;s decisions mart applies the same thresholds
          to every prediction and recomputes these numbers; a test fails on any disagreement.{' '}
          {losing.length > 0
            ? `The model policy is not recommended on ${losing.map((p) => POPULATION_LABEL[p].toLowerCase()).join(' and ')}.`
            : ''}{' '}
          <Link href="/" className="text-secondary hover:underline">
            Holdout results
          </Link>
          .
        </p>
      </section>
    </div>
  )
}
