import Link from 'next/link'
import { ExternalLink } from 'lucide-react'
import {
  ALL_ROWS_BASELINE,
  LINKS,
  POPULATION,
  REPLAY_SEASON,
  TARGETS,
  TARGET_LABEL,
  getGoldManifest,
  getGoldMart,
  getMetricsReport,
  getReplaySummary,
  goldSourceNote,
  modelIdentity,
  pinnedRows,
  type GoldHoldoutMetric,
  type Target,
} from '@/lib/data'

export const revalidate = 3600

const PREDICTORS = [
  { key: 'model', label: 'LightGBM model' },
  { key: 'baseline_last10', label: 'Last-10-game mean' },
  { key: 'baseline_season', label: 'Season-to-date mean' },
] as const

/**
 * The all-rows comparison: from the warehouse's mart_holdout_metrics (population `all` on the
 * replay rows, which a dbt test reconciles to the all-rows report) when the gold export is
 * published, otherwise from the committed copy of that report. Both carry the same numbers;
 * the page says which one it rendered.
 */
function allRowsView(gold: GoldHoldoutMetric[], manifest: Parameters<typeof goldSourceNote>[1]) {
  const rows = gold.filter((g) => g.population === 'all')
  const complete = TARGETS.every((t) => rows.some((g) => g.target === t))
  if (complete) {
    const model = {} as Record<Target, number | null>
    const baseline = {} as Record<Target, number | null>
    for (const t of TARGETS) {
      const g = rows.find((x) => x.target === t)!
      model[t] = g.model_mae
      baseline[t] = g.baseline_last10_mae
    }
    return {
      source: 'gold' as const,
      n: Math.max(...rows.map((g) => g.n)),
      nDates: null as number | null,
      model,
      baseline,
      baselineWins: TARGETS.filter((t) => baseline[t]! < model[t]!),
      note: `Source: ${goldSourceNote('mart_holdout_metrics', manifest)}, population all on the ${REPLAY_SEASON} replay rows; the warehouse test assert_daily_metrics_reconcile_to_all_rows_report holds these rows to the committed report ${ALL_ROWS_BASELINE.source_file}, which is the fallback.`,
    }
  }
  const d = ALL_ROWS_BASELINE
  return {
    source: 'committed' as const,
    n: d.n,
    nDates: d.n_dates as number | null,
    model: d.model_mae,
    baseline: d.baseline_last10_mae,
    baselineWins: d.baseline_wins,
    note: `Source: ${d.source_file} in the repository, a committed copy of the replay's daily MAE file in the dataset repo (the gold export mart_holdout_metrics.json was not available at render time).`,
  }
}

export default async function HomePage() {
  const [metrics, replay, holdout, manifest] = await Promise.all([
    getMetricsReport(),
    getReplaySummary(),
    getGoldMart<GoldHoldoutMetric>('mart_holdout_metrics'),
    getGoldManifest(),
  ])
  const m = metrics.data
  const r = replay.data
  const gold = pinnedRows(holdout.data, REPLAY_SEASON)
  const allRows = allRowsView(gold, manifest.data)
  const baselineWins = allRows.baselineWins
  const goldMin10 = TARGETS.map((t) => gold.find((g) => g.population === 'min10' && g.target === t) ?? null)

  return (
    <div className="space-y-8">
      <section className="glass-card p-8">
        <h1 className="gradient-text text-3xl font-bold">NBA Stat Predictor</h1>
        <p className="mt-4 max-w-3xl text-text-secondary">
          A batch pipeline that predicts a player&apos;s points, rebounds, and assists for a game
          from that player&apos;s history before the game. One LightGBM regressor per target is
          trained on four seasons of box scores and evaluated on a full held-out season against
          two simple baselines. Every number on this site is read from files published by the
          pipeline; nothing is typed in by hand.
        </p>
        <ul className="mt-6 flex flex-wrap gap-4 text-sm">
          {[
            ['Dataset (Hugging Face)', LINKS.dataset],
            ['Model (Hugging Face)', LINKS.model],
            ['Code (GitHub)', LINKS.github],
            ['Reconciliation notes', LINKS.reconciliation],
          ].map(([label, href]) => (
            <li key={href}>
              <a
                href={href}
                className="inline-flex items-center gap-1 text-secondary hover:underline"
                target="_blank"
                rel="noopener noreferrer"
              >
                {label}
                <ExternalLink className="h-3 w-3" aria-hidden="true" />
              </a>
            </li>
          ))}
        </ul>
      </section>

      <section className="glass-card p-8">
        <h2 className="text-xl font-semibold text-text-primary">Holdout results (training population)</h2>
        {m ? (
          <>
            <p className="mt-2 text-sm text-text-secondary">
              Trained on {m.split.train_seasons.join(', ')} ({m.split.train_dates.start} to{' '}
              {m.split.train_dates.end}, {m.split.n_train_rows.toLocaleString()} player-games).
              Held out: all of {m.split.holdout_season} ({m.split.holdout_dates.start} to{' '}
              {m.split.holdout_dates.end}). Rows are limited to games where the player logged at
              least {m.split.min_minutes} minutes; all three predictors are scored on the same{' '}
              {m.metrics.pts.model.n.toLocaleString()} holdout rows.
            </p>
            <div className="mt-4 overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-white/10 text-left text-xs uppercase tracking-wide text-text-secondary">
                    <th className="py-2 pr-4">Target</th>
                    <th className="py-2 pr-4">Predictor</th>
                    <th className="py-2 pr-4 text-right">MAE</th>
                    <th className="py-2 pr-4 text-right">RMSE</th>
                    <th className="py-2 pr-4 text-right">R²</th>
                  </tr>
                </thead>
                <tbody>
                  {TARGETS.flatMap((t) =>
                    PREDICTORS.map((p, i) => {
                      const v = m.metrics[t][p.key]
                      return (
                        <tr
                          key={`${t}-${p.key}`}
                          className={i === 0 ? 'border-t border-white/10 text-text-primary' : 'text-text-secondary'}
                        >
                          <td className="py-2 pr-4">{i === 0 ? TARGET_LABEL[t] : ''}</td>
                          <td className="py-2 pr-4">{p.label}</td>
                          <td className="py-2 pr-4 text-right tabular-nums">{v.mae.toFixed(3)}</td>
                          <td className="py-2 pr-4 text-right tabular-nums">{v.rmse.toFixed(3)}</td>
                          <td className="py-2 pr-4 text-right tabular-nums">{v.r2.toFixed(3)}</td>
                        </tr>
                      )
                    }),
                  )}
                </tbody>
              </table>
            </div>
            <p className="mt-3 text-xs text-text-secondary">
              Population: {POPULATION.headline(m.split.min_minutes)}. Source: metrics.json for model{' '}
              {modelIdentity(undefined, m.git_sha)}, trained {m.generated_at.slice(0, 10)}, dataset{' '}
              {m.dataset.version}.
              {goldMin10.every((g) => g !== null) && (
                <>
                  {' '}
                  Warehouse cross-check ({goldSourceNote('mart_holdout_metrics', manifest.data)}, population min10 on the
                  replay rows, n {goldMin10[0]!.n.toLocaleString()}): model MAE{' '}
                  {TARGETS.map((t, i) => `${TARGET_LABEL[t].toLowerCase()} ${goldMin10[i]!.model_mae.toFixed(3)}`).join(', ')}
                  ; the dbt test assert_holdout_metrics_reconcile_to_metrics_json holds these within 0.005 of the table
                  above (the replay never slated 169 of the holdout rows).
                </>
              )}
            </p>
          </>
        ) : (
          <p className="mt-2 text-text-secondary">
            The metrics report is not published at {metrics.url} (HTTP {metrics.status}).
          </p>
        )}
      </section>

      <section className="glass-card p-8">
        <h2 className="text-xl font-semibold text-text-primary">All rows (replay population)</h2>
        {allRows.n > 0 ? (
          <>
            <p className="mt-2 text-sm text-text-secondary">
              The same model and the last-10-game baseline on {POPULATION.allRows}:{' '}
              {allRows.n.toLocaleString()} rows{allRows.nDates ? ` over ${allRows.nDates} game dates` : ''} of {REPLAY_SEASON},
              row-weighted{allRows.source === 'gold' ? ' in the warehouse' : " from the committed copy of the replay's daily MAE file"}.
              This population is wider than the headline one (it includes games under 10 minutes), and on it{' '}
              {baselineWins.length === TARGETS.length
                ? 'the last-10 baseline has the lower MAE on every target'
                : baselineWins.length === 0
                  ? 'the model has the lower MAE on every target'
                  : `the last-10 baseline has the lower MAE on ${baselineWins.map((t) => TARGET_LABEL[t].toLowerCase()).join(' and ')}`}
              .
            </p>
            <div className="mt-4 overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-white/10 text-left text-xs uppercase tracking-wide text-text-secondary">
                    <th className="py-2 pr-4">Target</th>
                    <th className="py-2 pr-4 text-right">Model MAE</th>
                    <th className="py-2 pr-4 text-right">Last-10 MAE</th>
                    <th className="py-2 pr-4">Lower error</th>
                  </tr>
                </thead>
                <tbody>
                  {TARGETS.map((t) => (
                    <tr key={t} className="border-b border-white/5 text-text-primary">
                      <td className="py-2 pr-4">{TARGET_LABEL[t]}</td>
                      <td className="py-2 pr-4 text-right tabular-nums">
                        {allRows.model[t] === null ? '–' : allRows.model[t]!.toFixed(3)}
                      </td>
                      <td className="py-2 pr-4 text-right tabular-nums">
                        {allRows.baseline[t] === null ? '–' : allRows.baseline[t]!.toFixed(3)}
                      </td>
                      <td className="py-2 pr-4">
                        {allRows.model[t] === null || allRows.baseline[t] === null
                          ? '–'
                          : allRows.baseline[t]! < allRows.model[t]!
                            ? 'last-10 baseline'
                            : 'model'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="mt-3 text-xs text-text-secondary">
              Population: {POPULATION.allRows}. {allRows.note} The same numbers drive the{' '}
              <Link href="/replay" className="text-secondary hover:underline">replay page</Link>.
            </p>
          </>
        ) : (
          <p className="mt-2 text-text-secondary">The all-rows baseline report is empty.</p>
        )}
      </section>

      <section className="glass-card p-8">
        <h2 className="text-xl font-semibold text-text-primary">Replay equivalence</h2>
        {r ? (
          <>
            <p className="mt-2 text-sm text-text-secondary">
              The nightly slate was replayed for every game date of {r.season} using only data
              available before each date ({r.n_dates} dates, {r.n_predicted.toLocaleString()}{' '}
              slated player-games, {r.n_with_actuals.toLocaleString()} with a box score), with model{' '}
              {modelIdentity(r.model_revision)}. On the same population as the holdout report
              (the training population, {r.n_restricted.toLocaleString()} rows) the replayed MAE
              matches within the {r.tolerance} tolerance:{' '}
              <span className={r.passed ? 'text-success' : 'text-danger'}>
                {r.passed ? 'passed' : 'failed'}
              </span>
              .
            </p>
            <div className="mt-4 overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-white/10 text-left text-xs uppercase tracking-wide text-text-secondary">
                    <th className="py-2 pr-4">Target</th>
                    <th className="py-2 pr-4 text-right">Holdout MAE</th>
                    <th className="py-2 pr-4 text-right">Replayed MAE</th>
                    <th className="py-2 pr-4 text-right">Difference</th>
                    <th className="py-2 pr-4 text-right">Model MAE, all rows</th>
                  </tr>
                </thead>
                <tbody>
                  {TARGETS.map((t) => (
                    <tr key={t} className="border-b border-white/5 text-text-primary">
                      <td className="py-2 pr-4">{TARGET_LABEL[t]}</td>
                      <td className="py-2 pr-4 text-right tabular-nums">{r.reference.model_mae[t].toFixed(4)}</td>
                      <td className="py-2 pr-4 text-right tabular-nums">{r.mae_restricted[t].toFixed(4)}</td>
                      <td className="py-2 pr-4 text-right tabular-nums">
                        {r.diff_vs_metrics_json[t] >= 0 ? '+' : ''}
                        {r.diff_vs_metrics_json[t].toFixed(4)}
                      </td>
                      <td className="py-2 pr-4 text-right tabular-nums">{r.mae_unrestricted[t].toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="mt-3 text-xs text-text-secondary">
              Populations: the first three columns use the training population; the last column uses every
              replayed row with a box score ({r.n_with_actuals.toLocaleString()} rows; its baseline is in the
              table above). {r.unpredicted_actual_rows.all.toLocaleString()} actual player-games (
              {r.unpredicted_actual_rows.minutes_ge_min.toLocaleString()} with 10+ minutes) were never
              slated because the player had not appeared in his team&apos;s previous ten games.{' '}
              <Link href="/replay" className="text-secondary hover:underline">
                Daily chart and a sample slate
              </Link>
              . A tool-using analyst writes a nightly{' '}
              <Link href="/brief" className="text-secondary hover:underline">
                brief
              </Link>{' '}
              from these files.
            </p>
          </>
        ) : (
          <p className="mt-2 text-text-secondary">
            The {REPLAY_SEASON} replay summary is not published at {replay.url} (HTTP{' '}
            {replay.status}).
          </p>
        )}
      </section>
    </div>
  )
}
