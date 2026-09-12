import SlateTable from '@/components/SlateTable'
import { TARGETS, TARGET_LABEL, getLatestSlate, getReplayDaily, getRollingMetrics } from '@/lib/data'

export const revalidate = 3600

export default async function PredictionsPage() {
  const [latest, rolling, daily] = await Promise.all([
    getLatestSlate(),
    getRollingMetrics(),
    getReplayDaily(),
  ])
  const slate = latest.data
  const lastLine = rolling.data?.at(-1) ?? null

  return (
    <div className="space-y-8">
      <section className="glass-card p-8">
        <h1 className="text-2xl font-bold text-text-primary">Today&apos;s slate</h1>
        {slate ? (
          <>
            <p className="mt-2 text-sm text-text-secondary">
              {slate.date}: {slate.n_games} games, {slate.n_players} players. Generated{' '}
              {slate.generated_at.slice(0, 16).replace('T', ' ')} UTC with model{' '}
              {slate.model_revision.slice(0, 7)} on dataset {slate.dataset_revision.slice(0, 7)}.
              Predictions are the model&apos;s expected points, rebounds, and assists for players who
              appeared in their team&apos;s last ten games.
            </p>
            <div className="mt-6">
              <SlateTable rows={slate.predictions} />
            </div>
          </>
        ) : (
          <div className="mt-4 space-y-2 text-text-secondary">
            <p className="text-text-primary">No slate yet.</p>
            <p>
              The nightly job writes a slate only on days with scheduled regular-season games,
              and the 2026-27 season starts in October. The last replayed date was{' '}
              {daily.data?.last_date ?? 'not published'}
              {daily.data ? ` (${daily.data.season} replay)` : ''}.
            </p>
            <p className="text-xs">
              Checked {latest.url} (HTTP {latest.status}).
            </p>
          </div>
        )}
      </section>

      <section className="glass-card p-8">
        <h2 className="text-xl font-semibold text-text-primary">Recent accuracy</h2>
        {lastLine ? (
          <>
            <p className="mt-2 text-sm text-text-secondary">
              Latest scored date {lastLine.date}: {lastLine.n_with_actuals} of{' '}
              {lastLine.n_predicted} slated players had a box score ({lastLine.n_missing_actuals}{' '}
              missing). Rolling 30-day window: {lastLine.rolling_30d.days} days,{' '}
              {lastLine.rolling_30d.n.toLocaleString()} player-games.
            </p>
            <div className="mt-4 overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-white/10 text-left text-xs uppercase tracking-wide text-text-secondary">
                    <th className="py-2 pr-4">Target</th>
                    <th className="py-2 pr-4 text-right">MAE on {lastLine.date}</th>
                    <th className="py-2 pr-4 text-right">Rolling 30-day MAE</th>
                  </tr>
                </thead>
                <tbody>
                  {TARGETS.map((t) => (
                    <tr key={t} className="border-b border-white/5 text-text-primary">
                      <td className="py-2 pr-4">{TARGET_LABEL[t]}</td>
                      <td className="py-2 pr-4 text-right tabular-nums">
                        {lastLine.mae[t] === null ? '–' : lastLine.mae[t]!.toFixed(3)}
                      </td>
                      <td className="py-2 pr-4 text-right tabular-nums">
                        {lastLine.rolling_30d.mae[t] === null ? '–' : lastLine.rolling_30d.mae[t]!.toFixed(3)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        ) : (
          <p className="mt-2 text-text-secondary">
            No residuals have been scored yet; this table fills in the day after the first slate.
            Checked {rolling.url} (HTTP {rolling.status}).
          </p>
        )}
      </section>
    </div>
  )
}
