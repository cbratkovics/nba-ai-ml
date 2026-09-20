import ReplayChart from '@/components/ReplayChart'
import ReplaySlateTable from '@/components/ReplaySlateTable'
import Link from 'next/link'
import {
  ALL_ROWS_BASELINE,
  REPLAY_SEASON,
  TARGETS,
  TARGET_LABEL,
  getReplayDaily,
  getReplaySlate,
  getReplaySlateIndex,
  modelIdentity,
} from '@/lib/data'

// The date comes from the query string, so this route renders per request; the
// underlying fetches are still cached for an hour. Only the selected date's slate is
// fetched, never the whole season.
export const dynamic = 'force-dynamic'

const fmt = (v: number | null, digits = 3) => (v === null ? '–' : v.toFixed(digits))

export default async function ReplayPage({ searchParams }: { searchParams: { date?: string } }) {
  const [daily, index] = await Promise.all([getReplayDaily(), getReplaySlateIndex()])
  const d = daily.data
  const dates = index.data?.dates ?? []
  const known = new Set(dates.map((x) => x.date))
  const requested =
    searchParams?.date && /^\d{4}-\d{2}-\d{2}$/.test(searchParams.date) ? searchParams.date : null
  // Default to the latest replayed date; an unknown date is reported, not fetched.
  const selected = requested ?? index.data?.latest ?? dates.at(-1)?.date ?? null
  const slate = selected && known.has(selected) ? await getReplaySlate(selected) : null
  const s = slate?.data ?? null

  // Season numbers come from the committed report, not from the fetched daily file.
  const season = { n: ALL_ROWS_BASELINE.n, model: ALL_ROWS_BASELINE.model_mae, baseline: ALL_ROWS_BASELINE.baseline_last10_mae }
  const baselineWins = ALL_ROWS_BASELINE.baseline_wins

  return (
    <div className="space-y-8">
      <section className="glass-card p-8">
        <h1 className="text-2xl font-bold text-text-primary">Replay of {REPLAY_SEASON}</h1>
        {d ? (
          <>
            <p className="mt-2 text-sm text-text-secondary">
              Daily mean absolute error of the model and of the last-10-game baseline, replaying
              the nightly slate for {d.n_dates} game dates from {d.first_date} to {d.last_date}{' '}
              using only games played before each date. Population: {d.population}
              {` (${season.n.toLocaleString()} rows)`}. This is the all-rows population, not the
              training population behind the{' '}
              <Link href="/" className="text-secondary hover:underline">
                headline holdout table
              </Link>{' '}
              (at least 10 minutes, both baselines defined); on all rows{' '}
              {baselineWins.length === TARGETS.length
                ? 'the last-10 baseline has the lower season MAE on every target'
                : baselineWins.length === 0
                  ? 'the model has the lower season MAE on every target'
                  : `the last-10 baseline has the lower season MAE on ${baselineWins.map((t) => TARGET_LABEL[t].toLowerCase()).join(' and ')}`}
              .
            </p>
            <div className="mt-4 grid grid-cols-1 gap-3 text-sm sm:grid-cols-3">
              {TARGETS.map((t) => {
                const model = season.model[t]
                const base = season.baseline[t]
                return (
                  <div key={t} className="rounded-lg bg-white/5 p-3">
                    <div className="text-xs uppercase tracking-wide text-text-secondary">
                      {TARGET_LABEL[t]}, row-weighted season MAE (all rows)
                    </div>
                    <div className="mt-1 text-text-primary">
                      model {model === null ? '–' : model.toFixed(3)} · baseline{' '}
                      {base === null ? '–' : base.toFixed(3)}
                    </div>
                  </div>
                )
              })}
            </div>
            <p className="mt-3 text-xs text-text-secondary">
              The season figures above come from the committed copy of this daily file in the repository (the
              all-rows replay report cited on the overview); the chart below reads the published copy.
            </p>
            <div className="mt-6">
              <ReplayChart days={d.days} />
            </div>
          </>
        ) : (
          <p className="mt-2 text-text-secondary">
            The replay files are not published at {daily.url} (HTTP {daily.status}).
          </p>
        )}
      </section>

      <section className="glass-card p-8">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <h2 className="text-xl font-semibold text-text-primary">
              Replayed slate{s ? `: ${s.date}` : ''}
            </h2>
            <p className="mt-1 max-w-3xl text-sm text-text-secondary">
              Replay, not live: each date&apos;s slate was rebuilt in the backtest from games played
              before that date, with the pinned model; actuals come from the stored game logs.
              {dates.length > 0 && ` ${dates.length} dates are published; pick one to browse it.`}
            </p>
          </div>
          {dates.length > 0 && (
            <form method="get" className="flex items-center gap-2 text-sm">
              <label htmlFor="date" className="text-text-secondary">
                Date
              </label>
              <select
                id="date"
                name="date"
                defaultValue={s?.date ?? selected ?? ''}
                className="rounded-md border border-white/10 bg-card px-2 py-1 text-text-primary"
              >
                {[...dates].reverse().map((x) => (
                  <option key={x.date} value={x.date}>
                    {x.date} · {x.n_games} {x.n_games === 1 ? 'game' : 'games'}
                  </option>
                ))}
              </select>
              <button type="submit" className="rounded-md bg-primary px-3 py-1 text-white">
                Show
              </button>
            </form>
          )}
        </div>

        {s ? (
          <>
            <p className="mt-4 text-sm text-text-secondary">
              {s.n_games} games, {s.n_players} slated players, {s.n_with_actuals} with a box score,{' '}
              {s.n_did_not_play} did not play. Predictions were made with model{' '}
              {modelIdentity(s.model_revision)} on dataset {s.dataset_revision.slice(0, 7)} from games
              before {s.date}.
            </p>

            <h3 className="mt-6 text-sm font-semibold uppercase tracking-wide text-text-secondary">
              Games on {s.date}
            </h3>
            <ul className="mt-2 flex flex-wrap gap-2 text-sm">
              {s.games.map((g) => (
                <li key={g.game_id} className="rounded-lg bg-white/5 px-3 py-2 text-text-primary">
                  {g.away} @ {g.home}
                  <span className="ml-2 text-xs text-text-secondary">
                    {g.n_players} slated · {g.n_with_actuals} played
                  </span>
                </li>
              ))}
            </ul>

            <h3 className="mt-6 text-sm font-semibold uppercase tracking-wide text-text-secondary">
              MAE on {s.date} (replay)
            </h3>
            <p className="mt-1 text-xs text-text-secondary">
              Population: {s.metrics.population} ({s.metrics.n.toLocaleString()} rows).
            </p>
            <div className="mt-2 grid grid-cols-1 gap-3 text-sm sm:grid-cols-3">
              {TARGETS.map((t) => (
                <div key={t} className="rounded-lg bg-white/5 p-3">
                  <div className="text-xs uppercase tracking-wide text-text-secondary">{TARGET_LABEL[t]}</div>
                  <div className="mt-1 text-text-primary">
                    model {fmt(s.metrics.model[t])} · baseline {fmt(s.metrics.baseline_last10[t])}
                  </div>
                </div>
              ))}
            </div>

            <h3 className="mt-6 text-sm font-semibold uppercase tracking-wide text-text-secondary">
              Predicted vs actual
            </h3>
            <div className="mt-2">
              <ReplaySlateTable rows={s.rows} mae={s.metrics.model} />
            </div>
            <p className="mt-4 text-xs text-text-secondary">Source: {slate?.url}</p>
          </>
        ) : (
          <p className="mt-4 text-text-secondary">
            {index.data === null
              ? `The per-date replay slates are not published yet (${index.url}, HTTP ${index.status}).`
              : selected && !known.has(selected)
                ? `No replayed slate is published for ${selected}; pick a date from the list.`
                : slate
                  ? `The replayed slate for ${selected} is not published (${slate.url}, HTTP ${slate.status}).`
                  : 'No replayed dates are published yet.'}
          </p>
        )}
      </section>
    </div>
  )
}
