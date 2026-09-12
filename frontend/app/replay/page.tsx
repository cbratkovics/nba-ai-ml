import ReplayChart from '@/components/ReplayChart'
import SampleTable from '@/components/SampleTable'
import {
  REPLAY_SEASON,
  TARGETS,
  TARGET_LABEL,
  getReplayDaily,
  getReplaySample,
  type ReplayDay,
} from '@/lib/data'

export const revalidate = 3600

export default async function ReplayPage() {
  const daily = await getReplayDaily()
  const d = daily.data
  const sample = d?.sample_date ? await getReplaySample(d.sample_date) : null
  const s = sample?.data ?? null

  const seasonMean = (pick: (day: ReplayDay) => number) => {
    if (!d || d.days.length === 0) return null
    const total = d.days.reduce((acc, day) => acc + pick(day) * day.n, 0)
    const n = d.days.reduce((acc, day) => acc + day.n, 0)
    return n ? total / n : null
  }

  return (
    <div className="space-y-8">
      <section className="glass-card p-8">
        <h1 className="text-2xl font-bold text-text-primary">Replay of {REPLAY_SEASON}</h1>
        {d ? (
          <>
            <p className="mt-2 text-sm text-text-secondary">
              Daily mean absolute error of the model and of the last-10-game baseline, replaying
              the nightly slate for {d.n_dates} game dates from {d.first_date} to {d.last_date}{' '}
              using only games played before each date. Population: {d.population}.
            </p>
            <div className="mt-4 grid grid-cols-1 gap-3 text-sm sm:grid-cols-3">
              {TARGETS.map((t) => {
                const model = seasonMean((day) => day.model[t])
                const base = seasonMean((day) => day.baseline_last10[t])
                return (
                  <div key={t} className="rounded-lg bg-white/5 p-3">
                    <div className="text-xs uppercase tracking-wide text-text-secondary">
                      {TARGET_LABEL[t]}, row-weighted season MAE
                    </div>
                    <div className="mt-1 text-text-primary">
                      model {model === null ? '–' : model.toFixed(3)} · baseline{' '}
                      {base === null ? '–' : base.toFixed(3)}
                    </div>
                  </div>
                )
              })}
            </div>
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
        <h2 className="text-xl font-semibold text-text-primary">
          Sample slate{s ? `: ${s.date}` : ''}
        </h2>
        {s ? (
          <>
            <p className="mt-2 text-sm text-text-secondary">
              The last replayed date: {s.n_games} games, {s.n_players} slated players,{' '}
              {s.n_with_actuals} with a box score (the rest did not play). Predictions were made
              with model {s.model_revision.slice(0, 7)} from games before {s.date}; actuals come
              from the stored game logs.
            </p>
            <div className="mt-6">
              <SampleTable rows={s.rows} />
            </div>
          </>
        ) : (
          <p className="mt-2 text-text-secondary">
            {d?.sample_file
              ? `The sample slate is not published (${sample?.url ?? d.sample_file}, HTTP ${sample?.status ?? 'n/a'}).`
              : 'No sample date is published yet.'}
          </p>
        )}
      </section>
    </div>
  )
}
