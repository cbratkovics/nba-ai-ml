import Link from 'next/link'
import {
  briefMode,
  getBrief,
  getBriefIndex,
  getLatestBrief,
  getReplayDaily,
  type Brief,
  type BriefFinding,
} from '@/lib/data'

// The date comes from the query string, so this route renders per request; the
// underlying fetches are still cached for an hour.
export const dynamic = 'force-dynamic'

const SEVERITY_CLASS: Record<BriefFinding['severity'], string> = {
  info: 'bg-white/10 text-text-secondary',
  warning: 'bg-warning/20 text-warning',
  critical: 'bg-danger/20 text-danger',
}

const STATUS_TEXT: Record<Brief['status'], string> = {
  ok: 'every finding is grounded in tool output',
  ungrounded: 'one or more findings cited numbers not in their evidence and were dropped',
  agent_unavailable: 'the agent did not produce a brief',
}

function Finding({ f }: { f: BriefFinding }) {
  return (
    <li className="rounded-lg border border-white/10 bg-white/5 p-4">
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <span className={`rounded px-2 py-0.5 uppercase tracking-wide ${SEVERITY_CLASS[f.severity]}`}>
          {f.severity}
        </span>
        <span className="text-text-secondary">{f.kind}</span>
      </div>
      <p className="mt-2 text-text-primary">{f.text}</p>
      <details className="mt-2 text-xs text-text-secondary">
        <summary className="cursor-pointer">
          Evidence: {f.evidence.tool || 'no tool cited'}
          {Object.keys(f.evidence.args ?? {}).length ? ` ${JSON.stringify(f.evidence.args)}` : ''}
        </summary>
        <pre className="mt-2 overflow-x-auto rounded bg-black/40 p-3 text-[11px] leading-snug">
          {JSON.stringify(f.evidence.values, null, 2)}
        </pre>
      </details>
      {f.ungrounded_numbers && f.ungrounded_numbers.length > 0 && (
        <p className="mt-2 text-xs text-danger">
          Dropped: numbers not found in evidence: {f.ungrounded_numbers.join(', ')}
        </p>
      )}
    </li>
  )
}

export default async function BriefPage({
  searchParams,
}: {
  searchParams: { date?: string }
}) {
  const requested = searchParams?.date && /^\d{4}-\d{2}-\d{2}$/.test(searchParams.date) ? searchParams.date : null
  const [index, fetched, daily] = await Promise.all([
    getBriefIndex(),
    requested ? getBrief(requested) : getLatestBrief(),
    getReplayDaily(),
  ])
  const brief = fetched.data
  const dates = index.data?.dates ?? []
  const lastReplayDate = daily.data?.last_date ?? null
  const mode = brief ? briefMode(brief.date, lastReplayDate) : null

  return (
    <div className="space-y-8">
      <section className="glass-card p-8">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-text-primary">
              Analyst brief{brief ? ` for ${brief.date}` : ''}
            </h1>
            {brief && (
              <p className="mt-1 text-sm text-text-secondary">
                Brief date {brief.date}, written on run date {brief.run_date} ({mode}
                {mode === 'replay'
                  ? `: a replayed ${brief.date <= (lastReplayDate ?? '') ? daily.data?.season : ''} date briefed on demand`
                  : ': written by the nightly job'}
                ).
              </p>
            )}
            <p className="mt-2 max-w-3xl text-sm text-text-secondary">
              A tool-using agent reads the pipeline&apos;s published files (ingest report,
              residuals, rolling accuracy, game logs, known gaps) and writes a short brief. It
              may only report what the tools returned; every number is checked against the
              cited evidence before publication.
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
                defaultValue={brief?.date ?? index.data?.latest ?? ''}
                className="rounded-md border border-white/10 bg-card px-2 py-1 text-text-primary"
              >
                {[...dates].reverse().map((d) => (
                  <option key={d} value={d}>
                    {d} · {briefMode(d, lastReplayDate)}
                  </option>
                ))}
              </select>
              <button type="submit" className="rounded-md bg-primary px-3 py-1 text-white">
                Show
              </button>
            </form>
          )}
        </div>

        {brief ? (
          <>
            <dl className="mt-6 grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
              {[
                ['Status', `${brief.status} (${STATUS_TEXT[brief.status]})`],
                ['Model', brief.model_id],
                ['Tool calls', String(brief.tool_calls_made)],
                ['Latency', `${(brief.latency_ms / 1000).toFixed(1)} s`],
              ].map(([k, v]) => (
                <div key={k} className="rounded-lg bg-white/5 p-3">
                  <dt className="text-xs uppercase tracking-wide text-text-secondary">{k}</dt>
                  <dd className="mt-1 break-words text-text-primary">{v}</dd>
                </div>
              ))}
            </dl>
            <p className="mt-6 text-text-primary">{brief.summary}</p>
            {brief.error && <p className="mt-2 text-sm text-danger">{brief.error}</p>}
            <h2 className="mt-6 text-lg font-semibold text-text-primary">
              Findings ({brief.findings.length})
            </h2>
            {brief.findings.length ? (
              <ul className="mt-3 space-y-3">
                {brief.findings.map((f, i) => (
                  <Finding key={i} f={f} />
                ))}
              </ul>
            ) : (
              <p className="mt-2 text-sm text-text-secondary">No findings were published for this date.</p>
            )}
            {brief.dropped_findings && brief.dropped_findings.length > 0 && (
              <>
                <h2 className="mt-6 text-lg font-semibold text-text-primary">
                  Dropped by the grounding check ({brief.dropped_findings.length})
                </h2>
                <ul className="mt-3 space-y-3 opacity-70">
                  {brief.dropped_findings.map((f, i) => (
                    <Finding key={`d${i}`} f={f} />
                  ))}
                </ul>
              </>
            )}
            <p className="mt-6 text-xs text-text-secondary">
              Generated {brief.generated_at.slice(0, 16).replace('T', ' ')} UTC for run date{' '}
              {brief.run_date}. Source: {fetched.url}
            </p>
          </>
        ) : (
          <p className="mt-6 text-text-secondary">
            {requested
              ? `No brief is published for ${requested} (${fetched.url}, HTTP ${fetched.status}).`
              : `No brief has been published yet (${fetched.url}, HTTP ${fetched.status}). The nightly job writes one after each residual run; replay dates can be briefed on demand.`}
          </p>
        )}
      </section>
      <p className="text-xs text-text-secondary">
        Back to the <Link href="/" className="text-secondary hover:underline">overview</Link>.
      </p>
    </div>
  )
}
