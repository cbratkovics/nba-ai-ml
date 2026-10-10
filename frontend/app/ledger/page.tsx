import Link from 'next/link'
import {
  LEDGER_RULE,
  LEDGER_SEASON,
  LINKS,
  getGoldManifest,
  getGoldMart,
  goldSourceNote,
  type GoldLedgerBySide,
  type GoldLedgerCoverage,
  type GoldLedgerRow,
} from '@/lib/data'

export const revalidate = 3600

const pct = (v: number | null | undefined, digits = 1) => (v === null || v === undefined ? '–' : `${(v * 100).toFixed(digits)}%`)
const pm = (v: number | null | undefined, half: number | null | undefined, digits = 1) =>
  v === null || v === undefined ? '–' : `${pct(v, digits)}${half === null || half === undefined ? '' : ` ± ${(half * 100).toFixed(digits)}`}`
const units = (v: number | null | undefined) => (v === null || v === undefined ? '–' : `${v >= 0 ? '+' : ''}${v.toFixed(2)}`)
const MARKET_LABEL: Record<string, string> = {
  player_points: 'Points',
  player_rebounds: 'Rebounds',
  player_assists: 'Assists',
}

function LedgerTable({ rows }: { rows: GoldLedgerRow[] }) {
  return (
    <div className="mt-4 overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-white/10 text-left text-xs uppercase tracking-wide text-text-secondary">
            <th className="py-2 pr-4">Market</th>
            <th className="py-2 pr-4">Book</th>
            <th className="py-2 pr-4 text-right">Bets</th>
            <th className="py-2 pr-4 text-right">Settled</th>
            <th className="py-2 pr-4 text-right">Won / lost / push / void</th>
            <th className="py-2 pr-4 text-right">Pending</th>
            <th className="py-2 pr-4 text-right">Hit rate (95%)</th>
            <th className="py-2 pr-4 text-right">Break-even</th>
            <th className="py-2 pr-4 text-right">Units (95%)</th>
            <th className="py-2 pr-4">Verdict</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={`${r.source_group}-${r.book}-${r.market}`} className="border-b border-white/5 text-text-primary">
              <td className="py-2 pr-4">{MARKET_LABEL[r.market] ?? r.market}</td>
              <td className="py-2 pr-4">{r.book}</td>
              <td className="py-2 pr-4 text-right tabular-nums">{r.n_bets.toLocaleString()}</td>
              <td className="py-2 pr-4 text-right tabular-nums">{r.n_settled.toLocaleString()}</td>
              <td className="py-2 pr-4 text-right tabular-nums">
                {r.n_won} / {r.n_lost} / {r.n_push} / {r.n_void}
              </td>
              <td className="py-2 pr-4 text-right tabular-nums">{r.n_pending.toLocaleString()}</td>
              <td className="py-2 pr-4 text-right tabular-nums">{pm(r.hit_rate, r.hit_rate_half_width_95)}</td>
              <td className="py-2 pr-4 text-right tabular-nums">{pct(r.breakeven_hit_rate)}</td>
              <td className="py-2 pr-4 text-right tabular-nums">
                {units(r.units_profit)}
                {r.roi_half_width_95 === null ? '' : ` (ROI ${pm(r.roi_units, r.roi_half_width_95)})`}
              </td>
              <td className="py-2 pr-4">{r.verdict}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default async function LedgerPage() {
  const [ledger, bySide, coverage, manifest] = await Promise.all([
    getGoldMart<GoldLedgerRow>('mart_ledger'),
    getGoldMart<GoldLedgerBySide>('mart_ledger_by_side'),
    getGoldMart<GoldLedgerCoverage>('mart_ledger_coverage'),
    getGoldManifest(),
  ])
  const rule = LEDGER_RULE.rule
  const rows = (ledger.data ?? []).filter((r) => r.season === LEDGER_SEASON)
  const headline = rows.filter((r) => r.is_headline)
  const manual = rows.filter((r) => !r.is_headline)
  const sides = (bySide.data ?? []).filter((r) => r.season === LEDGER_SEASON && r.is_headline)
  const cov = (coverage.data ?? []).find((r) => r.season === LEDGER_SEASON) ?? null
  const published = ledger.data !== null
  const pending = cov?.bets_pending ?? headline.reduce((acc, r) => acc + r.n_pending, 0)
  const thresholds = Object.entries(rule.edge.threshold_by_market)

  return (
    <div className="space-y-8">
      <section className="glass-card p-8">
        <h1 className="text-2xl font-bold text-text-primary">Paper ledger, {LEDGER_SEASON}</h1>
        <p className="mt-4 max-w-3xl text-text-secondary">
          The frozen model against real sportsbook lines, out of sample, on paper: nothing here places a bet. One snapshot
          of DraftKings player-points lines a day from The Odds API&apos;s free plan, compared with the nightly prediction
          under a rule that was written down before the first game and is never tuned on this season&apos;s results.
          Raw lines and prices stay in a private repository; this page shows only the aggregates the warehouse computes
          from them. Every number below is labelled <span className="text-text-primary">out of sample</span>; the in-sample
          replay numbers on the{' '}
          <Link href="/decisions" className="text-secondary hover:underline">
            decisions page
          </Link>{' '}
          are never pooled with them.
        </p>
      </section>

      <section className="glass-card p-8">
        <h2 className="text-xl font-semibold text-text-primary">The pre-registered rule</h2>
        <p className="mt-2 text-sm text-text-secondary">
          {rule.version}, status {rule.status}, registered {rule.registered_on} for a first live date of {rule.first_live_date};
          sha256 {LEDGER_RULE.rule_hash.slice(0, 12)} of the rule text is written on every bet row. Model{' '}
          {rule.model.revision.slice(0, 7)} ({rule.model.feature_version}), unchanged.
        </p>
        <dl className="mt-4 grid grid-cols-1 gap-4 text-sm sm:grid-cols-2">
          <div className="rounded-lg bg-white/5 p-3">
            <dt className="text-xs uppercase tracking-wide text-text-secondary">Lines</dt>
            <dd className="mt-1 text-text-primary">
              {rule.books.api.join(', ')} {rule.markets.api.map((m) => MARKET_LABEL[m] ?? m).join(', ')} at {rule.snapshot.scheduled_utc} UTC
              ({rule.line_sources.api}); {rule.snapshot.scope}. Hand-typed {rule.books.manual.join(' / ')} lines for{' '}
              {rule.markets.manual.map((m) => (MARKET_LABEL[m] ?? m).toLowerCase()).join(', ')} are reported separately.
            </dd>
          </div>
          <div className="rounded-lg bg-white/5 p-3">
            <dt className="text-xs uppercase tracking-wide text-text-secondary">Eligibility</dt>
            <dd className="mt-1 text-text-primary">
              At least {rule.eligibility.min_prior_games_season} earlier game this season, a last-10 minutes mean of at least{' '}
              {rule.eligibility.min_minutes_mean_last10}, inputs at most {rule.eligibility.max_input_staleness_days} days old.
              <span className="block text-xs text-text-secondary">{rule.eligibility.why}.</span>
            </dd>
          </div>
          <div className="rounded-lg bg-white/5 p-3">
            <dt className="text-xs uppercase tracking-wide text-text-secondary">Bet when</dt>
            <dd className="mt-1 text-text-primary">
              |prediction − line| exceeds {thresholds.map(([m, v]) => `${(MARKET_LABEL[m] ?? m).toLowerCase()} ${v}`).join(', ')};
              over above the line, under below it. {rule.stake.risked}; {rule.stake.profit_on_win}.
              <span className="block text-xs text-text-secondary">{rule.edge.why}.</span>
            </dd>
          </div>
          <div className="rounded-lg bg-white/5 p-3">
            <dt className="text-xs uppercase tracking-wide text-text-secondary">Settlement and reading</dt>
            <dd className="mt-1 text-text-primary">
              Won, lost, push on the line, void only when the box score exists and the player has no row in it; pending until
              the box score arrives. Verdicts start at {rule.reading.too_early_below_settled_bets} settled bets and only when
              the 95% interval excludes the break-even rate implied by the prices.
              <span className="block text-xs text-text-secondary">{rule.reading.power_note}.</span>
            </dd>
          </div>
        </dl>
        <p className="mt-3 text-xs text-text-secondary">
          {rule.side_bias}. {rule.headline}. Full text: nba/ledger/rule.py and ADR-0030 in{' '}
          <a href={`${LINKS.github}/blob/main/docs/DECISIONS.md`} className="text-secondary hover:underline" target="_blank" rel="noopener noreferrer">
            docs/DECISIONS.md
          </a>
          .
        </p>
      </section>

      <section className="glass-card p-8">
        <h2 className="text-xl font-semibold text-text-primary">Coverage and open bets</h2>
        {cov ? (
          <>
            <div className="mt-4 grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
              {[
                ['Game nights so far', cov.game_nights],
                ['Nights captured', cov.nights_captured],
                ['Nights refused (budget)', cov.nights_refused],
                ['Games priced', cov.games_in_scope],
                ['Games out of scope (tipped before snapshot)', cov.games_out_of_scope],
                ['Credits charged', cov.credits_charged],
                ['Lines captured', cov.lines_total],
                ['Lines without a matched player', cov.lines_unmatched_player],
                ['Lines with a prediction', cov.lines_with_prediction],
                ['Ineligible under the rule', cov.lines_ineligible],
                ['Bets', cov.bets_total],
                ['Pending (no box score yet)', cov.bets_pending],
              ].map(([label, value]) => (
                <div key={String(label)} className="rounded-lg bg-white/5 p-3">
                  <div className="text-xs uppercase tracking-wide text-text-secondary">{label}</div>
                  <div className="mt-1 text-text-primary tabular-nums">{Number(value).toLocaleString()}</div>
                </div>
              ))}
            </div>
            <p className="mt-3 text-xs text-text-secondary">Source: {goldSourceNote('mart_ledger_coverage', manifest.data)}.</p>
          </>
        ) : (
          <p className="mt-2 text-text-secondary">
            {published
              ? `No ${LEDGER_SEASON} rows yet: the first live date is ${rule.first_live_date}, and the coverage mart has a row only once a game night has passed.`
              : `The gold export is not published at ${coverage.url} (HTTP ${coverage.status}).`}
          </p>
        )}
      </section>

      <section className="glass-card p-8">
        <h2 className="text-xl font-semibold text-text-primary">Headline ledger (API lines, DraftKings, points)</h2>
        {headline.length > 0 ? (
          <>
            <LedgerTable rows={headline} />
            <p className="mt-3 text-xs text-text-secondary">
              Out of sample. {pending.toLocaleString()} bets pending. Hit rate with its 95% half width; break-even is the mean
              implied probability of the prices actually bet; units are flat one-unit stakes; the verdict follows the reading
              rule above. Source: {goldSourceNote('mart_ledger', manifest.data)}.
            </p>
          </>
        ) : (
          <p className="mt-2 text-text-secondary">
            {published
              ? `No bets yet (0 settled, ${pending.toLocaleString()} pending). The rule expects none on opening night, when no player has an earlier game this season.`
              : `The gold export is not published at ${ledger.url} (HTTP ${ledger.status}).`}
          </p>
        )}
        {sides.length > 0 && (
          <>
            <h3 className="mt-6 text-sm font-semibold uppercase tracking-wide text-text-secondary">By side</h3>
            <div className="mt-2 overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-white/10 text-left text-xs uppercase tracking-wide text-text-secondary">
                    <th className="py-2 pr-4">Market</th>
                    <th className="py-2 pr-4">Side</th>
                    <th className="py-2 pr-4 text-right">Bets</th>
                    <th className="py-2 pr-4 text-right">Settled</th>
                    <th className="py-2 pr-4 text-right">Pending</th>
                    <th className="py-2 pr-4 text-right">Hit rate (95%)</th>
                    <th className="py-2 pr-4 text-right">Break-even</th>
                    <th className="py-2 pr-4 text-right">Units</th>
                  </tr>
                </thead>
                <tbody>
                  {sides.map((r) => (
                    <tr key={`${r.market}-${r.side}`} className="border-b border-white/5 text-text-primary">
                      <td className="py-2 pr-4">{MARKET_LABEL[r.market] ?? r.market}</td>
                      <td className="py-2 pr-4">{r.side}</td>
                      <td className="py-2 pr-4 text-right tabular-nums">{r.n_bets}</td>
                      <td className="py-2 pr-4 text-right tabular-nums">{r.n_settled}</td>
                      <td className="py-2 pr-4 text-right tabular-nums">{r.n_pending}</td>
                      <td className="py-2 pr-4 text-right tabular-nums">{pm(r.hit_rate, r.hit_rate_half_width_95)}</td>
                      <td className="py-2 pr-4 text-right tabular-nums">{pct(r.breakeven_hit_rate)}</td>
                      <td className="py-2 pr-4 text-right tabular-nums">{units(r.units_profit)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="mt-3 text-xs text-text-secondary">
              Reported by side because the model is a conditional mean and lines sit near a median (ADR-0030). Source:{' '}
              {goldSourceNote('mart_ledger_by_side', manifest.data)}.
            </p>
          </>
        )}
      </section>

      {manual.length > 0 && (
        <section className="glass-card p-8">
          <h2 className="text-xl font-semibold text-text-primary">Hand-entered lines (never in the headline)</h2>
          <LedgerTable rows={manual} />
          <p className="mt-3 text-xs text-text-secondary">
            Lines typed from the owner&apos;s own account under the same rule; a separate record, reported apart from the API
            capture. Source: {goldSourceNote('mart_ledger', manifest.data)}.
          </p>
        </section>
      )}

      <section className="glass-card p-8">
        <h2 className="text-xl font-semibold text-text-primary">What this is not</h2>
        <p className="mt-2 text-sm text-text-secondary">
          Not a betting product and not advice: a personal, non-commercial record kept where sports betting is legal, with
          no wager placed. Not a profitability claim: {rule.reporting}. The model knows nothing about injuries or rest; a
          line set after news the model has not seen is a line it will look &quot;wrong&quot; about. Lines are captured once a
          day and compared with a prediction made before them; a dbt test fails the build if any bet&apos;s prediction was
          generated after its line or any line was read after tip-off.
        </p>
      </section>
    </div>
  )
}
