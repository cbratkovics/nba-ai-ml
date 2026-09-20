'use client'

import SortableTable, { type Column } from '@/components/SortableTable'
import { TARGETS, TARGET_LABEL, type ReplaySlateRow, type Target } from '@/lib/data'
import { cn } from '@/lib/utils'

/** A slate row with its residuals (actual minus predicted) as sortable columns. */
type Row = ReplaySlateRow & Record<`resid_${Target}`, number | null>

/**
 * Absolute-error band relative to the date's MAE for that target: within one MAE,
 * within two, or beyond two. The number is always printed; colour only adds emphasis.
 */
function band(resid: number | null, mae: number | null): 0 | 1 | 2 | null {
  if (resid === null || mae === null || mae <= 0) return null
  const abs = Math.abs(resid)
  return abs <= mae ? 0 : abs <= 2 * mae ? 1 : 2
}

const BAND_CLASS = ['bg-success/15 text-success', 'bg-warning/15 text-warning', 'bg-danger/15 text-danger']

const num = (v: number | null, digits = 1) => (v === null ? '–' : v.toFixed(digits))
const signed = (v: number | null) => (v === null ? '–' : `${v > 0 ? '+' : v < 0 ? '−' : ''}${Math.abs(v).toFixed(1)}`)

interface Props {
  rows: ReplaySlateRow[]
  /** The date's MAE per target (from the slate file), used to scale the colour bands. */
  mae: Record<Target, number | null>
}

export default function ReplaySlateTable({ rows, mae }: Props) {
  const withResiduals: Row[] = rows.map((r) => ({
    ...r,
    resid_pts: r.actual_pts === null ? null : Number((r.actual_pts - r.pred_pts).toFixed(2)),
    resid_reb: r.actual_reb === null ? null : Number((r.actual_reb - r.pred_reb).toFixed(2)),
    resid_ast: r.actual_ast === null ? null : Number((r.actual_ast - r.pred_ast).toFixed(2)),
  }))

  const residual = (t: Target): Column<Row> => ({
    key: `resid_${t}`,
    label: `Δ ${t}`,
    numeric: true,
    render: (r) => {
      const b = band(r[`resid_${t}`], mae[t])
      return (
        <span className={cn('inline-block min-w-[3.5rem] rounded px-1.5 py-0.5', b !== null && BAND_CLASS[b])}>
          {signed(r[`resid_${t}`])}
        </span>
      )
    },
  })

  const columns: Column<Row>[] = [
    { key: 'player_name', label: 'Player' },
    { key: 'team', label: 'Team' },
    { key: 'home', label: 'Opponent', render: (r) => `${r.home ? 'vs' : '@'} ${r.opponent}` },
    {
      key: 'minutes',
      label: 'Min',
      numeric: true,
      render: (r) =>
        r.did_not_play ? (
          <span className="text-xs uppercase tracking-wide" title="Did not play: no box-score row">
            DNP
          </span>
        ) : r.has_actual ? (
          num(r.minutes)
        ) : (
          <span className="text-xs" title="No box score for this game in the stored logs">
            n/a
          </span>
        ),
    },
    { key: 'pred_pts', label: 'Pred pts', numeric: true, render: (r) => num(r.pred_pts) },
    { key: 'actual_pts', label: 'Pts', numeric: true, render: (r) => num(r.actual_pts, 0) },
    residual('pts'),
    { key: 'pred_reb', label: 'Pred reb', numeric: true, render: (r) => num(r.pred_reb) },
    { key: 'actual_reb', label: 'Reb', numeric: true, render: (r) => num(r.actual_reb, 0) },
    residual('reb'),
    { key: 'pred_ast', label: 'Pred ast', numeric: true, render: (r) => num(r.pred_ast) },
    { key: 'actual_ast', label: 'Ast', numeric: true, render: (r) => num(r.actual_ast, 0) },
    residual('ast'),
  ]

  return (
    <div>
      <SortableTable
        columns={columns}
        rows={withResiduals}
        initialSort={{ key: 'pred_pts', direction: 'desc' }}
        rowKey={(r) => `${r.game_id}-${r.player_id}`}
        rowClassName={(r) => (r.has_actual ? undefined : 'opacity-40')}
      />
      <p className="mt-3 text-xs text-text-secondary">
        Δ is actual minus predicted. Colour is the absolute error against this date&apos;s MAE for
        the same target:{' '}
        <span className="rounded bg-success/15 px-1 text-success">within one MAE</span>,{' '}
        <span className="rounded bg-warning/15 px-1 text-warning">within two</span>,{' '}
        <span className="rounded bg-danger/15 px-1 text-danger">beyond two</span>
        {' '}(this date&apos;s MAE:{' '}
        {TARGETS.map((t) => `${TARGET_LABEL[t].toLowerCase()} ${num(mae[t], 2)}`).join(', ')}).
        Greyed rows were slated but have no box score (DNP: did not play); their predictions
        are kept and they are excluded from the MAE.
      </p>
    </div>
  )
}
