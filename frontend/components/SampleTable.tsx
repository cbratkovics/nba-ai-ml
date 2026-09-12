'use client'

import SortableTable, { type Column } from '@/components/SortableTable'
import type { SampleRow } from '@/lib/data'

const num = (v: number | null, digits = 1) => (v === null ? '–' : v.toFixed(digits))

const COLUMNS: Column<SampleRow>[] = [
  { key: 'player_name', label: 'Player' },
  { key: 'team', label: 'Team' },
  { key: 'home', label: 'Opponent', render: (r) => `${r.home ? 'vs' : '@'} ${r.opponent}` },
  { key: 'minutes', label: 'Min', numeric: true, render: (r) => num(r.minutes) },
  { key: 'pred_pts', label: 'Pred pts', numeric: true, render: (r) => num(r.pred_pts) },
  { key: 'actual_pts', label: 'Pts', numeric: true, render: (r) => num(r.actual_pts, 0) },
  { key: 'pred_reb', label: 'Pred reb', numeric: true, render: (r) => num(r.pred_reb) },
  { key: 'actual_reb', label: 'Reb', numeric: true, render: (r) => num(r.actual_reb, 0) },
  { key: 'pred_ast', label: 'Pred ast', numeric: true, render: (r) => num(r.pred_ast) },
  { key: 'actual_ast', label: 'Ast', numeric: true, render: (r) => num(r.actual_ast, 0) },
]

export default function SampleTable({ rows }: { rows: SampleRow[] }) {
  return (
    <SortableTable
      columns={COLUMNS}
      rows={rows}
      initialSort={{ key: 'pred_pts', direction: 'desc' }}
      rowKey={(r) => `${r.game_id}-${r.player_id}`}
    />
  )
}
