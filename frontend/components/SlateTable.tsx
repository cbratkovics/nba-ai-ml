'use client'

import SortableTable, { type Column } from '@/components/SortableTable'
import type { SlatePrediction } from '@/lib/data'

const COLUMNS: Column<SlatePrediction>[] = [
  { key: 'player_name', label: 'Player' },
  { key: 'team', label: 'Team' },
  { key: 'home', label: 'Opponent', render: (r) => `${r.home ? 'vs' : '@'} ${r.opponent}` },
  { key: 'pred_pts', label: 'Pts', numeric: true, render: (r) => r.pred_pts.toFixed(1) },
  { key: 'pred_reb', label: 'Reb', numeric: true, render: (r) => r.pred_reb.toFixed(1) },
  { key: 'pred_ast', label: 'Ast', numeric: true, render: (r) => r.pred_ast.toFixed(1) },
]

export default function SlateTable({ rows }: { rows: SlatePrediction[] }) {
  return (
    <SortableTable
      columns={COLUMNS}
      rows={rows}
      initialSort={{ key: 'pred_pts', direction: 'desc' }}
      rowKey={(r) => `${r.player_id}`}
    />
  )
}
