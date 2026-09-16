'use client'

import SortableTable, { type Column } from '@/components/SortableTable'
import { TARGETS, TARGET_LABEL, type Call, type DecisionRow, type Population, type Target } from '@/lib/data'

interface Row {
  key: string
  player_name: string
  team: string
  home: boolean
  opponent: string
  pts_edge: number | null
  reb_edge: number | null
  ast_edge: number | null
  pts_call: Call
  reb_call: Call
  ast_call: Call
  pts_text: string
  reb_text: string
  ast_text: string
}

const CALL_TEXT: Record<Call, string> = { over: 'over', under: 'under', no_call: '–' }
const CALL_CLASS: Record<Call, string> = {
  over: 'text-success',
  under: 'text-warning',
  no_call: 'text-text-secondary',
}

function flatten(rows: DecisionRow[], population: Population): Row[] {
  return rows.map((r) => {
    const out: Partial<Row> = {
      key: `${r.player_id}-${r.game_id}`,
      player_name: r.player_name,
      team: r.team,
      home: r.home,
      opponent: r.opponent,
    }
    for (const t of TARGETS) {
      const d = r.targets[t]
      out[`${t}_edge`] = d.edge
      out[`${t}_call`] = d.populations[population].call
      out[`${t}_text`] = `${d.prediction.toFixed(1)} vs ${d.baseline_last10.toFixed(1)}`
    }
    return out as Row
  })
}

function callColumn(t: Target): Column<Row> {
  return {
    key: `${t}_edge` as keyof Row & string,
    label: `${TARGET_LABEL[t]} call`,
    numeric: true,
    render: (r) => {
      const call = r[`${t}_call`] as Call
      const edge = r[`${t}_edge`] as number | null
      return (
        <span className={CALL_CLASS[call]} title={r[`${t}_text`] as string}>
          {CALL_TEXT[call]}
          {call !== 'no_call' && edge !== null ? ` (${edge > 0 ? '+' : ''}${edge.toFixed(1)})` : ''}
        </span>
      )
    },
  }
}

const COLUMNS: Column<Row>[] = [
  { key: 'player_name', label: 'Player' },
  { key: 'team', label: 'Team' },
  { key: 'home', label: 'Opponent', render: (r) => `${r.home ? 'vs' : '@'} ${r.opponent}` },
  { key: 'pts_text', label: 'Pts pred vs last-10' },
  callColumn('pts'),
  { key: 'reb_text', label: 'Reb pred vs last-10' },
  callColumn('reb'),
  { key: 'ast_text', label: 'Ast pred vs last-10' },
  callColumn('ast'),
]

export default function DecisionsTable({ rows, population }: { rows: DecisionRow[]; population: Population }) {
  return (
    <SortableTable
      columns={COLUMNS}
      rows={flatten(rows, population)}
      initialSort={{ key: 'pts_edge', direction: 'desc' }}
      rowKey={(r) => r.key}
    />
  )
}
