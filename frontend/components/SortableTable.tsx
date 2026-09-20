'use client'

import { useMemo, useState } from 'react'
import { ChevronDown, ChevronUp, ChevronsUpDown } from 'lucide-react'
import { cn } from '@/lib/utils'

export interface Column<Row> {
  key: keyof Row & string
  label: string
  numeric?: boolean
  render?: (row: Row) => React.ReactNode
}

interface Props<Row> {
  columns: Column<Row>[]
  rows: Row[]
  initialSort: { key: keyof Row & string; direction: 'asc' | 'desc' }
  rowKey: (row: Row) => string
  caption?: string
  /** Extra classes for a row (e.g. to grey out rows without a box score). */
  rowClassName?: (row: Row) => string | undefined
}

function compare(a: unknown, b: unknown): number {
  if (a === null || a === undefined) return b === null || b === undefined ? 0 : 1
  if (b === null || b === undefined) return -1
  if (typeof a === 'number' && typeof b === 'number') return a - b
  if (typeof a === 'boolean' && typeof b === 'boolean') return Number(a) - Number(b)
  return String(a).localeCompare(String(b))
}

export default function SortableTable<Row extends object>({
  columns,
  rows,
  initialSort,
  rowKey,
  caption,
  rowClassName,
}: Props<Row>) {
  const [sort, setSort] = useState(initialSort)

  const sorted = useMemo(() => {
    const copy = [...rows]
    copy.sort((x, y) => {
      const c = compare(x[sort.key], y[sort.key])
      return sort.direction === 'asc' ? c : -c
    })
    return copy
  }, [rows, sort])

  const toggle = (key: keyof Row & string) => {
    setSort((s) =>
      s.key === key
        ? { key, direction: s.direction === 'asc' ? 'desc' : 'asc' }
        : { key, direction: columns.find((c) => c.key === key)?.numeric ? 'desc' : 'asc' },
    )
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        {caption && <caption className="mb-3 text-left text-text-secondary">{caption}</caption>}
        <thead>
          <tr className="border-b border-white/10 text-left text-xs uppercase tracking-wide text-text-secondary">
            {columns.map((col) => {
              const active = sort.key === col.key
              const Icon = active ? (sort.direction === 'asc' ? ChevronUp : ChevronDown) : ChevronsUpDown
              return (
                <th key={col.key} className={cn('py-2 pr-4', col.numeric && 'text-right')}>
                  <button
                    type="button"
                    onClick={() => toggle(col.key)}
                    className={cn(
                      'inline-flex items-center gap-1 hover:text-text-primary',
                      active && 'text-text-primary',
                    )}
                    aria-sort={active ? (sort.direction === 'asc' ? 'ascending' : 'descending') : 'none'}
                  >
                    {col.label}
                    <Icon className="h-3 w-3" aria-hidden="true" />
                  </button>
                </th>
              )
            })}
          </tr>
        </thead>
        <tbody>
          {sorted.map((row) => (
            <tr
              key={rowKey(row)}
              className={cn('border-b border-white/5 hover:bg-white/5', rowClassName?.(row))}
            >
              {columns.map((col) => (
                <td
                  key={col.key}
                  className={cn('py-2 pr-4 text-text-primary', col.numeric && 'text-right tabular-nums')}
                >
                  {col.render ? col.render(row) : String(row[col.key] ?? '')}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
