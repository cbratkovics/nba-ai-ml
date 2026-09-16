'use client'

import { useState } from 'react'
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { TARGETS, TARGET_LABEL, type PolicyTarget, type Target } from '@/lib/data'
import { cn } from '@/lib/utils'

interface Props {
  targets: Record<Target, PolicyTarget>
}

/**
 * Hit rate against coverage for every threshold on the grid: the model's calls, the
 * season-mean sign on the same rows, and the coin flip at 0.5. Reading left to right the
 * policy calls fewer rows (a higher threshold); the chosen threshold is marked.
 */
export default function CoverageCurveChart({ targets }: Props) {
  const [target, setTarget] = useState<Target>('pts')
  const tb = targets[target]
  const data = tb.coverage_curve
    .filter((p) => p.hit_rate !== null && p.n_resolved >= 30)
    .map((p) => ({
      threshold: p.threshold,
      coverage: Number((p.coverage * 100).toFixed(1)),
      model: p.hit_rate === null ? null : Number((p.hit_rate * 100).toFixed(2)),
      seasonMean:
        p.season_mean_same_rows_hit_rate === null
          ? null
          : Number((p.season_mean_same_rows_hit_rate * 100).toFixed(2)),
      n: p.n_resolved,
    }))
    .sort((a, b) => a.coverage - b.coverage)

  return (
    <div>
      <div className="mb-4 flex items-center gap-2" role="tablist" aria-label="Target">
        {TARGETS.map((t) => (
          <button
            key={t}
            type="button"
            role="tab"
            aria-selected={t === target}
            onClick={() => setTarget(t)}
            className={cn(
              'rounded-md px-3 py-1 text-sm transition-colors',
              t === target ? 'bg-primary text-white' : 'bg-white/5 text-text-secondary hover:text-text-primary',
            )}
          >
            {TARGET_LABEL[t]}
          </button>
        ))}
      </div>
      <div className="h-[340px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
            <CartesianGrid stroke="rgba(255,255,255,0.08)" vertical={false} />
            <XAxis
              dataKey="coverage"
              type="number"
              domain={[0, 100]}
              tick={{ fill: '#a1a1aa', fontSize: 11 }}
              tickFormatter={(v: number) => `${v}%`}
              label={{ value: 'coverage (share of rows called)', position: 'insideBottom', offset: -2, fill: '#a1a1aa', fontSize: 11 }}
            />
            <YAxis
              tick={{ fill: '#a1a1aa', fontSize: 11 }}
              width={44}
              domain={[40, 90]}
              tickFormatter={(v: number) => `${v}%`}
              label={{ value: 'hit rate', angle: -90, position: 'insideLeft', fill: '#a1a1aa', fontSize: 11 }}
            />
            <Tooltip
              contentStyle={{ background: '#1a1a2e', border: '1px solid rgba(255,255,255,0.1)' }}
              labelStyle={{ color: '#ffffff' }}
              formatter={(value: number, name: string) => [`${value.toFixed(1)}%`, name]}
              labelFormatter={(label: number) => {
                const p = data.find((d) => d.coverage === label)
                return p ? `coverage ${label}% (threshold ${p.threshold}, n=${p.n.toLocaleString()})` : `${label}%`
              }}
            />
            <Legend wrapperStyle={{ color: '#a1a1aa', fontSize: 12 }} />
            <ReferenceLine y={50} stroke="#71717a" strokeDasharray="4 4" label={{ value: 'coin flip', fill: '#a1a1aa', fontSize: 11, position: 'insideTopRight' }} />
            <ReferenceLine
              x={Number((tb.coverage * 100).toFixed(1))}
              stroke="#8b5cf6"
              strokeDasharray="2 4"
              label={{ value: `chosen ${tb.threshold}`, fill: '#a1a1aa', fontSize: 11, position: 'insideTopLeft' }}
            />
            <Line type="monotone" dataKey="model" name="Model calls" stroke="#8b5cf6" dot={false} strokeWidth={2} connectNulls />
            <Line
              type="monotone"
              dataKey="seasonMean"
              name="Season-mean sign, same rows"
              stroke="#f59e0b"
              dot={false}
              strokeWidth={2}
              connectNulls
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
