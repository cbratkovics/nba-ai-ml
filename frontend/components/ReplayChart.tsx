'use client'

import { useState } from 'react'
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { TARGETS, TARGET_LABEL, type ReplayDay, type Target } from '@/lib/data'
import { cn } from '@/lib/utils'

interface Props {
  days: ReplayDay[]
}

export default function ReplayChart({ days }: Props) {
  const [target, setTarget] = useState<Target>('pts')
  const data = days.map((d) => ({
    date: d.date,
    model: Number(d.model[target].toFixed(3)),
    baseline: Number(d.baseline_last10[target].toFixed(3)),
    n: d.n,
  }))

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
              t === target
                ? 'bg-primary text-white'
                : 'bg-white/5 text-text-secondary hover:text-text-primary',
            )}
          >
            {TARGET_LABEL[t]}
          </button>
        ))}
      </div>
      <div className="h-[380px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
            <CartesianGrid stroke="rgba(255,255,255,0.08)" vertical={false} />
            <XAxis
              dataKey="date"
              tick={{ fill: '#a1a1aa', fontSize: 11 }}
              tickFormatter={(d: string) => d.slice(5)}
              minTickGap={24}
            />
            <YAxis
              tick={{ fill: '#a1a1aa', fontSize: 11 }}
              width={40}
              label={{ value: 'MAE', angle: -90, position: 'insideLeft', fill: '#a1a1aa', fontSize: 11 }}
            />
            <Tooltip
              contentStyle={{ background: '#1a1a2e', border: '1px solid rgba(255,255,255,0.1)' }}
              labelStyle={{ color: '#ffffff' }}
              formatter={(value: number, name: string) => [value.toFixed(3), name]}
              labelFormatter={(label: string) => {
                const day = data.find((d) => d.date === label)
                return day ? `${label} (n=${day.n})` : label
              }}
            />
            <Legend wrapperStyle={{ color: '#a1a1aa', fontSize: 12 }} />
            <Line
              type="monotone"
              dataKey="model"
              name="Model"
              stroke="#8b5cf6"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="baseline"
              name="Last-10 baseline"
              stroke="#00d4ff"
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
