'use client'

import React from 'react'
import CountUp from 'react-countup'
import { motion } from 'framer-motion'
import { Activity, TrendingUp, Zap, Database, GitBranch, Shield, BarChart3, Clock } from 'lucide-react'
import { cn } from '@/lib/utils'

interface MetricCardProps {
  title: string
  value: number | string
  suffix?: string
  prefix?: string
  icon: React.ReactNode
  trend?: {
    value: number
    isPositive: boolean
  }
  status?: 'success' | 'warning' | 'danger'
  delay?: number
}

const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  suffix = '',
  prefix = '',
  icon,
  trend,
  status = 'success',
  delay = 0,
}) => {
  const statusColors = {
    success: 'bg-success/20 text-success',
    warning: 'bg-warning/20 text-warning',
    danger: 'bg-danger/20 text-danger',
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay }}
      className="metric-card group"
    >
      <div className="flex items-start justify-between mb-4">
        <div className={cn('p-3 rounded-lg', statusColors[status])}>
          {icon}
        </div>
        {trend && (
          <div className={cn(
            'flex items-center gap-1 text-sm',
            trend.isPositive ? 'text-success' : 'text-danger'
          )}>
            <TrendingUp className={cn('w-4 h-4', !trend.isPositive && 'rotate-180')} />
            <span>{trend.value}%</span>
          </div>
        )}
      </div>
      
      <h3 className="text-text-secondary text-sm font-medium mb-2">{title}</h3>
      
      <div className="text-3xl font-bold text-text-primary">
        {typeof value === 'number' ? (
          <>
            {prefix}
            <CountUp
              end={value}
              duration={2.5}
              separator=","
              decimals={suffix === '%' ? 1 : 0}
              delay={delay}
            />
            {suffix}
          </>
        ) : (
          value
        )}
      </div>
      
      <div className="absolute inset-0 bg-gradient-to-r from-primary/0 via-primary/5 to-primary/0 opacity-0 group-hover:opacity-100 transition-opacity duration-300 rounded-xl pointer-events-none" />
    </motion.div>
  )
}

export default function MetricsGrid() {
  const metrics = [
    {
      title: 'Model Accuracy',
      value: 94.2,
      suffix: '%',
      icon: <BarChart3 className="w-5 h-5" />,
      trend: { value: 2.3, isPositive: true },
      status: 'success' as const,
    },
    {
      title: 'Predictions Served',
      value: 1247893,
      suffix: '',
      icon: <Activity className="w-5 h-5" />,
      trend: { value: 12.5, isPositive: true },
      status: 'success' as const,
    },
    {
      title: 'Avg Latency',
      value: 87,
      suffix: 'ms',
      icon: <Zap className="w-5 h-5" />,
      trend: { value: 5.2, isPositive: false },
      status: 'warning' as const,
    },
    {
      title: 'Models in Production',
      value: 5,
      icon: <GitBranch className="w-5 h-5" />,
      status: 'success' as const,
    },
    {
      title: 'Active A/B Tests',
      value: 3,
      icon: <Shield className="w-5 h-5" />,
      status: 'success' as const,
    },
    {
      title: 'Data Pipeline',
      value: 'Healthy',
      icon: <Database className="w-5 h-5" />,
      status: 'success' as const,
    },
    {
      title: 'Model Drift',
      value: 0.03,
      suffix: '',
      icon: <TrendingUp className="w-5 h-5" />,
      trend: { value: 1.2, isPositive: false },
      status: 'success' as const,
    },
    {
      title: 'API Uptime',
      value: 99.99,
      suffix: '%',
      icon: <Clock className="w-5 h-5" />,
      status: 'success' as const,
    },
  ]

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      {metrics.map((metric, index) => (
        <MetricCard
          key={metric.title}
          {...metric}
          delay={index * 0.1}
        />
      ))}
    </div>
  )
}