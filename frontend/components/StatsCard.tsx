'use client'

import { motion } from 'framer-motion'
import { LucideIcon } from 'lucide-react'

interface StatsCardProps {
  title: string
  value: string | number
  subtitle?: string
  icon: LucideIcon
  iconColor?: string
  trend?: {
    value: number
    isPositive: boolean
  }
  delay?: number
}

export default function StatsCard({ 
  title, 
  value, 
  subtitle, 
  icon: Icon, 
  iconColor = 'primary',
  trend,
  delay = 0 
}: StatsCardProps) {
  const getIconColorClass = () => {
    switch (iconColor) {
      case 'primary': return 'bg-primary/20 text-primary'
      case 'secondary': return 'bg-secondary/20 text-secondary'
      case 'success': return 'bg-success/20 text-success'
      case 'warning': return 'bg-warning/20 text-warning'
      case 'danger': return 'bg-danger/20 text-danger'
      default: return 'bg-primary/20 text-primary'
    }
  }
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay }}
      className="glass-card p-6"
    >
      <div className="flex items-center justify-between">
        <div>
          <p className="text-text-secondary text-sm">{title}</p>
          <p className="text-2xl font-bold text-text-primary mt-1">{value}</p>
          {subtitle && (
            <p className="text-xs text-text-secondary mt-1">{subtitle}</p>
          )}
          {trend && (
            <p className={`text-xs mt-2 flex items-center gap-1 ${
              trend.isPositive ? 'text-success' : 'text-danger'
            }`}>
              {trend.isPositive ? '↑' : '↓'} {Math.abs(trend.value)}%
            </p>
          )}
        </div>
        <div className={`p-3 rounded-lg ${getIconColorClass()}`}>
          <Icon className="w-6 h-6" />
        </div>
      </div>
    </motion.div>
  )
}