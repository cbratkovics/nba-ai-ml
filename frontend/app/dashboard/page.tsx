'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { 
  LineChart, Line, AreaChart, Area, BarChart, Bar, ScatterChart, Scatter,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell
} from 'recharts'
import AnimatedCard from '@/components/ui/AnimatedCard'
import { 
  TrendingUp, AlertTriangle, CheckCircle, Clock, 
  Cpu, HardDrive, Activity, Zap
} from 'lucide-react'

export default function DashboardPage() {
  // Mock data for charts
  const accuracyData = Array.from({ length: 30 }, (_, i) => ({
    day: `Day ${i + 1}`,
    accuracy: 90 + Math.random() * 8,
    baseline: 92,
  }))

  const latencyData = Array.from({ length: 24 }, (_, i) => ({
    hour: `${i}:00`,
    p50: 50 + Math.random() * 20,
    p95: 80 + Math.random() * 40,
    p99: 120 + Math.random() * 60,
  }))

  const modelComparison = [
    { model: 'XGBoost', accuracy: 94.2, latency: 45, memory: 512 },
    { model: 'LightGBM', accuracy: 93.8, latency: 38, memory: 420 },
    { model: 'Random Forest', accuracy: 92.5, latency: 62, memory: 680 },
    { model: 'Neural Net', accuracy: 93.1, latency: 85, memory: 890 },
    { model: 'Ensemble', accuracy: 94.8, latency: 72, memory: 750 },
  ]

  const featureDrift = [
    { feature: 'Recent Form', current: 0.02, threshold: 0.1, status: 'stable' },
    { feature: 'Opponent Rank', current: 0.08, threshold: 0.1, status: 'warning' },
    { feature: 'Home/Away', current: 0.01, threshold: 0.1, status: 'stable' },
    { feature: 'Rest Days', current: 0.12, threshold: 0.1, status: 'critical' },
    { feature: 'Historical Avg', current: 0.03, threshold: 0.1, status: 'stable' },
  ]

  const apiMetrics = [
    { endpoint: '/predict', calls: 45230, avgLatency: 87, errorRate: 0.02 },
    { endpoint: '/batch', calls: 12340, avgLatency: 234, errorRate: 0.05 },
    { endpoint: '/health', calls: 89450, avgLatency: 12, errorRate: 0 },
    { endpoint: '/metrics', calls: 23450, avgLatency: 45, errorRate: 0.01 },
  ]

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="max-w-7xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-4xl font-bold text-text-primary mb-2">
            Model Monitoring Dashboard
          </h1>
          <p className="text-text-secondary">
            Real-time performance metrics and model health monitoring
          </p>
        </motion.div>

        {/* Key Metrics Row */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <AnimatedCard delay={0.1}>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-text-secondary text-sm">Avg Accuracy</p>
                <p className="text-2xl font-bold text-text-primary">94.2%</p>
                <p className="text-xs text-success flex items-center gap-1 mt-1">
                  <TrendingUp className="w-3 h-3" />
                  +2.3% from last week
                </p>
              </div>
              <div className="p-3 bg-success/20 rounded-lg">
                <CheckCircle className="w-6 h-6 text-success" />
              </div>
            </div>
          </AnimatedCard>

          <AnimatedCard delay={0.2}>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-text-secondary text-sm">Drift Score</p>
                <p className="text-2xl font-bold text-text-primary">0.03</p>
                <p className="text-xs text-warning flex items-center gap-1 mt-1">
                  <AlertTriangle className="w-3 h-3" />
                  Monitor closely
                </p>
              </div>
              <div className="p-3 bg-warning/20 rounded-lg">
                <Activity className="w-6 h-6 text-warning" />
              </div>
            </div>
          </AnimatedCard>

          <AnimatedCard delay={0.3}>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-text-secondary text-sm">API Latency</p>
                <p className="text-2xl font-bold text-text-primary">87ms</p>
                <p className="text-xs text-success flex items-center gap-1 mt-1">
                  <Zap className="w-3 h-3" />
                  Within SLA
                </p>
              </div>
              <div className="p-3 bg-primary/20 rounded-lg">
                <Clock className="w-6 h-6 text-primary" />
              </div>
            </div>
          </AnimatedCard>

          <AnimatedCard delay={0.4}>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-text-secondary text-sm">Error Rate</p>
                <p className="text-2xl font-bold text-text-primary">0.02%</p>
                <p className="text-xs text-success flex items-center gap-1 mt-1">
                  <CheckCircle className="w-3 h-3" />
                  All systems normal
                </p>
              </div>
              <div className="p-3 bg-success/20 rounded-lg">
                <Cpu className="w-6 h-6 text-success" />
              </div>
            </div>
          </AnimatedCard>
        </div>

        {/* Charts Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* Accuracy Over Time */}
          <AnimatedCard delay={0.5} className="p-6">
            <h3 className="text-lg font-semibold text-text-primary mb-4">
              Model Accuracy Trend
            </h3>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={accuracyData}>
                <defs>
                  <linearGradient id="accuracyGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="day" stroke="#666" fontSize={10} />
                <YAxis stroke="#666" fontSize={10} domain={[88, 100]} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1a1a2e', border: 'none' }}
                  labelStyle={{ color: '#a1a1aa' }}
                />
                <Line 
                  type="monotone" 
                  dataKey="baseline" 
                  stroke="#666" 
                  strokeDasharray="5 5"
                  dot={false}
                />
                <Area 
                  type="monotone" 
                  dataKey="accuracy" 
                  stroke="#8b5cf6" 
                  fillOpacity={1} 
                  fill="url(#accuracyGradient)" 
                />
              </AreaChart>
            </ResponsiveContainer>
          </AnimatedCard>

          {/* Latency Distribution */}
          <AnimatedCard delay={0.6} className="p-6">
            <h3 className="text-lg font-semibold text-text-primary mb-4">
              API Latency Distribution
            </h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={latencyData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="hour" stroke="#666" fontSize={10} />
                <YAxis stroke="#666" fontSize={10} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1a1a2e', border: 'none' }}
                  labelStyle={{ color: '#a1a1aa' }}
                />
                <Legend />
                <Line type="monotone" dataKey="p50" stroke="#10b981" name="P50" />
                <Line type="monotone" dataKey="p95" stroke="#f59e0b" name="P95" />
                <Line type="monotone" dataKey="p99" stroke="#ef4444" name="P99" />
              </LineChart>
            </ResponsiveContainer>
          </AnimatedCard>
        </div>

        {/* Model Comparison */}
        <AnimatedCard delay={0.7} className="p-6 mb-8">
          <h3 className="text-lg font-semibold text-text-primary mb-4">
            Model Performance Comparison
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={modelComparison}>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis dataKey="model" stroke="#666" fontSize={10} />
              <YAxis stroke="#666" fontSize={10} />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1a1a2e', border: 'none' }}
                labelStyle={{ color: '#a1a1aa' }}
              />
              <Legend />
              <Bar dataKey="accuracy" fill="#8b5cf6" name="Accuracy (%)" />
              <Bar dataKey="latency" fill="#00d4ff" name="Latency (ms)" />
            </BarChart>
          </ResponsiveContainer>
        </AnimatedCard>

        {/* Feature Drift Monitor */}
        <AnimatedCard delay={0.8} className="p-6">
          <h3 className="text-lg font-semibold text-text-primary mb-4">
            Feature Drift Monitor
          </h3>
          <div className="space-y-4">
            {featureDrift.map((feature, index) => (
              <motion.div
                key={feature.feature}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.8 + index * 0.1 }}
                className="flex items-center justify-between"
              >
                <div className="flex-1">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-text-secondary">{feature.feature}</span>
                    <span className="text-sm text-text-primary">
                      {feature.current.toFixed(3)} / {feature.threshold}
                    </span>
                  </div>
                  <div className="w-full bg-card-hover rounded-full h-2">
                    <div
                      className={`h-full rounded-full ${
                        feature.status === 'stable' ? 'bg-success' :
                        feature.status === 'warning' ? 'bg-warning' : 'bg-danger'
                      }`}
                      style={{ width: `${(feature.current / feature.threshold) * 100}%` }}
                    />
                  </div>
                </div>
                <div className={`ml-4 ${
                  feature.status === 'stable' ? 'text-success' :
                  feature.status === 'warning' ? 'text-warning' : 'text-danger'
                }`}>
                  {feature.status === 'stable' ? <CheckCircle className="w-5 h-5" /> :
                   feature.status === 'warning' ? <AlertTriangle className="w-5 h-5" /> :
                   <AlertTriangle className="w-5 h-5" />}
                </div>
              </motion.div>
            ))}
          </div>
        </AnimatedCard>
      </div>
    </div>
  )
}