'use client'

import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  LineChart, Line, AreaChart, Area, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts'
import AnimatedCard from '@/components/ui/AnimatedCard'
import { 
  PlayCircle, PauseCircle, TrendingUp, TrendingDown, 
  Activity, Clock, Users, BarChart3, Plus, ChevronDown,
  ChevronUp, Filter, Search, Calendar, Settings, CheckCircle,
  AlertCircle, XCircle
} from 'lucide-react'

export default function ExperimentsPage() {
  const [selectedExperiment, setSelectedExperiment] = useState<string | null>(null)
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [trafficAllocation, setTrafficAllocation] = useState(50)
  const [expandedHistory, setExpandedHistory] = useState<string | null>(null)
  const [searchTerm, setSearchTerm] = useState('')
  const [filterStatus, setFilterStatus] = useState('all')

  // Simulated real-time updates
  const [activeExperiments, setActiveExperiments] = useState([
    {
      id: 'exp-001',
      name: 'XGBoost vs Ensemble Q1-2025',
      status: 'running',
      startDate: '2025-01-15',
      progress: 8432,
      total: 10000,
      control: {
        name: 'XGBoost v2.1',
        accuracy: 91.3,
        latency: 95,
        predictions: 4216,
        errors: 12
      },
      variant: {
        name: 'Ensemble v3.0',
        accuracy: 93.7,
        latency: 87,
        predictions: 4216,
        errors: 8
      },
      lift: {
        accuracy: 2.4,
        latency: -8.4
      },
      pValue: 0.023,
      trafficSplit: 50,
      confidence: 97.7
    },
    {
      id: 'exp-002',
      name: 'Neural Net Feature Engineering',
      status: 'running',
      startDate: '2025-01-18',
      progress: 3245,
      total: 5000,
      control: {
        name: 'Basic Features',
        accuracy: 89.8,
        latency: 120,
        predictions: 1622,
        errors: 34
      },
      variant: {
        name: 'Enhanced Features',
        accuracy: 91.2,
        latency: 135,
        predictions: 1623,
        errors: 28
      },
      lift: {
        accuracy: 1.4,
        latency: 12.5
      },
      pValue: 0.087,
      trafficSplit: 50,
      confidence: 91.3
    },
    {
      id: 'exp-003',
      name: 'Cache Strategy Optimization',
      status: 'scheduled',
      startDate: '2025-01-25',
      progress: 0,
      total: 15000,
      control: {
        name: 'Redis Standard',
        accuracy: 94.2,
        latency: 45,
        predictions: 0,
        errors: 0
      },
      variant: {
        name: 'Redis + Edge Cache',
        accuracy: 94.2,
        latency: 28,
        predictions: 0,
        errors: 0
      },
      lift: {
        accuracy: 0,
        latency: -37.8
      },
      pValue: null,
      trafficSplit: 30,
      confidence: 0
    }
  ])

  const experimentHistory = [
    {
      id: 'hist-001',
      name: 'LightGBM Parameter Tuning',
      dateRange: '2024-12-01 - 2024-12-15',
      winner: 'Variant B',
      lift: '+3.2% accuracy',
      sampleSize: 25000,
      decision: 'Deployed to Production',
      details: {
        control: { accuracy: 90.1, latency: 78, errorRate: 0.03 },
        variant: { accuracy: 93.3, latency: 82, errorRate: 0.02 },
        pValue: 0.001,
        confidence: 99.9
      }
    },
    {
      id: 'hist-002',
      name: 'Feature Selection Experiment',
      dateRange: '2024-11-15 - 2024-11-30',
      winner: 'Control',
      lift: '-0.8% accuracy',
      sampleSize: 18500,
      decision: 'Rejected',
      details: {
        control: { accuracy: 92.5, latency: 65, errorRate: 0.025 },
        variant: { accuracy: 91.7, latency: 58, errorRate: 0.028 },
        pValue: 0.142,
        confidence: 85.8
      }
    },
    {
      id: 'hist-003',
      name: 'Batch Prediction Optimization',
      dateRange: '2024-11-01 - 2024-11-14',
      winner: 'Variant A',
      lift: '-45% latency',
      sampleSize: 32000,
      decision: 'Deployed to Production',
      details: {
        control: { accuracy: 93.8, latency: 234, errorRate: 0.02 },
        variant: { accuracy: 93.9, latency: 128, errorRate: 0.019 },
        pValue: 0.0001,
        confidence: 99.99
      }
    }
  ]

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      setActiveExperiments(prev => prev.map(exp => {
        if (exp.status === 'running' && exp.progress < exp.total) {
          const increment = Math.floor(Math.random() * 50) + 10
          const newProgress = Math.min(exp.progress + increment, exp.total)
          const halfIncrement = Math.floor(increment / 2)
          
          return {
            ...exp,
            progress: newProgress,
            control: {
              ...exp.control,
              predictions: exp.control.predictions + halfIncrement,
              accuracy: exp.control.accuracy + (Math.random() - 0.5) * 0.1,
              latency: exp.control.latency + (Math.random() - 0.5) * 2,
              errors: exp.control.errors + Math.floor(Math.random() * 2)
            },
            variant: {
              ...exp.variant,
              predictions: exp.variant.predictions + halfIncrement,
              accuracy: exp.variant.accuracy + (Math.random() - 0.5) * 0.1,
              latency: exp.variant.latency + (Math.random() - 0.5) * 2,
              errors: exp.variant.errors + Math.floor(Math.random() * 2)
            }
          }
        }
        return exp
      }))
    }, 3000)

    return () => clearInterval(interval)
  }, [])

  // Time series data for experiment metrics
  const getTimeSeriesData = () => {
    return Array.from({ length: 24 }, (_, i) => ({
      hour: `${i}:00`,
      control: 90 + Math.random() * 4,
      variant: 92 + Math.random() * 4,
    }))
  }

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'running':
        return (
          <span className="flex items-center gap-1 px-3 py-1 bg-success/20 text-success rounded-full text-xs font-medium">
            <div className="w-2 h-2 bg-success rounded-full animate-pulse" />
            Running
          </span>
        )
      case 'completed':
        return (
          <span className="flex items-center gap-1 px-3 py-1 bg-primary/20 text-primary rounded-full text-xs font-medium">
            <CheckCircle className="w-3 h-3" />
            Completed
          </span>
        )
      case 'scheduled':
        return (
          <span className="flex items-center gap-1 px-3 py-1 bg-warning/20 text-warning rounded-full text-xs font-medium">
            <Clock className="w-3 h-3" />
            Scheduled
          </span>
        )
      default:
        return null
    }
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 95) return 'text-success'
    if (confidence >= 90) return 'text-warning'
    return 'text-danger'
  }

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-4xl font-bold text-text-primary mb-2">
            ML Experiments Hub
          </h1>
          <p className="text-text-secondary">
            {activeExperiments.filter(e => e.status === 'running').length} active experiments running
          </p>
        </motion.div>

        {/* Quick Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <AnimatedCard delay={0.1}>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-text-secondary text-sm">Active Experiments</p>
                <p className="text-2xl font-bold text-text-primary">
                  {activeExperiments.filter(e => e.status === 'running').length}
                </p>
                <p className="text-xs text-success mt-1">2 completing soon</p>
              </div>
              <div className="p-3 bg-primary/20 rounded-lg">
                <Activity className="w-6 h-6 text-primary" />
              </div>
            </div>
          </AnimatedCard>

          <AnimatedCard delay={0.2}>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-text-secondary text-sm">Success Rate</p>
                <p className="text-2xl font-bold text-text-primary">87.5%</p>
                <p className="text-xs text-success flex items-center gap-1 mt-1">
                  <TrendingUp className="w-3 h-3" />
                  +5.2% this month
                </p>
              </div>
              <div className="p-3 bg-success/20 rounded-lg">
                <CheckCircle className="w-6 h-6 text-success" />
              </div>
            </div>
          </AnimatedCard>

          <AnimatedCard delay={0.3}>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-text-secondary text-sm">Avg Lift</p>
                <p className="text-2xl font-bold text-text-primary">+2.8%</p>
                <p className="text-xs text-primary mt-1">Accuracy improvement</p>
              </div>
              <div className="p-3 bg-secondary/20 rounded-lg">
                <TrendingUp className="w-6 h-6 text-secondary" />
              </div>
            </div>
          </AnimatedCard>

          <AnimatedCard delay={0.4}>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-text-secondary text-sm">Total Comparisons</p>
                <p className="text-2xl font-bold text-text-primary">1,247</p>
                <p className="text-xs text-text-secondary mt-1">This quarter</p>
              </div>
              <div className="p-3 bg-warning/20 rounded-lg">
                <BarChart3 className="w-6 h-6 text-warning" />
              </div>
            </div>
          </AnimatedCard>
        </div>

        {/* Active Experiments */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-semibold text-text-primary">Active Experiments</h2>
            <button
              onClick={() => setShowCreateModal(true)}
              className="flex items-center gap-2 px-4 py-2 bg-primary hover:bg-primary-hover text-white rounded-lg transition-colors"
            >
              <Plus className="w-4 h-4" />
              New Experiment
            </button>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {activeExperiments.map((exp, index) => (
              <AnimatedCard key={exp.id} delay={0.5 + index * 0.1} className="p-6">
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <h3 className="text-lg font-semibold text-text-primary mb-1">{exp.name}</h3>
                    <div className="flex items-center gap-3">
                      {getStatusBadge(exp.status)}
                      <span className="text-xs text-text-secondary">Started {exp.startDate}</span>
                    </div>
                  </div>
                  {exp.status === 'running' ? (
                    <PauseCircle className="w-6 h-6 text-text-secondary hover:text-warning cursor-pointer" />
                  ) : (
                    <PlayCircle className="w-6 h-6 text-text-secondary hover:text-success cursor-pointer" />
                  )}
                </div>

                {/* Progress Bar */}
                <div className="mb-4">
                  <div className="flex items-center justify-between text-xs text-text-secondary mb-1">
                    <span>{exp.progress.toLocaleString()} / {exp.total.toLocaleString()} predictions</span>
                    <span>{((exp.progress / exp.total) * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-card-hover rounded-full h-2">
                    <div
                      className="h-full bg-gradient-to-r from-primary to-secondary rounded-full transition-all duration-500"
                      style={{ width: `${(exp.progress / exp.total) * 100}%` }}
                    />
                  </div>
                </div>

                {/* Metrics Comparison */}
                <div className="space-y-3">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-card-hover rounded-lg p-3">
                      <p className="text-xs text-text-secondary mb-1">Control: {exp.control.name}</p>
                      <div className="space-y-1">
                        <div className="flex justify-between text-sm">
                          <span className="text-text-secondary">Accuracy:</span>
                          <span className="text-text-primary font-medium">{exp.control.accuracy.toFixed(1)}%</span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span className="text-text-secondary">Latency:</span>
                          <span className="text-text-primary font-medium">{exp.control.latency}ms</span>
                        </div>
                      </div>
                    </div>
                    <div className="bg-card-hover rounded-lg p-3">
                      <p className="text-xs text-text-secondary mb-1">Variant: {exp.variant.name}</p>
                      <div className="space-y-1">
                        <div className="flex justify-between text-sm">
                          <span className="text-text-secondary">Accuracy:</span>
                          <span className="text-text-primary font-medium">{exp.variant.accuracy.toFixed(1)}%</span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span className="text-text-secondary">Latency:</span>
                          <span className="text-text-primary font-medium">{exp.variant.latency}ms</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Lift Metrics */}
                  <div className="bg-gradient-to-r from-primary/10 to-secondary/10 rounded-lg p-3">
                    <p className="text-xs text-text-secondary mb-2">Performance Lift</p>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-text-secondary">Accuracy:</span>
                        <span className={`text-sm font-medium flex items-center gap-1 ${
                          exp.lift.accuracy > 0 ? 'text-success' : 'text-danger'
                        }`}>
                          {exp.lift.accuracy > 0 ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
                          {exp.lift.accuracy > 0 ? '+' : ''}{exp.lift.accuracy.toFixed(1)}%
                        </span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-text-secondary">Latency:</span>
                        <span className={`text-sm font-medium flex items-center gap-1 ${
                          exp.lift.latency < 0 ? 'text-success' : 'text-danger'
                        }`}>
                          {exp.lift.latency < 0 ? <TrendingDown className="w-3 h-3" /> : <TrendingUp className="w-3 h-3" />}
                          {exp.lift.latency > 0 ? '+' : ''}{exp.lift.latency.toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Statistical Significance */}
                  {exp.pValue !== null && (
                    <div className="flex items-center justify-between pt-2 border-t border-white/10">
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-text-secondary">p-value:</span>
                        <span className={`text-xs font-medium ${
                          exp.pValue < 0.05 ? 'text-success' : 'text-warning'
                        }`}>
                          {exp.pValue.toFixed(3)}
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-text-secondary">Confidence:</span>
                        <span className={`text-xs font-medium ${getConfidenceColor(exp.confidence)}`}>
                          {exp.confidence.toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  )}

                  {/* Traffic Allocation */}
                  <div className="flex items-center gap-2 text-xs">
                    <span className="text-text-secondary">Traffic Split:</span>
                    <div className="flex-1 flex items-center gap-2">
                      <div className="flex-1 bg-card-hover rounded-full h-2 relative overflow-hidden">
                        <div 
                          className="absolute left-0 top-0 h-full bg-primary"
                          style={{ width: `${exp.trafficSplit}%` }}
                        />
                        <div 
                          className="absolute right-0 top-0 h-full bg-secondary"
                          style={{ width: `${100 - exp.trafficSplit}%` }}
                        />
                      </div>
                      <span className="text-text-primary">{exp.trafficSplit}/{100 - exp.trafficSplit}</span>
                    </div>
                  </div>
                </div>
              </AnimatedCard>
            ))}
          </div>
        </div>

        {/* Experiment Results History */}
        <AnimatedCard delay={0.8} className="p-6">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-semibold text-text-primary">Experiment History</h2>
            <div className="flex items-center gap-3">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-secondary" />
                <input
                  type="text"
                  placeholder="Search experiments..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10 pr-4 py-2 bg-card-hover text-text-primary rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                />
              </div>
              <select
                value={filterStatus}
                onChange={(e) => setFilterStatus(e.target.value)}
                className="px-4 py-2 bg-card-hover text-text-primary rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary"
              >
                <option value="all">All Results</option>
                <option value="deployed">Deployed</option>
                <option value="rejected">Rejected</option>
              </select>
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left py-3 px-4 text-sm font-medium text-text-secondary">Experiment Name</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-text-secondary">Date Range</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-text-secondary">Winner</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-text-secondary">Lift</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-text-secondary">Sample Size</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-text-secondary">Decision</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-text-secondary"></th>
                </tr>
              </thead>
              <tbody>
                {experimentHistory.map((exp) => (
                  <React.Fragment key={exp.id}>
                    <tr className="border-b border-white/5 hover:bg-card-hover transition-colors cursor-pointer"
                        onClick={() => setExpandedHistory(expandedHistory === exp.id ? null : exp.id)}>
                      <td className="py-3 px-4 text-sm text-text-primary font-medium">{exp.name}</td>
                      <td className="py-3 px-4 text-sm text-text-secondary">{exp.dateRange}</td>
                      <td className="py-3 px-4 text-sm text-text-primary">{exp.winner}</td>
                      <td className="py-3 px-4">
                        <span className={`text-sm font-medium ${
                          exp.lift.includes('+') ? 'text-success' : 'text-danger'
                        }`}>
                          {exp.lift}
                        </span>
                      </td>
                      <td className="py-3 px-4 text-sm text-text-secondary">{exp.sampleSize.toLocaleString()}</td>
                      <td className="py-3 px-4">
                        <span className={`inline-flex px-2 py-1 text-xs rounded-full ${
                          exp.decision === 'Deployed to Production' 
                            ? 'bg-success/20 text-success' 
                            : 'bg-danger/20 text-danger'
                        }`}>
                          {exp.decision}
                        </span>
                      </td>
                      <td className="py-3 px-4">
                        {expandedHistory === exp.id ? (
                          <ChevronUp className="w-4 h-4 text-text-secondary" />
                        ) : (
                          <ChevronDown className="w-4 h-4 text-text-secondary" />
                        )}
                      </td>
                    </tr>
                    {expandedHistory === exp.id && (
                      <tr>
                        <td colSpan={7} className="p-4 bg-card-hover">
                          <div className="grid grid-cols-3 gap-4">
                            <div>
                              <p className="text-xs text-text-secondary mb-2">Control Metrics</p>
                              <div className="space-y-1">
                                <div className="flex justify-between text-sm">
                                  <span className="text-text-secondary">Accuracy:</span>
                                  <span className="text-text-primary">{exp.details.control.accuracy}%</span>
                                </div>
                                <div className="flex justify-between text-sm">
                                  <span className="text-text-secondary">Latency:</span>
                                  <span className="text-text-primary">{exp.details.control.latency}ms</span>
                                </div>
                                <div className="flex justify-between text-sm">
                                  <span className="text-text-secondary">Error Rate:</span>
                                  <span className="text-text-primary">{exp.details.control.errorRate}%</span>
                                </div>
                              </div>
                            </div>
                            <div>
                              <p className="text-xs text-text-secondary mb-2">Variant Metrics</p>
                              <div className="space-y-1">
                                <div className="flex justify-between text-sm">
                                  <span className="text-text-secondary">Accuracy:</span>
                                  <span className="text-text-primary">{exp.details.variant.accuracy}%</span>
                                </div>
                                <div className="flex justify-between text-sm">
                                  <span className="text-text-secondary">Latency:</span>
                                  <span className="text-text-primary">{exp.details.variant.latency}ms</span>
                                </div>
                                <div className="flex justify-between text-sm">
                                  <span className="text-text-secondary">Error Rate:</span>
                                  <span className="text-text-primary">{exp.details.variant.errorRate}%</span>
                                </div>
                              </div>
                            </div>
                            <div>
                              <p className="text-xs text-text-secondary mb-2">Statistical Analysis</p>
                              <div className="space-y-1">
                                <div className="flex justify-between text-sm">
                                  <span className="text-text-secondary">p-value:</span>
                                  <span className={`font-medium ${
                                    exp.details.pValue < 0.05 ? 'text-success' : 'text-warning'
                                  }`}>
                                    {exp.details.pValue}
                                  </span>
                                </div>
                                <div className="flex justify-between text-sm">
                                  <span className="text-text-secondary">Confidence:</span>
                                  <span className={`font-medium ${getConfidenceColor(exp.details.confidence)}`}>
                                    {exp.details.confidence}%
                                  </span>
                                </div>
                              </div>
                            </div>
                          </div>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                ))}
              </tbody>
            </table>
          </div>
        </AnimatedCard>

        {/* Experiment Metrics Chart */}
        <AnimatedCard delay={0.9} className="p-6 mt-6">
          <h3 className="text-lg font-semibold text-text-primary mb-4">
            Real-time Performance Comparison
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={getTimeSeriesData()}>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis dataKey="hour" stroke="#666" fontSize={10} />
              <YAxis stroke="#666" fontSize={10} domain={[88, 96]} />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1a1a2e', border: 'none' }}
                labelStyle={{ color: '#a1a1aa' }}
              />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="control" 
                stroke="#8b5cf6" 
                name="Control"
                strokeWidth={2}
              />
              <Line 
                type="monotone" 
                dataKey="variant" 
                stroke="#00d4ff" 
                name="Variant"
                strokeWidth={2}
              />
            </LineChart>
          </ResponsiveContainer>
        </AnimatedCard>
      </div>

      {/* Create Experiment Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-card rounded-xl p-6 max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto"
          >
            <h3 className="text-xl font-semibold text-text-primary mb-4">Create New Experiment</h3>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm text-text-secondary mb-2">Experiment Name</label>
                <input
                  type="text"
                  className="w-full px-4 py-2 bg-card-hover text-text-primary rounded-lg focus:outline-none focus:ring-2 focus:ring-primary"
                  placeholder="e.g., Random Forest vs XGBoost Q1-2025"
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm text-text-secondary mb-2">Control Model</label>
                  <select className="w-full px-4 py-2 bg-card-hover text-text-primary rounded-lg focus:outline-none focus:ring-2 focus:ring-primary">
                    <option>XGBoost v2.1</option>
                    <option>Random Forest v1.8</option>
                    <option>Neural Net v3.0</option>
                    <option>Ensemble v2.5</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm text-text-secondary mb-2">Variant Model</label>
                  <select className="w-full px-4 py-2 bg-card-hover text-text-primary rounded-lg focus:outline-none focus:ring-2 focus:ring-primary">
                    <option>XGBoost v2.2</option>
                    <option>Random Forest v2.0</option>
                    <option>Neural Net v3.1</option>
                    <option>Ensemble v3.0</option>
                  </select>
                </div>
              </div>

              <div>
                <label className="block text-sm text-text-secondary mb-2">Success Metrics</label>
                <div className="grid grid-cols-3 gap-2">
                  <label className="flex items-center gap-2">
                    <input type="checkbox" className="rounded" defaultChecked />
                    <span className="text-sm text-text-primary">Accuracy</span>
                  </label>
                  <label className="flex items-center gap-2">
                    <input type="checkbox" className="rounded" defaultChecked />
                    <span className="text-sm text-text-primary">Latency</span>
                  </label>
                  <label className="flex items-center gap-2">
                    <input type="checkbox" className="rounded" />
                    <span className="text-sm text-text-primary">Error Rate</span>
                  </label>
                </div>
              </div>

              <div>
                <label className="block text-sm text-text-secondary mb-2">
                  Traffic Allocation: {trafficAllocation}% / {100 - trafficAllocation}%
                </label>
                <input
                  type="range"
                  min="10"
                  max="90"
                  value={trafficAllocation}
                  onChange={(e) => setTrafficAllocation(Number(e.target.value))}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-text-secondary mt-1">
                  <span>Control: {trafficAllocation}%</span>
                  <span>Variant: {100 - trafficAllocation}%</span>
                </div>
              </div>

              <div>
                <label className="block text-sm text-text-secondary mb-2">Minimum Sample Size</label>
                <input
                  type="number"
                  className="w-full px-4 py-2 bg-card-hover text-text-primary rounded-lg focus:outline-none focus:ring-2 focus:ring-primary"
                  placeholder="10000"
                  defaultValue="10000"
                />
              </div>

              <div className="bg-primary/10 rounded-lg p-4">
                <p className="text-sm text-text-secondary mb-2">Estimated Duration</p>
                <p className="text-lg font-semibold text-text-primary">7-10 days</p>
                <p className="text-xs text-text-secondary mt-1">Based on current traffic and sample size</p>
              </div>
            </div>

            <div className="flex gap-3 mt-6">
              <button
                onClick={() => setShowCreateModal(false)}
                className="flex-1 px-4 py-2 bg-card-hover text-text-primary rounded-lg hover:bg-card transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={() => setShowCreateModal(false)}
                className="flex-1 px-4 py-2 bg-primary hover:bg-primary-hover text-white rounded-lg transition-colors"
              >
                Start Experiment
              </button>
            </div>
          </motion.div>
        </div>
      )}
    </div>
  )
}