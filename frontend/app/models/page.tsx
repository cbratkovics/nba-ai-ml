'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  LineChart, Line, AreaChart, Area, BarChart, Bar, RadarChart, Radar,
  PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell
} from 'recharts'
import AnimatedCard from '@/components/ui/AnimatedCard'
import { 
  Database, Brain, Zap, Clock, AlertCircle, CheckCircle,
  TrendingUp, TrendingDown, RefreshCw, ChevronRight, Settings,
  Download, Upload, GitBranch, Shield, Cpu, HardDrive,
  Activity, BarChart3, Eye, ArrowRight, ArrowLeft, Play, XCircle
} from 'lucide-react'

export default function ModelsPage() {
  const [selectedModel, setSelectedModel] = useState<string | null>(null)
  const [showComparisonMatrix, setShowComparisonMatrix] = useState(false)
  const [activeTab, setActiveTab] = useState('production')
  const [draggedModel, setDraggedModel] = useState<string | null>(null)

  // Production Models
  const productionModels = [
    {
      id: 'model-001',
      name: 'points_predictor_v2.3',
      version: 'v2.3',
      algorithm: 'Random Forest',
      status: 'healthy',
      accuracy: 94.7,
      precision: 93.2,
      recall: 95.8,
      f1Score: 94.5,
      mae: 2.34,
      predictionsToday: 145234,
      avgLatency: 82,
      lastRetrained: '2 days ago',
      deployedDate: '2025-01-15',
      memoryUsage: 512,
      cpuUsage: 23,
      accuracyTrend: [92, 93, 93.5, 94, 94.2, 94.5, 94.7],
      features: ['player_avg_30d', 'opponent_rank', 'home_away', 'rest_days', 'injury_status'],
      hyperparameters: {
        n_estimators: 200,
        max_depth: 15,
        min_samples_split: 5
      }
    },
    {
      id: 'model-002',
      name: 'assists_predictor_v1.8',
      version: 'v1.8',
      algorithm: 'XGBoost',
      status: 'healthy',
      accuracy: 92.3,
      precision: 91.8,
      recall: 92.9,
      f1Score: 92.3,
      mae: 1.87,
      predictionsToday: 98456,
      avgLatency: 95,
      lastRetrained: '5 days ago',
      deployedDate: '2025-01-10',
      memoryUsage: 420,
      cpuUsage: 18,
      accuracyTrend: [90, 90.5, 91, 91.5, 92, 92.1, 92.3],
      features: ['team_pace', 'player_usage_rate', 'defensive_rating', 'minutes_played'],
      hyperparameters: {
        learning_rate: 0.1,
        n_estimators: 150,
        max_depth: 8
      }
    },
    {
      id: 'model-003',
      name: 'rebounds_predictor_v3.1',
      version: 'v3.1',
      algorithm: 'Ensemble',
      status: 'degrading',
      accuracy: 89.8,
      precision: 88.9,
      recall: 90.6,
      f1Score: 89.7,
      mae: 3.12,
      predictionsToday: 67890,
      avgLatency: 124,
      lastRetrained: '12 days ago',
      deployedDate: '2025-01-05',
      memoryUsage: 750,
      cpuUsage: 31,
      accuracyTrend: [91, 91, 90.5, 90.2, 90, 89.9, 89.8],
      features: ['height', 'position', 'opponent_pace', 'team_rebounds_avg'],
      hyperparameters: {
        base_models: ['rf', 'xgboost', 'lightgbm'],
        meta_learner: 'logistic_regression'
      }
    },
    {
      id: 'model-004',
      name: 'player_impact_v2.0',
      version: 'v2.0',
      algorithm: 'Neural Network',
      status: 'healthy',
      accuracy: 93.1,
      precision: 92.5,
      recall: 93.7,
      f1Score: 93.1,
      mae: 2.67,
      predictionsToday: 234567,
      avgLatency: 108,
      lastRetrained: '1 day ago',
      deployedDate: '2025-01-18',
      memoryUsage: 890,
      cpuUsage: 42,
      accuracyTrend: [91.5, 92, 92.3, 92.6, 92.8, 93, 93.1],
      features: ['player_efficiency', 'win_shares', 'plus_minus', 'usage_rate'],
      hyperparameters: {
        layers: [128, 64, 32],
        activation: 'relu',
        dropout: 0.3
      }
    },
    {
      id: 'model-005',
      name: 'game_outcome_v1.5',
      version: 'v1.5',
      algorithm: 'LightGBM',
      status: 'healthy',
      accuracy: 91.2,
      precision: 90.8,
      recall: 91.6,
      f1Score: 91.2,
      mae: 2.98,
      predictionsToday: 56789,
      avgLatency: 76,
      lastRetrained: '7 days ago',
      deployedDate: '2025-01-12',
      memoryUsage: 380,
      cpuUsage: 15,
      accuracyTrend: [89.5, 90, 90.3, 90.6, 90.9, 91.1, 91.2],
      features: ['team_form_10g', 'head_to_head', 'injury_count', 'rest_advantage'],
      hyperparameters: {
        num_leaves: 31,
        learning_rate: 0.05,
        feature_fraction: 0.9
      }
    }
  ]

  // Staging Models
  const stagingModels = [
    {
      id: 'staging-001',
      name: 'points_predictor_v3.0',
      version: 'v3.0-beta',
      algorithm: 'Transformer',
      accuracy: 95.2,
      latency: 145,
      status: 'testing'
    },
    {
      id: 'staging-002',
      name: 'assists_predictor_v2.0',
      version: 'v2.0-alpha',
      algorithm: 'XGBoost',
      accuracy: 93.8,
      latency: 88,
      status: 'validation'
    },
    {
      id: 'staging-003',
      name: 'injury_risk_v1.0',
      version: 'v1.0-rc',
      algorithm: 'Random Forest',
      accuracy: 87.5,
      latency: 92,
      status: 'ready'
    }
  ]

  // Training Jobs
  const [trainingJobs, setTrainingJobs] = useState([
    {
      id: 'job-001',
      modelName: 'points_predictor_v3.1',
      status: 'running',
      progress: 67,
      startTime: '10:23 AM',
      eta: '12 min',
      cpuUsage: 78,
      memoryUsage: 4.2,
      currentEpoch: 67,
      totalEpochs: 100,
      loss: 0.0234
    },
    {
      id: 'job-002',
      modelName: 'rebounds_predictor_v3.2',
      status: 'queued',
      progress: 0,
      startTime: 'Pending',
      eta: '45 min',
      cpuUsage: 0,
      memoryUsage: 0,
      currentEpoch: 0,
      totalEpochs: 150,
      loss: null
    }
  ])

  // Feature importance data
  const getFeatureImportance = (modelId: string) => {
    const importanceMap: { [key: string]: any[] } = {
      'model-001': [
        { feature: 'player_avg_30d', importance: 0.35 },
        { feature: 'opponent_rank', importance: 0.25 },
        { feature: 'home_away', importance: 0.18 },
        { feature: 'rest_days', importance: 0.12 },
        { feature: 'injury_status', importance: 0.10 }
      ],
      'model-002': [
        { feature: 'team_pace', importance: 0.40 },
        { feature: 'player_usage_rate', importance: 0.28 },
        { feature: 'defensive_rating', importance: 0.20 },
        { feature: 'minutes_played', importance: 0.12 }
      ]
    }
    return importanceMap[modelId] || []
  }

  // Model comparison matrix data
  const comparisonMetrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'MAE', 'Latency']
  const modelComparison = productionModels.map(model => ({
    name: model.name.split('_')[0],
    accuracy: model.accuracy,
    precision: model.precision,
    recall: model.recall,
    f1: model.f1Score,
    mae: model.mae,
    latency: model.avgLatency
  }))

  // Simulate training progress
  useEffect(() => {
    const interval = setInterval(() => {
      setTrainingJobs(prev => prev.map(job => {
        if (job.status === 'running' && job.progress < 100) {
          const newProgress = Math.min(job.progress + Math.random() * 2, 100)
          return {
            ...job,
            progress: newProgress,
            currentEpoch: Math.floor((newProgress / 100) * job.totalEpochs),
            loss: Math.max(0.0234 - (newProgress / 100) * 0.015, 0.008),
            cpuUsage: 70 + Math.random() * 20,
            memoryUsage: 3.5 + Math.random() * 2
          }
        }
        return job
      }))
    }, 2000)

    return () => clearInterval(interval)
  }, [])

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'healthy':
        return (
          <span className="flex items-center gap-1 px-2 py-1 bg-success/20 text-success rounded-full text-xs">
            <CheckCircle className="w-3 h-3" />
            Healthy
          </span>
        )
      case 'degrading':
        return (
          <span className="flex items-center gap-1 px-2 py-1 bg-warning/20 text-warning rounded-full text-xs">
            <AlertCircle className="w-3 h-3" />
            Degrading
          </span>
        )
      default:
        return null
    }
  }

  const getAlgorithmBadge = (algorithm: string) => {
    const colors: { [key: string]: string } = {
      'Random Forest': 'bg-green-500/20 text-green-400',
      'XGBoost': 'bg-blue-500/20 text-blue-400',
      'Ensemble': 'bg-purple-500/20 text-purple-400',
      'Neural Network': 'bg-pink-500/20 text-pink-400',
      'LightGBM': 'bg-cyan-500/20 text-cyan-400',
      'Transformer': 'bg-orange-500/20 text-orange-400'
    }
    return (
      <span className={`px-2 py-1 rounded-full text-xs font-medium ${colors[algorithm] || 'bg-gray-500/20 text-gray-400'}`}>
        {algorithm}
      </span>
    )
  }

  const handleDragStart = (e: React.DragEvent, modelId: string) => {
    setDraggedModel(modelId)
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
  }

  const handleDrop = (e: React.DragEvent, stage: string) => {
    e.preventDefault()
    console.log(`Moving model ${draggedModel} to ${stage}`)
    setDraggedModel(null)
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
            Model Registry
          </h1>
          <p className="text-text-secondary">
            Version control and lifecycle management for ML models
          </p>
        </motion.div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <AnimatedCard delay={0.1}>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-text-secondary text-sm">Models in Production</p>
                <p className="text-2xl font-bold text-text-primary">5</p>
                <p className="text-xs text-success mt-1">All healthy</p>
              </div>
              <div className="p-3 bg-success/20 rounded-lg">
                <Database className="w-6 h-6 text-success" />
              </div>
            </div>
          </AnimatedCard>

          <AnimatedCard delay={0.2}>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-text-secondary text-sm">Staging Models</p>
                <p className="text-2xl font-bold text-text-primary">3</p>
                <p className="text-xs text-warning mt-1">1 ready to deploy</p>
              </div>
              <div className="p-3 bg-warning/20 rounded-lg">
                <GitBranch className="w-6 h-6 text-warning" />
              </div>
            </div>
          </AnimatedCard>

          <AnimatedCard delay={0.3}>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-text-secondary text-sm">Total Versions</p>
                <p className="text-2xl font-bold text-text-primary">47</p>
                <p className="text-xs text-text-secondary mt-1">Across all models</p>
              </div>
              <div className="p-3 bg-primary/20 rounded-lg">
                <RefreshCw className="w-6 h-6 text-primary" />
              </div>
            </div>
          </AnimatedCard>

          <AnimatedCard delay={0.4}>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-text-secondary text-sm">Avg Accuracy</p>
                <p className="text-2xl font-bold text-text-primary">94.2%</p>
                <p className="text-xs text-success flex items-center gap-1 mt-1">
                  <TrendingUp className="w-3 h-3" />
                  +1.3% this week
                </p>
              </div>
              <div className="p-3 bg-secondary/20 rounded-lg">
                <Brain className="w-6 h-6 text-secondary" />
              </div>
            </div>
          </AnimatedCard>
        </div>

        {/* Production Models */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-semibold text-text-primary">Production Models</h2>
            <button
              onClick={() => setShowComparisonMatrix(!showComparisonMatrix)}
              className="flex items-center gap-2 px-4 py-2 bg-card-hover text-text-primary rounded-lg hover:bg-card transition-colors"
            >
              <BarChart3 className="w-4 h-4" />
              {showComparisonMatrix ? 'Hide' : 'Show'} Comparison
            </button>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
            {productionModels.map((model, index) => (
              <AnimatedCard 
                key={model.id} 
                delay={0.5 + index * 0.1} 
                className="p-6"
              >
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <h3 className="text-lg font-semibold text-text-primary">{model.name}</h3>
                    <div className="flex items-center gap-2 mt-1">
                      <span className="text-xs text-text-secondary">{model.version}</span>
                      {getAlgorithmBadge(model.algorithm)}
                    </div>
                  </div>
                  {getStatusBadge(model.status)}
                </div>

                {/* Performance Metrics */}
                <div className="grid grid-cols-2 gap-3 mb-4">
                  <div className="bg-card-hover rounded-lg p-2">
                    <p className="text-xs text-text-secondary">Accuracy</p>
                    <p className="text-lg font-semibold text-text-primary">{model.accuracy}%</p>
                  </div>
                  <div className="bg-card-hover rounded-lg p-2">
                    <p className="text-xs text-text-secondary">Latency</p>
                    <p className="text-lg font-semibold text-text-primary">{model.avgLatency}ms</p>
                  </div>
                </div>

                {/* Usage Stats */}
                <div className="space-y-2 mb-4">
                  <div className="flex justify-between text-sm">
                    <span className="text-text-secondary">Predictions today:</span>
                    <span className="text-text-primary font-medium">{model.predictionsToday.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-text-secondary">Last retrained:</span>
                    <span className="text-text-primary">{model.lastRetrained}</span>
                  </div>
                </div>

                {/* Accuracy Trend Sparkline */}
                <div className="mb-4">
                  <p className="text-xs text-text-secondary mb-2">Accuracy Trend (7 days)</p>
                  <ResponsiveContainer width="100%" height={40}>
                    <LineChart data={model.accuracyTrend.map((val, i) => ({ day: i, value: val }))}>
                      <Line 
                        type="monotone" 
                        dataKey="value" 
                        stroke={model.status === 'degrading' ? '#f59e0b' : '#10b981'}
                        strokeWidth={2}
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                {/* Resource Usage */}
                <div className="flex items-center gap-4 pt-3 border-t border-white/10">
                  <div className="flex items-center gap-1 text-xs">
                    <Cpu className="w-3 h-3 text-text-secondary" />
                    <span className="text-text-secondary">CPU:</span>
                    <span className="text-text-primary">{model.cpuUsage}%</span>
                  </div>
                  <div className="flex items-center gap-1 text-xs">
                    <HardDrive className="w-3 h-3 text-text-secondary" />
                    <span className="text-text-secondary">Mem:</span>
                    <span className="text-text-primary">{model.memoryUsage}MB</span>
                  </div>
                </div>

                {/* Actions */}
                <div className="flex gap-2 mt-4">
                  <button 
                    onClick={() => setSelectedModel(model.id)}
                    className="flex-1 px-3 py-1.5 bg-primary/20 text-primary rounded-lg text-sm hover:bg-primary/30 transition-colors"
                  >
                    View Details
                  </button>
                  <button className="px-3 py-1.5 bg-card-hover text-text-secondary rounded-lg text-sm hover:bg-card transition-colors">
                    <RefreshCw className="w-4 h-4" />
                  </button>
                  <button className="px-3 py-1.5 bg-card-hover text-text-secondary rounded-lg text-sm hover:bg-card transition-colors">
                    <ArrowLeft className="w-4 h-4" />
                  </button>
                </div>
              </AnimatedCard>
            ))}
          </div>
        </div>

        {/* Model Comparison Matrix */}
        {showComparisonMatrix && (
          <AnimatedCard delay={0.7} className="p-6 mb-8">
            <h3 className="text-lg font-semibold text-text-primary mb-4">Model Comparison Matrix</h3>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-white/10">
                    <th className="text-left py-3 px-4 text-sm font-medium text-text-secondary">Model</th>
                    {comparisonMetrics.map(metric => (
                      <th key={metric} className="text-center py-3 px-4 text-sm font-medium text-text-secondary">
                        {metric}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {modelComparison.map((model, index) => (
                    <tr key={index} className="border-b border-white/5">
                      <td className="py-3 px-4 text-sm text-text-primary font-medium">{model.name}</td>
                      <td className="py-3 px-4 text-center">
                        <span className={`inline-block px-2 py-1 rounded text-xs font-medium ${
                          model.accuracy === Math.max(...modelComparison.map(m => m.accuracy))
                            ? 'bg-success/20 text-success'
                            : model.accuracy === Math.min(...modelComparison.map(m => m.accuracy))
                            ? 'bg-danger/20 text-danger'
                            : 'bg-card-hover text-text-primary'
                        }`}>
                          {model.accuracy}%
                        </span>
                      </td>
                      <td className="py-3 px-4 text-center">
                        <span className={`inline-block px-2 py-1 rounded text-xs font-medium ${
                          model.precision === Math.max(...modelComparison.map(m => m.precision))
                            ? 'bg-success/20 text-success'
                            : model.precision === Math.min(...modelComparison.map(m => m.precision))
                            ? 'bg-danger/20 text-danger'
                            : 'bg-card-hover text-text-primary'
                        }`}>
                          {model.precision}%
                        </span>
                      </td>
                      <td className="py-3 px-4 text-center">
                        <span className={`inline-block px-2 py-1 rounded text-xs font-medium ${
                          model.recall === Math.max(...modelComparison.map(m => m.recall))
                            ? 'bg-success/20 text-success'
                            : model.recall === Math.min(...modelComparison.map(m => m.recall))
                            ? 'bg-danger/20 text-danger'
                            : 'bg-card-hover text-text-primary'
                        }`}>
                          {model.recall}%
                        </span>
                      </td>
                      <td className="py-3 px-4 text-center">
                        <span className={`inline-block px-2 py-1 rounded text-xs font-medium ${
                          model.f1 === Math.max(...modelComparison.map(m => m.f1))
                            ? 'bg-success/20 text-success'
                            : model.f1 === Math.min(...modelComparison.map(m => m.f1))
                            ? 'bg-danger/20 text-danger'
                            : 'bg-card-hover text-text-primary'
                        }`}>
                          {model.f1}%
                        </span>
                      </td>
                      <td className="py-3 px-4 text-center">
                        <span className={`inline-block px-2 py-1 rounded text-xs font-medium ${
                          model.mae === Math.min(...modelComparison.map(m => m.mae))
                            ? 'bg-success/20 text-success'
                            : model.mae === Math.max(...modelComparison.map(m => m.mae))
                            ? 'bg-danger/20 text-danger'
                            : 'bg-card-hover text-text-primary'
                        }`}>
                          {model.mae}
                        </span>
                      </td>
                      <td className="py-3 px-4 text-center">
                        <span className={`inline-block px-2 py-1 rounded text-xs font-medium ${
                          model.latency === Math.min(...modelComparison.map(m => m.latency))
                            ? 'bg-success/20 text-success'
                            : model.latency === Math.max(...modelComparison.map(m => m.latency))
                            ? 'bg-danger/20 text-danger'
                            : 'bg-card-hover text-text-primary'
                        }`}>
                          {model.latency}ms
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </AnimatedCard>
        )}

        {/* Model Lifecycle Pipeline */}
        <AnimatedCard delay={0.8} className="p-6 mb-8">
          <h3 className="text-lg font-semibold text-text-primary mb-4">Model Lifecycle Pipeline</h3>
          <div className="relative">
            <div className="flex items-center justify-between">
              {['Development', 'Staging', 'Shadow', 'Production'].map((stage, index) => (
                <div
                  key={stage}
                  className="flex-1 relative"
                  onDragOver={handleDragOver}
                  onDrop={(e) => handleDrop(e, stage)}
                >
                  <div className={`text-center p-4 rounded-lg border-2 border-dashed ${
                    index === 3 ? 'border-success bg-success/10' : 'border-white/20 bg-card-hover'
                  }`}>
                    <p className="text-sm font-medium text-text-primary mb-2">{stage}</p>
                    <p className="text-xs text-text-secondary">
                      {index === 0 && '12 models'}
                      {index === 1 && '3 models'}
                      {index === 2 && '1 model'}
                      {index === 3 && '5 models'}
                    </p>
                  </div>
                  {index < 3 && (
                    <ArrowRight className="absolute -right-4 top-1/2 -translate-y-1/2 w-6 h-6 text-text-secondary z-10" />
                  )}
                </div>
              ))}
            </div>

            {/* Validation Gates */}
            <div className="flex justify-between mt-4 px-8">
              <div className="flex items-center gap-2 text-xs">
                <CheckCircle className="w-4 h-4 text-success" />
                <span className="text-text-secondary">Unit Tests</span>
              </div>
              <div className="flex items-center gap-2 text-xs">
                <CheckCircle className="w-4 h-4 text-success" />
                <span className="text-text-secondary">A/B Testing</span>
              </div>
              <div className="flex items-center gap-2 text-xs">
                <CheckCircle className="w-4 h-4 text-success" />
                <span className="text-text-secondary">Shadow Mode</span>
              </div>
              <div className="flex items-center gap-2 text-xs">
                <Shield className="w-4 h-4 text-primary" />
                <span className="text-text-secondary">Production</span>
              </div>
            </div>
          </div>
        </AnimatedCard>

        {/* Training Jobs Monitor */}
        <AnimatedCard delay={0.9} className="p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-text-primary">Training Jobs Monitor</h3>
            <button className="flex items-center gap-2 px-4 py-2 bg-primary hover:bg-primary-hover text-white rounded-lg transition-colors">
              <Play className="w-4 h-4" />
              New Training Job
            </button>
          </div>

          <div className="space-y-4">
            {trainingJobs.map((job) => (
              <div key={job.id} className="bg-card-hover rounded-lg p-4">
                <div className="flex items-start justify-between mb-3">
                  <div>
                    <h4 className="text-sm font-medium text-text-primary">{job.modelName}</h4>
                    <div className="flex items-center gap-3 mt-1">
                      <span className={`text-xs px-2 py-1 rounded-full ${
                        job.status === 'running' 
                          ? 'bg-success/20 text-success' 
                          : 'bg-warning/20 text-warning'
                      }`}>
                        {job.status}
                      </span>
                      <span className="text-xs text-text-secondary">Started: {job.startTime}</span>
                      <span className="text-xs text-text-secondary">ETA: {job.eta}</span>
                    </div>
                  </div>
                  {job.loss && (
                    <div className="text-right">
                      <p className="text-xs text-text-secondary">Loss</p>
                      <p className="text-sm font-medium text-text-primary">{job.loss.toFixed(4)}</p>
                    </div>
                  )}
                </div>

                {/* Progress Bar */}
                <div className="mb-3">
                  <div className="flex items-center justify-between text-xs text-text-secondary mb-1">
                    <span>Epoch {job.currentEpoch}/{job.totalEpochs}</span>
                    <span>{job.progress.toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-background rounded-full h-2">
                    <div
                      className="h-full bg-gradient-to-r from-primary to-secondary rounded-full transition-all duration-500"
                      style={{ width: `${job.progress}%` }}
                    />
                  </div>
                </div>

                {/* Resource Usage */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-text-secondary flex items-center gap-1">
                      <Cpu className="w-3 h-3" />
                      CPU Usage
                    </span>
                    <span className="text-xs font-medium text-text-primary">{job.cpuUsage.toFixed(0)}%</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-text-secondary flex items-center gap-1">
                      <HardDrive className="w-3 h-3" />
                      Memory
                    </span>
                    <span className="text-xs font-medium text-text-primary">{job.memoryUsage.toFixed(1)} GB</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </AnimatedCard>

        {/* Feature Importance Visualization */}
        {selectedModel && (
          <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="bg-card rounded-xl p-6 max-w-4xl w-full mx-4 max-h-[90vh] overflow-y-auto"
            >
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-xl font-semibold text-text-primary">
                  Model Details: {productionModels.find(m => m.id === selectedModel)?.name}
                </h3>
                <button
                  onClick={() => setSelectedModel(null)}
                  className="text-text-secondary hover:text-text-primary"
                >
                  <XCircle className="w-6 h-6" />
                </button>
              </div>

              <div className="grid grid-cols-2 gap-6">
                {/* Feature Importance */}
                <div>
                  <h4 className="text-sm font-medium text-text-secondary mb-3">Feature Importance</h4>
                  <ResponsiveContainer width="100%" height={200}>
                    <BarChart data={getFeatureImportance(selectedModel)} layout="horizontal">
                      <XAxis type="number" stroke="#666" fontSize={10} />
                      <YAxis dataKey="feature" type="category" stroke="#666" fontSize={10} width={100} />
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#1a1a2e', border: 'none' }}
                        labelStyle={{ color: '#a1a1aa' }}
                      />
                      <Bar dataKey="importance" fill="#8b5cf6" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                {/* Hyperparameters */}
                <div>
                  <h4 className="text-sm font-medium text-text-secondary mb-3">Hyperparameters</h4>
                  <div className="bg-card-hover rounded-lg p-4 space-y-2">
                    {Object.entries(productionModels.find(m => m.id === selectedModel)?.hyperparameters || {}).map(([key, value]) => (
                      <div key={key} className="flex justify-between text-sm">
                        <span className="text-text-secondary">{key}:</span>
                        <span className="text-text-primary font-medium">
                          {typeof value === 'object' ? JSON.stringify(value) : value}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="mt-6 flex gap-3">
                <button className="flex-1 px-4 py-2 bg-primary hover:bg-primary-hover text-white rounded-lg transition-colors">
                  Retrain Model
                </button>
                <button className="flex-1 px-4 py-2 bg-card-hover text-text-primary rounded-lg hover:bg-card transition-colors">
                  View Logs
                </button>
                <button className="flex-1 px-4 py-2 bg-card-hover text-text-primary rounded-lg hover:bg-card transition-colors">
                  Download
                </button>
              </div>
            </motion.div>
          </div>
        )}
      </div>
    </div>
  )
}