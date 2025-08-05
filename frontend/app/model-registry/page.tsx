'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  GitBranch, Package, Activity, Shield, Download, Upload,
  ChevronRight, Clock, CheckCircle, AlertCircle, XCircle,
  ArrowRight, ArrowLeft, Play, Pause, RefreshCw, Eye,
  Database, Brain, Zap, TrendingUp, TrendingDown, Settings
} from 'lucide-react'
import { 
  LineChart, Line, AreaChart, Area, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts'
import BackButton from '@/components/BackButton'
import StatsCard from '@/components/StatsCard'
import AnimatedCard from '@/components/ui/AnimatedCard'
import CodeBlock from '@/components/CodeBlock'

export default function ModelRegistry() {
  const [selectedModel, setSelectedModel] = useState('points_predictor')
  const [showUploadModal, setShowUploadModal] = useState(false)
  const [selectedVersion, setSelectedVersion] = useState('v2.3')
  const [deploymentStage, setDeploymentStage] = useState<string | null>(null)

  // Production models
  const productionModels = [
    {
      id: 'points_predictor',
      name: 'points_predictor_v2.3',
      algorithm: 'Random Forest',
      version: 'v2.3',
      status: 'production',
      accuracy: 94.7,
      deployedAt: '2025-01-15',
      lastUpdated: '2 days ago',
      predictions: 145234,
      latency: 82
    },
    {
      id: 'rebounds_predictor',
      name: 'rebounds_predictor_v2.1',
      algorithm: 'XGBoost',
      version: 'v2.1',
      status: 'production',
      accuracy: 92.3,
      deployedAt: '2025-01-10',
      lastUpdated: '5 days ago',
      predictions: 98456,
      latency: 95
    },
    {
      id: 'assists_predictor',
      name: 'assists_predictor_v2.4',
      algorithm: 'Ensemble',
      version: 'v2.4',
      status: 'production',
      accuracy: 93.1,
      deployedAt: '2025-01-12',
      lastUpdated: '3 days ago',
      predictions: 112345,
      latency: 104
    }
  ]

  // Model versions timeline
  const modelVersions = {
    points_predictor: [
      { version: 'v2.3', date: '2025-01-15', accuracy: 94.7, status: 'production' },
      { version: 'v2.2', date: '2025-01-08', accuracy: 94.2, status: 'archived' },
      { version: 'v2.1', date: '2024-12-28', accuracy: 93.8, status: 'archived' },
      { version: 'v2.0', date: '2024-12-15', accuracy: 93.1, status: 'archived' },
      { version: 'v1.9', date: '2024-12-01', accuracy: 92.5, status: 'archived' }
    ],
    rebounds_predictor: [
      { version: 'v2.1', date: '2025-01-10', accuracy: 92.3, status: 'production' },
      { version: 'v2.0', date: '2024-12-25', accuracy: 91.8, status: 'archived' },
      { version: 'v1.9', date: '2024-12-10', accuracy: 91.2, status: 'archived' }
    ],
    assists_predictor: [
      { version: 'v2.4', date: '2025-01-12', accuracy: 93.1, status: 'production' },
      { version: 'v2.3', date: '2024-12-30', accuracy: 92.7, status: 'archived' },
      { version: 'v2.2', date: '2024-12-18', accuracy: 92.3, status: 'archived' }
    ]
  }

  // Staging models
  const stagingModels = [
    {
      id: 'points_predictor_v3.0',
      name: 'points_predictor_v3.0',
      algorithm: 'Transformer',
      status: 'staging',
      accuracy: 95.2,
      validationStatus: 'passed',
      testResults: {
        unitTests: 'passed',
        integrationTests: 'passed',
        performanceTests: 'running'
      }
    },
    {
      id: 'rebounds_predictor_v2.2',
      name: 'rebounds_predictor_v2.2',
      algorithm: 'LightGBM',
      status: 'staging',
      accuracy: 92.8,
      validationStatus: 'in_progress',
      testResults: {
        unitTests: 'passed',
        integrationTests: 'running',
        performanceTests: 'pending'
      }
    }
  ]

  // Performance comparison data
  const performanceData = modelVersions[selectedModel]?.map(v => ({
    version: v.version,
    accuracy: v.accuracy,
    latency: 80 + Math.random() * 40
  })) || []

  // Model metadata
  const modelMetadata = {
    trainingData: {
      samples: 1234567,
      features: 52,
      dateRange: '2023-01 to 2024-12',
      splitRatio: '80/10/10'
    },
    hyperparameters: {
      n_estimators: 200,
      max_depth: 15,
      min_samples_split: 5,
      learning_rate: 0.1
    },
    featureImportance: [
      { feature: 'player_avg_30d', importance: 0.35 },
      { feature: 'opponent_rank', importance: 0.25 },
      { feature: 'home_away', importance: 0.18 },
      { feature: 'rest_days', importance: 0.12 },
      { feature: 'injury_status', importance: 0.10 }
    ]
  }

  // A/B Test configuration
  const abTestConfig = {
    name: 'Model Performance Test',
    control: 'v2.3',
    variant: 'v3.0',
    trafficSplit: 20,
    duration: '7 days',
    successMetrics: ['accuracy', 'latency', 'error_rate']
  }

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'production':
        return <span className="px-2 py-1 bg-success/20 text-success rounded-full text-xs">Production</span>
      case 'staging':
        return <span className="px-2 py-1 bg-warning/20 text-warning rounded-full text-xs">Staging</span>
      case 'archived':
        return <span className="px-2 py-1 bg-gray-500/20 text-gray-400 rounded-full text-xs">Archived</span>
      default:
        return null
    }
  }

  const getTestStatusIcon = (status: string) => {
    switch (status) {
      case 'passed':
        return <CheckCircle className="w-4 h-4 text-success" />
      case 'running':
        return <RefreshCw className="w-4 h-4 text-warning animate-spin" />
      case 'failed':
        return <XCircle className="w-4 h-4 text-danger" />
      default:
        return <Clock className="w-4 h-4 text-text-secondary" />
    }
  }

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="max-w-7xl mx-auto">
        <BackButton href="/" />
        
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-4xl font-bold text-text-primary mb-2">Model Registry</h1>
          <p className="text-text-secondary">Version control and deployment management for ML models</p>
        </motion.div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <StatsCard
            title="Production Models"
            value="3"
            subtitle="Active in production"
            icon={Shield}
            iconColor="success"
            delay={0.1}
          />
          <StatsCard
            title="Staging Models"
            value="2"
            subtitle="Ready for deployment"
            icon={Package}
            iconColor="warning"
            delay={0.2}
          />
          <StatsCard
            title="Total Versions"
            value="47"
            subtitle="Across all models"
            icon={GitBranch}
            iconColor="primary"
            delay={0.3}
          />
          <StatsCard
            title="Avg Accuracy"
            value="93.4%"
            subtitle="Production models"
            icon={TrendingUp}
            iconColor="secondary"
            trend={{ value: 1.2, isPositive: true }}
            delay={0.4}
          />
        </div>

        {/* Current Production Models */}
        <AnimatedCard delay={0.5} className="p-6 mb-8">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-text-primary">Current Production Models</h3>
            <button
              onClick={() => setShowUploadModal(true)}
              className="flex items-center gap-2 px-4 py-2 bg-primary hover:bg-primary-hover text-white rounded-lg transition-colors"
            >
              <Upload className="w-4 h-4" />
              Upload New Model
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {productionModels.map((model) => (
              <motion.div
                key={model.id}
                whileHover={{ scale: 1.02 }}
                onClick={() => setSelectedModel(model.id)}
                className={`p-4 rounded-lg border cursor-pointer transition-all ${
                  selectedModel === model.id
                    ? 'bg-primary/10 border-primary'
                    : 'bg-card-hover border-white/10 hover:border-white/20'
                }`}
              >
                <div className="flex items-center justify-between mb-3">
                  <h4 className="text-sm font-semibold text-text-primary">{model.name}</h4>
                  {getStatusBadge(model.status)}
                </div>
                
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-text-secondary">Algorithm:</span>
                    <span className="text-text-primary">{model.algorithm}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-text-secondary">Accuracy:</span>
                    <span className="text-success">{model.accuracy}%</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-text-secondary">Latency:</span>
                    <span className="text-text-primary">{model.latency}ms</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-text-secondary">Predictions:</span>
                    <span className="text-text-primary">{model.predictions.toLocaleString()}</span>
                  </div>
                </div>

                <div className="mt-4 flex gap-2">
                  <button className="flex-1 px-3 py-1.5 bg-card hover:bg-background rounded text-xs text-text-secondary transition-colors">
                    <Eye className="w-3 h-3 inline mr-1" />
                    View
                  </button>
                  <button className="flex-1 px-3 py-1.5 bg-card hover:bg-background rounded text-xs text-text-secondary transition-colors">
                    <ArrowLeft className="w-3 h-3 inline mr-1" />
                    Rollback
                  </button>
                </div>
              </motion.div>
            ))}
          </div>
        </AnimatedCard>

        {/* Model Versions Timeline */}
        <AnimatedCard delay={0.6} className="p-6 mb-8">
          <h3 className="text-lg font-semibold text-text-primary mb-6">Model Version Timeline</h3>
          
          <div className="relative">
            {/* Git-like branch visualization */}
            <div className="absolute left-8 top-0 bottom-0 w-0.5 bg-card-hover" />
            
            <div className="space-y-6">
              {modelVersions[selectedModel]?.map((version, index) => (
                <motion.div
                  key={version.version}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="relative flex items-center gap-4"
                >
                  <div className={`absolute left-6 w-4 h-4 rounded-full border-2 ${
                    version.status === 'production' 
                      ? 'bg-success border-success' 
                      : 'bg-card border-white/20'
                  }`} />
                  
                  <div className="ml-14 flex-1">
                    <div className={`p-4 rounded-lg ${
                      version.status === 'production' 
                        ? 'bg-success/10 border border-success/30' 
                        : 'bg-card-hover'
                    }`}>
                      <div className="flex items-center justify-between">
                        <div>
                          <div className="flex items-center gap-3">
                            <span className="font-mono text-sm text-text-primary">{version.version}</span>
                            {getStatusBadge(version.status)}
                            <span className="text-xs text-text-secondary">{version.date}</span>
                          </div>
                          <div className="mt-2 flex items-center gap-4 text-sm">
                            <span className="text-text-secondary">Accuracy: <span className="text-text-primary">{version.accuracy}%</span></span>
                            {version.status === 'production' && (
                              <span className="text-success flex items-center gap-1">
                                <CheckCircle className="w-3 h-3" />
                                Current
                              </span>
                            )}
                          </div>
                        </div>
                        
                        {version.status !== 'production' && (
                          <button className="px-3 py-1.5 bg-primary/20 text-primary rounded text-xs hover:bg-primary/30 transition-colors">
                            Deploy
                          </button>
                        )}
                      </div>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </AnimatedCard>

        {/* Performance Comparison */}
        <AnimatedCard delay={0.7} className="p-6 mb-8">
          <h3 className="text-lg font-semibold text-text-primary mb-4">Performance Comparison</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={performanceData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis dataKey="version" stroke="#666" fontSize={10} />
              <YAxis yAxisId="left" stroke="#666" fontSize={10} />
              <YAxis yAxisId="right" orientation="right" stroke="#666" fontSize={10} />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1a1a2e', border: 'none' }}
                labelStyle={{ color: '#a1a1aa' }}
              />
              <Legend />
              <Line 
                yAxisId="left"
                type="monotone" 
                dataKey="accuracy" 
                stroke="#8b5cf6" 
                strokeWidth={2}
                name="Accuracy (%)"
              />
              <Line 
                yAxisId="right"
                type="monotone" 
                dataKey="latency" 
                stroke="#00d4ff" 
                strokeWidth={2}
                name="Latency (ms)"
              />
            </LineChart>
          </ResponsiveContainer>
        </AnimatedCard>

        {/* Staging Area */}
        <AnimatedCard delay={0.8} className="p-6 mb-8">
          <h3 className="text-lg font-semibold text-text-primary mb-4">Staging Area</h3>
          
          <div className="space-y-4">
            {stagingModels.map((model) => (
              <div key={model.id} className="bg-card-hover rounded-lg p-4">
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <h4 className="text-sm font-semibold text-text-primary">{model.name}</h4>
                    <div className="flex items-center gap-3 mt-1">
                      <span className="text-xs text-text-secondary">Algorithm: {model.algorithm}</span>
                      <span className="text-xs text-text-secondary">Accuracy: {model.accuracy}%</span>
                    </div>
                  </div>
                  {getStatusBadge(model.status)}
                </div>

                {/* Validation Results */}
                <div className="space-y-2 mb-4">
                  <p className="text-xs text-text-secondary mb-2">Validation Status:</p>
                  <div className="grid grid-cols-3 gap-3">
                    <div className="flex items-center gap-2 text-xs">
                      {getTestStatusIcon(model.testResults.unitTests)}
                      <span className="text-text-secondary">Unit Tests</span>
                    </div>
                    <div className="flex items-center gap-2 text-xs">
                      {getTestStatusIcon(model.testResults.integrationTests)}
                      <span className="text-text-secondary">Integration Tests</span>
                    </div>
                    <div className="flex items-center gap-2 text-xs">
                      {getTestStatusIcon(model.testResults.performanceTests)}
                      <span className="text-text-secondary">Performance Tests</span>
                    </div>
                  </div>
                </div>

                {/* A/B Test Configuration */}
                <div className="p-3 bg-background rounded-lg mb-4">
                  <p className="text-xs text-text-secondary mb-2">A/B Test Configuration:</p>
                  <div className="grid grid-cols-2 gap-3 text-xs">
                    <div>
                      <span className="text-text-secondary">Traffic Split: </span>
                      <span className="text-text-primary">{abTestConfig.trafficSplit}%</span>
                    </div>
                    <div>
                      <span className="text-text-secondary">Duration: </span>
                      <span className="text-text-primary">{abTestConfig.duration}</span>
                    </div>
                  </div>
                </div>

                <div className="flex gap-2">
                  <button className="flex-1 px-3 py-1.5 bg-primary hover:bg-primary-hover text-white rounded text-xs transition-colors">
                    Start A/B Test
                  </button>
                  <button className="flex-1 px-3 py-1.5 bg-success hover:bg-success/80 text-white rounded text-xs transition-colors">
                    Promote to Production
                  </button>
                </div>
              </div>
            ))}
          </div>
        </AnimatedCard>

        {/* Model Metadata */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Training Data Info */}
          <AnimatedCard delay={0.9} className="p-6">
            <h3 className="text-lg font-semibold text-text-primary mb-4">Training Data</h3>
            <div className="space-y-3">
              <div className="flex justify-between text-sm">
                <span className="text-text-secondary">Total Samples:</span>
                <span className="text-text-primary">{modelMetadata.trainingData.samples.toLocaleString()}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-text-secondary">Features:</span>
                <span className="text-text-primary">{modelMetadata.trainingData.features}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-text-secondary">Date Range:</span>
                <span className="text-text-primary">{modelMetadata.trainingData.dateRange}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-text-secondary">Train/Val/Test Split:</span>
                <span className="text-text-primary">{modelMetadata.trainingData.splitRatio}</span>
              </div>
            </div>

            <div className="mt-6">
              <h4 className="text-sm font-medium text-text-secondary mb-3">Hyperparameters</h4>
              <div className="bg-card-hover rounded-lg p-3">
                <CodeBlock 
                  code={JSON.stringify(modelMetadata.hyperparameters, null, 2)}
                  language="json"
                  showLineNumbers={false}
                />
              </div>
            </div>
          </AnimatedCard>

          {/* Feature Importance */}
          <AnimatedCard delay={1.0} className="p-6">
            <h3 className="text-lg font-semibold text-text-primary mb-4">Feature Importance</h3>
            <div className="space-y-3">
              {modelMetadata.featureImportance.map((item, index) => (
                <div key={item.feature}>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm text-text-secondary">{item.feature}</span>
                    <span className="text-sm text-text-primary">{(item.importance * 100).toFixed(0)}%</span>
                  </div>
                  <div className="w-full bg-card-hover rounded-full h-2">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${item.importance * 100}%` }}
                      transition={{ delay: 1.0 + index * 0.1, duration: 0.5 }}
                      className="h-full bg-gradient-to-r from-primary to-secondary rounded-full"
                    />
                  </div>
                </div>
              ))}
            </div>

            <div className="mt-6 flex gap-2">
              <button className="flex-1 px-3 py-2 bg-card-hover text-text-primary rounded-lg text-sm hover:bg-card transition-colors">
                <Eye className="w-4 h-4 inline mr-1" />
                View SHAP
              </button>
              <button className="flex-1 px-3 py-2 bg-card-hover text-text-primary rounded-lg text-sm hover:bg-card transition-colors">
                <Download className="w-4 h-4 inline mr-1" />
                Export
              </button>
            </div>
          </AnimatedCard>
        </div>

        {/* Upload Model Modal */}
        {showUploadModal && (
          <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="bg-card rounded-xl p-6 max-w-md w-full mx-4"
            >
              <h3 className="text-xl font-semibold text-text-primary mb-4">Upload New Model</h3>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm text-text-secondary mb-2">Model Name</label>
                  <input
                    type="text"
                    className="w-full px-4 py-2 bg-card-hover text-text-primary rounded-lg focus:outline-none focus:ring-2 focus:ring-primary"
                    placeholder="e.g., points_predictor_v3.0"
                  />
                </div>

                <div>
                  <label className="block text-sm text-text-secondary mb-2">Algorithm</label>
                  <select className="w-full px-4 py-2 bg-card-hover text-text-primary rounded-lg focus:outline-none focus:ring-2 focus:ring-primary">
                    <option>Random Forest</option>
                    <option>XGBoost</option>
                    <option>LightGBM</option>
                    <option>Neural Network</option>
                    <option>Ensemble</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm text-text-secondary mb-2">Model File</label>
                  <div className="border-2 border-dashed border-white/20 rounded-lg p-8 text-center hover:border-primary/50 transition-colors cursor-pointer">
                    <Upload className="w-8 h-8 text-text-secondary mx-auto mb-2" />
                    <p className="text-sm text-text-secondary">Click to upload or drag and drop</p>
                    <p className="text-xs text-text-secondary mt-1">.pkl, .h5, .pt up to 500MB</p>
                  </div>
                </div>

                <div>
                  <label className="block text-sm text-text-secondary mb-2">Deployment Stage</label>
                  <select className="w-full px-4 py-2 bg-card-hover text-text-primary rounded-lg focus:outline-none focus:ring-2 focus:ring-primary">
                    <option>Development</option>
                    <option>Staging</option>
                    <option>Shadow</option>
                  </select>
                </div>
              </div>

              <div className="flex gap-3 mt-6">
                <button
                  onClick={() => setShowUploadModal(false)}
                  className="flex-1 px-4 py-2 bg-card-hover text-text-primary rounded-lg hover:bg-card transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={() => setShowUploadModal(false)}
                  className="flex-1 px-4 py-2 bg-primary hover:bg-primary-hover text-white rounded-lg transition-colors"
                >
                  Upload Model
                </button>
              </div>
            </motion.div>
          </div>
        )}
      </div>
    </div>
  )
}