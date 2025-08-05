'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  Database, Zap, Clock, TrendingUp, RefreshCw, Search, Filter,
  ChevronRight, GitBranch, Settings, Play, CheckCircle, AlertCircle,
  Code, Activity, Layers, Package, ArrowRight, Eye, Download
} from 'lucide-react'
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import BackButton from '@/components/BackButton'
import StatsCard from '@/components/StatsCard'
import DataTable from '@/components/DataTable'
import CodeBlock from '@/components/CodeBlock'
import AnimatedCard from '@/components/ui/AnimatedCard'

export default function FeatureStore() {
  const [selectedCategory, setSelectedCategory] = useState('player')
  const [selectedFeature, setSelectedFeature] = useState<any>(null)
  const [pipelineStatus, setPipelineStatus] = useState('running')
  const [showCodeEditor, setShowCodeEditor] = useState(false)
  const [featureCode, setFeatureCode] = useState('')

  // Simulate real-time pipeline updates
  useEffect(() => {
    const interval = setInterval(() => {
      setPipelineStatus(prev => prev === 'running' ? 'validating' : prev === 'validating' ? 'storing' : 'running')
    }, 3000)
    return () => clearInterval(interval)
  }, [])

  // Feature categories
  const featureCategories = [
    {
      id: 'player',
      name: 'Player Features',
      count: 23,
      icon: Activity,
      color: 'primary',
      description: 'Individual player statistics and metrics'
    },
    {
      id: 'team',
      name: 'Team Features',
      count: 15,
      icon: Layers,
      color: 'secondary',
      description: 'Team-level aggregated features'
    },
    {
      id: 'game',
      name: 'Game Context',
      count: 8,
      icon: Package,
      color: 'success',
      description: 'Game situation and context features'
    },
    {
      id: 'historical',
      name: 'Historical Features',
      count: 6,
      icon: Clock,
      color: 'warning',
      description: 'Time-series and historical data'
    }
  ]

  // Feature details data
  const featuresData = [
    {
      name: 'player_avg_30d',
      type: 'float',
      category: 'Player',
      updateFreq: 'Daily',
      importance: 0.92,
      status: 'active',
      lastUpdate: '5 min ago',
      nullRate: 0.02,
      cardinality: 'continuous',
      description: '30-day rolling average of player points'
    },
    {
      name: 'opponent_defensive_rating',
      type: 'float',
      category: 'Team',
      updateFreq: 'Hourly',
      importance: 0.78,
      status: 'active',
      lastUpdate: '12 min ago',
      nullRate: 0.00,
      cardinality: 'continuous',
      description: 'Opponent team defensive efficiency rating'
    },
    {
      name: 'rest_days',
      type: 'integer',
      category: 'Game',
      updateFreq: 'Real-time',
      importance: 0.65,
      status: 'active',
      lastUpdate: '2 min ago',
      nullRate: 0.00,
      cardinality: 'discrete',
      description: 'Days of rest since last game'
    },
    {
      name: 'home_away_flag',
      type: 'boolean',
      category: 'Game',
      updateFreq: 'Static',
      importance: 0.54,
      status: 'active',
      lastUpdate: '1 hour ago',
      nullRate: 0.00,
      cardinality: 'binary',
      description: 'Home (1) or away (0) game indicator'
    },
    {
      name: 'injury_status',
      type: 'categorical',
      category: 'Player',
      updateFreq: 'Real-time',
      importance: 0.88,
      status: 'warning',
      lastUpdate: '8 min ago',
      nullRate: 0.15,
      cardinality: 'low',
      description: 'Current injury status of player'
    },
    {
      name: 'win_streak',
      type: 'integer',
      category: 'Team',
      updateFreq: 'Daily',
      importance: 0.42,
      status: 'active',
      lastUpdate: '3 hours ago',
      nullRate: 0.00,
      cardinality: 'discrete',
      description: 'Current team winning streak'
    },
    {
      name: 'player_efficiency_rating',
      type: 'float',
      category: 'Player',
      updateFreq: 'Daily',
      importance: 0.83,
      status: 'active',
      lastUpdate: '25 min ago',
      nullRate: 0.03,
      cardinality: 'continuous',
      description: 'PER metric for player performance'
    },
    {
      name: 'usage_rate',
      type: 'float',
      category: 'Player',
      updateFreq: 'Hourly',
      importance: 0.71,
      status: 'active',
      lastUpdate: '45 min ago',
      nullRate: 0.01,
      cardinality: 'continuous',
      description: 'Percentage of team plays used by player'
    }
  ]

  // Pipeline stages
  const pipelineStages = [
    { id: 'raw', name: 'Raw Data', status: pipelineStatus === 'running' ? 'active' : 'completed' },
    { id: 'transform', name: 'Transform', status: pipelineStatus === 'validating' ? 'active' : pipelineStatus === 'running' ? 'pending' : 'completed' },
    { id: 'validate', name: 'Validate', status: pipelineStatus === 'storing' ? 'active' : pipelineStatus === 'validating' ? 'pending' : 'completed' },
    { id: 'store', name: 'Store', status: pipelineStatus === 'storing' ? 'pending' : 'completed' }
  ]

  // Feature computation time series
  const computeTimeData = Array.from({ length: 24 }, (_, i) => ({
    hour: `${i}:00`,
    time: 20 + Math.random() * 10
  }))

  const tableColumns = [
    { 
      key: 'name', 
      label: 'Feature Name', 
      sortable: true,
      render: (value: string) => (
        <div className="flex items-center gap-2">
          <Code className="w-4 h-4 text-primary" />
          <span className="font-mono text-sm">{value}</span>
        </div>
      )
    },
    { key: 'type', label: 'Type', sortable: true },
    { key: 'category', label: 'Category', sortable: true },
    { key: 'updateFreq', label: 'Update Frequency', sortable: true },
    { 
      key: 'importance', 
      label: 'Importance', 
      sortable: true,
      render: (value: number) => (
        <div className="flex items-center gap-2">
          <div className="w-20 bg-card-hover rounded-full h-2">
            <div 
              className="h-full bg-gradient-to-r from-primary to-secondary rounded-full"
              style={{ width: `${value * 100}%` }}
            />
          </div>
          <span className="text-xs">{(value * 100).toFixed(0)}%</span>
        </div>
      )
    },
    { 
      key: 'status', 
      label: 'Status',
      render: (value: string) => (
        <span className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs ${
          value === 'active' ? 'bg-success/20 text-success' : 'bg-warning/20 text-warning'
        }`}>
          {value === 'active' ? <CheckCircle className="w-3 h-3" /> : <AlertCircle className="w-3 h-3" />}
          {value}
        </span>
      )
    },
    { key: 'lastUpdate', label: 'Last Update' }
  ]

  const exampleFeatureCode = `# Custom Feature: Player Momentum Score
def calculate_player_momentum(player_id: str, games: int = 5) -> float:
    """
    Calculate player momentum based on recent performance trend
    """
    recent_games = get_recent_games(player_id, games)
    
    # Calculate weighted average with recency bias
    weights = np.exp(np.linspace(-1, 0, games))
    scores = [g.points + 0.5 * g.assists + 0.3 * g.rebounds 
              for g in recent_games]
    
    momentum = np.average(scores, weights=weights)
    baseline = get_season_average(player_id)
    
    return momentum / baseline  # Normalized momentum score`

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
          <h1 className="text-4xl font-bold text-text-primary mb-2">Feature Store</h1>
          <p className="text-text-secondary">Centralized feature engineering and management platform</p>
        </motion.div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <StatsCard
            title="Total Features"
            value="52"
            subtitle="Across all categories"
            icon={Database}
            iconColor="primary"
            delay={0.1}
          />
          <StatsCard
            title="Active Pipelines"
            value="7"
            subtitle="Processing in real-time"
            icon={Activity}
            iconColor="success"
            trend={{ value: 2, isPositive: true }}
            delay={0.2}
          />
          <StatsCard
            title="Compute Time"
            value="23ms"
            subtitle="Average latency"
            icon={Zap}
            iconColor="secondary"
            delay={0.3}
          />
          <StatsCard
            title="Last Update"
            value="5 min"
            subtitle="ago"
            icon={Clock}
            iconColor="warning"
            delay={0.4}
          />
        </div>

        {/* Feature Categories */}
        <AnimatedCard delay={0.5} className="p-6 mb-8">
          <h3 className="text-lg font-semibold text-text-primary mb-4">Feature Categories</h3>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {featureCategories.map((category) => (
              <motion.div
                key={category.id}
                whileHover={{ scale: 1.02 }}
                onClick={() => setSelectedCategory(category.id)}
                className={`p-4 rounded-lg border cursor-pointer transition-all ${
                  selectedCategory === category.id 
                    ? 'bg-primary/20 border-primary' 
                    : 'bg-card-hover border-white/10 hover:border-white/20'
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <category.icon className={`w-6 h-6 ${
                    selectedCategory === category.id ? 'text-primary' : 'text-text-secondary'
                  }`} />
                  <span className="text-2xl font-bold text-text-primary">{category.count}</span>
                </div>
                <h4 className="text-sm font-medium text-text-primary mb-1">{category.name}</h4>
                <p className="text-xs text-text-secondary">{category.description}</p>
              </motion.div>
            ))}
          </div>
        </AnimatedCard>

        {/* Real-time Feature Pipeline */}
        <AnimatedCard delay={0.6} className="p-6 mb-8">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-text-primary">Real-time Feature Pipeline</h3>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-success rounded-full animate-pulse" />
              <span className="text-sm text-text-secondary">Processing</span>
            </div>
          </div>
          
          <div className="relative">
            <div className="flex items-center justify-between mb-8">
              {pipelineStages.map((stage, index) => (
                <div key={stage.id} className="flex-1 relative">
                  <div className={`flex flex-col items-center ${
                    stage.status === 'active' ? 'scale-110' : ''
                  } transition-transform`}>
                    <div className={`w-16 h-16 rounded-full flex items-center justify-center mb-2 ${
                      stage.status === 'active' ? 'bg-primary/20 ring-2 ring-primary animate-pulse' :
                      stage.status === 'completed' ? 'bg-success/20' : 'bg-card-hover'
                    }`}>
                      {stage.status === 'completed' ? (
                        <CheckCircle className="w-8 h-8 text-success" />
                      ) : stage.status === 'active' ? (
                        <RefreshCw className="w-8 h-8 text-primary animate-spin" />
                      ) : (
                        <Clock className="w-8 h-8 text-text-secondary" />
                      )}
                    </div>
                    <span className={`text-sm font-medium ${
                      stage.status === 'active' ? 'text-primary' : 
                      stage.status === 'completed' ? 'text-success' : 'text-text-secondary'
                    }`}>
                      {stage.name}
                    </span>
                  </div>
                  {index < pipelineStages.length - 1 && (
                    <div className="absolute top-8 left-[60%] w-full h-0.5">
                      <div className={`h-full transition-all duration-1000 ${
                        stage.status === 'completed' ? 'bg-success w-full' : 'bg-card-hover w-full'
                      }`} />
                    </div>
                  )}
                </div>
              ))}
            </div>
            
            <div className="grid grid-cols-4 gap-4 text-center">
              <div>
                <p className="text-xs text-text-secondary">Records/sec</p>
                <p className="text-lg font-semibold text-text-primary">1,234</p>
              </div>
              <div>
                <p className="text-xs text-text-secondary">Transform Time</p>
                <p className="text-lg font-semibold text-text-primary">12ms</p>
              </div>
              <div>
                <p className="text-xs text-text-secondary">Validation Rate</p>
                <p className="text-lg font-semibold text-success">99.8%</p>
              </div>
              <div>
                <p className="text-xs text-text-secondary">Storage Latency</p>
                <p className="text-lg font-semibold text-text-primary">8ms</p>
              </div>
            </div>
          </div>
        </AnimatedCard>

        {/* Feature Details Table */}
        <AnimatedCard delay={0.7} className="p-6 mb-8">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-text-primary">Feature Details</h3>
            <div className="flex items-center gap-2">
              <button className="p-2 bg-card-hover rounded-lg hover:bg-card transition-colors">
                <Filter className="w-4 h-4 text-text-secondary" />
              </button>
              <button className="p-2 bg-card-hover rounded-lg hover:bg-card transition-colors">
                <Download className="w-4 h-4 text-text-secondary" />
              </button>
            </div>
          </div>
          
          <DataTable
            columns={tableColumns}
            data={featuresData}
            searchPlaceholder="Search features..."
          />
        </AnimatedCard>

        {/* Feature Engineering Workspace */}
        <AnimatedCard delay={0.8} className="p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-text-primary">Feature Engineering Workspace</h3>
            <button
              onClick={() => setShowCodeEditor(!showCodeEditor)}
              className="flex items-center gap-2 px-4 py-2 bg-primary hover:bg-primary-hover text-white rounded-lg transition-colors"
            >
              <Code className="w-4 h-4" />
              {showCodeEditor ? 'Hide Editor' : 'Create New Feature'}
            </button>
          </div>

          {showCodeEditor ? (
            <div className="space-y-4">
              <div className="bg-card-hover rounded-lg p-4">
                <textarea
                  value={featureCode || exampleFeatureCode}
                  onChange={(e) => setFeatureCode(e.target.value)}
                  className="w-full h-64 bg-background text-text-primary font-mono text-sm p-4 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary"
                  placeholder="# Write your feature engineering code here..."
                />
              </div>
              
              <div className="flex gap-3">
                <button className="flex items-center gap-2 px-4 py-2 bg-secondary hover:bg-secondary/80 text-white rounded-lg transition-colors">
                  <Play className="w-4 h-4" />
                  Test Feature
                </button>
                <button className="flex items-center gap-2 px-4 py-2 bg-success hover:bg-success/80 text-white rounded-lg transition-colors">
                  <CheckCircle className="w-4 h-4" />
                  Deploy to Production
                </button>
                <button 
                  onClick={() => setShowCodeEditor(false)}
                  className="px-4 py-2 bg-card-hover text-text-primary rounded-lg hover:bg-card transition-colors"
                >
                  Cancel
                </button>
              </div>
            </div>
          ) : (
            <div>
              <p className="text-sm text-text-secondary mb-4">
                Example feature implementation:
              </p>
              <CodeBlock code={exampleFeatureCode} language="python" />
            </div>
          )}
        </AnimatedCard>

        {/* Compute Time Chart */}
        <AnimatedCard delay={0.9} className="p-6 mt-8">
          <h3 className="text-lg font-semibold text-text-primary mb-4">Feature Compute Time (24h)</h3>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={computeTimeData}>
              <defs>
                <linearGradient id="computeGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis dataKey="hour" stroke="#666" fontSize={10} />
              <YAxis stroke="#666" fontSize={10} />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1a1a2e', border: 'none' }}
                labelStyle={{ color: '#a1a1aa' }}
              />
              <Area 
                type="monotone" 
                dataKey="time" 
                stroke="#8b5cf6" 
                fillOpacity={1} 
                fill="url(#computeGradient)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </AnimatedCard>
      </div>
    </div>
  )
}