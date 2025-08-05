'use client'

import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { LineChart, Line, BarChart, Bar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area, AreaChart } from 'recharts'
import { Users, TrendingUp, AlertCircle, Zap, Target, Brain } from 'lucide-react'
import AnimatedCard from '@/components/ui/AnimatedCard'
import { cn } from '@/lib/utils'

interface PlayerPrediction {
  player: string
  team: string
  prediction: number
  confidence: number
  features: {
    name: string
    value: number
    importance: number
  }[]
  ensemble: {
    model: string
    prediction: number
    weight: number
  }[]
}

const mockPredictions: PlayerPrediction[] = [
  {
    player: 'LeBron James',
    team: 'Lakers',
    prediction: 28.5,
    confidence: 92.3,
    features: [
      { name: 'Recent Form', value: 85, importance: 0.25 },
      { name: 'Opponent Defense', value: 65, importance: 0.20 },
      { name: 'Home/Away', value: 90, importance: 0.15 },
      { name: 'Rest Days', value: 95, importance: 0.10 },
      { name: 'Historical Avg', value: 88, importance: 0.30 },
    ],
    ensemble: [
      { model: 'XGBoost', prediction: 28.3, weight: 0.4 },
      { model: 'LightGBM', prediction: 28.7, weight: 0.35 },
      { model: 'Random Forest', prediction: 28.5, weight: 0.25 },
    ],
  },
  {
    player: 'Stephen Curry',
    team: 'Warriors',
    prediction: 31.2,
    confidence: 88.7,
    features: [
      { name: 'Recent Form', value: 92, importance: 0.28 },
      { name: 'Opponent Defense', value: 70, importance: 0.18 },
      { name: 'Home/Away', value: 85, importance: 0.12 },
      { name: 'Rest Days', value: 88, importance: 0.15 },
      { name: 'Historical Avg', value: 90, importance: 0.27 },
    ],
    ensemble: [
      { model: 'XGBoost', prediction: 31.5, weight: 0.4 },
      { model: 'LightGBM', prediction: 30.9, weight: 0.35 },
      { model: 'Random Forest', prediction: 31.1, weight: 0.25 },
    ],
  },
]

const PredictionCard: React.FC<{ prediction: PlayerPrediction; index: number }> = ({ prediction, index }) => {
  const [showDetails, setShowDetails] = useState(false)

  const confidenceColor = prediction.confidence > 90 ? 'text-success' : 
                          prediction.confidence > 80 ? 'text-warning' : 'text-danger'

  const distributionData = Array.from({ length: 20 }, (_, i) => {
    const x = prediction.prediction - 10 + i
    const variance = 3
    const y = Math.exp(-0.5 * Math.pow((x - prediction.prediction) / variance, 2))
    return { x, y: y * 100 }
  })

  return (
    <AnimatedCard delay={index * 0.1} className="relative overflow-hidden">
      <div className="absolute top-0 right-0 w-32 h-32 bg-gradient-to-br from-primary/20 to-transparent rounded-full blur-3xl" />
      
      <div className="relative z-10">
        <div className="flex items-start justify-between mb-4">
          <div>
            <h3 className="text-xl font-bold text-text-primary">{prediction.player}</h3>
            <p className="text-text-secondary text-sm">{prediction.team}</p>
          </div>
          <div className="text-right">
            <div className="text-3xl font-bold gradient-text">
              {prediction.prediction.toFixed(1)}
            </div>
            <div className={cn('text-sm font-medium', confidenceColor)}>
              {prediction.confidence.toFixed(1)}% confidence
            </div>
          </div>
        </div>

        {/* Confidence Interval Visualization */}
        <div className="h-24 mb-4">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={distributionData}>
              <defs>
                <linearGradient id={`gradient-${index}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <Area 
                type="monotone" 
                dataKey="y" 
                stroke="#8b5cf6" 
                fillOpacity={1} 
                fill={`url(#gradient-${index})`} 
              />
              <XAxis dataKey="x" hide />
              <YAxis hide />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Feature Importance */}
        <div className="space-y-2 mb-4">
          <div className="flex items-center justify-between text-sm">
            <span className="text-text-secondary">Feature Importance</span>
            <button
              onClick={() => setShowDetails(!showDetails)}
              className="text-primary hover:text-primary-hover transition-colors"
            >
              {showDetails ? 'Hide' : 'Show'} Details
            </button>
          </div>
          
          {prediction.features.slice(0, 3).map((feature, i) => (
            <div key={i} className="space-y-1">
              <div className="flex justify-between text-xs">
                <span className="text-text-secondary">{feature.name}</span>
                <span className="text-text-primary">{feature.value}%</span>
              </div>
              <div className="w-full bg-card-hover rounded-full h-2">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${feature.value}%` }}
                  transition={{ duration: 1, delay: index * 0.1 + i * 0.1 }}
                  className="h-full bg-gradient-to-r from-primary to-secondary rounded-full"
                />
              </div>
            </div>
          ))}
        </div>

        <AnimatePresence>
          {showDetails && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="border-t border-white/10 pt-4 mt-4"
            >
              {/* Ensemble Voting */}
              <div className="mb-4">
                <h4 className="text-sm font-medium text-text-secondary mb-2">Model Ensemble Voting</h4>
                <div className="space-y-2">
                  {prediction.ensemble.map((model, i) => (
                    <div key={i} className="flex items-center justify-between text-sm">
                      <span className="text-text-secondary">{model.model}</span>
                      <div className="flex items-center gap-2">
                        <span className="text-text-primary">{model.prediction.toFixed(1)}</span>
                        <span className="text-xs text-text-secondary">({(model.weight * 100).toFixed(0)}%)</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Radar Chart for Features */}
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart data={prediction.features}>
                    <PolarGrid stroke="#ffffff20" />
                    <PolarAngleAxis dataKey="name" tick={{ fill: '#a1a1aa', fontSize: 10 }} />
                    <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fill: '#a1a1aa', fontSize: 10 }} />
                    <Radar name="Value" dataKey="value" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.6} />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </AnimatedCard>
  )
}

export default function PredictionInterface() {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-2xl font-bold text-text-primary mb-2">Live Predictions</h2>
          <p className="text-text-secondary">Real-time player performance predictions with confidence intervals</p>
        </div>
        <div className="flex items-center gap-2">
          <div className="pulse-dot" />
          <span className="text-sm text-text-secondary">Live</span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {mockPredictions.map((prediction, index) => (
          <PredictionCard key={prediction.player} prediction={prediction} index={index} />
        ))}
      </div>
    </div>
  )
}