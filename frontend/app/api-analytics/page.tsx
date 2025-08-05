'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  LineChart, Line, AreaChart, Area, BarChart, Bar, PieChart, Pie,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell,
  RadialBarChart, RadialBar, ScatterChart, Scatter
} from 'recharts'
import AnimatedCard from '@/components/ui/AnimatedCard'
import { 
  Activity, Clock, AlertTriangle, TrendingUp, TrendingDown,
  Globe, Users, DollarSign, Zap, Database, Shield,
  CheckCircle, XCircle, AlertCircle, BarChart3, Settings,
  Filter, Download, RefreshCw, ChevronUp, ChevronDown,
  Wifi, WifiOff, Server, Cpu, HardDrive, Bell
} from 'lucide-react'

export default function ApiAnalyticsPage() {
  const [timeRange, setTimeRange] = useState('1h')
  const [selectedEndpoint, setSelectedEndpoint] = useState<string | null>(null)
  const [showAlertConfig, setShowAlertConfig] = useState(false)
  const [expandedClient, setExpandedClient] = useState<string | null>(null)

  // Real-time metrics state
  const [currentQPS, setCurrentQPS] = useState(2345)
  const [totalRequests, setTotalRequests] = useState(2300000)
  const [avgLatency, setAvgLatency] = useState(87)
  const [errorRate, setErrorRate] = useState(0.02)
  const [uptime, setUptime] = useState(99.98)

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentQPS(prev => prev + (Math.random() - 0.5) * 100)
      setTotalRequests(prev => prev + Math.floor(Math.random() * 100))
      setAvgLatency(prev => Math.max(50, Math.min(150, prev + (Math.random() - 0.5) * 5)))
      setErrorRate(prev => Math.max(0, Math.min(0.1, prev + (Math.random() - 0.5) * 0.005)))
    }, 2000)

    return () => clearInterval(interval)
  }, [])

  // Time series data for requests per second
  const getRequestsData = () => {
    const points = timeRange === '1h' ? 60 : timeRange === '24h' ? 24 : 7
    return Array.from({ length: points }, (_, i) => ({
      time: timeRange === '1h' ? `${i}m` : timeRange === '24h' ? `${i}h` : `Day ${i + 1}`,
      requests: 2000 + Math.random() * 1000,
      errors: Math.random() * 50,
      cache_hits: 1800 + Math.random() * 800
    }))
  }

  // Latency distribution histogram data
  const latencyDistribution = [
    { range: '0-50ms', count: 4234, percentage: 18 },
    { range: '50-100ms', count: 12456, percentage: 52 },
    { range: '100-150ms', count: 5678, percentage: 24 },
    { range: '150-200ms', count: 1234, percentage: 5 },
    { range: '200ms+', count: 234, percentage: 1 }
  ]

  // Endpoint performance data
  const endpointMetrics = [
    {
      endpoint: '/api/predict',
      method: 'POST',
      requests: 892345,
      avgLatency: 95,
      p95Latency: 142,
      errorRate: 0.02,
      cost: 234.56,
      trend: [80, 85, 90, 95, 92, 94, 95],
      status: 'healthy'
    },
    {
      endpoint: '/api/players',
      method: 'GET',
      requests: 654321,
      avgLatency: 45,
      p95Latency: 78,
      errorRate: 0.01,
      cost: 123.45,
      trend: [40, 42, 44, 45, 46, 44, 45],
      status: 'healthy'
    },
    {
      endpoint: '/api/batch',
      method: 'POST',
      requests: 234567,
      avgLatency: 234,
      p95Latency: 456,
      errorRate: 0.05,
      cost: 567.89,
      trend: [220, 225, 230, 235, 240, 235, 234],
      status: 'warning'
    },
    {
      endpoint: '/api/experiments',
      method: 'GET',
      requests: 123456,
      avgLatency: 67,
      p95Latency: 98,
      errorRate: 0.03,
      cost: 89.12,
      trend: [65, 66, 67, 68, 67, 66, 67],
      status: 'healthy'
    },
    {
      endpoint: '/api/health',
      method: 'GET',
      requests: 1234567,
      avgLatency: 12,
      p95Latency: 18,
      errorRate: 0,
      cost: 12.34,
      trend: [10, 11, 12, 11, 12, 13, 12],
      status: 'healthy'
    },
    {
      endpoint: '/api/metrics',
      method: 'GET',
      requests: 345678,
      avgLatency: 156,
      p95Latency: 234,
      errorRate: 0.08,
      cost: 234.56,
      trend: [150, 155, 160, 158, 156, 154, 156],
      status: 'degraded'
    }
  ]

  // Geographic distribution data
  const geoDistribution = [
    { region: 'North America', requests: 45, color: '#8b5cf6' },
    { region: 'Europe', requests: 25, color: '#00d4ff' },
    { region: 'Asia', requests: 20, color: '#10b981' },
    { region: 'South America', requests: 5, color: '#f59e0b' },
    { region: 'Other', requests: 5, color: '#ef4444' }
  ]

  // Client usage data
  const clientUsage = [
    {
      clientId: 'client-001',
      name: 'Production Web App',
      apiKey: 'pk_live_xxxxxxxxxxx',
      requests: 1234567,
      rateLimit: 5000,
      currentRate: 3456,
      billing: 1234.56,
      status: 'active',
      lastSeen: '2 min ago'
    },
    {
      clientId: 'client-002',
      name: 'Mobile App iOS',
      apiKey: 'pk_live_yyyyyyyyyyy',
      requests: 876543,
      rateLimit: 3000,
      currentRate: 2345,
      billing: 876.54,
      status: 'active',
      lastSeen: '5 min ago'
    },
    {
      clientId: 'client-003',
      name: 'Mobile App Android',
      apiKey: 'pk_live_zzzzzzzzzzz',
      requests: 765432,
      rateLimit: 3000,
      currentRate: 2890,
      billing: 765.43,
      status: 'warning',
      lastSeen: '1 min ago'
    },
    {
      clientId: 'client-004',
      name: 'Analytics Dashboard',
      apiKey: 'pk_live_aaaaaaaaaaa',
      requests: 234567,
      rateLimit: 1000,
      currentRate: 456,
      billing: 234.56,
      status: 'active',
      lastSeen: '30 sec ago'
    }
  ]

  // Performance insights
  const performanceInsights = [
    {
      type: 'slow_query',
      title: 'Slow Database Query Detected',
      description: 'Query to player_stats table taking >500ms',
      impact: 'high',
      recommendation: 'Add index on player_id and game_date columns'
    },
    {
      type: 'cache_miss',
      title: 'Low Cache Hit Rate',
      description: 'Cache hit rate dropped to 78% in last hour',
      impact: 'medium',
      recommendation: 'Increase cache TTL for frequently accessed data'
    },
    {
      type: 'rate_limit',
      title: 'Client Approaching Rate Limit',
      description: 'Mobile App Android at 96% of rate limit',
      impact: 'low',
      recommendation: 'Contact client or increase rate limit'
    }
  ]

  // Recent alerts
  const recentAlerts = [
    {
      id: 'alert-001',
      timestamp: '10:23 AM',
      type: 'error_spike',
      message: 'Error rate exceeded 0.1% threshold',
      status: 'resolved'
    },
    {
      id: 'alert-002',
      timestamp: '9:45 AM',
      type: 'latency',
      message: 'P95 latency exceeded 200ms',
      status: 'resolved'
    },
    {
      id: 'alert-003',
      timestamp: '8:30 AM',
      type: 'rate_limit',
      message: 'Client-003 hit rate limit',
      status: 'acknowledged'
    }
  ]

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'text-success'
      case 'warning': return 'text-warning'
      case 'degraded': return 'text-danger'
      default: return 'text-text-secondary'
    }
  }

  const getMethodBadge = (method: string) => {
    const colors: { [key: string]: string } = {
      'GET': 'bg-green-500/20 text-green-400',
      'POST': 'bg-blue-500/20 text-blue-400',
      'PUT': 'bg-yellow-500/20 text-yellow-400',
      'DELETE': 'bg-red-500/20 text-red-400'
    }
    return (
      <span className={`px-2 py-0.5 rounded text-xs font-medium ${colors[method]}`}>
        {method}
      </span>
    )
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
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold text-text-primary mb-2">API Analytics</h1>
              <p className="text-text-secondary">Real-time monitoring and performance insights</p>
            </div>
            <div className="flex items-center gap-2">
              <span className="flex items-center gap-2 px-3 py-1.5 bg-success/20 text-success rounded-lg text-sm">
                <div className="w-2 h-2 bg-success rounded-full animate-pulse" />
                {currentQPS.toFixed(0)} QPS
              </span>
            </div>
          </div>
        </motion.div>

        {/* Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-5 gap-6 mb-8">
          <AnimatedCard delay={0.1}>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-text-secondary text-sm">Requests Today</p>
                <p className="text-2xl font-bold text-text-primary">
                  {(totalRequests / 1000000).toFixed(1)}M
                </p>
                <p className="text-xs text-success flex items-center gap-1 mt-1">
                  <TrendingUp className="w-3 h-3" />
                  +12.3% vs yesterday
                </p>
              </div>
              <div className="p-3 bg-primary/20 rounded-lg">
                <Activity className="w-6 h-6 text-primary" />
              </div>
            </div>
          </AnimatedCard>

          <AnimatedCard delay={0.2}>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-text-secondary text-sm">Avg Latency</p>
                <p className="text-2xl font-bold text-text-primary">{avgLatency}ms</p>
                <p className="text-xs text-success mt-1">Within SLA</p>
              </div>
              <div className="p-3 bg-secondary/20 rounded-lg">
                <Clock className="w-6 h-6 text-secondary" />
              </div>
            </div>
          </AnimatedCard>

          <AnimatedCard delay={0.3}>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-text-secondary text-sm">Error Rate</p>
                <p className="text-2xl font-bold text-text-primary">{errorRate.toFixed(2)}%</p>
                <p className="text-xs text-success mt-1">Below threshold</p>
              </div>
              <div className="p-3 bg-success/20 rounded-lg">
                <CheckCircle className="w-6 h-6 text-success" />
              </div>
            </div>
          </AnimatedCard>

          <AnimatedCard delay={0.4}>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-text-secondary text-sm">Uptime</p>
                <p className="text-2xl font-bold text-text-primary">{uptime}%</p>
                <p className="text-xs text-text-secondary mt-1">Last 30 days</p>
              </div>
              <div className="p-3 bg-success/20 rounded-lg">
                <Wifi className="w-6 h-6 text-success" />
              </div>
            </div>
          </AnimatedCard>

          <AnimatedCard delay={0.5}>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-text-secondary text-sm">Cache Hit Rate</p>
                <p className="text-2xl font-bold text-text-primary">94%</p>
                <p className="text-xs text-warning flex items-center gap-1 mt-1">
                  <TrendingDown className="w-3 h-3" />
                  -2% last hour
                </p>
              </div>
              <div className="p-3 bg-warning/20 rounded-lg">
                <Database className="w-6 h-6 text-warning" />
              </div>
            </div>
          </AnimatedCard>
        </div>

        {/* Real-time Metrics Dashboard */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* Requests per Second Chart */}
          <AnimatedCard delay={0.6} className="p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-text-primary">Request Volume</h3>
              <div className="flex gap-2">
                {['1h', '24h', '7d'].map((range) => (
                  <button
                    key={range}
                    onClick={() => setTimeRange(range)}
                    className={`px-3 py-1 rounded-lg text-sm transition-colors ${
                      timeRange === range
                        ? 'bg-primary text-white'
                        : 'bg-card-hover text-text-secondary hover:text-text-primary'
                    }`}
                  >
                    {range}
                  </button>
                ))}
              </div>
            </div>
            <ResponsiveContainer width="100%" height={250}>
              <AreaChart data={getRequestsData()}>
                <defs>
                  <linearGradient id="requestsGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0}/>
                  </linearGradient>
                  <linearGradient id="cacheGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#00d4ff" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#00d4ff" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="time" stroke="#666" fontSize={10} />
                <YAxis stroke="#666" fontSize={10} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1a1a2e', border: 'none' }}
                  labelStyle={{ color: '#a1a1aa' }}
                />
                <Area 
                  type="monotone" 
                  dataKey="requests" 
                  stroke="#8b5cf6" 
                  fillOpacity={1} 
                  fill="url(#requestsGradient)"
                  name="Requests"
                />
                <Area 
                  type="monotone" 
                  dataKey="cache_hits" 
                  stroke="#00d4ff" 
                  fillOpacity={1} 
                  fill="url(#cacheGradient)"
                  name="Cache Hits"
                />
              </AreaChart>
            </ResponsiveContainer>
          </AnimatedCard>

          {/* Latency Distribution */}
          <AnimatedCard delay={0.7} className="p-6">
            <h3 className="text-lg font-semibold text-text-primary mb-4">Latency Distribution</h3>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={latencyDistribution}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="range" stroke="#666" fontSize={10} />
                <YAxis stroke="#666" fontSize={10} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1a1a2e', border: 'none' }}
                  labelStyle={{ color: '#a1a1aa' }}
                />
                <Bar dataKey="count" fill="#8b5cf6">
                  {latencyDistribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={
                      index === 0 ? '#10b981' :
                      index === 1 ? '#10b981' :
                      index === 2 ? '#f59e0b' :
                      index === 3 ? '#ef4444' : '#ef4444'
                    } />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div className="mt-4 grid grid-cols-3 gap-4 text-sm">
              <div>
                <p className="text-text-secondary">P50</p>
                <p className="text-lg font-semibold text-text-primary">75ms</p>
              </div>
              <div>
                <p className="text-text-secondary">P95</p>
                <p className="text-lg font-semibold text-text-primary">142ms</p>
              </div>
              <div>
                <p className="text-text-secondary">P99</p>
                <p className="text-lg font-semibold text-text-primary">198ms</p>
              </div>
            </div>
          </AnimatedCard>
        </div>

        {/* Geographic Distribution and Error Rate */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          {/* Geographic Heat Map */}
          <AnimatedCard delay={0.8} className="p-6">
            <h3 className="text-lg font-semibold text-text-primary mb-4">Geographic Distribution</h3>
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie
                  data={geoDistribution}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="requests"
                >
                  {geoDistribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1a1a2e', border: 'none' }}
                  formatter={(value: any) => `${value}%`}
                />
              </PieChart>
            </ResponsiveContainer>
            <div className="space-y-2 mt-4">
              {geoDistribution.map((region) => (
                <div key={region.region} className="flex items-center justify-between text-sm">
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full" style={{ backgroundColor: region.color }} />
                    <span className="text-text-secondary">{region.region}</span>
                  </div>
                  <span className="text-text-primary font-medium">{region.requests}%</span>
                </div>
              ))}
            </div>
          </AnimatedCard>

          {/* Error Rate Trend */}
          <AnimatedCard delay={0.9} className="p-6 lg:col-span-2">
            <h3 className="text-lg font-semibold text-text-primary mb-4">Error Rate Trend</h3>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={getRequestsData()}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="time" stroke="#666" fontSize={10} />
                <YAxis stroke="#666" fontSize={10} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1a1a2e', border: 'none' }}
                  labelStyle={{ color: '#a1a1aa' }}
                />
                <Line 
                  type="monotone" 
                  dataKey="errors" 
                  stroke="#ef4444" 
                  strokeWidth={2}
                  dot={false}
                  name="Errors"
                />
              </LineChart>
            </ResponsiveContainer>
            <div className="mt-4 p-3 bg-danger/10 rounded-lg flex items-center justify-between">
              <div className="flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-danger" />
                <span className="text-sm text-text-secondary">Anomaly detected at 9:45 AM</span>
              </div>
              <button className="text-xs text-danger hover:text-danger/80">View Details</button>
            </div>
          </AnimatedCard>
        </div>

        {/* Endpoint Performance Table */}
        <AnimatedCard delay={1.0} className="p-6 mb-8">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-text-primary">Endpoint Performance</h3>
            <div className="flex items-center gap-3">
              <button className="p-2 bg-card-hover rounded-lg hover:bg-card transition-colors">
                <Filter className="w-4 h-4 text-text-secondary" />
              </button>
              <button className="p-2 bg-card-hover rounded-lg hover:bg-card transition-colors">
                <Download className="w-4 h-4 text-text-secondary" />
              </button>
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left py-3 px-4 text-sm font-medium text-text-secondary">Endpoint</th>
                  <th className="text-right py-3 px-4 text-sm font-medium text-text-secondary">Requests</th>
                  <th className="text-right py-3 px-4 text-sm font-medium text-text-secondary">Avg Latency</th>
                  <th className="text-right py-3 px-4 text-sm font-medium text-text-secondary">P95 Latency</th>
                  <th className="text-right py-3 px-4 text-sm font-medium text-text-secondary">Error Rate</th>
                  <th className="text-right py-3 px-4 text-sm font-medium text-text-secondary">Cost</th>
                  <th className="text-center py-3 px-4 text-sm font-medium text-text-secondary">Trend</th>
                </tr>
              </thead>
              <tbody>
                {endpointMetrics.map((endpoint) => (
                  <tr key={endpoint.endpoint} className="border-b border-white/5 hover:bg-card-hover transition-colors">
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2">
                        {getMethodBadge(endpoint.method)}
                        <span className="text-sm text-text-primary font-medium">{endpoint.endpoint}</span>
                        <div className={`w-2 h-2 rounded-full ${
                          endpoint.status === 'healthy' ? 'bg-success' :
                          endpoint.status === 'warning' ? 'bg-warning' : 'bg-danger'
                        }`} />
                      </div>
                    </td>
                    <td className="py-3 px-4 text-right text-sm text-text-primary">
                      {endpoint.requests.toLocaleString()}
                    </td>
                    <td className="py-3 px-4 text-right text-sm text-text-primary">
                      {endpoint.avgLatency}ms
                    </td>
                    <td className="py-3 px-4 text-right text-sm text-text-primary">
                      {endpoint.p95Latency}ms
                    </td>
                    <td className="py-3 px-4 text-right">
                      <span className={`text-sm ${
                        endpoint.errorRate === 0 ? 'text-success' :
                        endpoint.errorRate < 0.05 ? 'text-warning' : 'text-danger'
                      }`}>
                        {endpoint.errorRate.toFixed(2)}%
                      </span>
                    </td>
                    <td className="py-3 px-4 text-right text-sm text-text-primary">
                      ${endpoint.cost.toFixed(2)}
                    </td>
                    <td className="py-3 px-4">
                      <ResponsiveContainer width={60} height={30}>
                        <LineChart data={endpoint.trend.map((val, i) => ({ value: val }))}>
                          <Line 
                            type="monotone" 
                            dataKey="value" 
                            stroke={endpoint.status === 'healthy' ? '#10b981' :
                                   endpoint.status === 'warning' ? '#f59e0b' : '#ef4444'}
                            strokeWidth={2}
                            dot={false}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </AnimatedCard>

        {/* Client Usage and Performance Insights */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* Client Usage Analytics */}
          <AnimatedCard delay={1.1} className="p-6">
            <h3 className="text-lg font-semibold text-text-primary mb-4">Client Usage Analytics</h3>
            <div className="space-y-3">
              {clientUsage.map((client) => (
                <div key={client.clientId} className="bg-card-hover rounded-lg p-4">
                  <div 
                    className="flex items-center justify-between cursor-pointer"
                    onClick={() => setExpandedClient(expandedClient === client.clientId ? null : client.clientId)}
                  >
                    <div className="flex items-center gap-3">
                      <div className={`w-2 h-2 rounded-full ${
                        client.status === 'active' ? 'bg-success animate-pulse' :
                        client.status === 'warning' ? 'bg-warning animate-pulse' : 'bg-danger'
                      }`} />
                      <div>
                        <p className="text-sm font-medium text-text-primary">{client.name}</p>
                        <p className="text-xs text-text-secondary">Last seen: {client.lastSeen}</p>
                      </div>
                    </div>
                    {expandedClient === client.clientId ? 
                      <ChevronUp className="w-4 h-4 text-text-secondary" /> :
                      <ChevronDown className="w-4 h-4 text-text-secondary" />
                    }
                  </div>

                  {expandedClient === client.clientId && (
                    <div className="mt-4 pt-4 border-t border-white/10 space-y-3">
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <p className="text-text-secondary">API Key</p>
                          <p className="text-text-primary font-mono text-xs">{client.apiKey}</p>
                        </div>
                        <div>
                          <p className="text-text-secondary">Total Requests</p>
                          <p className="text-text-primary font-medium">{client.requests.toLocaleString()}</p>
                        </div>
                      </div>
                      
                      <div>
                        <div className="flex justify-between text-xs mb-1">
                          <span className="text-text-secondary">Rate Limit Usage</span>
                          <span className={`${
                            (client.currentRate / client.rateLimit) > 0.9 ? 'text-danger' :
                            (client.currentRate / client.rateLimit) > 0.7 ? 'text-warning' : 'text-success'
                          }`}>
                            {client.currentRate} / {client.rateLimit} req/s
                          </span>
                        </div>
                        <div className="w-full bg-background rounded-full h-2">
                          <div
                            className={`h-full rounded-full transition-all ${
                              (client.currentRate / client.rateLimit) > 0.9 ? 'bg-danger' :
                              (client.currentRate / client.rateLimit) > 0.7 ? 'bg-warning' : 'bg-success'
                            }`}
                            style={{ width: `${(client.currentRate / client.rateLimit) * 100}%` }}
                          />
                        </div>
                      </div>

                      <div className="flex justify-between text-sm">
                        <span className="text-text-secondary">Billing (MTD)</span>
                        <span className="text-text-primary font-medium">${client.billing.toFixed(2)}</span>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>

            <div className="mt-4 p-3 bg-primary/10 rounded-lg">
              <p className="text-xs text-text-secondary mb-1">Total API Revenue (MTD)</p>
              <p className="text-xl font-bold text-text-primary">$3,012.48</p>
            </div>
          </AnimatedCard>

          {/* Performance Insights */}
          <AnimatedCard delay={1.2} className="p-6">
            <h3 className="text-lg font-semibold text-text-primary mb-4">Performance Insights</h3>
            <div className="space-y-3">
              {performanceInsights.map((insight, index) => (
                <div key={index} className="bg-card-hover rounded-lg p-4">
                  <div className="flex items-start gap-3">
                    <div className={`p-2 rounded-lg ${
                      insight.impact === 'high' ? 'bg-danger/20' :
                      insight.impact === 'medium' ? 'bg-warning/20' : 'bg-primary/20'
                    }`}>
                      {insight.type === 'slow_query' && <Database className={`w-4 h-4 ${
                        insight.impact === 'high' ? 'text-danger' :
                        insight.impact === 'medium' ? 'text-warning' : 'text-primary'
                      }`} />}
                      {insight.type === 'cache_miss' && <Zap className={`w-4 h-4 ${
                        insight.impact === 'high' ? 'text-danger' :
                        insight.impact === 'medium' ? 'text-warning' : 'text-primary'
                      }`} />}
                      {insight.type === 'rate_limit' && <AlertCircle className={`w-4 h-4 ${
                        insight.impact === 'high' ? 'text-danger' :
                        insight.impact === 'medium' ? 'text-warning' : 'text-primary'
                      }`} />}
                    </div>
                    <div className="flex-1">
                      <p className="text-sm font-medium text-text-primary">{insight.title}</p>
                      <p className="text-xs text-text-secondary mt-1">{insight.description}</p>
                      <div className="mt-3 p-2 bg-background rounded flex items-start gap-2">
                        <Zap className="w-3 h-3 text-primary mt-0.5" />
                        <p className="text-xs text-text-secondary">
                          <span className="text-primary font-medium">Recommendation:</span> {insight.recommendation}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            <div className="mt-4 grid grid-cols-2 gap-3">
              <div className="bg-card-hover rounded-lg p-3">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs text-text-secondary">Cache Hit Rate</span>
                  <span className="text-xs font-medium text-text-primary">94%</span>
                </div>
                <div className="w-full bg-background rounded-full h-1.5">
                  <div className="h-full bg-success rounded-full" style={{ width: '94%' }} />
                </div>
              </div>
              <div className="bg-card-hover rounded-lg p-3">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs text-text-secondary">DB Connections</span>
                  <span className="text-xs font-medium text-text-primary">45/100</span>
                </div>
                <div className="w-full bg-background rounded-full h-1.5">
                  <div className="h-full bg-primary rounded-full" style={{ width: '45%' }} />
                </div>
              </div>
            </div>
          </AnimatedCard>
        </div>

        {/* Alert Configuration */}
        <AnimatedCard delay={1.3} className="p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-text-primary">Alert Configuration</h3>
            <button
              onClick={() => setShowAlertConfig(!showAlertConfig)}
              className="flex items-center gap-2 px-4 py-2 bg-primary hover:bg-primary-hover text-white rounded-lg transition-colors"
            >
              <Settings className="w-4 h-4" />
              Configure Alerts
            </button>
          </div>

          {showAlertConfig ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h4 className="text-sm font-medium text-text-secondary mb-3">Threshold Settings</h4>
                <div className="space-y-3">
                  <div>
                    <label className="flex items-center justify-between text-sm">
                      <span className="text-text-secondary">Error Rate Threshold</span>
                      <input
                        type="number"
                        className="w-20 px-2 py-1 bg-card-hover text-text-primary rounded text-sm"
                        defaultValue="0.1"
                        step="0.01"
                      />
                    </label>
                  </div>
                  <div>
                    <label className="flex items-center justify-between text-sm">
                      <span className="text-text-secondary">P95 Latency (ms)</span>
                      <input
                        type="number"
                        className="w-20 px-2 py-1 bg-card-hover text-text-primary rounded text-sm"
                        defaultValue="200"
                      />
                    </label>
                  </div>
                  <div>
                    <label className="flex items-center justify-between text-sm">
                      <span className="text-text-secondary">Min Cache Hit Rate (%)</span>
                      <input
                        type="number"
                        className="w-20 px-2 py-1 bg-card-hover text-text-primary rounded text-sm"
                        defaultValue="80"
                      />
                    </label>
                  </div>
                </div>
              </div>

              <div>
                <h4 className="text-sm font-medium text-text-secondary mb-3">Notification Channels</h4>
                <div className="space-y-3">
                  <label className="flex items-center gap-3">
                    <input type="checkbox" className="rounded" defaultChecked />
                    <span className="text-sm text-text-primary">Email notifications</span>
                  </label>
                  <label className="flex items-center gap-3">
                    <input type="checkbox" className="rounded" defaultChecked />
                    <span className="text-sm text-text-primary">Slack integration</span>
                  </label>
                  <label className="flex items-center gap-3">
                    <input type="checkbox" className="rounded" />
                    <span className="text-sm text-text-primary">PagerDuty alerts</span>
                  </label>
                  <label className="flex items-center gap-3">
                    <input type="checkbox" className="rounded" />
                    <span className="text-sm text-text-primary">Webhook notifications</span>
                  </label>
                </div>
              </div>
            </div>
          ) : (
            <div>
              <h4 className="text-sm font-medium text-text-secondary mb-3">Recent Alerts</h4>
              <div className="space-y-2">
                {recentAlerts.map((alert) => (
                  <div key={alert.id} className="flex items-center justify-between p-3 bg-card-hover rounded-lg">
                    <div className="flex items-center gap-3">
                      {alert.status === 'resolved' ? (
                        <CheckCircle className="w-4 h-4 text-success" />
                      ) : (
                        <AlertCircle className="w-4 h-4 text-warning" />
                      )}
                      <div>
                        <p className="text-sm text-text-primary">{alert.message}</p>
                        <p className="text-xs text-text-secondary">{alert.timestamp}</p>
                      </div>
                    </div>
                    <span className={`text-xs px-2 py-1 rounded-full ${
                      alert.status === 'resolved' ? 'bg-success/20 text-success' : 'bg-warning/20 text-warning'
                    }`}>
                      {alert.status}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </AnimatedCard>
      </div>
    </div>
  )
}