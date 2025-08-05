'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  Globe, Key, Shield, Zap, AlertCircle, Activity,
  CheckCircle, XCircle, Clock, TrendingUp, TrendingDown,
  Lock, Unlock, RefreshCw, Settings, Plus, Copy,
  Eye, EyeOff, Database, Server, Wifi, WifiOff
} from 'lucide-react'
import { 
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, RadarChart,
  PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar
} from 'recharts'
import BackButton from '@/components/BackButton'
import StatsCard from '@/components/StatsCard'
import DataTable from '@/components/DataTable'
import AnimatedCard from '@/components/ui/AnimatedCard'

export default function APIGateway() {
  const [currentRPS, setCurrentRPS] = useState(2456)
  const [selectedEndpoint, setSelectedEndpoint] = useState<string | null>(null)
  const [showKeyModal, setShowKeyModal] = useState(false)
  const [copiedKey, setCopiedKey] = useState<string | null>(null)
  const [selectedTimeRange, setSelectedTimeRange] = useState('1h')

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentRPS(prev => Math.max(1000, Math.min(5000, prev + (Math.random() - 0.5) * 200)))
    }, 2000)
    return () => clearInterval(interval)
  }, [])

  // Endpoints data
  const endpoints = [
    {
      path: '/api/predictions',
      method: 'POST',
      status: 'healthy',
      requests: 892345,
      avgLatency: 95,
      p99Latency: 142,
      errorRate: 0.02,
      cacheHitRate: 0.78,
      rateLimit: 1000,
      currentLoad: 0.67
    },
    {
      path: '/api/players',
      method: 'GET',
      status: 'healthy',
      requests: 654321,
      avgLatency: 45,
      p99Latency: 78,
      errorRate: 0.01,
      cacheHitRate: 0.94,
      rateLimit: 5000,
      currentLoad: 0.23
    },
    {
      path: '/api/batch',
      method: 'POST',
      status: 'warning',
      requests: 234567,
      avgLatency: 234,
      p99Latency: 456,
      errorRate: 0.05,
      cacheHitRate: 0.65,
      rateLimit: 100,
      currentLoad: 0.89
    },
    {
      path: '/api/experiments',
      method: 'GET',
      status: 'healthy',
      requests: 123456,
      avgLatency: 67,
      p99Latency: 98,
      errorRate: 0.03,
      cacheHitRate: 0.82,
      rateLimit: 1000,
      currentLoad: 0.45
    },
    {
      path: '/api/health',
      method: 'GET',
      status: 'healthy',
      requests: 1234567,
      avgLatency: 12,
      p99Latency: 18,
      errorRate: 0,
      cacheHitRate: 1.0,
      rateLimit: 10000,
      currentLoad: 0.12
    }
  ]

  // API Keys
  const apiKeys = [
    {
      id: 'key-001',
      name: 'Production Web App',
      key: 'pk_live_xxxxxxxxxxxxxxxxxxx',
      created: '2024-12-01',
      lastUsed: '2 min ago',
      status: 'active',
      requests: 2345678,
      quota: 5000,
      currentUsage: 3456,
      permissions: ['read', 'write', 'predict']
    },
    {
      id: 'key-002',
      name: 'Mobile App',
      key: 'pk_live_yyyyyyyyyyyyyyyyyyy',
      created: '2024-12-15',
      lastUsed: '5 min ago',
      status: 'active',
      requests: 1234567,
      quota: 3000,
      currentUsage: 2100,
      permissions: ['read', 'predict']
    },
    {
      id: 'key-003',
      name: 'Analytics Dashboard',
      key: 'pk_live_zzzzzzzzzzzzzzzzzzz',
      created: '2025-01-01',
      lastUsed: '1 hour ago',
      status: 'active',
      requests: 456789,
      quota: 1000,
      currentUsage: 234,
      permissions: ['read']
    },
    {
      id: 'key-004',
      name: 'Test Environment',
      key: 'pk_test_aaaaaaaaaaaaaaaaaaa',
      created: '2025-01-10',
      lastUsed: '3 days ago',
      status: 'inactive',
      requests: 12345,
      quota: 500,
      currentUsage: 0,
      permissions: ['read', 'write', 'predict', 'admin']
    }
  ]

  // Security events
  const securityEvents = [
    { time: '10:23 AM', type: 'rate_limit', ip: '192.168.1.100', endpoint: '/api/batch', status: 'blocked' },
    { time: '10:15 AM', type: 'auth_failed', ip: '10.0.0.45', endpoint: '/api/predictions', status: 'blocked' },
    { time: '9:58 AM', type: 'suspicious', ip: '203.0.113.0', endpoint: '/api/players', status: 'flagged' },
    { time: '9:45 AM', type: 'rate_limit', ip: '172.16.0.1', endpoint: '/api/experiments', status: 'blocked' },
    { time: '9:30 AM', type: 'auth_failed', ip: '192.168.2.50', endpoint: '/api/batch', status: 'blocked' }
  ]

  // Geographic distribution
  const geoData = [
    { region: 'North America', value: 45, requests: 1234567 },
    { region: 'Europe', value: 25, requests: 687234 },
    { region: 'Asia', value: 20, requests: 548923 },
    { region: 'South America', value: 7, requests: 192034 },
    { region: 'Other', value: 3, requests: 82345 }
  ]

  // Time series data for requests
  const requestsData = Array.from({ length: 24 }, (_, i) => ({
    time: selectedTimeRange === '1h' ? `${i * 2.5}m` : `${i}:00`,
    requests: 2000 + Math.random() * 1000,
    errors: Math.random() * 50
  }))

  // Response time distribution
  const responseTimeData = [
    { range: '0-50ms', count: 45 },
    { range: '50-100ms', count: 30 },
    { range: '100-200ms', count: 15 },
    { range: '200-500ms', count: 7 },
    { range: '500ms+', count: 3 }
  ]

  // Cache performance
  const cacheData = [
    { metric: 'Hit Rate', value: 0.89 },
    { metric: 'Miss Rate', value: 0.11 },
    { metric: 'Eviction Rate', value: 0.05 },
    { metric: 'Fill Rate', value: 0.92 }
  ]

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy':
        return <CheckCircle className="w-4 h-4 text-success" />
      case 'warning':
        return <AlertCircle className="w-4 h-4 text-warning" />
      case 'error':
        return <XCircle className="w-4 h-4 text-danger" />
      default:
        return <Clock className="w-4 h-4 text-text-secondary" />
    }
  }

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text)
    setCopiedKey(id)
    setTimeout(() => setCopiedKey(null), 2000)
  }

  const endpointColumns = [
    {
      key: 'path',
      label: 'Endpoint',
      render: (value: string, row: any) => (
        <div className="flex items-center gap-2">
          {getStatusIcon(row.status)}
          <span className={`px-2 py-0.5 rounded text-xs font-medium ${
            row.method === 'GET' ? 'bg-green-500/20 text-green-400' :
            row.method === 'POST' ? 'bg-blue-500/20 text-blue-400' :
            row.method === 'PUT' ? 'bg-yellow-500/20 text-yellow-400' :
            'bg-red-500/20 text-red-400'
          }`}>
            {row.method}
          </span>
          <span className="font-mono text-sm">{value}</span>
        </div>
      )
    },
    { 
      key: 'requests', 
      label: 'Requests',
      sortable: true,
      render: (value: number) => value.toLocaleString()
    },
    { key: 'avgLatency', label: 'Avg Latency', sortable: true, render: (value: number) => `${value}ms` },
    { key: 'p99Latency', label: 'P99 Latency', sortable: true, render: (value: number) => `${value}ms` },
    { 
      key: 'errorRate', 
      label: 'Error Rate',
      sortable: true,
      render: (value: number) => (
        <span className={value > 0.03 ? 'text-warning' : 'text-success'}>
          {(value * 100).toFixed(2)}%
        </span>
      )
    },
    {
      key: 'cacheHitRate',
      label: 'Cache Hit',
      render: (value: number) => (
        <div className="flex items-center gap-2">
          <div className="w-16 bg-card-hover rounded-full h-2">
            <div 
              className="h-full bg-gradient-to-r from-primary to-secondary rounded-full"
              style={{ width: `${value * 100}%` }}
            />
          </div>
          <span className="text-xs">{(value * 100).toFixed(0)}%</span>
        </div>
      )
    }
  ]

  const COLORS = ['#8b5cf6', '#00d4ff', '#10b981', '#f59e0b', '#ef4444']

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
          <h1 className="text-4xl font-bold text-text-primary mb-2">API Gateway</h1>
          <p className="text-text-secondary">Real-time API management and monitoring dashboard</p>
        </motion.div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <StatsCard
            title="Requests/Second"
            value={currentRPS.toLocaleString()}
            subtitle="Current throughput"
            icon={Activity}
            iconColor="primary"
            delay={0.1}
          />
          <StatsCard
            title="Active API Keys"
            value="12"
            subtitle="3 near quota limit"
            icon={Key}
            iconColor="success"
            delay={0.2}
          />
          <StatsCard
            title="Security Events"
            value="5"
            subtitle="Last hour"
            icon={Shield}
            iconColor="warning"
            delay={0.3}
          />
          <StatsCard
            title="Avg Response Time"
            value="87ms"
            subtitle="P99: 142ms"
            icon={Zap}
            iconColor="secondary"
            trend={{ value: 8.5, isPositive: false }}
            delay={0.4}
          />
        </div>

        {/* Live Request Monitoring */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <AnimatedCard delay={0.5} className="p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-text-primary">Request Volume</h3>
              <div className="flex gap-2">
                {['1h', '24h', '7d'].map((range) => (
                  <button
                    key={range}
                    onClick={() => setSelectedTimeRange(range)}
                    className={`px-3 py-1 rounded text-xs transition-colors ${
                      selectedTimeRange === range
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
              <LineChart data={requestsData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="time" stroke="#666" fontSize={10} />
                <YAxis stroke="#666" fontSize={10} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1a1a2e', border: 'none' }}
                  labelStyle={{ color: '#a1a1aa' }}
                />
                <Line 
                  type="monotone" 
                  dataKey="requests" 
                  stroke="#8b5cf6" 
                  strokeWidth={2}
                  dot={false}
                  name="Requests"
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
          </AnimatedCard>

          <AnimatedCard delay={0.6} className="p-6">
            <h3 className="text-lg font-semibold text-text-primary mb-4">Geographic Distribution</h3>
            <div className="flex items-center justify-between">
              <div className="flex-1">
                <ResponsiveContainer width="100%" height={200}>
                  <PieChart>
                    <Pie
                      data={geoData}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={80}
                      paddingAngle={5}
                      dataKey="value"
                    >
                      {geoData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div className="space-y-2">
                {geoData.map((item, index) => (
                  <div key={item.region} className="flex items-center gap-2 text-sm">
                    <div 
                      className="w-3 h-3 rounded-full" 
                      style={{ backgroundColor: COLORS[index % COLORS.length] }}
                    />
                    <span className="text-text-secondary w-24">{item.region}</span>
                    <span className="text-text-primary font-medium">{item.value}%</span>
                  </div>
                ))}
              </div>
            </div>
          </AnimatedCard>
        </div>

        {/* Endpoint Management */}
        <AnimatedCard delay={0.7} className="p-6 mb-8">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-text-primary">Endpoint Performance</h3>
            <button className="flex items-center gap-2 px-4 py-2 bg-card-hover text-text-primary rounded-lg hover:bg-card transition-colors">
              <Settings className="w-4 h-4" />
              Configure
            </button>
          </div>
          
          <DataTable
            columns={endpointColumns}
            data={endpoints}
            searchable={false}
          />

          <div className="grid grid-cols-4 gap-4 mt-6">
            <div className="text-center">
              <p className="text-xs text-text-secondary mb-1">Total Endpoints</p>
              <p className="text-2xl font-bold text-text-primary">{endpoints.length}</p>
            </div>
            <div className="text-center">
              <p className="text-xs text-text-secondary mb-1">Healthy</p>
              <p className="text-2xl font-bold text-success">
                {endpoints.filter(e => e.status === 'healthy').length}
              </p>
            </div>
            <div className="text-center">
              <p className="text-xs text-text-secondary mb-1">Warnings</p>
              <p className="text-2xl font-bold text-warning">
                {endpoints.filter(e => e.status === 'warning').length}
              </p>
            </div>
            <div className="text-center">
              <p className="text-xs text-text-secondary mb-1">Errors</p>
              <p className="text-2xl font-bold text-danger">
                {endpoints.filter(e => e.status === 'error').length}
              </p>
            </div>
          </div>
        </AnimatedCard>

        {/* API Key Management */}
        <AnimatedCard delay={0.8} className="p-6 mb-8">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-text-primary">API Key Management</h3>
            <button
              onClick={() => setShowKeyModal(true)}
              className="flex items-center gap-2 px-4 py-2 bg-primary hover:bg-primary-hover text-white rounded-lg transition-colors"
            >
              <Plus className="w-4 h-4" />
              Generate New Key
            </button>
          </div>

          <div className="space-y-3">
            {apiKeys.map((apiKey) => (
              <div key={apiKey.id} className="bg-card-hover rounded-lg p-4">
                <div className="flex items-start justify-between mb-3">
                  <div>
                    <div className="flex items-center gap-3">
                      <h4 className="text-sm font-semibold text-text-primary">{apiKey.name}</h4>
                      <span className={`px-2 py-0.5 rounded-full text-xs ${
                        apiKey.status === 'active' 
                          ? 'bg-success/20 text-success' 
                          : 'bg-gray-500/20 text-gray-400'
                      }`}>
                        {apiKey.status}
                      </span>
                    </div>
                    <div className="flex items-center gap-2 mt-1">
                      <code className="text-xs text-text-secondary font-mono">{apiKey.key}</code>
                      <button
                        onClick={() => copyToClipboard(apiKey.key, apiKey.id)}
                        className="p-1 hover:bg-card rounded transition-colors"
                      >
                        {copiedKey === apiKey.id ? (
                          <CheckCircle className="w-3 h-3 text-success" />
                        ) : (
                          <Copy className="w-3 h-3 text-text-secondary" />
                        )}
                      </button>
                    </div>
                  </div>
                  <button className="p-2 bg-card rounded-lg hover:bg-background transition-colors">
                    {apiKey.status === 'active' ? (
                      <Lock className="w-4 h-4 text-text-secondary" />
                    ) : (
                      <Unlock className="w-4 h-4 text-text-secondary" />
                    )}
                  </button>
                </div>

                <div className="grid grid-cols-4 gap-4 text-xs">
                  <div>
                    <p className="text-text-secondary">Created</p>
                    <p className="text-text-primary">{apiKey.created}</p>
                  </div>
                  <div>
                    <p className="text-text-secondary">Last Used</p>
                    <p className="text-text-primary">{apiKey.lastUsed}</p>
                  </div>
                  <div>
                    <p className="text-text-secondary">Total Requests</p>
                    <p className="text-text-primary">{apiKey.requests.toLocaleString()}</p>
                  </div>
                  <div>
                    <p className="text-text-secondary">Quota Usage</p>
                    <div className="flex items-center gap-2 mt-1">
                      <div className="flex-1 bg-background rounded-full h-1.5">
                        <div 
                          className={`h-full rounded-full ${
                            (apiKey.currentUsage / apiKey.quota) > 0.8 
                              ? 'bg-danger' 
                              : 'bg-success'
                          }`}
                          style={{ width: `${(apiKey.currentUsage / apiKey.quota) * 100}%` }}
                        />
                      </div>
                      <span className="text-text-primary">
                        {apiKey.currentUsage}/{apiKey.quota}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-2 mt-3">
                  <span className="text-xs text-text-secondary">Permissions:</span>
                  {apiKey.permissions.map((perm) => (
                    <span key={perm} className="px-2 py-0.5 bg-primary/20 text-primary rounded text-xs">
                      {perm}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </AnimatedCard>

        {/* Security Dashboard */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <AnimatedCard delay={0.9} className="p-6">
            <h3 className="text-lg font-semibold text-text-primary mb-4">Security Events</h3>
            <div className="space-y-2">
              {securityEvents.map((event, index) => (
                <div key={index} className="flex items-center justify-between p-3 bg-card-hover rounded-lg">
                  <div className="flex items-center gap-3">
                    {event.status === 'blocked' ? (
                      <XCircle className="w-4 h-4 text-danger" />
                    ) : (
                      <AlertCircle className="w-4 h-4 text-warning" />
                    )}
                    <div>
                      <p className="text-sm text-text-primary">
                        {event.type === 'rate_limit' && 'Rate limit exceeded'}
                        {event.type === 'auth_failed' && 'Authentication failed'}
                        {event.type === 'suspicious' && 'Suspicious activity detected'}
                      </p>
                      <p className="text-xs text-text-secondary">
                        {event.ip} • {event.endpoint} • {event.time}
                      </p>
                    </div>
                  </div>
                  <span className={`text-xs px-2 py-1 rounded-full ${
                    event.status === 'blocked' 
                      ? 'bg-danger/20 text-danger' 
                      : 'bg-warning/20 text-warning'
                  }`}>
                    {event.status}
                  </span>
                </div>
              ))}
            </div>
            
            <div className="mt-4 p-3 bg-danger/10 rounded-lg">
              <p className="text-xs text-danger font-medium">5 threats blocked in the last hour</p>
            </div>
          </AnimatedCard>

          <AnimatedCard delay={1.0} className="p-6">
            <h3 className="text-lg font-semibold text-text-primary mb-4">Performance Optimization</h3>
            
            {/* Response Time Distribution */}
            <div className="mb-4">
              <p className="text-sm text-text-secondary mb-2">Response Time Distribution</p>
              <ResponsiveContainer width="100%" height={120}>
                <BarChart data={responseTimeData}>
                  <XAxis dataKey="range" stroke="#666" fontSize={10} />
                  <YAxis stroke="#666" fontSize={10} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#1a1a2e', border: 'none' }}
                    labelStyle={{ color: '#a1a1aa' }}
                  />
                  <Bar dataKey="count" fill="#8b5cf6" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Cache Performance */}
            <div>
              <p className="text-sm text-text-secondary mb-2">Cache Performance</p>
              <div className="grid grid-cols-2 gap-3">
                {cacheData.map((item) => (
                  <div key={item.metric} className="bg-card-hover rounded-lg p-3">
                    <p className="text-xs text-text-secondary">{item.metric}</p>
                    <p className="text-lg font-semibold text-text-primary">
                      {(item.value * 100).toFixed(0)}%
                    </p>
                  </div>
                ))}
              </div>
            </div>

            <div className="mt-4 flex gap-2">
              <button className="flex-1 px-3 py-2 bg-card-hover text-text-primary rounded-lg text-sm hover:bg-card transition-colors">
                Clear Cache
              </button>
              <button className="flex-1 px-3 py-2 bg-card-hover text-text-primary rounded-lg text-sm hover:bg-card transition-colors">
                Optimize Routes
              </button>
            </div>
          </AnimatedCard>
        </div>

        {/* Database Connection Pool */}
        <AnimatedCard delay={1.1} className="p-6">
          <h3 className="text-lg font-semibold text-text-primary mb-4">System Resources</h3>
          <div className="grid grid-cols-4 gap-6">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-text-secondary">DB Connections</span>
                <Database className="w-4 h-4 text-primary" />
              </div>
              <p className="text-2xl font-bold text-text-primary">45/100</p>
              <div className="w-full bg-card-hover rounded-full h-2 mt-2">
                <div className="h-full bg-primary rounded-full" style={{ width: '45%' }} />
              </div>
            </div>
            
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-text-secondary">Redis Memory</span>
                <Server className="w-4 h-4 text-secondary" />
              </div>
              <p className="text-2xl font-bold text-text-primary">2.3GB</p>
              <div className="w-full bg-card-hover rounded-full h-2 mt-2">
                <div className="h-full bg-secondary rounded-full" style={{ width: '58%' }} />
              </div>
            </div>
            
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-text-secondary">Network I/O</span>
                <Wifi className="w-4 h-4 text-success" />
              </div>
              <p className="text-2xl font-bold text-text-primary">124MB/s</p>
              <div className="w-full bg-card-hover rounded-full h-2 mt-2">
                <div className="h-full bg-success rounded-full" style={{ width: '31%' }} />
              </div>
            </div>
            
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-text-secondary">CPU Usage</span>
                <Activity className="w-4 h-4 text-warning" />
              </div>
              <p className="text-2xl font-bold text-text-primary">67%</p>
              <div className="w-full bg-card-hover rounded-full h-2 mt-2">
                <div className="h-full bg-warning rounded-full" style={{ width: '67%' }} />
              </div>
            </div>
          </div>
        </AnimatedCard>

        {/* Generate API Key Modal */}
        {showKeyModal && (
          <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="bg-card rounded-xl p-6 max-w-md w-full mx-4"
            >
              <h3 className="text-xl font-semibold text-text-primary mb-4">Generate New API Key</h3>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm text-text-secondary mb-2">Key Name</label>
                  <input
                    type="text"
                    className="w-full px-4 py-2 bg-card-hover text-text-primary rounded-lg focus:outline-none focus:ring-2 focus:ring-primary"
                    placeholder="e.g., Production Server"
                  />
                </div>

                <div>
                  <label className="block text-sm text-text-secondary mb-2">Rate Limit (req/hour)</label>
                  <input
                    type="number"
                    className="w-full px-4 py-2 bg-card-hover text-text-primary rounded-lg focus:outline-none focus:ring-2 focus:ring-primary"
                    placeholder="5000"
                    defaultValue="5000"
                  />
                </div>

                <div>
                  <label className="block text-sm text-text-secondary mb-2">Permissions</label>
                  <div className="space-y-2">
                    <label className="flex items-center gap-2">
                      <input type="checkbox" className="rounded" defaultChecked />
                      <span className="text-sm text-text-primary">Read</span>
                    </label>
                    <label className="flex items-center gap-2">
                      <input type="checkbox" className="rounded" defaultChecked />
                      <span className="text-sm text-text-primary">Write</span>
                    </label>
                    <label className="flex items-center gap-2">
                      <input type="checkbox" className="rounded" defaultChecked />
                      <span className="text-sm text-text-primary">Predict</span>
                    </label>
                    <label className="flex items-center gap-2">
                      <input type="checkbox" className="rounded" />
                      <span className="text-sm text-text-primary">Admin</span>
                    </label>
                  </div>
                </div>

                <div>
                  <label className="block text-sm text-text-secondary mb-2">Expiration</label>
                  <select className="w-full px-4 py-2 bg-card-hover text-text-primary rounded-lg focus:outline-none focus:ring-2 focus:ring-primary">
                    <option>Never</option>
                    <option>30 days</option>
                    <option>90 days</option>
                    <option>1 year</option>
                  </select>
                </div>
              </div>

              <div className="flex gap-3 mt-6">
                <button
                  onClick={() => setShowKeyModal(false)}
                  className="flex-1 px-4 py-2 bg-card-hover text-text-primary rounded-lg hover:bg-card transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={() => setShowKeyModal(false)}
                  className="flex-1 px-4 py-2 bg-primary hover:bg-primary-hover text-white rounded-lg transition-colors"
                >
                  Generate Key
                </button>
              </div>
            </motion.div>
          </div>
        )}
      </div>
    </div>
  )
}