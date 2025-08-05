'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import AnimatedCard from '@/components/ui/AnimatedCard'
import { 
  Server, Database, Cloud, GitBranch, 
  Activity, HardDrive, Cpu, Zap,
  CheckCircle, AlertTriangle, XCircle
} from 'lucide-react'
import { cn } from '@/lib/utils'

interface ServiceStatus {
  name: string
  status: 'healthy' | 'degraded' | 'down'
  uptime: number
  latency: number
  icon: React.ReactNode
  metrics: {
    cpu?: number
    memory?: number
    connections?: number
    requests?: number
  }
}

export default function InfrastructureHealth() {
  const [services, setServices] = useState<ServiceStatus[]>([
    {
      name: 'FastAPI Backend',
      status: 'healthy',
      uptime: 99.99,
      latency: 87,
      icon: <Server className="w-5 h-5" />,
      metrics: { cpu: 42, memory: 68, requests: 1247 }
    },
    {
      name: 'PostgreSQL (Supabase)',
      status: 'healthy',
      uptime: 99.98,
      latency: 23,
      icon: <Database className="w-5 h-5" />,
      metrics: { connections: 85, memory: 72 }
    },
    {
      name: 'Redis Cache',
      status: 'healthy',
      uptime: 99.95,
      latency: 2,
      icon: <HardDrive className="w-5 h-5" />,
      metrics: { memory: 45, connections: 120 }
    },
    {
      name: 'Railway Platform',
      status: 'healthy',
      uptime: 100,
      latency: 45,
      icon: <Cloud className="w-5 h-5" />,
      metrics: { cpu: 38, memory: 52 }
    },
    {
      name: 'GitHub Actions',
      status: 'healthy',
      uptime: 100,
      latency: 0,
      icon: <GitBranch className="w-5 h-5" />,
      metrics: { requests: 24 }
    },
    {
      name: 'Model Registry',
      status: 'degraded',
      uptime: 98.5,
      latency: 156,
      icon: <Cpu className="w-5 h-5" />,
      metrics: { cpu: 78, memory: 84 }
    },
  ])

  const [selectedService, setSelectedService] = useState<ServiceStatus | null>(null)

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      setServices(prev => prev.map(service => ({
        ...service,
        latency: service.latency + (Math.random() - 0.5) * 10,
        metrics: {
          ...service.metrics,
          cpu: service.metrics.cpu ? Math.min(100, Math.max(0, service.metrics.cpu + (Math.random() - 0.5) * 5)) : undefined,
          memory: service.metrics.memory ? Math.min(100, Math.max(0, service.metrics.memory + (Math.random() - 0.5) * 3)) : undefined,
          requests: service.metrics.requests ? Math.floor(service.metrics.requests + Math.random() * 10) : undefined,
        }
      })))
    }, 3000)

    return () => clearInterval(interval)
  }, [])

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy':
        return <CheckCircle className="w-4 h-4 text-success" />
      case 'degraded':
        return <AlertTriangle className="w-4 h-4 text-warning" />
      case 'down':
        return <XCircle className="w-4 h-4 text-danger" />
      default:
        return null
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'border-success/50 bg-success/10'
      case 'degraded':
        return 'border-warning/50 bg-warning/10'
      case 'down':
        return 'border-danger/50 bg-danger/10'
      default:
        return ''
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-2xl font-bold text-text-primary mb-2">Infrastructure Health</h2>
          <p className="text-text-secondary">Real-time monitoring of production services</p>
        </div>
        <div className="flex items-center gap-2">
          <div className="pulse-dot" />
          <span className="text-sm text-text-secondary">Live monitoring</span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {services.map((service, index) => (
          <motion.div
            key={service.name}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: index * 0.1 }}
            onClick={() => setSelectedService(service)}
            className={cn(
              'glass-card p-4 cursor-pointer transition-all hover:scale-105',
              'border-2',
              getStatusColor(service.status)
            )}
          >
            <div className="flex items-start justify-between mb-3">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-card-hover rounded-lg">
                  {service.icon}
                </div>
                <div>
                  <h3 className="font-semibold text-text-primary">{service.name}</h3>
                  <div className="flex items-center gap-2 mt-1">
                    {getStatusIcon(service.status)}
                    <span className="text-xs text-text-secondary capitalize">{service.status}</span>
                  </div>
                </div>
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-text-secondary">Uptime</span>
                <span className="text-text-primary font-medium">{service.uptime}%</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-text-secondary">Latency</span>
                <span className="text-text-primary font-medium">{service.latency.toFixed(0)}ms</span>
              </div>
              
              {service.metrics.cpu !== undefined && (
                <div className="mt-3">
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-text-secondary">CPU</span>
                    <span className="text-text-primary">{service.metrics.cpu.toFixed(0)}%</span>
                  </div>
                  <div className="w-full bg-card-hover rounded-full h-1.5">
                    <motion.div
                      className="h-full bg-gradient-to-r from-primary to-secondary rounded-full"
                      initial={{ width: 0 }}
                      animate={{ width: `${service.metrics.cpu}%` }}
                      transition={{ duration: 1 }}
                    />
                  </div>
                </div>
              )}

              {service.metrics.memory !== undefined && (
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-text-secondary">Memory</span>
                    <span className="text-text-primary">{service.metrics.memory.toFixed(0)}%</span>
                  </div>
                  <div className="w-full bg-card-hover rounded-full h-1.5">
                    <motion.div
                      className="h-full bg-gradient-to-r from-secondary to-primary rounded-full"
                      initial={{ width: 0 }}
                      animate={{ width: `${service.metrics.memory}%` }}
                      transition={{ duration: 1 }}
                    />
                  </div>
                </div>
              )}
            </div>

            <div className="mt-3 pt-3 border-t border-white/10 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Activity className="w-3 h-3 text-text-secondary" />
                <span className="text-xs text-text-secondary">
                  {service.metrics.requests ? `${service.metrics.requests} req/s` : 
                   service.metrics.connections ? `${service.metrics.connections} conn` : 
                   'Active'}
                </span>
              </div>
              <Zap className="w-3 h-3 text-success" />
            </div>
          </motion.div>
        ))}
      </div>

      {/* Recent Events */}
      <AnimatedCard className="p-6">
        <h3 className="text-lg font-semibold text-text-primary mb-4">Recent Events</h3>
        <div className="space-y-3">
          {[
            { time: '2 min ago', event: 'Auto-scaling triggered on Railway', type: 'info' },
            { time: '15 min ago', event: 'Cache hit rate dropped to 85%', type: 'warning' },
            { time: '1 hour ago', event: 'Model v2.1.0 deployed to production', type: 'success' },
            { time: '3 hours ago', event: 'Database connection pool expanded', type: 'info' },
            { time: '5 hours ago', event: 'GitHub Actions workflow completed', type: 'success' },
          ].map((event, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
              className="flex items-center justify-between p-3 bg-card-hover rounded-lg"
            >
              <div className="flex items-center gap-3">
                <div className={cn(
                  'w-2 h-2 rounded-full',
                  event.type === 'success' ? 'bg-success' :
                  event.type === 'warning' ? 'bg-warning' :
                  event.type === 'error' ? 'bg-danger' : 'bg-primary'
                )} />
                <span className="text-sm text-text-primary">{event.event}</span>
              </div>
              <span className="text-xs text-text-secondary">{event.time}</span>
            </motion.div>
          ))}
        </div>
      </AnimatedCard>
    </div>
  )
}