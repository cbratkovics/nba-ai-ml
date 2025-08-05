'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import MetricsGrid from '@/components/dashboard/MetricsGrid'
import PredictionInterface from '@/components/dashboard/PredictionInterface'
import { Toaster } from 'react-hot-toast'
import { 
  Brain, 
  GitBranch, 
  Database, 
  Cpu, 
  Activity, 
  BarChart3,
  Terminal,
  Github,
  Star,
  GitFork,
  ExternalLink,
  ChevronRight
} from 'lucide-react'
import Link from 'next/link'

export default function Home() {
  const [githubStars, setGithubStars] = useState(0)
  const [githubForks, setGithubForks] = useState(0)

  useEffect(() => {
    // Fetch GitHub stats
    fetch('https://api.github.com/repos/cbratkovics/nba-ai-ml')
      .then(res => res.json())
      .then(data => {
        setGithubStars(data.stargazers_count || 0)
        setGithubForks(data.forks_count || 0)
      })
      .catch(console.error)
  }, [])

  const navItems = [
    { label: 'Dashboard', href: '/dashboard', icon: BarChart3 },
    { label: 'Experiments', href: '/experiments', icon: Brain },
    { label: 'Models', href: '/models', icon: GitBranch },
    { label: 'API Analytics', href: '/api-analytics', icon: Activity },
  ]

  const techStack = [
    { name: 'FastAPI', color: 'from-green-400 to-emerald-600' },
    { name: 'XGBoost', color: 'from-blue-400 to-blue-600' },
    { name: 'LightGBM', color: 'from-purple-400 to-purple-600' },
    { name: 'Redis', color: 'from-red-400 to-red-600' },
    { name: 'PostgreSQL', color: 'from-cyan-400 to-blue-600' },
    { name: 'Railway', color: 'from-pink-400 to-purple-600' },
    { name: 'Next.js', color: 'from-gray-400 to-gray-600' },
    { name: 'TypeScript', color: 'from-blue-400 to-blue-600' },
  ]

  return (
    <main className="min-h-screen bg-background relative">
      <Toaster position="top-right" />
      
      {/* Navigation */}
      <nav className="glass-card border-b border-white/10 sticky top-0 z-50">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-8">
              <motion.div 
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                className="flex items-center gap-3"
              >
                <div className="w-10 h-10 bg-gradient-to-r from-primary to-secondary rounded-lg flex items-center justify-center">
                  <Brain className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h1 className="text-xl font-bold text-text-primary">NBA ML Platform</h1>
                  <p className="text-xs text-text-secondary">Enterprise MLOps Dashboard</p>
                </div>
              </motion.div>

              <div className="hidden md:flex items-center gap-6">
                {navItems.map((item, index) => (
                  <motion.div
                    key={item.label}
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.1 }}
                  >
                    <Link 
                      href={item.href}
                      className="flex items-center gap-2 text-text-secondary hover:text-text-primary transition-colors"
                    >
                      <item.icon className="w-4 h-4" />
                      <span className="text-sm font-medium">{item.label}</span>
                    </Link>
                  </motion.div>
                ))}
              </div>
            </div>

            <div className="flex items-center gap-4">
              <motion.div 
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                className="flex items-center gap-4"
              >
                <a 
                  href="https://github.com/cbratkovics/nba-ai-ml"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-2 text-text-secondary hover:text-text-primary transition-colors"
                >
                  <Github className="w-5 h-5" />
                  <div className="flex items-center gap-3 text-sm">
                    <span className="flex items-center gap-1">
                      <Star className="w-3 h-3" />
                      {githubStars}
                    </span>
                    <span className="flex items-center gap-1">
                      <GitFork className="w-3 h-3" />
                      {githubForks}
                    </span>
                  </div>
                </a>
                <button className="neon-button text-sm">
                  <Terminal className="w-4 h-4 mr-2" />
                  API Docs
                </button>
              </motion.div>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="container mx-auto px-6 py-12">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <h2 className="text-5xl font-bold mb-4">
            <span className="gradient-text">Production ML Platform</span>
          </h2>
          <p className="text-xl text-text-secondary max-w-3xl mx-auto">
            Enterprise-grade machine learning infrastructure serving 1.2M+ predictions 
            with 94.2% accuracy and sub-100ms latency
          </p>
        </motion.div>

        {/* Metrics Grid */}
        <MetricsGrid />

        {/* Quick Actions */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-12"
        >
          <div className="glass-card p-6 hover:scale-105 transition-transform cursor-pointer">
            <div className="flex items-center justify-between mb-4">
              <Database className="w-8 h-8 text-secondary" />
              <ChevronRight className="w-5 h-5 text-text-secondary" />
            </div>
            <h3 className="text-lg font-semibold text-text-primary mb-2">Feature Store</h3>
            <p className="text-sm text-text-secondary">
              50+ engineered features with real-time computation
            </p>
          </div>

          <div className="glass-card p-6 hover:scale-105 transition-transform cursor-pointer">
            <div className="flex items-center justify-between mb-4">
              <Cpu className="w-8 h-8 text-primary" />
              <ChevronRight className="w-5 h-5 text-text-secondary" />
            </div>
            <h3 className="text-lg font-semibold text-text-primary mb-2">Model Registry</h3>
            <p className="text-sm text-text-secondary">
              Version control for ML models with automatic rollback
            </p>
          </div>

          <div className="glass-card p-6 hover:scale-105 transition-transform cursor-pointer">
            <div className="flex items-center justify-between mb-4">
              <Activity className="w-8 h-8 text-success" />
              <ChevronRight className="w-5 h-5 text-text-secondary" />
            </div>
            <h3 className="text-lg font-semibold text-text-primary mb-2">API Gateway</h3>
            <p className="text-sm text-text-secondary">
              RESTful API with automatic documentation and monitoring
            </p>
          </div>
        </motion.div>
      </section>

      {/* Prediction Interface */}
      <section className="container mx-auto px-6 py-12">
        <PredictionInterface />
      </section>

      {/* Tech Stack */}
      <section className="container mx-auto px-6 py-12">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
        >
          <h3 className="text-2xl font-bold text-text-primary mb-8 text-center">
            Powered By Industry-Leading Technology
          </h3>
          <div className="flex flex-wrap justify-center gap-4">
            {techStack.map((tech, index) => (
              <motion.div
                key={tech.name}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.5 + index * 0.05 }}
                whileHover={{ scale: 1.1 }}
                className={`px-6 py-3 rounded-full bg-gradient-to-r ${tech.color} text-white font-medium text-sm shadow-lg`}
              >
                {tech.name}
              </motion.div>
            ))}
          </div>
        </motion.div>
      </section>

      {/* Footer */}
      <footer className="glass-card border-t border-white/10 mt-20">
        <div className="container mx-auto px-6 py-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-6">
              <span className="text-sm text-text-secondary">
                © 2024 NBA ML Platform
              </span>
              <div className="flex items-center gap-4">
                <a href="#" className="text-sm text-text-secondary hover:text-primary transition-colors">
                  Documentation
                </a>
                <a href="#" className="text-sm text-text-secondary hover:text-primary transition-colors">
                  API Reference
                </a>
                <a href="#" className="text-sm text-text-secondary hover:text-primary transition-colors">
                  GitHub
                </a>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className="pulse-dot" />
              <span className="text-sm text-text-secondary">All systems operational</span>
            </div>
          </div>
        </div>
      </footer>
    </main>
  )
}