'use client'

import Link from 'next/link'
import { motion } from 'framer-motion'
import { Brain, BarChart3, Github, FlaskConical, Database } from 'lucide-react'
import PredictionInterface from '@/components/dashboard/PredictionInterface'

export default function Home() {
  return (
    <main className="min-h-screen bg-background relative">
      <nav className="glass-card border-b border-white/10 sticky top-0 z-50" aria-label="Primary navigation">
        <div className="container mx-auto px-6 py-4 flex items-center justify-between">
          <Link href="/" className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-r from-primary to-secondary rounded-lg flex items-center justify-center">
              <Brain className="w-6 h-6 text-white" aria-hidden="true" />
            </div>
            <div><h1 className="text-xl font-bold text-text-primary">NBA Performance Prediction System</h1><p className="text-xs text-text-secondary">Applied data-science portfolio</p></div>
          </Link>
          <div className="flex items-center gap-5 text-sm">
            <Link href="/dashboard" className="text-text-secondary hover:text-text-primary">Evaluation</Link>
            <Link href="/models" className="text-text-secondary hover:text-text-primary">Models</Link>
            <a href="https://github.com/cbratkovics/nba-ai-ml" target="_blank" rel="noopener noreferrer" className="text-text-secondary hover:text-text-primary" aria-label="View source on GitHub"><Github className="w-5 h-5" /></a>
          </div>
        </div>
      </nav>

      <section className="container mx-auto px-6 py-16">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="text-center max-w-4xl mx-auto">
          <h2 className="text-4xl md:text-5xl font-bold mb-5"><span className="gradient-text">NBA player-performance forecasting</span></h2>
          <p className="text-xl text-text-secondary">Explore NBA player-performance forecasts, feature engineering, and model evaluation through an interactive portfolio demonstration.</p>
          <div role="note" className="mt-8 glass-card border border-primary/40 p-4 text-left text-sm text-text-secondary">
            <strong className="text-text-primary">Portfolio demonstration.</strong> Example predictions and charts use deterministic synthetic fixtures unless a panel identifies a measured evaluation source. They do not represent live production performance.
          </div>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-12">
          {[
            { icon: Database, title: 'Feature engineering', body: 'Rolling and contextual feature code for historical player game logs.' },
            { icon: FlaskConical, title: 'Model experiments', body: 'Random Forest, XGBoost, LightGBM, and experimental stacking implementations.' },
            { icon: BarChart3, title: 'Evidence first', body: 'Measured results appear only when a reproducible evaluation artifact is available.' },
          ].map(({ icon: Icon, title, body }) => <div key={title} className="glass-card p-6"><Icon className="w-7 h-7 text-primary mb-4" aria-hidden="true"/><h3 className="font-semibold text-text-primary mb-2">{title}</h3><p className="text-sm text-text-secondary">{body}</p></div>)}
        </div>
      </section>

      <section className="container mx-auto px-6 py-10"><PredictionInterface /></section>
      <footer className="glass-card border-t border-white/10 mt-16"><div className="container mx-auto px-6 py-8 text-sm text-text-secondary">NBA Performance Prediction System · source-code portfolio demonstration</div></footer>
    </main>
  )
}
