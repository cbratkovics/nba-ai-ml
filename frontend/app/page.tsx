'use client'

import Link from 'next/link'
import { motion } from 'framer-motion'
import {
  ArrowRight,
  BarChart3,
  Braces,
  Check,
  ChevronDown,
  Database,
  Github,
  Layers3,
  LockKeyhole,
  Orbit,
} from 'lucide-react'
import PredictionInterface from '@/components/dashboard/PredictionInterface'

const capabilities = [
  {
    number: '01',
    icon: Database,
    title: 'Build context',
    body: 'Rolling form, matchup context, rest, and historical player-game features are assembled from database or NBA client records.',
    meta: 'Historical feature pipeline',
  },
  {
    number: '02',
    icon: Layers3,
    title: 'Run experiments',
    body: 'The repository includes Random Forest and multi-library ensemble experiments, with saved-artifact inference kept as a distinct path.',
    meta: 'Multiple model families',
  },
  {
    number: '03',
    icon: BarChart3,
    title: 'Demand evidence',
    body: 'Quality claims require an identified dataset, chronological split, metric definition, artifact hash, and reproducible command.',
    meta: 'Evidence-gated reporting',
  },
]

const contract = [
  ['Input', 'Player + game context'],
  ['Targets', 'PTS · REB · AST'],
  ['Serving', 'Trusted .pkl artifacts'],
  ['Fallback', 'Explicit error, never a hidden guess'],
]

export default function Home() {
  return (
    <main className="min-h-screen relative overflow-hidden">
      <nav className="site-nav" aria-label="Primary navigation">
        <div className="page-shell flex h-20 items-center justify-between">
          <Link href="/" className="brand-lockup" aria-label="Court Vision home">
            <span className="brand-mark"><Orbit className="h-5 w-5" aria-hidden="true" /></span>
            <span><strong>COURT VISION</strong><small>NBA · ML LAB</small></span>
          </Link>
          <div className="hidden items-center gap-8 md:flex text-sm">
            <Link href="#system" className="nav-link">System</Link>
            <Link href="/dashboard" className="nav-link">Evaluation</Link>
            <Link href="/models" className="nav-link">Models</Link>
            <a href="https://github.com/cbratkovics/nba-ai-ml" target="_blank" rel="noopener noreferrer" className="source-link">
              <Github className="h-4 w-4" aria-hidden="true" /> Source
            </a>
          </div>
          <a href="https://github.com/cbratkovics/nba-ai-ml" target="_blank" rel="noopener noreferrer" className="md:hidden text-white" aria-label="View source on GitHub"><Github className="h-5 w-5" /></a>
        </div>
      </nav>

      <section className="hero-court">
        <div className="court-lines" aria-hidden="true" />
        <div className="page-shell relative z-10 grid gap-14 py-20 lg:grid-cols-[1.15fr_.85fr] lg:items-end lg:py-28">
          <motion.div initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: .55 }}>
            <div className="eyebrow"><span /> OPEN-SOURCE APPLIED ML</div>
            <h1 className="hero-title">See the game<br /><em>before the box score.</em></h1>
            <p className="hero-copy">A rigorous workspace for exploring NBA player-performance forecasting—from historical features to artifact-backed inference.</p>
            <div className="mt-9 flex flex-wrap gap-3">
              <Link href="#demo" className="primary-cta">Explore the interface <ArrowRight className="h-4 w-4" /></Link>
              <Link href="/dashboard" className="secondary-cta">Review evaluation</Link>
            </div>
            <p className="mt-5 flex items-center gap-2 text-xs text-slate-400"><LockKeyhole className="h-3.5 w-3.5 text-cyan-300" /> Portfolio demonstration: no invented production metrics. Fixtures are labeled at the point of use.</p>
          </motion.div>

          <motion.aside initial={{ opacity: 0, x: 18 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: .6, delay: .12 }} className="scouting-card" aria-label="Serving contract summary">
            <div className="flex items-center justify-between border-b border-white/10 pb-5">
              <div><p className="data-label">SERVING CONTRACT</p><h2 className="mt-1 text-lg font-semibold">Forecast, with provenance</h2></div>
              <span className="status-pill"><span /> CODE PATH PRESENT</span>
            </div>
            <dl className="divide-y divide-white/10">
              {contract.map(([term, value]) => <div key={term} className="flex items-center justify-between gap-4 py-4"><dt className="text-sm text-slate-400">{term}</dt><dd className="text-right text-sm font-medium text-slate-100">{value}</dd></div>)}
            </dl>
            <div className="mt-2 rounded-xl bg-cyan-300/[.06] p-4 text-xs leading-5 text-cyan-100/75">
              This checkout contains no qualifying benchmark artifact. The evaluation view remains unavailable until a reproducible run is committed.
            </div>
          </motion.aside>
        </div>
        <a href="#system" className="scroll-cue" aria-label="Scroll to system overview">SCROLL TO READ <ChevronDown className="h-4 w-4" /></a>
      </section>

      <section id="system" className="page-shell py-24">
        <div className="section-heading">
          <div><p className="data-label text-cyan-300">THE SYSTEM</p><h2>From raw game logs to<br />an inspectable forecast.</h2></div>
          <p>Not a black-box accuracy claim. Each layer is represented in source, while unverified operational and quality claims stay explicitly out of bounds.</p>
        </div>
        <div className="mt-12 grid gap-px overflow-hidden rounded-2xl border border-white/10 bg-white/10 lg:grid-cols-3">
          {capabilities.map(({ number, icon: Icon, title, body, meta }) => (
            <article key={title} className="capability-card">
              <div className="flex items-center justify-between"><span className="step-number">{number}</span><Icon className="h-6 w-6 text-cyan-300" aria-hidden="true" /></div>
              <h3>{title}</h3><p>{body}</p>
              <div className="capability-meta"><Check className="h-3.5 w-3.5" /> {meta}</div>
            </article>
          ))}
        </div>
      </section>

      <section className="evidence-band">
        <div className="page-shell grid gap-10 py-14 md:grid-cols-[auto_1fr_auto] md:items-center">
          <Braces className="h-9 w-9 text-orange-400" aria-hidden="true" />
          <div><p className="data-label text-orange-300">HONEST BY DESIGN</p><h2 className="mt-2 text-2xl font-semibold">Missing artifacts fail loudly.</h2><p className="mt-2 max-w-2xl text-sm leading-6 text-slate-400">The serving route does not silently replace a missing trusted model with synthetic output. Demo fixtures live in the presentation layer and say exactly what they are.</p></div>
          <Link href="/dashboard" className="text-link">Read the evidence contract <ArrowRight className="h-4 w-4" /></Link>
        </div>
      </section>

      <section id="demo" className="page-shell py-24">
        <div className="mb-10 flex flex-wrap items-end justify-between gap-6">
          <div><p className="data-label text-cyan-300">INTERFACE PREVIEW</p><h2 className="mt-3 text-3xl font-semibold md:text-4xl">Forecast presentation, explored.</h2></div>
          <span className="fixture-badge"><span /> SYNTHETIC FIXTURE · NOT MODEL OUTPUT</span>
        </div>
        <PredictionInterface />
      </section>

      <footer className="border-t border-white/10">
        <div className="page-shell flex flex-col gap-5 py-10 text-sm text-slate-500 md:flex-row md:items-center md:justify-between">
          <div className="brand-lockup opacity-80"><span className="brand-mark"><Orbit className="h-5 w-5" /></span><span><strong>COURT VISION</strong><small>NBA · ML LAB</small></span></div>
          <p>Applied data science, documented without fictional results.</p>
          <div className="flex gap-6"><Link href="/models" className="hover:text-white">Models</Link><Link href="/api-docs" className="hover:text-white">API</Link><a href="https://github.com/cbratkovics/nba-ai-ml" className="hover:text-white">GitHub</a></div>
        </div>
      </footer>
    </main>
  )
}
