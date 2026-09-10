import Link from 'next/link'
import DemoDataBanner from '@/components/DemoDataBanner'

const targets = ['Points', 'Rebounds', 'Assists']

export default function DashboardPage() {
  return <main className="min-h-screen bg-background p-6"><div className="max-w-5xl mx-auto">
    <DemoDataBanner />
    <h1 className="text-4xl font-bold text-text-primary mt-8 mb-3">Model evaluation</h1>
    <p className="text-text-secondary mb-8">No eligible recorded evaluation artifact is committed. Run an evaluation on a defined dataset and untouched chronological test split before publishing model quality.</p>
    <section className="glass-card p-6 mb-8" aria-labelledby="status-heading"><h2 id="status-heading" className="text-xl font-semibold text-text-primary mb-4">Recorded evaluation status</h2><dl className="grid grid-cols-1 md:grid-cols-3 gap-4">{targets.map(target => <div key={target} className="bg-card-hover rounded-lg p-4"><dt className="text-sm text-text-secondary">{target}</dt><dd className="text-lg font-semibold text-text-primary">Unavailable</dd><p className="text-xs text-text-secondary mt-2">MAE, RMSE, and R² require a reproducible run.</p></div>)}</dl></section>
    <section className="glass-card p-6"><h2 className="text-xl font-semibold text-text-primary mb-3">What qualifies as evidence</h2><p className="text-text-secondary">A result must identify its input hash, chronological split, feature contract, model artifact, metric definition, and execution record. Synthetic smoke checks validate plumbing only.</p><Link href="https://github.com/cbratkovics/nba-ai-ml/blob/master/docs/PORTFOLIO_EVIDENCE.md" className="inline-block mt-4 text-primary hover:underline">Read the evidence and model-path map</Link></section>
  </div></main>
}
