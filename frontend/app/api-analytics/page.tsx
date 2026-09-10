import DemoDataBanner from '@/components/DemoDataBanner'

const items = [{ title: 'Liveness', body: 'A successful /health response shows only that one request was served.' }, { title: 'Readiness', body: 'Database, feature data, and a trusted model artifact are separate requirements.' }, { title: 'History', body: 'Traffic, uptime, and latency history are unavailable because no eligible telemetry artifact is committed.' }]

export default function Page() {
  return <main className="min-h-screen bg-background p-6"><div className="max-w-5xl mx-auto"><DemoDataBanner /><h1 className="text-4xl font-bold text-text-primary mt-8 mb-3">API evidence</h1><p className="text-text-secondary mb-8">Operational measurements</p><div className="grid grid-cols-1 md:grid-cols-3 gap-6">{items.map(item => <section key={item.title} className="glass-card p-6"><h2 className="text-lg font-semibold text-text-primary mb-3">{item.title}</h2><p className="text-sm text-text-secondary">{item.body}</p></section>)}</div></div></main>
}
