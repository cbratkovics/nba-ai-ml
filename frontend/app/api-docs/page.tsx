import DemoDataBanner from '@/components/DemoDataBanner'

const items = [{ title: 'POST /v1/predict', body: 'Uses database-backed features and a trusted saved artifact. Missing dependencies return an error, not synthetic success.' }, { title: 'GET /health', body: 'Liveness check only; it does not establish model readiness or historical uptime.' }, { title: 'GET /demo/metrics', body: 'Returns an explicit unavailable evidence state. Interactive OpenAPI documentation is exposed by a running backend at /docs.' }]

export default function Page() {
  return <main className="min-h-screen bg-background p-6"><div className="max-w-5xl mx-auto"><DemoDataBanner /><h1 className="text-4xl font-bold text-text-primary mt-8 mb-3">API contract</h1><p className="text-text-secondary mb-8">Demonstration endpoints</p><div className="grid grid-cols-1 md:grid-cols-3 gap-6">{items.map(item => <section key={item.title} className="glass-card p-6"><h2 className="text-lg font-semibold text-text-primary mb-3">{item.title}</h2><p className="text-sm text-text-secondary">{item.body}</p></section>)}</div></div></main>
}
