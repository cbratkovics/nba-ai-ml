import DemoDataBanner from '@/components/DemoDataBanner'

const items = [{ title: 'Serving artifacts', body: 'Unavailable in this checkout; existing external pickle/joblib provenance is unknown.' }, { title: 'Synthetic smoke artifacts', body: 'Generated only into an explicitly isolated directory with hashes and source_kind=synthetic_fixture.' }, { title: 'Selection boundary', body: 'The API fails when a trusted artifact is missing instead of generating an unlabeled fallback model.' }]

export default function Page() {
  return <main className="min-h-screen bg-background p-6"><div className="max-w-5xl mx-auto"><DemoDataBanner /><h1 className="text-4xl font-bold text-text-primary mt-8 mb-3">Artifact evidence</h1><p className="text-text-secondary mb-8">Saved model status</p><div className="grid grid-cols-1 md:grid-cols-3 gap-6">{items.map(item => <section key={item.title} className="glass-card p-6"><h2 className="text-lg font-semibold text-text-primary mb-3">{item.title}</h2><p className="text-sm text-text-secondary">{item.body}</p></section>)}</div></div></main>
}
