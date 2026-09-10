import DemoDataBanner from '@/components/DemoDataBanner'

const items = [{ title: 'Implementation', body: 'Experiment-management and multi-model training code is present.' }, { title: 'Recorded results', body: 'Unavailable: no committed run links dataset, split, model identity, metrics, and prediction rows.' }, { title: 'Methodological limit', body: 'The Bayesian-ridge meta-learner currently receives in-sample base predictions and is therefore experimental, not validated out-of-fold stacking.' }]

export default function Page() {
  return <main className="min-h-screen bg-background p-6"><div className="max-w-5xl mx-auto"><DemoDataBanner /><h1 className="text-4xl font-bold text-text-primary mt-8 mb-3">Model experiments</h1><p className="text-text-secondary mb-8">Experiment evidence</p><div className="grid grid-cols-1 md:grid-cols-3 gap-6">{items.map(item => <section key={item.title} className="glass-card p-6"><h2 className="text-lg font-semibold text-text-primary mb-3">{item.title}</h2><p className="text-sm text-text-secondary">{item.body}</p></section>)}</div></div></main>
}
