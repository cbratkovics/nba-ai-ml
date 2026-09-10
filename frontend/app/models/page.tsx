import DemoDataBanner from '@/components/DemoDataBanner'

const items = [{ title: 'Random Forest serving convention', body: 'The loader supports target-specific Random Forest artifacts, but no provenance-backed binary is committed.' }, { title: 'Experimental ensemble', body: 'NBAEnsemble implements Random Forest, XGBoost, LightGBM and a Bayesian ridge meta-learner. Its current fit is not out-of-fold stacking.' }, { title: 'Optional paths', body: 'Neural-network, SHAP, MLflow, and tuning code is retained as optional or experimental implementation.' }]

export default function Page() {
  return <main className="min-h-screen bg-background p-6"><div className="max-w-5xl mx-auto"><DemoDataBanner /><h1 className="text-4xl font-bold text-text-primary mt-8 mb-3">Model implementations</h1><p className="text-text-secondary mb-8">Implemented model families</p><div className="grid grid-cols-1 md:grid-cols-3 gap-6">{items.map(item => <section key={item.title} className="glass-card p-6"><h2 className="text-lg font-semibold text-text-primary mb-3">{item.title}</h2><p className="text-sm text-text-secondary">{item.body}</p></section>)}</div></div></main>
}
