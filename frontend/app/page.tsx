export default function Home() {
  return (
    <main style={{ padding: '2rem', textAlign: 'center' }}>
      <h1>NBA ML Predictions</h1>
      <p>API Status: Connected to Railway</p>
      <a href={`${process.env.NEXT_PUBLIC_API_URL || 'https://nba-ai-ml-production.up.railway.app'}/docs`}>
        View API Documentation
      </a>
    </main>
  )
}
