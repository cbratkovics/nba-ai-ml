import './globals.css'
import GradientBackground from '@/components/ui/GradientBackground'


export const metadata = {
  title: 'Court Vision — NBA ML Lab',
  description: 'Explore NBA player-performance forecasts, feature engineering, and model evaluation in an applied data-science portfolio demonstration.',
  openGraph: {
    title: 'NBA Performance Prediction System',
    description: 'An applied data-science portfolio for NBA forecasting and evaluation.',
    type: 'website',
  },
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return <html lang="en" className="dark"><body className="bg-background min-h-screen antialiased"><GradientBackground />{children}</body></html>
}
