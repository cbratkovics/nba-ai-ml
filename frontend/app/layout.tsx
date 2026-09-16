import './globals.css'
import { Inter } from 'next/font/google'
import Link from 'next/link'
import GradientBackground from '@/components/ui/GradientBackground'

const inter = Inter({ subsets: ['latin'] })

export const metadata = {
  title: 'NBA Stat Predictor',
  description:
    'Batch pipeline predicting NBA player points, rebounds, and assists from pre-game history.',
}

const NAV = [
  { href: '/', label: 'Overview' },
  { href: '/predictions', label: 'Predictions' },
  { href: '/replay', label: 'Replay' },
  { href: '/decisions', label: 'Decisions' },
  { href: '/brief', label: 'Brief' },
]

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} bg-background min-h-screen`}>
        <GradientBackground />
        <header className="glass-card border-b border-white/10 sticky top-0 z-50 rounded-none">
          <nav className="container mx-auto flex items-center justify-between px-6 py-4">
            <Link href="/" className="text-lg font-semibold text-text-primary">
              NBA Stat Predictor
            </Link>
            <ul className="flex items-center gap-6 text-sm">
              {NAV.map((item) => (
                <li key={item.href}>
                  <Link
                    href={item.href}
                    className="text-text-secondary transition-colors hover:text-text-primary"
                  >
                    {item.label}
                  </Link>
                </li>
              ))}
            </ul>
          </nav>
        </header>
        <main className="container mx-auto px-6 py-10">{children}</main>
        <footer className="container mx-auto px-6 py-8 text-xs text-text-secondary">
          Personal, non-commercial project. Data derived from NBA.com box scores via a
          CC0 Kaggle dump; see the reconciliation notes for provenance.
        </footer>
      </body>
    </html>
  )
}
