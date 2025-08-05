import './globals.css'
import { Inter } from 'next/font/google'
import GradientBackground from '@/components/ui/GradientBackground'

const inter = Inter({ subsets: ['latin'] })

export const metadata = {
  title: 'NBA ML Platform | Enterprise MLOps Dashboard',
  description: 'Production-grade machine learning platform for NBA predictions with 94.2% accuracy',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} bg-background min-h-screen`}>
        <GradientBackground />
        {children}
      </body>
    </html>
  )
}