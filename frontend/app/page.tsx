import { SignInButton, SignedIn, SignedOut, UserButton } from '@clerk/nextjs'
import PredictionForm from '@/components/PredictionForm'

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-b from-gray-900 to-gray-800 text-white">
      <nav className="p-4 flex justify-between items-center border-b border-gray-700">
        <h1 className="text-2xl font-bold">NBA ML Predictions</h1>
        <SignedOut>
          <SignInButton className="bg-blue-600 px-4 py-2 rounded hover:bg-blue-700" />
        </SignedOut>
        <SignedIn>
          <UserButton />
        </SignedIn>
      </nav>
      
      <div className="container mx-auto px-4 py-8">
        <SignedOut>
          <div className="text-center py-20">
            <h2 className="text-4xl font-bold mb-4">Professional NBA Predictions</h2>
            <p className="text-xl mb-8">94%+ accuracy powered by advanced ML</p>
            <SignInButton className="bg-blue-600 px-6 py-3 rounded-lg text-lg hover:bg-blue-700" />
          </div>
        </SignedOut>
        
        <SignedIn>
          <PredictionForm />
        </SignedIn>
      </div>
    </main>
  )
}