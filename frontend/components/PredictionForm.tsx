'use client'
import { useState } from 'react'
import { useAuth } from '@clerk/nextjs'

export default function PredictionForm() {
  const { getToken } = useAuth()
  const [playerId, setPlayerId] = useState('')
  const [gameDate, setGameDate] = useState('')
  const [opponent, setOpponent] = useState('')
  const [prediction, setPrediction] = useState<any>(null)
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    
    try {
      const token = await getToken()
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/v1/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          player_id: playerId,
          game_date: gameDate,
          opponent_team: opponent
        })
      })
      
      const data = await response.json()
      setPrediction(data)
    } catch (error) {
      console.error('Prediction failed:', error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-2xl mx-auto">
      <form onSubmit={handleSubmit} className="bg-gray-800 p-6 rounded-lg shadow-lg">
        <h2 className="text-2xl font-bold mb-6">Get Player Prediction</h2>
        
        <div className="mb-4">
          <label className="block mb-2">Player ID</label>
          <input
            type="text"
            value={playerId}
            onChange={(e) => setPlayerId(e.target.value)}
            className="w-full p-2 rounded bg-gray-700 text-white"
            placeholder="e.g., 203999 (Jokic)"
            required
          />
        </div>
        
        <div className="mb-4">
          <label className="block mb-2">Game Date</label>
          <input
            type="date"
            value={gameDate}
            onChange={(e) => setGameDate(e.target.value)}
            className="w-full p-2 rounded bg-gray-700 text-white"
            required
          />
        </div>
        
        <div className="mb-6">
          <label className="block mb-2">Opponent Team</label>
          <input
            type="text"
            value={opponent}
            onChange={(e) => setOpponent(e.target.value)}
            className="w-full p-2 rounded bg-gray-700 text-white"
            placeholder="e.g., LAL"
            required
          />
        </div>
        
        <button
          type="submit"
          disabled={loading}
          className="w-full bg-blue-600 py-3 rounded font-semibold hover:bg-blue-700 disabled:opacity-50"
        >
          {loading ? 'Predicting...' : 'Get Prediction'}
        </button>
      </form>
      
      {prediction && (
        <div className="mt-8 bg-gray-800 p-6 rounded-lg">
          <h3 className="text-xl font-bold mb-4">Prediction Results</h3>
          <div className="grid grid-cols-3 gap-4">
            <div className="text-center">
              <p className="text-3xl font-bold text-blue-400">{prediction.points?.toFixed(1)}</p>
              <p className="text-gray-400">Points</p>
            </div>
            <div className="text-center">
              <p className="text-3xl font-bold text-green-400">{prediction.rebounds?.toFixed(1)}</p>
              <p className="text-gray-400">Rebounds</p>
            </div>
            <div className="text-center">
              <p className="text-3xl font-bold text-purple-400">{prediction.assists?.toFixed(1)}</p>
              <p className="text-gray-400">Assists</p>
            </div>
          </div>
          <p className="mt-4 text-center text-gray-400">
            Confidence: {(prediction.confidence * 100).toFixed(1)}%
          </p>
        </div>
      )}
    </div>
  )
}