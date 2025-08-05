// Vercel Edge Function for ultra-low latency predictions
import type { VercelRequest, VercelResponse } from '@vercel/node'

export const config = {
  runtime: 'edge',
  regions: ['iad1', 'sfo1'], // US East and West
}

// Redis client for edge
const REDIS_URL = process.env.REDIS_URL || ''
const RAILWAY_API_URL = process.env.RAILWAY_API_URL || ''

interface PredictionRequest {
  player_id: string
  game_date?: string
  opponent_team?: string
  include_confidence?: boolean
  include_explanation?: boolean
}

export default async function handler(req: Request) {
  // Handle CORS
  if (req.method === 'OPTIONS') {
    return new Response(null, {
      status: 200,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization',
      },
    })
  }

  if (req.method !== 'POST') {
    return new Response('Method not allowed', { status: 405 })
  }

  try {
    const body: PredictionRequest = await req.json()
    
    // Generate cache key
    const cacheKey = `pred:${body.player_id}:${body.game_date || 'today'}:${body.opponent_team || 'default'}`
    
    // Check Redis cache at edge
    const cached = await getRedisCached(cacheKey)
    if (cached) {
      return new Response(cached, {
        headers: {
          'Content-Type': 'application/json',
          'X-Cache': 'HIT',
          'Cache-Control': 'public, max-age=300', // 5 minutes
        },
      })
    }
    
    // Forward to Railway API
    const response = await fetch(`${RAILWAY_API_URL}/v1/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Forwarded-For': req.headers.get('X-Forwarded-For') || '',
      },
      body: JSON.stringify(body),
    })
    
    if (!response.ok) {
      throw new Error(`API responded with ${response.status}`)
    }
    
    const data = await response.json()
    
    // Cache successful predictions
    if (response.status === 200) {
      await setRedisCache(cacheKey, JSON.stringify(data), 300) // 5 minutes
    }
    
    return new Response(JSON.stringify(data), {
      headers: {
        'Content-Type': 'application/json',
        'X-Cache': 'MISS',
        'Cache-Control': 'public, max-age=300',
      },
    })
    
  } catch (error) {
    console.error('Edge function error:', error)
    
    // Fallback response
    return new Response(
      JSON.stringify({
        error: 'Prediction service temporarily unavailable',
        fallback: true,
      }),
      {
        status: 503,
        headers: {
          'Content-Type': 'application/json',
          'Retry-After': '60',
        },
      }
    )
  }
}

// Simplified Redis operations for edge runtime
async function getRedisCached(key: string): Promise<string | null> {
  if (!REDIS_URL) return null
  
  try {
    // In a real implementation, you'd use an edge-compatible Redis client
    // For now, we'll use HTTP-based Redis operations
    const response = await fetch(`${REDIS_URL}/get/${key}`, {
      headers: {
        'Authorization': `Bearer ${process.env.REDIS_TOKEN}`,
      },
    })
    
    if (response.ok) {
      return await response.text()
    }
  } catch (error) {
    console.error('Redis get error:', error)
  }
  
  return null
}

async function setRedisCache(key: string, value: string, ttl: number): Promise<void> {
  if (!REDIS_URL) return
  
  try {
    await fetch(`${REDIS_URL}/set/${key}`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${process.env.REDIS_TOKEN}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ value, ttl }),
    })
  } catch (error) {
    console.error('Redis set error:', error)
  }
}