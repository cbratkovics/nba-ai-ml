// Supabase Edge Function for real-time anomaly detection
import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

interface Prediction {
  id: number
  player_id: string
  player_name: string
  points_predicted: number
  rebounds_predicted: number
  assists_predicted: number
  confidence: number
  model_version: string
  created_at: string
}

interface AnomalyResult {
  anomaly_detected: boolean
  anomaly_type?: string
  severity?: string
  details?: any
}

serve(async (req) => {
  // Handle CORS
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    // Get Supabase client
    const supabaseUrl = Deno.env.get('SUPABASE_URL') ?? ''
    const supabaseKey = Deno.env.get('SUPABASE_ANON_KEY') ?? ''
    const supabase = createClient(supabaseUrl, supabaseKey)

    const { prediction } = await req.json()

    // Real-time anomaly detection
    const anomalies = await detectAnomalies(prediction, supabase)

    if (anomalies.anomaly_detected) {
      // Create alert in database
      await createAnomalyAlert(prediction, anomalies, supabase)
      
      // Send webhook notification for critical anomalies
      if (anomalies.severity === 'critical') {
        await sendWebhookNotification(anomalies)
      }
    }

    return new Response(
      JSON.stringify({ 
        success: true, 
        anomalies 
      }),
      { 
        headers: { 
          ...corsHeaders, 
          'Content-Type': 'application/json' 
        } 
      }
    )

  } catch (error) {
    console.error('Error in anomaly detection:', error)
    return new Response(
      JSON.stringify({ 
        success: false, 
        error: error.message 
      }),
      { 
        status: 500,
        headers: { 
          ...corsHeaders, 
          'Content-Type': 'application/json' 
        }
      }
    )
  }
})

async function detectAnomalies(
  prediction: Prediction, 
  supabase: any
): Promise<AnomalyResult> {
  const anomalies: string[] = []
  let severity = 'low'

  // Get statistical baselines
  const { data: stats } = await supabase
    .from('prediction_metrics_hourly')
    .select('avg_points_pred:avg(points_predicted), std_points_pred:stddev(points_predicted)')
    .single()

  // Check for impossible values
  if (prediction.points_predicted < 0 || prediction.points_predicted > 60) {
    anomalies.push('impossible_value')
    severity = 'critical'
  }

  // Check for statistical outliers (3 standard deviations)
  if (stats && stats.std_points_pred > 0) {
    const zScore = Math.abs(prediction.points_predicted - stats.avg_points_pred) / stats.std_points_pred
    if (zScore > 3) {
      anomalies.push('statistical_outlier')
      severity = severity === 'critical' ? 'critical' : 'high'
    }
  }

  // Check confidence levels
  if (prediction.confidence < 0.3) {
    anomalies.push('low_confidence')
    severity = severity === 'low' ? 'medium' : severity
  } else if (prediction.confidence > 0.99) {
    anomalies.push('suspiciously_high_confidence')
    severity = severity === 'low' ? 'medium' : severity
  }

  // Check for unrealistic stat combinations
  const totalStats = prediction.points_predicted + prediction.rebounds_predicted + prediction.assists_predicted
  if (totalStats > 70) {
    anomalies.push('unrealistic_total_stats')
    severity = 'high'
  }

  return {
    anomaly_detected: anomalies.length > 0,
    anomaly_type: anomalies.join(','),
    severity,
    details: {
      prediction_id: prediction.id,
      player: prediction.player_name,
      anomalies,
      values: {
        points: prediction.points_predicted,
        rebounds: prediction.rebounds_predicted,
        assists: prediction.assists_predicted,
        confidence: prediction.confidence
      }
    }
  }
}

async function createAnomalyAlert(
  prediction: Prediction,
  anomaly: AnomalyResult,
  supabase: any
): Promise<void> {
  await supabase
    .from('anomaly_alerts')
    .insert({
      anomaly_type: anomaly.anomaly_type,
      severity: anomaly.severity,
      prediction_id: prediction.id,
      details: anomaly.details,
      created_at: new Date().toISOString()
    })
}

async function sendWebhookNotification(anomaly: AnomalyResult): Promise<void> {
  const webhookUrl = Deno.env.get('ALERT_WEBHOOK_URL')
  if (!webhookUrl) return

  try {
    await fetch(webhookUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: `🚨 Critical Anomaly Detected`,
        attachments: [{
          color: 'danger',
          fields: [
            { title: 'Type', value: anomaly.anomaly_type, short: true },
            { title: 'Severity', value: anomaly.severity, short: true },
            { title: 'Details', value: JSON.stringify(anomaly.details, null, 2) }
          ]
        }]
      })
    })
  } catch (error) {
    console.error('Failed to send webhook:', error)
  }
}