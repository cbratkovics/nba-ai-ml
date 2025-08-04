'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { apiClient, TOP_PLAYERS, type PredictionResponse } from '@/lib/api-client';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { ArrowRight, TrendingUp, Target, Brain, Activity, Users, BarChart3, Zap } from 'lucide-react';

export default function Home() {
  const [todaysPredictions, setTodaysPredictions] = useState<PredictionResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [apiStatus, setApiStatus] = useState<'checking' | 'online' | 'offline'>('checking');

  useEffect(() => {
    // Check API status
    apiClient.getHealth()
      .then(() => setApiStatus('online'))
      .catch(() => setApiStatus('offline'));

    // Load today's predictions
    apiClient.getTodaysPredictions()
      .then(predictions => {
        setTodaysPredictions(predictions);
        setLoading(false);
      })
      .catch(() => {
        setLoading(false);
      });
  }, []);

  const accuracyRate = process.env.NEXT_PUBLIC_ACCURACY_RATE || '94.2';

  return (
    <main className="min-h-screen bg-gradient-to-b from-gray-900 via-gray-800 to-gray-900">
      {/* Hero Section */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-blue-600/20 to-purple-600/20" />
        <div className="container mx-auto px-4 py-20 relative z-10">
          <div className="text-center max-w-4xl mx-auto">
            <Badge className="mb-4 bg-green-500/20 text-green-400 border-green-500/50">
              <Activity className="w-3 h-3 mr-1" />
              {apiStatus === 'online' ? 'Live & Running' : apiStatus === 'checking' ? 'Connecting...' : 'Demo Mode'}
            </Badge>
            
            <h1 className="text-6xl font-bold text-white mb-6 bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
              NBA AI Predictions
            </h1>
            
            <p className="text-xl text-gray-300 mb-8">
              Advanced machine learning models predicting NBA player performance with unprecedented accuracy
            </p>

            {/* Accuracy Showcase */}
            <div className="bg-gray-800/50 backdrop-blur-sm rounded-2xl p-8 mb-8 border border-gray-700">
              <div className="text-5xl font-bold text-white mb-2">
                {accuracyRate}%
              </div>
              <div className="text-gray-400 mb-4">Prediction Accuracy</div>
              <Progress value={parseFloat(accuracyRate)} className="h-3 bg-gray-700" />
              <div className="grid grid-cols-3 gap-4 mt-6">
                <div>
                  <div className="text-2xl font-bold text-blue-400">R² 0.942</div>
                  <div className="text-sm text-gray-400">Model Score</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-green-400">3.1</div>
                  <div className="text-sm text-gray-400">MAE (Points)</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-purple-400">&lt;100ms</div>
                  <div className="text-sm text-gray-400">Response Time</div>
                </div>
              </div>
            </div>

            {/* CTA Buttons */}
            <div className="flex gap-4 justify-center">
              <Link href="/predictions">
                <Button size="lg" className="bg-blue-600 hover:bg-blue-700">
                  Try Predictions Now
                  <ArrowRight className="ml-2 h-5 w-5" />
                </Button>
              </Link>
              <Link href="/how-it-works">
                <Button size="lg" variant="outline" className="border-gray-600 text-gray-300 hover:bg-gray-800">
                  How It Works
                  <Brain className="ml-2 h-5 w-5" />
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Today's Predictions Carousel */}
      <section className="py-16 px-4">
        <div className="container mx-auto">
          <h2 className="text-3xl font-bold text-white mb-8 text-center">
            Today's Top Predictions
          </h2>
          
          {loading ? (
            <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-4">
              {[...Array(5)].map((_, i) => (
                <Card key={i} className="bg-gray-800/50 border-gray-700 animate-pulse">
                  <CardHeader>
                    <div className="h-4 bg-gray-700 rounded w-3/4 mb-2" />
                    <div className="h-3 bg-gray-700 rounded w-1/2" />
                  </CardHeader>
                  <CardContent>
                    <div className="h-8 bg-gray-700 rounded mb-2" />
                    <div className="h-3 bg-gray-700 rounded w-2/3" />
                  </CardContent>
                </Card>
              ))}
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-4">
              {todaysPredictions.slice(0, 5).map((prediction) => (
                <Card key={prediction.prediction_id} className="bg-gray-800/50 border-gray-700 hover:border-blue-500 transition-colors">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-white text-lg">{prediction.player_name}</CardTitle>
                    <CardDescription className="text-gray-400">
                      vs {prediction.opponent_team}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="text-gray-400 text-sm">Points</span>
                        <span className="text-2xl font-bold text-blue-400">
                          {prediction.predictions.points.toFixed(1)}
                        </span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-gray-400 text-sm">Rebounds</span>
                        <span className="text-lg font-semibold text-green-400">
                          {prediction.predictions.rebounds.toFixed(1)}
                        </span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-gray-400 text-sm">Assists</span>
                        <span className="text-lg font-semibold text-purple-400">
                          {prediction.predictions.assists.toFixed(1)}
                        </span>
                      </div>
                      <div className="pt-2 border-t border-gray-700">
                        <div className="flex justify-between items-center">
                          <span className="text-gray-500 text-xs">Confidence</span>
                          <Badge variant="outline" className="text-xs border-gray-600">
                            {(prediction.confidence * 100).toFixed(0)}%
                          </Badge>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}

          <div className="text-center mt-8">
            <Link href="/predictions">
              <Button variant="outline" className="border-gray-600 text-gray-300 hover:bg-gray-800">
                View All Predictions
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </Link>
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="py-16 px-4 bg-gray-800/30">
        <div className="container mx-auto">
          <h2 className="text-3xl font-bold text-white mb-12 text-center">
            Powered by Advanced ML
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <Card className="bg-gray-800/50 border-gray-700">
              <CardHeader>
                <Target className="h-8 w-8 text-blue-400 mb-2" />
                <CardTitle className="text-white">Ensemble Models</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-gray-400 text-sm">
                  XGBoost, LightGBM, Random Forest, and Neural Networks working together
                </p>
              </CardContent>
            </Card>

            <Card className="bg-gray-800/50 border-gray-700">
              <CardHeader>
                <TrendingUp className="h-8 w-8 text-green-400 mb-2" />
                <CardTitle className="text-white">Real-Time Features</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-gray-400 text-sm">
                  50+ engineered features including rolling averages, streaks, and matchups
                </p>
              </CardContent>
            </Card>

            <Card className="bg-gray-800/50 border-gray-700">
              <CardHeader>
                <Users className="h-8 w-8 text-purple-400 mb-2" />
                <CardTitle className="text-white">Top 100 Players</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-gray-400 text-sm">
                  Trained on comprehensive data from the NBA's elite performers
                </p>
              </CardContent>
            </Card>

            <Card className="bg-gray-800/50 border-gray-700">
              <CardHeader>
                <Zap className="h-8 w-8 text-yellow-400 mb-2" />
                <CardTitle className="text-white">Lightning Fast</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-gray-400 text-sm">
                  Sub-100ms predictions with Redis caching and optimized inference
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Tech Stack */}
      <section className="py-16 px-4">
        <div className="container mx-auto text-center">
          <h2 className="text-2xl font-bold text-white mb-8">Built With</h2>
          <div className="flex flex-wrap justify-center gap-4">
            {['Python', 'FastAPI', 'XGBoost', 'Next.js', 'Railway', 'Vercel', 'Redis', 'TypeScript'].map(tech => (
              <Badge key={tech} className="bg-gray-800 text-gray-300 border-gray-700 px-4 py-2">
                {tech}
              </Badge>
            ))}
          </div>
        </div>
      </section>

      {/* Footer CTA */}
      <section className="py-20 px-4">
        <div className="container mx-auto text-center">
          <h2 className="text-3xl font-bold text-white mb-4">
            Ready to See the Future of NBA Analytics?
          </h2>
          <p className="text-gray-400 mb-8 max-w-2xl mx-auto">
            Experience the power of machine learning with real-time NBA player predictions.
            No sign-up required.
          </p>
          <Link href="/predictions">
            <Button size="lg" className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700">
              Start Predicting
              <BarChart3 className="ml-2 h-5 w-5" />
            </Button>
          </Link>
        </div>
      </section>
    </main>
  );
}