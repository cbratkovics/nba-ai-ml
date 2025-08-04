'use client';

import { useState } from 'react';
import { apiClient, TOP_PLAYERS, NBA_TEAMS, type PredictionRequest, type PredictionResponse } from '@/lib/api-client';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Calendar, Search, TrendingUp, Target, Brain, Loader2, BarChart3, Info, Share2 } from 'lucide-react';
import { format } from 'date-fns';

export default function PredictionsPage() {
  const [selectedPlayer, setSelectedPlayer] = useState<string>('');
  const [selectedOpponent, setSelectedOpponent] = useState<string>('');
  const [selectedDate, setSelectedDate] = useState<string>(format(new Date(), 'yyyy-MM-dd'));
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string>('');

  const filteredPlayers = searchQuery
    ? TOP_PLAYERS.filter(p => p.name.toLowerCase().includes(searchQuery.toLowerCase()))
    : TOP_PLAYERS;

  const handlePredict = async () => {
    if (!selectedPlayer || !selectedOpponent || !selectedDate) {
      setError('Please select a player, opponent, and date');
      return;
    }

    setLoading(true);
    setError('');
    setPrediction(null);

    try {
      const request: PredictionRequest = {
        player_id: selectedPlayer,
        game_date: selectedDate,
        opponent_team: selectedOpponent,
        include_explanation: true,
        include_confidence_intervals: true,
      };

      const response = await apiClient.getPrediction(request);
      setPrediction(response);
    } catch (err) {
      setError('Failed to get prediction. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleShare = () => {
    if (!prediction) return;
    
    const text = `NBA AI Prediction: ${prediction.player_name} vs ${prediction.opponent_team}
Points: ${prediction.predictions.points.toFixed(1)}
Rebounds: ${prediction.predictions.rebounds.toFixed(1)}
Assists: ${prediction.predictions.assists.toFixed(1)}
Confidence: ${(prediction.confidence * 100).toFixed(0)}%`;
    
    navigator.clipboard.writeText(text);
  };

  return (
    <main className="min-h-screen bg-gradient-to-b from-gray-900 via-gray-800 to-gray-900 py-12 px-4">
      <div className="container mx-auto max-w-6xl">
        {/* Header */}
        <div className="text-center mb-10">
          <h1 className="text-4xl font-bold text-white mb-4">
            NBA Player Predictions
          </h1>
          <p className="text-gray-400 text-lg">
            Get AI-powered predictions for any NBA player's upcoming performance
          </p>
        </div>

        {/* Prediction Form */}
        <Card className="bg-gray-800/50 border-gray-700 mb-8">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <Target className="h-5 w-5 text-blue-400" />
              Configure Prediction
            </CardTitle>
            <CardDescription className="text-gray-400">
              Select a player, opponent, and game date to generate predictions
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Player Selection */}
            <div className="space-y-2">
              <Label className="text-gray-300">Player</Label>
              <div className="relative">
                <Search className="absolute left-3 top-3 h-4 w-4 text-gray-500" />
                <Input
                  placeholder="Search players..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-10 bg-gray-700 border-gray-600 text-white"
                />
              </div>
              {searchQuery && (
                <div className="grid grid-cols-2 md:grid-cols-3 gap-2 mt-2 max-h-48 overflow-y-auto">
                  {filteredPlayers.slice(0, 9).map(player => (
                    <Button
                      key={player.id}
                      variant={selectedPlayer === player.id ? 'default' : 'outline'}
                      size="sm"
                      onClick={() => {
                        setSelectedPlayer(player.id);
                        setSearchQuery(player.name);
                      }}
                      className={selectedPlayer === player.id 
                        ? 'bg-blue-600 hover:bg-blue-700'
                        : 'border-gray-600 text-gray-300 hover:bg-gray-700'
                      }
                    >
                      {player.name}
                    </Button>
                  ))}
                </div>
              )}
            </div>

            {/* Opponent Selection */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label className="text-gray-300">Opponent Team</Label>
                <Select value={selectedOpponent} onValueChange={setSelectedOpponent}>
                  <SelectTrigger className="bg-gray-700 border-gray-600 text-white">
                    <SelectValue placeholder="Select opponent..." />
                  </SelectTrigger>
                  <SelectContent className="bg-gray-800 border-gray-700">
                    {NBA_TEAMS.map(team => (
                      <SelectItem key={team} value={team} className="text-gray-300">
                        {team}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label className="text-gray-300">Game Date</Label>
                <div className="relative">
                  <Calendar className="absolute left-3 top-3 h-4 w-4 text-gray-500" />
                  <Input
                    type="date"
                    value={selectedDate}
                    onChange={(e) => setSelectedDate(e.target.value)}
                    className="pl-10 bg-gray-700 border-gray-600 text-white"
                  />
                </div>
              </div>
            </div>

            {/* Error Alert */}
            {error && (
              <Alert className="bg-red-900/20 border-red-800">
                <AlertDescription className="text-red-400">
                  {error}
                </AlertDescription>
              </Alert>
            )}

            {/* Predict Button */}
            <Button
              onClick={handlePredict}
              disabled={loading || !selectedPlayer || !selectedOpponent}
              className="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700"
              size="lg"
            >
              {loading ? (
                <>
                  <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                  Generating Prediction...
                </>
              ) : (
                <>
                  <Brain className="mr-2 h-5 w-5" />
                  Generate Prediction
                </>
              )}
            </Button>
          </CardContent>
        </Card>

        {/* Prediction Results */}
        {prediction && (
          <Card className="bg-gray-800/50 border-gray-700">
            <CardHeader>
              <div className="flex justify-between items-start">
                <div>
                  <CardTitle className="text-white text-2xl">
                    {prediction.player_name}
                  </CardTitle>
                  <CardDescription className="text-gray-400">
                    vs {prediction.opponent_team} • {format(new Date(prediction.game_date), 'MMM dd, yyyy')}
                  </CardDescription>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleShare}
                  className="border-gray-600 text-gray-300 hover:bg-gray-700"
                >
                  <Share2 className="h-4 w-4 mr-1" />
                  Share
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="predictions" className="w-full">
                <TabsList className="grid w-full grid-cols-3 bg-gray-700">
                  <TabsTrigger value="predictions">Predictions</TabsTrigger>
                  <TabsTrigger value="confidence">Confidence</TabsTrigger>
                  <TabsTrigger value="factors">Key Factors</TabsTrigger>
                </TabsList>

                {/* Predictions Tab */}
                <TabsContent value="predictions" className="space-y-4 mt-6">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <Card className="bg-gray-700/50 border-gray-600">
                      <CardHeader className="pb-3">
                        <CardTitle className="text-blue-400 flex items-center gap-2">
                          <TrendingUp className="h-5 w-5" />
                          Points
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="text-3xl font-bold text-white">
                          {prediction.predictions.points.toFixed(1)}
                        </div>
                        {prediction.confidence_intervals?.points && (
                          <div className="text-sm text-gray-400 mt-1">
                            {prediction.confidence_intervals.points.lower.toFixed(0)} - {prediction.confidence_intervals.points.upper.toFixed(0)}
                          </div>
                        )}
                      </CardContent>
                    </Card>

                    <Card className="bg-gray-700/50 border-gray-600">
                      <CardHeader className="pb-3">
                        <CardTitle className="text-green-400 flex items-center gap-2">
                          <BarChart3 className="h-5 w-5" />
                          Rebounds
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="text-3xl font-bold text-white">
                          {prediction.predictions.rebounds.toFixed(1)}
                        </div>
                        {prediction.confidence_intervals?.rebounds && (
                          <div className="text-sm text-gray-400 mt-1">
                            {prediction.confidence_intervals.rebounds.lower.toFixed(0)} - {prediction.confidence_intervals.rebounds.upper.toFixed(0)}
                          </div>
                        )}
                      </CardContent>
                    </Card>

                    <Card className="bg-gray-700/50 border-gray-600">
                      <CardHeader className="pb-3">
                        <CardTitle className="text-purple-400 flex items-center gap-2">
                          <Target className="h-5 w-5" />
                          Assists
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="text-3xl font-bold text-white">
                          {prediction.predictions.assists.toFixed(1)}
                        </div>
                        {prediction.confidence_intervals?.assists && (
                          <div className="text-sm text-gray-400 mt-1">
                            {prediction.confidence_intervals.assists.lower.toFixed(0)} - {prediction.confidence_intervals.assists.upper.toFixed(0)}
                          </div>
                        )}
                      </CardContent>
                    </Card>
                  </div>

                  {prediction.explanation && (
                    <Alert className="bg-blue-900/20 border-blue-800">
                      <Info className="h-4 w-4 text-blue-400" />
                      <AlertDescription className="text-gray-300">
                        {prediction.explanation}
                      </AlertDescription>
                    </Alert>
                  )}
                </TabsContent>

                {/* Confidence Tab */}
                <TabsContent value="confidence" className="space-y-4 mt-6">
                  <div className="space-y-4">
                    <div>
                      <div className="flex justify-between mb-2">
                        <span className="text-gray-400">Overall Confidence</span>
                        <span className="text-white font-semibold">
                          {(prediction.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                      <Progress value={prediction.confidence * 100} className="h-3 bg-gray-700" />
                    </div>

                    <div className="grid grid-cols-3 gap-4 pt-4">
                      <div className="text-center">
                        <div className="text-2xl font-bold text-blue-400">
                          {prediction.model_accuracy.r2_score.toFixed(3)}
                        </div>
                        <div className="text-sm text-gray-400">R² Score</div>
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold text-green-400">
                          {prediction.model_accuracy.mae.toFixed(1)}
                        </div>
                        <div className="text-sm text-gray-400">MAE</div>
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold text-purple-400">
                          {prediction.model_accuracy.rmse.toFixed(1)}
                        </div>
                        <div className="text-sm text-gray-400">RMSE</div>
                      </div>
                    </div>

                    <Alert className="bg-gray-700/50 border-gray-600">
                      <AlertDescription className="text-gray-300">
                        Model version: {prediction.model_version} • 
                        Trained on 100+ NBA players with 50+ engineered features
                      </AlertDescription>
                    </Alert>
                  </div>
                </TabsContent>

                {/* Factors Tab */}
                <TabsContent value="factors" className="space-y-4 mt-6">
                  {prediction.factors && prediction.factors.length > 0 ? (
                    <div className="space-y-3">
                      {prediction.factors.map((factor, index) => (
                        <div key={index} className="flex items-center justify-between p-3 bg-gray-700/50 rounded-lg">
                          <div className="flex items-center gap-3">
                            <div className={`w-2 h-2 rounded-full ${
                              factor.impact === 'positive' ? 'bg-green-400' :
                              factor.impact === 'negative' ? 'bg-red-400' :
                              'bg-yellow-400'
                            }`} />
                            <span className="text-gray-300">{factor.feature}</span>
                          </div>
                          <div className="flex items-center gap-3">
                            <span className="text-white font-semibold">
                              {typeof factor.value === 'number' ? factor.value.toFixed(1) : factor.value}
                            </span>
                            <Badge variant="outline" className={`text-xs ${
                              factor.impact === 'positive' ? 'border-green-600 text-green-400' :
                              factor.impact === 'negative' ? 'border-red-600 text-red-400' :
                              'border-yellow-600 text-yellow-400'
                            }`}>
                              {factor.impact}
                            </Badge>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <Alert className="bg-gray-700/50 border-gray-600">
                      <AlertDescription className="text-gray-300">
                        Key factors include recent performance, rest days, home/away status, and opponent strength.
                      </AlertDescription>
                    </Alert>
                  )}
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>
        )}
      </div>
    </main>
  );
}