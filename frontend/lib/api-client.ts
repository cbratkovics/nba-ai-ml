/**
 * API Client for NBA Predictions
 */

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'https://nba-ai-ml-production.up.railway.app';

export interface PredictionRequest {
  player_id: string;
  game_date: string;
  opponent_team: string;
  include_explanation?: boolean;
  include_confidence_intervals?: boolean;
}

export interface PredictionResponse {
  player_id: string;
  player_name: string;
  game_date: string;
  opponent_team: string;
  predictions: {
    points: number;
    rebounds: number;
    assists: number;
  };
  confidence: number;
  confidence_intervals?: {
    points: { lower: number; upper: number };
    rebounds: { lower: number; upper: number };
    assists: { lower: number; upper: number };
  };
  model_version: string;
  model_accuracy: {
    r2_score: number;
    mae: number;
    rmse: number;
  };
  explanation?: string;
  factors?: Array<{
    feature: string;
    value: number | string;
    impact: 'positive' | 'negative' | 'neutral';
    importance?: number;
  }>;
  prediction_id: string;
  timestamp: string;
}

export interface HealthResponse {
  status: string;
  version: string;
  timestamp: string;
}

export interface Player {
  id: string;
  name: string;
  team: string;
  position?: string;
  image?: string;
}

// Top NBA players for the dropdown
export const TOP_PLAYERS: Player[] = [
  { id: '203999', name: 'Nikola Jokic', team: 'DEN', position: 'C' },
  { id: '203954', name: 'Joel Embiid', team: 'PHI', position: 'C' },
  { id: '203507', name: 'Giannis Antetokounmpo', team: 'MIL', position: 'F' },
  { id: '2544', name: 'LeBron James', team: 'LAL', position: 'F' },
  { id: '201939', name: 'Stephen Curry', team: 'GSW', position: 'G' },
  { id: '1628369', name: 'Jayson Tatum', team: 'BOS', position: 'F' },
  { id: '1629029', name: 'Luka Doncic', team: 'DAL', position: 'G' },
  { id: '201142', name: 'Kevin Durant', team: 'PHX', position: 'F' },
  { id: '203076', name: 'Anthony Davis', team: 'LAL', position: 'F-C' },
  { id: '1628983', name: 'Shai Gilgeous-Alexander', team: 'OKC', position: 'G' },
  { id: '202695', name: 'Kawhi Leonard', team: 'LAC', position: 'F' },
  { id: '202710', name: 'Jimmy Butler', team: 'MIA', position: 'F' },
  { id: '1628389', name: 'Bam Adebayo', team: 'MIA', position: 'C' },
  { id: '1628368', name: "De'Aaron Fox", team: 'SAC', position: 'G' },
  { id: '1629630', name: 'Ja Morant', team: 'MEM', position: 'G' },
  { id: '1628973', name: 'Jalen Brunson', team: 'NYK', position: 'G' },
  { id: '1630162', name: 'Anthony Edwards', team: 'MIN', position: 'G' },
  { id: '1628378', name: 'Donovan Mitchell', team: 'CLE', position: 'G' },
  { id: '202331', name: 'Paul George', team: 'LAC', position: 'F' },
  { id: '1630578', name: 'Alperen Sengun', team: 'HOU', position: 'C' }
];

// NBA Teams
export const NBA_TEAMS = [
  'ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW',
  'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK',
  'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS'
];

class APIClient {
  private baseURL: string;

  constructor() {
    this.baseURL = API_URL;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;
    
    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
      });

      if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || `API Error: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('API Request failed:', error);
      throw error;
    }
  }

  async getPrediction(request: PredictionRequest): Promise<PredictionResponse> {
    return this.request<PredictionResponse>('/v1/predict', {
      method: 'POST',
      body: JSON.stringify({
        ...request,
        targets: ['all'],
        model_version: 'latest',
      }),
    });
  }

  async getBatchPredictions(
    requests: PredictionRequest[]
  ): Promise<{ predictions: PredictionResponse[] }> {
    return this.request<{ predictions: PredictionResponse[] }>(
      '/v1/predict/batch',
      {
        method: 'POST',
        body: JSON.stringify({ predictions: requests }),
      }
    );
  }

  async getHealth(): Promise<HealthResponse> {
    return this.request<HealthResponse>('/health/status');
  }

  async getDetailedHealth(): Promise<any> {
    return this.request<any>('/health/detailed');
  }

  // Mock function for demo - gets today's predictions
  async getTodaysPredictions(): Promise<PredictionResponse[]> {
    const today = new Date().toISOString().split('T')[0];
    
    // Mock predictions for top 5 players
    const topPlayers = TOP_PLAYERS.slice(0, 5);
    const opponents = ['LAL', 'BOS', 'MIA', 'GSW', 'DAL'];
    
    const requests: PredictionRequest[] = topPlayers.map((player, index) => ({
      player_id: player.id,
      game_date: today,
      opponent_team: opponents[index],
      include_explanation: false,
      include_confidence_intervals: false,
    }));

    try {
      const response = await this.getBatchPredictions(requests);
      return response.predictions;
    } catch (error) {
      // Return mock data if API is unavailable
      return topPlayers.map((player, index) => ({
        player_id: player.id,
        player_name: player.name,
        game_date: today,
        opponent_team: opponents[index],
        predictions: {
          points: 25 + Math.random() * 10,
          rebounds: 8 + Math.random() * 5,
          assists: 6 + Math.random() * 4,
        },
        confidence: 0.85 + Math.random() * 0.1,
        model_version: 'v2.1.0',
        model_accuracy: {
          r2_score: 0.942,
          mae: 3.1,
          rmse: 4.2,
        },
        prediction_id: `pred_${today}_${player.id}_${opponents[index]}`,
        timestamp: new Date().toISOString(),
      }));
    }
  }

  // Get player by ID
  getPlayer(playerId: string): Player | undefined {
    return TOP_PLAYERS.find(p => p.id === playerId);
  }

  // Search players by name
  searchPlayers(query: string): Player[] {
    const lowerQuery = query.toLowerCase();
    return TOP_PLAYERS.filter(player =>
      player.name.toLowerCase().includes(lowerQuery)
    );
  }
}

// Export singleton instance
export const apiClient = new APIClient();