import React, { useEffect, useState } from 'react';
import {
  Grid,
  Paper,
  Typography,
  Box,
  Card,
  CardContent,
  CircularProgress,
  Alert,
  Chip,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  SportsBa sketball,
  Analytics as AnalyticsIcon,
  Speed,
  CheckCircle,
  Warning,
  Refresh,
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { useSelector, useDispatch } from 'react-redux';
import { RootState } from '../store/store';
import { fetchDashboardData } from '../store/dashboardSlice';
import { useWebSocket } from '../contexts/WebSocketContext';
import MetricCard from '../components/MetricCard';
import ModelPerformanceChart from '../components/ModelPerformanceChart';
import PredictionFeed from '../components/PredictionFeed';
import SystemHealth from '../components/SystemHealth';

interface DashboardMetrics {
  totalPredictions: number;
  todayPredictions: number;
  averageAccuracy: number;
  activeModels: number;
  apiLatency: number;
  systemUptime: number;
  predictionTrend: number;
  accuracyTrend: number;
}

interface ModelMetrics {
  name: string;
  accuracy: number;
  predictions: number;
  lastUpdated: string;
  status: 'active' | 'training' | 'inactive';
}

const Dashboard: React.FC = () => {
  const dispatch = useDispatch();
  const { ws, isConnected } = useWebSocket();
  const [metrics, setMetrics] = useState<DashboardMetrics>({
    totalPredictions: 0,
    todayPredictions: 0,
    averageAccuracy: 0,
    activeModels: 0,
    apiLatency: 0,
    systemUptime: 0,
    predictionTrend: 0,
    accuracyTrend: 0,
  });
  
  const [modelMetrics, setModelMetrics] = useState<ModelMetrics[]>([
    { name: 'Points (PTS)', accuracy: 94.5, predictions: 125430, lastUpdated: '2 hours ago', status: 'active' },
    { name: 'Rebounds (REB)', accuracy: 93.8, predictions: 125430, lastUpdated: '2 hours ago', status: 'active' },
    { name: 'Assists (AST)', accuracy: 94.2, predictions: 125430, lastUpdated: '2 hours ago', status: 'active' },
  ]);
  
  const [recentPredictions, setRecentPredictions] = useState<any[]>([]);
  const [performanceData, setPerformanceData] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadDashboardData();
    
    // Subscribe to real-time updates
    if (ws && isConnected) {
      ws.send(JSON.stringify({
        type: 'subscribe',
        channels: ['metrics', 'predictions', 'system'],
      }));
    }
    
    // Set up auto-refresh
    const interval = setInterval(loadDashboardData, 30000); // Refresh every 30 seconds
    
    return () => clearInterval(interval);
  }, [ws, isConnected]);
  
  const loadDashboardData = async () => {
    try {
      setLoading(true);
      
      // Fetch dashboard metrics
      const response = await fetch('/api/v1/dashboard/metrics');
      const data = await response.json();
      
      setMetrics({
        totalPredictions: data.total_predictions || 1250000,
        todayPredictions: data.today_predictions || 45230,
        averageAccuracy: data.average_accuracy || 94.2,
        activeModels: data.active_models || 3,
        apiLatency: data.api_latency || 42,
        systemUptime: data.system_uptime || 99.9,
        predictionTrend: data.prediction_trend || 12.5,
        accuracyTrend: data.accuracy_trend || 1.2,
      });
      
      // Generate mock performance data
      const perfData = generatePerformanceData();
      setPerformanceData(perfData);
      
      // Generate mock recent predictions
      const predictions = generateRecentPredictions();
      setRecentPredictions(predictions);
      
      setError(null);
    } catch (err) {
      setError('Failed to load dashboard data');
      console.error('Dashboard error:', err);
    } finally {
      setLoading(false);
    }
  };
  
  const generatePerformanceData = () => {
    const data = [];
    const now = new Date();
    
    for (let i = 23; i >= 0; i--) {
      const hour = new Date(now.getTime() - i * 60 * 60 * 1000);
      data.push({
        time: hour.toLocaleTimeString('en-US', { hour: '2-digit' }),
        predictions: Math.floor(Math.random() * 2000) + 1000,
        accuracy: 92 + Math.random() * 4,
        latency: 30 + Math.random() * 30,
      });
    }
    
    return data;
  };
  
  const generateRecentPredictions = () => {
    const players = ['LeBron James', 'Stephen Curry', 'Nikola Jokic', 'Giannis Antetokounmpo', 'Joel Embiid'];
    const predictions = [];
    
    for (let i = 0; i < 10; i++) {
      predictions.push({
        id: i,
        player: players[Math.floor(Math.random() * players.length)],
        target: ['PTS', 'REB', 'AST'][Math.floor(Math.random() * 3)],
        prediction: Math.floor(Math.random() * 30) + 10,
        confidence: 85 + Math.random() * 10,
        timestamp: new Date(Date.now() - Math.random() * 3600000).toISOString(),
      });
    }
    
    return predictions;
  };
  
  const formatNumber = (num: number): string => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
    return num.toString();
  };
  
  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="80vh">
        <CircularProgress size={60} />
      </Box>
    );
  }
  
  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" component="h1" gutterBottom>
          NBA ML Dashboard
        </Typography>
        <Box>
          <Chip
            icon={<CheckCircle />}
            label={isConnected ? 'Connected' : 'Disconnected'}
            color={isConnected ? 'success' : 'error'}
            sx={{ mr: 2 }}
          />
          <Tooltip title="Refresh data">
            <IconButton onClick={loadDashboardData} color="primary">
              <Refresh />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>
      
      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}
      
      {/* Metrics Cards */}
      <Grid container spacing={3} mb={3}>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Total Predictions"
            value={formatNumber(metrics.totalPredictions)}
            trend={metrics.predictionTrend}
            icon={<SportsBasketball />}
            color="primary"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Today's Predictions"
            value={formatNumber(metrics.todayPredictions)}
            trend={15.3}
            icon={<TrendingUp />}
            color="success"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Average Accuracy"
            value={`${metrics.averageAccuracy.toFixed(1)}%`}
            trend={metrics.accuracyTrend}
            icon={<AnalyticsIcon />}
            color="info"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="API Latency"
            value={`${metrics.apiLatency}ms`}
            trend={-5.2}
            icon={<Speed />}
            color="warning"
          />
        </Grid>
      </Grid>
      
      {/* Model Performance */}
      <Grid container spacing={3} mb={3}>
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3, height: '400px' }}>
            <Typography variant="h6" gutterBottom>
              Prediction Volume & Accuracy
            </Typography>
            <ResponsiveContainer width="100%" height="90%">
              <LineChart data={performanceData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="time" stroke="#9CA3AF" />
                <YAxis yAxisId="left" stroke="#9CA3AF" />
                <YAxis yAxisId="right" orientation="right" stroke="#9CA3AF" />
                <RechartsTooltip
                  contentStyle={{ backgroundColor: '#1F2937', border: 'none' }}
                />
                <Legend />
                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="predictions"
                  stroke="#3B82F6"
                  strokeWidth={2}
                  name="Predictions"
                />
                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="accuracy"
                  stroke="#10B981"
                  strokeWidth={2}
                  name="Accuracy %"
                />
              </LineChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3, height: '400px' }}>
            <Typography variant="h6" gutterBottom>
              Model Status
            </Typography>
            <Box sx={{ mt: 2 }}>
              {modelMetrics.map((model, index) => (
                <Box key={index} sx={{ mb: 2 }}>
                  <Box display="flex" justifyContent="space-between" alignItems="center">
                    <Typography variant="body2">{model.name}</Typography>
                    <Chip
                      label={model.status}
                      size="small"
                      color={model.status === 'active' ? 'success' : 'default'}
                    />
                  </Box>
                  <Box display="flex" justifyContent="space-between" alignItems="center" mt={1}>
                    <Typography variant="caption" color="text.secondary">
                      Accuracy: {model.accuracy}%
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {model.lastUpdated}
                    </Typography>
                  </Box>
                  <Box sx={{ width: '100%', mt: 1 }}>
                    <Box
                      sx={{
                        height: 8,
                        borderRadius: 1,
                        backgroundColor: 'rgba(255, 255, 255, 0.1)',
                        overflow: 'hidden',
                      }}
                    >
                      <Box
                        sx={{
                          height: '100%',
                          width: `${model.accuracy}%`,
                          backgroundColor: model.accuracy > 94 ? '#10B981' : '#F59E0B',
                          transition: 'width 0.3s ease',
                        }}
                      />
                    </Box>
                  </Box>
                </Box>
              ))}
            </Box>
          </Paper>
        </Grid>
      </Grid>
      
      {/* Recent Predictions and System Health */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3, height: '400px', overflow: 'hidden' }}>
            <Typography variant="h6" gutterBottom>
              Recent Predictions
            </Typography>
            <PredictionFeed predictions={recentPredictions} />
          </Paper>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3, height: '400px' }}>
            <Typography variant="h6" gutterBottom>
              System Health
            </Typography>
            <SystemHealth />
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;