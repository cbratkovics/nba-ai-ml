import React, { useEffect, useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { Provider } from 'react-redux';
import { store } from './store/store';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import Predictions from './pages/Predictions';
import Models from './pages/Models';
import LiveGames from './pages/LiveGames';
import Analytics from './pages/Analytics';
import Settings from './pages/Settings';
import { WebSocketProvider } from './contexts/WebSocketContext';
import './App.css';

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#0a0e27',
      paper: '#1a1f3a',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 600,
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 500,
    },
  },
});

const App: React.FC = () => {
  return (
    <Provider store={store}>
      <ThemeProvider theme={darkTheme}>
        <CssBaseline />
        <WebSocketProvider>
          <Router>
            <Layout>
              <Routes>
                <Route path="/" element={<Navigate to="/dashboard" />} />
                <Route path="/dashboard" element={<Dashboard />} />
                <Route path="/predictions" element={<Predictions />} />
                <Route path="/models" element={<Models />} />
                <Route path="/live" element={<LiveGames />} />
                <Route path="/analytics" element={<Analytics />} />
                <Route path="/settings" element={<Settings />} />
              </Routes>
            </Layout>
          </Router>
        </WebSocketProvider>
      </ThemeProvider>
    </Provider>
  );
};

export default App;