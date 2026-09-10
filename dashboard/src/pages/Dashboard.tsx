import React from 'react';
import { Alert, Box, Card, CardContent, Grid, Typography } from '@mui/material';

const targets = ['Points', 'Rebounds', 'Assists'];

/**
 * Legacy Create React App dashboard retained for repository history.
 * The deployed configuration in this checkout starts the FastAPI image; the
 * maintained portfolio UI is in frontend/. This surface intentionally makes no
 * live-service or model-quality claims.
 */
const Dashboard: React.FC = () => (
  <Box sx={{ flexGrow: 1, p: 3 }}>
    <Typography variant="h4" component="h1" gutterBottom>NBA Performance Prediction System</Typography>
    <Alert severity="info" sx={{ mb: 3 }}>
      Legacy dashboard implementation. No measured evaluation or live operational telemetry is connected on this surface.
    </Alert>
    <Typography variant="h5" component="h2" gutterBottom>Recorded evaluation</Typography>
    <Grid container spacing={3}>
      {targets.map((target) => (
        <Grid item xs={12} sm={4} key={target}>
          <Card><CardContent><Typography color="text.secondary">{target}</Typography><Typography variant="h6">Unavailable</Typography><Typography variant="body2">Requires a reproducible evaluation artifact.</Typography></CardContent></Card>
        </Grid>
      ))}
    </Grid>
  </Box>
);

export default Dashboard;
