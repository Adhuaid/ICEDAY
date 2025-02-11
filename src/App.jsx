import { useState, useEffect } from 'react';
import axios from 'axios';
import {
  Container,
  Box,
  TextField,
  Button,
  Typography,
  Paper,
  List,
  ListItem,
  ListItemText,
  CircularProgress,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Grid,
  LinearProgress,
  Card,
  CardContent,
} from '@mui/material';

const API_URL = 'http://localhost:5000';  // or 5001 if you changed it

function App() {
  const [studentData, setStudentData] = useState({
    raisedhands: '',
    VisITedResources: '',
    Discussion: '',
  });
  const [prediction, setPrediction] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [sampleData, setSampleData] = useState({ data: [], columns: [] });
  const [dataLoading, setDataLoading] = useState(true);
  const [analysis, setAnalysis] = useState(null);

  const handleInputChange = (e) => {
    setStudentData({
      ...studentData,
      [e.target.name]: parseFloat(e.target.value) || 0,
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post(`${API_URL}/predict`, studentData);
      console.log('Prediction response:', response.data); // Debug log
      setPrediction(response.data.prediction);
      setRecommendations(response.data.recommendations);
      setAnalysis(response.data);
    } catch (error) {
      console.error('Error:', error);
      const errorMessage = error.response?.data?.error || 'Failed to get prediction. Please try again.';
      setError(errorMessage);
      alert(errorMessage); // Add alert for visibility
    } finally {
      setLoading(false);
    }
  };

  const handleTrain = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post(`${API_URL}/train`);
      alert(response.data.message);
      // After successful training, enable predictions
      console.log('Training successful');
    } catch (error) {
      console.error('Error:', error);
      const errorMessage = error.response?.data?.error || 'Failed to train model. Please try again.';
      setError(errorMessage);
      alert(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSampleData();
  }, []);

  const fetchSampleData = async () => {
    try {
      const response = await axios.get(`${API_URL}/get-sample-data`);
      setSampleData(response.data);
    } catch (error) {
      console.error('Error fetching sample data:', error);
      setError('Failed to load sample data');
    } finally {
      setDataLoading(false);
    }
  };

  const renderAnalysis = () => {
    if (!analysis) return null;

    return (
      <Paper sx={{ p: 3, mt: 3 }}>
        <Typography variant="h5" gutterBottom>
          Performance Analysis
        </Typography>
        
        <Typography variant="body1" sx={{ mb: 3 }}>
          {analysis.analysis_summary}
        </Typography>

        <Grid container spacing={3} sx={{ mb: 3 }}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Performance Metrics
                </Typography>
                
                <Typography variant="subtitle2">Participation Score</Typography>
                <LinearProgress 
                  variant="determinate" 
                  value={analysis.metrics.participation_score} 
                  sx={{ mb: 2 }}
                />
                
                <Typography variant="subtitle2">Resource Usage</Typography>
                <LinearProgress 
                  variant="determinate" 
                  value={analysis.metrics.resource_usage} 
                  sx={{ mb: 2 }}
                />
                
                <Typography variant="subtitle2">Discussion Engagement</Typography>
                <LinearProgress 
                  variant="determinate" 
                  value={analysis.metrics.discussion_engagement} 
                  sx={{ mb: 2 }}
                />
                
                <Typography variant="subtitle2">Overall Engagement</Typography>
                <LinearProgress 
                  variant="determinate" 
                  value={analysis.metrics.overall_engagement} 
                />
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Performance Indicators
                </Typography>
                <List>
                  <ListItem>
                    <ListItemText 
                      primary="Class Participation" 
                      secondary={analysis.indicators.participation}
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemText 
                      primary="Resource Utilization" 
                      secondary={analysis.indicators.resource_usage}
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemText 
                      primary="Discussion Engagement" 
                      secondary={analysis.indicators.discussion}
                    />
                  </ListItem>
                </List>
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        <Typography variant="h6" gutterBottom>
          Recommendations
        </Typography>
        <List>
          {recommendations.map((rec, index) => (
            <ListItem key={index}>
              <ListItemText primary={rec} />
            </ListItem>
          ))}
        </List>
      </Paper>
    );
  };

  return (
    <Container maxWidth="md">
      <Box sx={{ my: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Student Progress Tracker
        </Typography>

        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        {/* Sample Data Table */}
        <Paper sx={{ p: 3, mb: 3, overflowX: 'auto' }}>
          <Typography variant="h6" gutterBottom>
            Sample Dataset (First 5 Rows)
          </Typography>
          {dataLoading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
              <CircularProgress />
            </Box>
          ) : (
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    {sampleData.columns.map((column) => (
                      <TableCell key={column}>{column}</TableCell>
                    ))}
                  </TableRow>
                </TableHead>
                <TableBody>
                  {sampleData.data.map((row, index) => (
                    <TableRow key={index}>
                      {sampleData.columns.map((column) => (
                        <TableCell key={column}>
                          {row[column]}
                        </TableCell>
                      ))}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </Paper>

        <Paper sx={{ p: 3, mb: 3 }}>
          <form onSubmit={handleSubmit}>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <TextField
                name="raisedhands"
                label="Raised Hands"
                type="number"
                value={studentData.raisedhands}
                onChange={handleInputChange}
                required
                disabled={loading}
                inputProps={{ min: 0, max: 100 }}
              />
              <TextField
                name="VisITedResources"
                label="Visited Resources"
                type="number"
                value={studentData.VisITedResources}
                onChange={handleInputChange}
                required
                disabled={loading}
                inputProps={{ min: 0, max: 100 }}
              />
              <TextField
                name="Discussion"
                label="Discussion Participation"
                type="number"
                value={studentData.Discussion}
                onChange={handleInputChange}
                required
                disabled={loading}
                inputProps={{ min: 0, max: 100 }}
              />
              <Button 
                variant="contained" 
                type="submit"
                disabled={loading}
              >
                {loading ? <CircularProgress size={24} /> : 'Predict Progress'}
              </Button>
            </Box>
          </form>
        </Paper>

        <Button 
          variant="outlined" 
          onClick={handleTrain} 
          sx={{ mb: 2 }}
          disabled={loading}
        >
          {loading ? <CircularProgress size={24} /> : 'Train Model'}
        </Button>

        {analysis && renderAnalysis()}
      </Box>
    </Container>
  );
}

export default App; 