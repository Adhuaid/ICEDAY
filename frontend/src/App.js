import React, { useState } from 'react';
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
} from '@mui/material';

function App() {
  const [studentData, setStudentData] = useState({
    raisedhands: '',
    VisITedResources: '',
    Discussion: '',
  });
  const [prediction, setPrediction] = useState(null);
  const [recommendations, setRecommendations] = useState([]);

  const handleInputChange = (e) => {
    setStudentData({
      ...studentData,
      [e.target.name]: parseFloat(e.target.value),
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post('http://localhost:5000/predict', studentData);
      setPrediction(response.data.prediction);
      setRecommendations(response.data.recommendations);
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <Container maxWidth="md">
      <Box sx={{ my: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Student Progress Tracker
        </Typography>

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
              />
              <TextField
                name="VisITedResources"
                label="Visited Resources"
                type="number"
                value={studentData.VisITedResources}
                onChange={handleInputChange}
                required
              />
              <TextField
                name="Discussion"
                label="Discussion Participation"
                type="number"
                value={studentData.Discussion}
                onChange={handleInputChange}
                required
              />
              <Button variant="contained" type="submit">
                Predict Progress
              </Button>
            </Box>
          </form>
        </Paper>

        {prediction && (
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Predicted Performance Level: {prediction}
            </Typography>
            
            <Typography variant="h6" gutterBottom>
              Recommendations:
            </Typography>
            <List>
              {recommendations.map((rec, index) => (
                <ListItem key={index}>
                  <ListItemText primary={rec} />
                </ListItem>
              ))}
            </List>
          </Paper>
        )}
      </Box>
    </Container>
  );
}

export default App; 