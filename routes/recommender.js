const express = require('express');
const axios = require('axios');
const router = express.Router();

router.post('/recommend', async (req, res) => {
  const { food_name } = req.body;

  try {
    const response = await axios.post('http://localhost:5001/recommend', { food_name });
    res.json(response.data);
  } catch (error) {
    console.error('Error communicating with FastAPI:', error.message);
    res.status(500).json({ error: 'Failed to fetch recommendations' });
  }
});

module.exports = router;
