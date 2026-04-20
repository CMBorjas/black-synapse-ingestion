const express = require('express');
const axios = require('axios');
const requireAuth = require('../middleware/requireAuth');
const { getDb } = require('../services/db');

const router = express.Router();

router.post('/send', requireAuth, async (req, res) => {
  const { user_input, to } = req.body;
  const db = getDb();

  const connection = db.prepare(`
    SELECT service, access_token, refresh_token
    FROM connections
    WHERE user_id = ? AND service = ?
  `).get(req.session.userId, 'google');

  if (!connection || !connection.access_token) {
    return res.status(400).json({ error: 'Google account not connected' });
  }

  try {
    const response = await axios.post(
      process.env.N8N_WEBHOOK_URL || 'http://localhost:5678/webhook-test/send-email',
      {
        user_input,
        to,
        provider: 'gmail',
        access_token: connection.access_token,
      }
    );

    return res.json(response.data);
  } catch (error) {
    console.error('Failed to call n8n:', error.response?.data || error.message);
    return res.status(500).json({ error: 'Failed to execute email workflow' });
  }
});

module.exports = router;