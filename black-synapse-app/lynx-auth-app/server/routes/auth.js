const express = require('express');
const bcrypt = require('bcrypt');
const { getDb } = require('../services/db');

const router = express.Router();

// ── POST /api/auth/login ──────────────────────────────────────────────────────
router.post('/login', async (req, res) => {
  const { username, password } = req.body;

  if (!username || !password) {
    return res.status(400).json({ error: 'Username and password are required' });
  }

  const db = getDb();
  const user = db.prepare('SELECT * FROM users WHERE username = ?').get(username);

  if (!user) {
    return res.status(401).json({ error: 'Invalid credentials' });
  }

  const valid = await bcrypt.compare(password, user.password_hash);
  if (!valid) {
    return res.status(401).json({ error: 'Invalid credentials' });
  }

  req.session.userId = user.id;
  req.session.username = user.username;

  res.json({ success: true, username: user.username });
});

// ── POST /api/auth/logout ─────────────────────────────────────────────────────
router.post('/logout', (req, res) => {
  req.session.destroy(() => {
    res.json({ success: true });
  });
});

// ── GET /api/auth/me ──────────────────────────────────────────────────────────
// Returns the current session user — React uses this on load to check auth state.
router.get('/me', (req, res) => {
  if (!req.session || !req.session.userId) {
    return res.status(401).json({ error: 'Not authenticated' });
  }
  res.json({ userId: req.session.userId, username: req.session.username });
});

// ── POST /api/auth/register ───────────────────────────────────────────────────
// One-time team member registration. Gated by REGISTER_SECRET env var.
// Once all team accounts are created you can remove this route.
router.post('/register', async (req, res) => {
  const { username, password, registerSecret } = req.body;

  if (!registerSecret || registerSecret !== process.env.REGISTER_SECRET) {
    return res.status(403).json({ error: 'Invalid registration secret' });
  }

  if (!username || !password || password.length < 8) {
    return res.status(400).json({ error: 'Username required and password must be at least 8 characters' });
  }

  const hash = await bcrypt.hash(password, 10);
  const db = getDb();

  try {
    db.prepare('INSERT INTO users (username, password_hash) VALUES (?, ?)').run(username, hash);
    res.json({ success: true, message: `User "${username}" created` });
  } catch (e) {
    if (e.message.includes('UNIQUE')) {
      return res.status(409).json({ error: 'Username already exists' });
    }
    throw e;
  }
});

module.exports = router;
