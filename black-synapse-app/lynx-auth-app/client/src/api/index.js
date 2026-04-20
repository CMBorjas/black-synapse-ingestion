import axios from 'axios';

/**
 * Pre-configured axios instance.
 * In development, Vite proxies /api → localhost:4000.
 * In production, set VITE_API_URL to the Express server's origin.
 */
const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || '',
  withCredentials: true, // Required: sends the session cookie with every request
});

// ── Auth ──────────────────────────────────────────────────────────────────────

export const login = (username, password) =>
  api.post('/api/auth/login', { username, password }).then((r) => r.data);

export const logout = () =>
  api.post('/api/auth/logout').then((r) => r.data);

export const getMe = () =>
  api.get('/api/auth/me').then((r) => r.data);

export const register = (username, password, registerSecret) =>
  api.post('/api/auth/register', { username, password, registerSecret }).then((r) => r.data);

// ── Credentials ───────────────────────────────────────────────────────────────

export const getConnections = () =>
  api.get('/api/credentials').then((r) => r.data);

export const disconnectService = (service) =>
  api.delete(`/api/credentials/${service}`).then((r) => r.data);

// ── OAuth (redirect-based, not axios) ────────────────────────────────────────
// These just build the URL — the browser does the redirect itself.

export const getOAuthUrl = (service) => `/api/oauth/${service}`;
