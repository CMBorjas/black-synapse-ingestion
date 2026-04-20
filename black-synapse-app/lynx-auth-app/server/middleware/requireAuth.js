/**
 * Session guard middleware.
 * Rejects any request that doesn't have a valid logged-in session.
 */
function requireAuth(req, res, next) {
  if (!req.session || !req.session.userId) {
    return res.status(401).json({ error: 'Not authenticated' });
  }
  next();
}

module.exports = requireAuth;
