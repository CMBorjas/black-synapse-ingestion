/**
 * Upload access middleware.
 *
 * Supports two auth modes:
 * - Browser session: req.session.userId (normal portal users)
 * - Service key: X-Lynx-Service-Key header + a target userId (for n8n automation)
 *
 * When authorized, sets req.uploadUserId (number).
 */
function requireUploadAccess(req, res, next) {
  // Normal session-based access
  const sessionUserId = req.session?.userId;
  if (Number.isFinite(sessionUserId)) {
    req.uploadUserId = sessionUserId;
    return next();
  }

  // Service-key access (for n8n)
  const serviceKey = req.get('x-lynx-service-key');
  const expected = process.env.LYNX_UPLOADS_SERVICE_KEY;

  if (!expected || expected.length < 16) {
    return res.status(500).json({ error: 'Uploads service key not configured' });
  }
  if (!serviceKey || serviceKey !== expected) {
    return res.status(401).json({ error: 'Not authenticated' });
  }

  const rawUserId = req.get('x-lynx-user-id') ?? req.query.userId;
  const userId = parseInt(String(rawUserId || ''), 10);
  if (!Number.isFinite(userId) || userId <= 0) {
    return res.status(400).json({ error: 'Missing or invalid userId (set X-Lynx-User-Id or ?userId=)' });
  }

  req.uploadUserId = userId;
  next();
}

module.exports = requireUploadAccess;

