const path = require('path');
const fs = require('fs');
const crypto = require('crypto');
const express = require('express');
const multer = require('multer');
const FormData = require('form-data');
const axios = require('axios');
const requireUploadAccess = require('../middleware/requireUploadAccess');
const { getDb } = require('../services/db');

const WORKER_URL = process.env.WORKER_URL || 'http://localhost:8000';

const router = express.Router();

const UPLOADS_DIR = path.join(__dirname, '../data/uploads');
const MAX_FILES_PER_REQUEST = 10;

function ensureUploadsDir() {
  fs.mkdirSync(UPLOADS_DIR, { recursive: true });
}

function maxBytes() {
  const mb = parseInt(process.env.UPLOAD_MAX_MB || '25', 10);
  const safe = Number.isFinite(mb) && mb > 0 ? mb : 25;
  return safe * 1024 * 1024;
}

const storage = multer.diskStorage({
  destination: (_req, _file, cb) => {
    ensureUploadsDir();
    cb(null, UPLOADS_DIR);
  },
  filename: (_req, file, cb) => {
    const ext = path.extname(file.originalname || '').slice(0, 32);
    cb(null, `${crypto.randomUUID()}${ext}`);
  },
});

function pdfOnly(_req, file, cb) {
  if (file.mimetype === 'application/pdf' || file.originalname.toLowerCase().endsWith('.pdf')) {
    cb(null, true);
  } else {
    cb(new multer.MulterError('LIMIT_UNEXPECTED_FILE', 'Only PDF files are accepted'));
  }
}

const upload = multer({
  storage,
  fileFilter: pdfOnly,
  limits: { fileSize: maxBytes(), files: MAX_FILES_PER_REQUEST },
});

function diskPath(diskFilename) {
  const resolved = path.resolve(UPLOADS_DIR, diskFilename);
  if (!resolved.startsWith(path.resolve(UPLOADS_DIR) + path.sep)) {
    return null;
  }
  return resolved;
}

// ── GET /api/uploads ──────────────────────────────────────────────────────────
router.get('/', requireUploadAccess, (req, res) => {
  const db = getDb();
  const rows = db.prepare(`
    SELECT id, original_filename AS originalFilename, mime_type AS mimeType,
           size_bytes AS sizeBytes, created_at AS createdAt
    FROM user_uploads
    WHERE user_id = ?
    ORDER BY created_at DESC
  `).all(req.uploadUserId);
  res.json(rows);
});

// ── GET /api/uploads/:id/file ────────────────────────────────────────────────
router.get('/:id/file', requireUploadAccess, (req, res) => {
  const id = parseInt(req.params.id, 10);
  if (!Number.isFinite(id)) {
    return res.status(400).json({ error: 'Invalid id' });
  }

  const db = getDb();
  const row = db.prepare(
    'SELECT disk_filename, original_filename, mime_type FROM user_uploads WHERE id = ? AND user_id = ?'
  ).get(id, req.uploadUserId);

  if (!row) {
    return res.status(404).json({ error: 'File not found' });
  }

  const abs = diskPath(row.disk_filename);
  if (!abs || !fs.existsSync(abs)) {
    return res.status(404).json({ error: 'File missing on disk' });
  }

  if (row.mime_type) {
    res.setHeader('Content-Type', row.mime_type);
  }
  res.download(abs, row.original_filename);
});

// ── POST /api/uploads ─────────────────────────────────────────────────────────
router.post('/', requireUploadAccess, (req, res, next) => {
  upload.array('files', MAX_FILES_PER_REQUEST)(req, res, (err) => {
    if (err) {
      if (err instanceof multer.MulterError) {
        if (err.code === 'LIMIT_FILE_SIZE') {
          return res.status(413).json({ error: `Each file must be under ${process.env.UPLOAD_MAX_MB || 25} MB` });
        }
        if (err.code === 'LIMIT_FILE_COUNT') {
          return res.status(400).json({ error: `At most ${MAX_FILES_PER_REQUEST} files per upload` });
        }
        if (err.code === 'LIMIT_UNEXPECTED_FILE') {
          return res.status(415).json({ error: 'Only PDF files are accepted' });
        }
        return res.status(400).json({ error: err.message });
      }
      return next(err);
    }
    next();
  });
}, (req, res) => {
  const files = req.files;
  if (!files || files.length === 0) {
    return res.status(400).json({ error: 'No files received (use field name "files")' });
  }

  const db = getDb();
  const insert = db.prepare(`
    INSERT INTO user_uploads (user_id, original_filename, disk_filename, mime_type, size_bytes)
    VALUES (?, ?, ?, ?, ?)
  `);

  const created = [];
  const insertMany = db.transaction(() => {
    for (const f of files) {
      const info = insert.run(
        req.uploadUserId,
        f.originalname || 'unnamed',
        f.filename,
        f.mimetype || null,
        f.size
      );
      created.push({
        id: info.lastInsertRowid,
        originalFilename: f.originalname || 'unnamed',
        mimeType: f.mimetype || null,
        sizeBytes: f.size,
        diskFilename: f.filename,
      });
    }
  });

  try {
    insertMany();
  } catch (e) {
    for (const f of files) {
      const abs = diskPath(f.filename);
      if (abs && fs.existsSync(abs)) {
        try { fs.unlinkSync(abs); } catch (_) { /* ignore */ }
      }
    }
    console.error('upload insert failed:', e);
    return res.status(500).json({ error: 'Failed to save upload metadata' });
  }

  // Forward each PDF to the ingestion worker in the background
  for (const f of created) {
    const abs = diskPath(f.diskFilename);
    if (!abs) continue;
    const form = new FormData();
    form.append('file', fs.createReadStream(abs), {
      filename: f.originalFilename,
      contentType: 'application/pdf',
    });
    axios.post(`${WORKER_URL}/ingest/pdf`, form, { headers: form.getHeaders() })
      .then((r) => console.log(`[ingestion] ${f.originalFilename}:`, r.data.message || r.data))
      .catch((err) => console.error(`[ingestion] failed for ${f.originalFilename}:`, err.message));
  }

  res.status(201).json({ files: created.map(({ diskFilename: _d, ...rest }) => rest) });
});

// ── DELETE /api/uploads/:id ──────────────────────────────────────────────────
router.delete('/:id', requireUploadAccess, (req, res) => {
  const id = parseInt(req.params.id, 10);
  if (!Number.isFinite(id)) {
    return res.status(400).json({ error: 'Invalid id' });
  }

  const db = getDb();
  const row = db.prepare(
    'SELECT disk_filename FROM user_uploads WHERE id = ? AND user_id = ?'
  ).get(id, req.uploadUserId);

  if (!row) {
    return res.status(404).json({ error: 'File not found' });
  }

  db.prepare('DELETE FROM user_uploads WHERE id = ? AND user_id = ?').run(id, req.uploadUserId);

  const abs = diskPath(row.disk_filename);
  if (abs && fs.existsSync(abs)) {
    try {
      fs.unlinkSync(abs);
    } catch (e) {
      console.warn('Could not delete file from disk:', e.message);
    }
  }

  res.json({ success: true });
});

module.exports = router;
