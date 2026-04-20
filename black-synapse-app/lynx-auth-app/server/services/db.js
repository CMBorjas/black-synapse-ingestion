const Database = require('better-sqlite3');
const path = require('path');

let db;

function getDb() {
  if (!db) {
    db = new Database(path.join(__dirname, '../data/lynx.db'));
    db.pragma('journal_mode = WAL'); // Better concurrent read performance
  }
  return db;
}

function initDb() {
  const db = getDb();

  db.exec(`
    CREATE TABLE IF NOT EXISTS users (
      id            INTEGER PRIMARY KEY AUTOINCREMENT,
      username      TEXT UNIQUE NOT NULL,
      password_hash TEXT NOT NULL,
      created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS connections (
      id                   INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id              INTEGER NOT NULL,
      service              TEXT NOT NULL,
      access_token         TEXT,
      refresh_token        TEXT,
      n8n_credential_id    TEXT,
      connected_at         DATETIME DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY (user_id) REFERENCES users(id),
      UNIQUE(user_id, service)
    );
  `);

  const cols = db.prepare('PRAGMA table_info(connections)').all();
  if (!cols.some((c) => c.name === 'n8n_credential_id')) {
    db.exec('ALTER TABLE connections ADD COLUMN n8n_credential_id TEXT');
  }

  console.log('📦 Database initialized');
}

module.exports = { getDb, initDb };
