import { useState } from 'react';
import { disconnectService, getOAuthUrl } from '../api';

const SERVICE_META = {
  google: {
    label: 'Google',
    description: 'Gmail · Calendar · Drive',
    icon: '🔵',
    color: '#4285f4',
  },
  microsoft: {
    label: 'Microsoft',
    description: 'Outlook · Teams · OneDrive',
    icon: '🟦',
    color: '#00a4ef',
  },
  discord: {
    label: 'Discord',
    description: 'Messaging · Guilds',
    icon: '🟣',
    color: '#5865f2',
  },
  notion: {
    label: 'Notion',
    description: 'Pages · Databases',
    icon: '⬜',
    color: '#e2e8f0',
  },
};

export default function ServiceCard({ service, connection, onDisconnect }) {
  const meta = SERVICE_META[service] || { label: service, description: '', icon: '🔌', color: '#718096' };
  const isConnected = Boolean(connection);
  const inN8n = Boolean(connection?.n8n_credential_id);
  const [disconnecting, setDisconnecting] = useState(false);
  const [error, setError] = useState('');

  async function handleDisconnect() {
    setDisconnecting(true);
    setError('');
    try {
      await disconnectService(service);
      onDisconnect(service);
    } catch (err) {
      setError(err.response?.data?.error || 'Disconnect failed');
    } finally {
      setDisconnecting(false);
    }
  }

  function handleConnect() {
    // Redirect the browser to the Express OAuth start route
    window.location.href = getOAuthUrl(service);
  }

  return (
    <div style={{ ...styles.card, borderColor: isConnected ? meta.color + '55' : '#2d3748' }}>
      {/* Service icon + info */}
      <div style={styles.top}>
        <span style={styles.icon}>{meta.icon}</span>
        <div>
          <div style={styles.label}>{meta.label}</div>
          <div style={styles.description}>{meta.description}</div>
        </div>
        {/* Status: Lynx always stores tokens locally; n8n row only exists if provision returned an id */}
        <div
          style={{
            ...styles.badge,
            background: !isConnected ? '#2d3748' : inN8n ? '#22543d' : '#744210',
            color: !isConnected ? '#718096' : inN8n ? '#68d391' : '#f6e05e',
          }}
        >
          {!isConnected
            ? '○ Not connected'
            : inN8n
              ? '● In n8n'
              : '● Local only'}
        </div>
      </div>

      {/* Connected metadata */}
      {isConnected && connection.connected_at && (
        <p style={styles.meta}>
          Connected {new Date(connection.connected_at).toLocaleDateString()}
        </p>
      )}

      {isConnected && !inN8n && (
        <p style={styles.n8nWarn}>
          {meta.label} is linked in Lynx, but no credential was saved in n8n (provision failed or never ran).
          Fix <code style={styles.code}>N8N_BASE_URL</code> / API key, then Reconnect. For Gmail, use the Gmail node —
          it expects a &quot;Gmail OAuth2 API&quot; credential (what Lynx creates).
        </p>
      )}

      {error && <p style={styles.error}>{error}</p>}

      {/* Action button */}
      {isConnected ? (
        <div style={styles.actions}>
          <button style={styles.reconnectBtn} onClick={handleConnect}>
            Reconnect
          </button>
          <button
            style={styles.disconnectBtn}
            onClick={handleDisconnect}
            disabled={disconnecting}
          >
            {disconnecting ? 'Disconnecting…' : 'Disconnect'}
          </button>
        </div>
      ) : (
        <button style={{ ...styles.connectBtn, background: meta.color }} onClick={handleConnect}>
          Connect {meta.label}
        </button>
      )}
    </div>
  );
}

const styles = {
  card: {
    background: '#1a1d27',
    border: '1px solid',
    borderRadius: 12,
    padding: 24,
    display: 'flex',
    flexDirection: 'column',
    gap: 14,
    transition: 'border-color 0.2s',
  },
  top: {
    display: 'flex',
    alignItems: 'center',
    gap: 14,
  },
  icon: { fontSize: 28, lineHeight: 1 },
  label: {
    fontWeight: 600,
    fontSize: 16,
    color: '#e2e8f0',
  },
  description: {
    fontSize: 13,
    color: '#718096',
    marginTop: 2,
  },
  badge: {
    marginLeft: 'auto',
    fontSize: 12,
    fontWeight: 500,
    padding: '4px 10px',
    borderRadius: 20,
    whiteSpace: 'nowrap',
  },
  meta: {
    fontSize: 12,
    color: '#4a5568',
  },
  n8nWarn: {
    fontSize: 12,
    color: '#ecc94b',
    lineHeight: 1.5,
    margin: 0,
    padding: '10px 12px',
    background: '#3d2f00',
    borderRadius: 8,
    border: '1px solid #744210',
  },
  code: {
    fontSize: 11,
    background: '#1a202c',
    padding: '1px 5px',
    borderRadius: 4,
  },
  error: {
    fontSize: 13,
    color: '#fc8181',
  },
  actions: {
    display: 'flex',
    gap: 8,
  },
  connectBtn: {
    color: '#fff',
    border: 'none',
    borderRadius: 8,
    padding: '10px 16px',
    fontWeight: 600,
    fontSize: 14,
    cursor: 'pointer',
    width: '100%',
  },
  reconnectBtn: {
    flex: 1,
    background: 'transparent',
    border: '1px solid #2d3748',
    borderRadius: 8,
    color: '#a0aec0',
    padding: '9px 12px',
    fontSize: 13,
    cursor: 'pointer',
  },
  disconnectBtn: {
    flex: 1,
    background: 'transparent',
    border: '1px solid #742a2a',
    borderRadius: 8,
    color: '#fc8181',
    padding: '9px 12px',
    fontSize: 13,
    cursor: 'pointer',
  },
};
