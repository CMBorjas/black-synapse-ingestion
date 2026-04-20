import { useState, useEffect, useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';
import { getConnections } from '../api';
import Header from '../components/Header';
import ServiceCard from '../components/ServiceCard';
import FileUploads from '../components/FileUploads';

const SERVICES = ['google', 'microsoft', 'discord', 'notion'];

export default function Dashboard({ user, onLogout }) {
  const [connections, setConnections] = useState([]); // array of { service, n8n_credential_id, connected_at }
  const [loading, setLoading] = useState(true);
  const [toast, setToast] = useState(null); // { type: 'success' | 'error', message }
  const [searchParams, setSearchParams] = useSearchParams();

  // Build a lookup map: { google: {...}, microsoft: {...}, ... }
  const connectionMap = connections.reduce((acc, c) => {
    acc[c.service] = c;
    return acc;
  }, {});

  const fetchConnections = useCallback(async () => {
    try {
      const data = await getConnections();
      setConnections(data);
    } catch {
      // If fetch fails, leave connections empty
    } finally {
      setLoading(false);
    }
  }, []);

  // Load connections on mount
  useEffect(() => {
    fetchConnections();
  }, [fetchConnections]);

  // Handle OAuth redirect callbacks — Express redirects back here with query params
  useEffect(() => {
    const success = searchParams.get('oauth_success');
    const error = searchParams.get('oauth_error');
    const service = searchParams.get('service');

    if (success === 'true' && service) {
      showToast('success', `${capitalize(service)} connected successfully`);
      fetchConnections(); // Refresh the list
    } else if (error && service) {
      const http = searchParams.get('n8n_http');
      const hint = searchParams.get('n8n_hint');
      let msg = `Failed to connect ${capitalize(service)}: ${error}`;
      if (http) msg += ` (HTTP ${http})`;
      if (hint) msg += ` — ${hint}`;
      showToast('error', msg);
    }

    // Clear query params so re-visiting the page doesn't re-trigger
    if (success || error) {
      setSearchParams({});
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  function showToast(type, message) {
    setToast({ type, message });
    setTimeout(() => setToast(null), 4000);
  }

  function handleDisconnect(service) {
    setConnections((prev) => prev.filter((c) => c.service !== service));
    showToast('success', `${capitalize(service)} disconnected`);
  }

  return (
    <div style={styles.page}>
      <Header user={user} onLogout={onLogout} />

      <main style={styles.main}>
        <div style={styles.hero}>
          <h1 style={styles.title}>Connected Services</h1>
          <p style={styles.subtitle}>
            Manage the accounts Lynx can access on your behalf. Each connection is stored
            as a credential in n8n and available to any workflow immediately.
          </p>
        </div>

        {/* Toast notification */}
        {toast && (
          <div style={{ ...styles.toast, background: toast.type === 'success' ? '#22543d' : '#742a2a', color: toast.type === 'success' ? '#68d391' : '#fc8181' }}>
            {toast.type === 'success' ? '✓ ' : '✗ '}{toast.message}
          </div>
        )}

        {loading ? (
          <div style={styles.loadingRow}>
            <div style={styles.spinner} />
            <span>Loading connections…</span>
          </div>
        ) : (
          <div style={styles.grid}>
            {SERVICES.map((service) => (
              <ServiceCard
                key={service}
                service={service}
                connection={connectionMap[service] || null}
                onDisconnect={handleDisconnect}
              />
            ))}
          </div>
        )}

        {/* n8n status note */}
        <FileUploads onToast={(type, message) => showToast(type, message)} />

        <p style={styles.footer}>
          Credentials are provisioned directly into your n8n instance and encrypted at rest.
          Disconnecting removes them from n8n immediately.
        </p>
      </main>
    </div>
  );
}

function capitalize(s) {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

const styles = {
  page: {
    minHeight: '100vh',
    background: '#0f1117',
  },
  main: {
    maxWidth: 860,
    margin: '0 auto',
    padding: '40px 24px 80px',
  },
  hero: {
    marginBottom: 32,
  },
  title: {
    fontSize: 26,
    fontWeight: 700,
    color: '#e2e8f0',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 15,
    color: '#718096',
    lineHeight: 1.6,
    maxWidth: 580,
  },
  toast: {
    borderRadius: 8,
    padding: '12px 16px',
    fontSize: 14,
    fontWeight: 500,
    marginBottom: 24,
  },
  loadingRow: {
    display: 'flex',
    alignItems: 'center',
    gap: 12,
    color: '#718096',
    fontSize: 14,
    marginTop: 40,
  },
  spinner: {
    width: 20,
    height: 20,
    border: '2px solid #2d3748',
    borderTop: '2px solid #667eea',
    borderRadius: '50%',
    animation: 'spin 0.8s linear infinite',
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(380px, 1fr))',
    gap: 16,
  },
  footer: {
    marginTop: 40,
    fontSize: 13,
    color: '#4a5568',
    borderTop: '1px solid #2d3748',
    paddingTop: 20,
  },
};
