import { logout } from '../api';

export default function Header({ user, onLogout }) {
  async function handleLogout() {
    await logout();
    onLogout();
  }

  return (
    <header style={styles.header}>
      <div style={styles.brand}>
        <span style={styles.icon}>🤖</span>
        <span style={styles.name}>Lynx</span>
        <span style={styles.sep}>·</span>
        <span style={styles.section}>Connected Services</span>
      </div>
      <div style={styles.right}>
        <span style={styles.username}>{user.username}</span>
        <button style={styles.logoutBtn} onClick={handleLogout}>
          Sign out
        </button>
      </div>
    </header>
  );
}

const styles = {
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '0 32px',
    height: 60,
    background: '#1a1d27',
    borderBottom: '1px solid #2d3748',
    position: 'sticky',
    top: 0,
    zIndex: 10,
  },
  brand: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
  },
  icon: { fontSize: 20 },
  name: {
    fontWeight: 700,
    fontSize: 18,
    color: '#e2e8f0',
  },
  sep: { color: '#4a5568', fontSize: 18 },
  section: { color: '#718096', fontSize: 14 },
  right: {
    display: 'flex',
    alignItems: 'center',
    gap: 16,
  },
  username: {
    fontSize: 14,
    color: '#a0aec0',
  },
  logoutBtn: {
    background: 'transparent',
    border: '1px solid #2d3748',
    borderRadius: 6,
    color: '#a0aec0',
    padding: '5px 12px',
    fontSize: 13,
    cursor: 'pointer',
  },
};
