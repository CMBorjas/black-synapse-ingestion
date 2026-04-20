import { useState, useEffect, useCallback, useRef } from 'react';
import { getUploads, uploadFiles, deleteUpload } from '../api';

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function FileUploads({ onToast }) {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef(null);

  const base = import.meta.env.VITE_API_URL || '';

  const load = useCallback(async () => {
    try {
      const data = await getUploads();
      setItems(Array.isArray(data) ? data : []);
    } catch {
      onToast?.('error', 'Could not load your files');
      setItems([]);
    } finally {
      setLoading(false);
    }
  }, [onToast]);

  useEffect(() => {
    load();
  }, [load]);

  async function handleFiles(fileList) {
    const files = Array.from(fileList || []).filter(Boolean);
    if (files.length === 0) return;
    setUploading(true);
    try {
      await uploadFiles(files);
      onToast?.('success', files.length === 1 ? 'File uploaded' : `${files.length} files uploaded`);
      await load();
    } catch (err) {
      const msg = err.response?.data?.error || err.message || 'Upload failed';
      onToast?.('error', msg);
    } finally {
      setUploading(false);
    }
  }

  async function handleDelete(id, name) {
    if (!window.confirm(`Remove “${name}” from the portal?`)) return;
    try {
      await deleteUpload(id);
      onToast?.('success', 'File removed');
      setItems((prev) => prev.filter((x) => x.id !== id));
    } catch {
      onToast?.('error', 'Could not delete file');
    }
  }

  return (
    <section style={styles.section}>
      <div style={styles.sectionHead}>
        <h2 style={styles.sectionTitle}>Your uploads</h2>
        <p style={styles.sectionBlurb}>
          Drop files here for Lynx workflows or your team. Stored on this server with your account only.
        </p>
      </div>

      <div
        style={{
          ...styles.dropzone,
          borderColor: dragOver ? '#667eea' : '#2d3748',
          background: dragOver ? 'rgba(102, 126, 234, 0.08)' : '#161922',
        }}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          handleFiles(e.dataTransfer.files);
        }}
        onClick={() => !uploading && inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            inputRef.current?.click();
          }
        }}
      >
        <input
          ref={inputRef}
          type="file"
          multiple
          style={{ display: 'none' }}
          onChange={(e) => {
            handleFiles(e.target.files);
            e.target.value = '';
          }}
        />
        <span style={styles.dropIcon}>⬆</span>
        <p style={styles.dropTitle}>{uploading ? 'Uploading…' : 'Click or drag files here'}</p>
        <p style={styles.dropHint}>Up to 10 files per batch · max size set on server (default 25 MB each)</p>
      </div>

      {loading ? (
        <div style={styles.loadingRow}>
          <div style={styles.spinner} />
          <span>Loading files…</span>
        </div>
      ) : items.length === 0 ? (
        <p style={styles.empty}>No files yet.</p>
      ) : (
        <ul style={styles.list}>
          {items.map((f) => (
            <li key={f.id} style={styles.row}>
              <div style={styles.rowMain}>
                <span style={styles.fileName} title={f.originalFilename}>
                  {f.originalFilename}
                </span>
                <span style={styles.meta}>
                  {formatSize(f.sizeBytes)}
                  {f.createdAt ? ` · ${new Date(f.createdAt).toLocaleString()}` : ''}
                </span>
              </div>
              <div style={styles.rowActions}>
                <a
                  href={`${base}/api/uploads/${f.id}/file`}
                  download={f.originalFilename}
                  style={styles.linkBtn}
                >
                  Download
                </a>
                <button type="button" style={styles.dangerBtn} onClick={() => handleDelete(f.id, f.originalFilename)}>
                  Remove
                </button>
              </div>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

const styles = {
  section: {
    marginTop: 48,
    paddingTop: 32,
    borderTop: '1px solid #2d3748',
  },
  sectionHead: {
    marginBottom: 20,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 700,
    color: '#e2e8f0',
    marginBottom: 6,
  },
  sectionBlurb: {
    fontSize: 14,
    color: '#718096',
    lineHeight: 1.55,
    maxWidth: 560,
    margin: 0,
  },
  dropzone: {
    border: '2px dashed #2d3748',
    borderRadius: 12,
    padding: '28px 20px',
    textAlign: 'center',
    cursor: 'pointer',
    transition: 'border-color 0.15s, background 0.15s',
  },
  dropIcon: {
    fontSize: 28,
    display: 'block',
    marginBottom: 8,
    opacity: 0.85,
  },
  dropTitle: {
    color: '#e2e8f0',
    fontSize: 15,
    fontWeight: 600,
    margin: '0 0 6px',
  },
  dropHint: {
    color: '#4a5568',
    fontSize: 13,
    margin: 0,
  },
  loadingRow: {
    display: 'flex',
    alignItems: 'center',
    gap: 12,
    color: '#718096',
    fontSize: 14,
    marginTop: 24,
  },
  spinner: {
    width: 18,
    height: 18,
    border: '2px solid #2d3748',
    borderTop: '2px solid #667eea',
    borderRadius: '50%',
    animation: 'spin 0.8s linear infinite',
  },
  empty: {
    color: '#4a5568',
    fontSize: 14,
    marginTop: 20,
  },
  list: {
    listStyle: 'none',
    margin: '24px 0 0',
    padding: 0,
    display: 'flex',
    flexDirection: 'column',
    gap: 10,
  },
  row: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 16,
    flexWrap: 'wrap',
    background: '#161922',
    border: '1px solid #2d3748',
    borderRadius: 8,
    padding: '12px 14px',
  },
  rowMain: {
    minWidth: 0,
    flex: 1,
  },
  fileName: {
    display: 'block',
    color: '#e2e8f0',
    fontSize: 14,
    fontWeight: 500,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  meta: {
    display: 'block',
    fontSize: 12,
    color: '#718096',
    marginTop: 4,
  },
  rowActions: {
    display: 'flex',
    gap: 10,
    flexShrink: 0,
  },
  linkBtn: {
    fontSize: 13,
    color: '#90cdf4',
    textDecoration: 'none',
    padding: '6px 10px',
    borderRadius: 6,
    border: '1px solid #2c5282',
    background: 'rgba(44, 82, 130, 0.25)',
  },
  dangerBtn: {
    fontSize: 13,
    color: '#feb2b2',
    padding: '6px 10px',
    borderRadius: 6,
    border: '1px solid #742a2a',
    background: 'rgba(116, 42, 42, 0.35)',
    cursor: 'pointer',
  },
};
