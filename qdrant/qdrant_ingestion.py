
import os
import re
import sys
import json
import time
import random
import uuid
import argparse
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Iterable, Optional, Tuple

import requests
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct


# -----------------------------
# Config
# -----------------------------
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "cu_denver_database")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "embeddinggemma")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))

DEFAULT_TOPIC = "cu_denver_database"

# Local state/caches (safe to delete; will just re-embed everything)
DEFAULT_CACHE_DIR = ".ingest_cache"
DEFAULT_PAGE_CACHE_FILE = "page_hashes.json"
DEFAULT_STATE_FILE = "ingest_state.json"


# -----------------------------
# Utilities
# -----------------------------
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sha256_text(s: str) -> str:
    return sha256_bytes(s.encode("utf-8", errors="ignore"))

def canonicalize_url(url: str) -> str:
    """Basic URL canonicalization: strip fragment + common tracking params."""
    if not url:
        return url
    # strip fragment
    url = url.split("#", 1)[0]
    # strip common query params
    if "?" in url:
        base, q = url.split("?", 1)
        params = q.split("&")
        keep = []
        for p in params:
            key = p.split("=", 1)[0].lower()
            if key in {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "gclid", "fbclid"}:
                continue
            keep.append(p)
        url = base + ("?" + "&".join(keep) if keep else "")
    # normalize trailing slash lightly
    if url.endswith("/") and len(url) > 8:
        url = url[:-1]
    return url

def safe_get(d: Dict[str, Any], key: str, default=""):
    v = d.get(key, default)
    return v if v is not None else default

def load_json(path: Path, default):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default

def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


# -----------------------------
# Reading crawl JSON
# -----------------------------
def read_crawl_response_json(file_path: str) -> List[Dict[str, Any]]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        return data["data"]
    if isinstance(data, list):
        return data
    raise ValueError(
        f"Unexpected JSON structure in {file_path}. Expected object with 'data' array or array of objects."
    )


# -----------------------------
# Chunking (heading-aware)
# -----------------------------
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)\s*$", re.MULTILINE)

def normalize_markdown(md: str) -> str:
    """Light cleanup: remove excessive whitespace, trim."""
    md = md.replace("\r\n", "\n").replace("\r", "\n")
    md = re.sub(r"\n{4,}", "\n\n\n", md)  # keep some separation
    return md.strip()

def split_by_headings(md: str) -> List[Tuple[str, str]]:
    """
    Returns list of (heading, section_text).
    heading = "H2: Financial Aid" etc. or "" for preface content.
    """
    md = normalize_markdown(md)
    matches = list(_HEADING_RE.finditer(md))
    if not matches:
        return [("", md)]

    sections: List[Tuple[str, str]] = []

    # Preface before first heading
    first_start = matches[0].start()
    if first_start > 0:
        pre = md[:first_start].strip()
        if pre:
            sections.append(("", pre))

    for i, m in enumerate(matches):
        level = len(m.group(1))
        title = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        body = md[start:end].strip()
        heading = f"H{level}: {title}"
        if body:
            sections.append((heading, body))
        else:
            # even if empty, keep heading as context-less section
            sections.append((heading, ""))

    return sections

def chunk_paragraphs(text: str, max_chars: int, overlap_chars: int) -> List[str]:
    """
    Chunk using paragraph boundaries. Keeps overlap by reusing trailing paragraphs until overlap_chars reached.
    """
    text = text.strip()
    if not text:
        return []

    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []

    cur: List[str] = []
    cur_len = 0

    def flush():
        nonlocal cur, cur_len
        if cur:
            chunks.append("\n\n".join(cur).strip())
            cur = []
            cur_len = 0

    for p in paras:
        p_len = len(p) + (2 if cur else 0)
        if cur_len + p_len <= max_chars:
            cur.append(p)
            cur_len += p_len
        else:
            # flush current chunk
            flush()

            # if paragraph itself is too large, hard-split it
            if len(p) > max_chars:
                start = 0
                while start < len(p):
                    chunks.append(p[start:start + max_chars])
                    start += max_chars - overlap_chars
            else:
                cur.append(p)
                cur_len = len(p)

    flush()

    # Apply overlap across chunk boundaries by reusing trailing text (approx by chars)
    if overlap_chars <= 0 or len(chunks) <= 1:
        return chunks

    overlapped: List[str] = []
    for i, ch in enumerate(chunks):
        if i == 0:
            overlapped.append(ch)
            continue
        prev = overlapped[-1]
        # take tail of prev up to overlap_chars
        tail = prev[-overlap_chars:]
        overlapped.append((tail + "\n\n" + ch).strip())
    return overlapped

def chunk_markdown(md: str, max_chars: int = 2200, overlap_chars: int = 250) -> List[Dict[str, Any]]:
    """
    Produce chunks with section heading metadata.
    Returns list of dicts: {text, section_heading}
    """
    out: List[Dict[str, Any]] = []
    for heading, body in split_by_headings(md):
        # include heading text as context inside the chunk too (helps retrieval)
        section_text = (f"{heading}\n\n{body}".strip() if heading else body).strip()
        if not section_text:
            continue

        for ch in chunk_paragraphs(section_text, max_chars=max_chars, overlap_chars=overlap_chars):
            out.append({"text": ch, "section_heading": heading})
    return out


# -----------------------------
# Embeddings (Ollama) with retries
# -----------------------------
def embed_texts(
    texts: List[str],
    model: str = EMBED_MODEL,
    max_retries: int = 6,
    timeout_s: int = 120,
) -> List[List[float]]:
    """
    Embeds a list of texts via Ollama.

    Tries the batch endpoint /api/embed (newer Ollama),
    and falls back to /api/embeddings (widely supported).
    """
    batch_url = f"{OLLAMA_HOST}/api/embed"
    single_url = f"{OLLAMA_HOST}/api/embed"
    last_err = None

    for attempt in range(max_retries):
        try:
            # Try batch endpoint first
            resp = requests.post(batch_url, json={"model": model, "input": texts}, timeout=timeout_s)

            if resp.status_code == 404:
                raise FileNotFoundError("Batch endpoint /api/embed not found (404).")

            resp.raise_for_status()
            data = resp.json()
            vectors = data.get("embeddings")
            if not isinstance(vectors, list):
                raise RuntimeError(f"Unexpected /api/embed response: {data}")
            return vectors

        except FileNotFoundError:
            # Fallback: /api/embeddings (one-by-one)
            try:
                vectors = []
                for t in texts:
                    r = requests.post(single_url, json={"model": model, "prompt": t}, timeout=timeout_s)
                    r.raise_for_status()
                    d = r.json()
                    v = d.get("embedding")
                    if not isinstance(v, list):
                        raise RuntimeError(f"Unexpected /api/embeddings response: {d}")
                    vectors.append(v)
                return vectors
            except Exception as e:
                last_err = e

        except Exception as e:
            last_err = e

        if attempt < max_retries - 1:
            sleep_s = (2 ** attempt) + random.random()
            print(f"[embed retry {attempt+1}/{max_retries}] {last_err} -> sleep {sleep_s:.1f}s")
            time.sleep(sleep_s)

    raise RuntimeError(f"Failed to get embeddings from Ollama after retries: {last_err}")
# -----------------------------
# Deterministic IDs (no duplicates)
# -----------------------------

QDRANT_ID_NAMESPACE = uuid.UUID("6ba7b811-9dad-11d1-80b4-00c04fd430c8")  # DNS namespace (standard)

def stable_point_id(url: str, chunk_index: int, chunk_text: str) -> str:
    """
    Deterministic, Qdrant-valid ID (UUIDv5).
    Same (url, chunk_index, chunk_text) -> same UUID.
    """
    url = canonicalize_url(url or "")
    h_chunk = sha256_text(chunk_text)
    raw = f"{url}|{chunk_index}|{h_chunk}"
    return str(uuid.uuid5(QDRANT_ID_NAMESPACE, raw))

# -----------------------------
# Qdrant setup
# -----------------------------
def ensure_collection(client: QdrantClient, collection_name: str) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if collection_name in existing:
        print(f"Using existing collection '{collection_name}'...")
        return
    print(f"Creating collection '{collection_name}' (dim={EMBED_DIM})...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )


# -----------------------------
# Main ingestion
# -----------------------------
def ingest_crawl_response(
    json_path: str,
    collection_name: str = COLLECTION,
    topic: str = DEFAULT_TOPIC,
    source_name: Optional[str] = None,
    chunk_size_chars: int = 2200,
    chunk_overlap_chars: int = 250,
    embed_batch_size: int = 32,
    upsert_batch_size: int = 128,
    cache_dir: str = DEFAULT_CACHE_DIR,
    resume: bool = True,
) -> None:
    client = QdrantClient(url=QDRANT_URL)
    ensure_collection(client, collection_name)

    pages = read_crawl_response_json(json_path)
    print(f"Loaded {len(pages)} pages from {json_path}")

    cache_path = Path(cache_dir)
    page_hashes_path = cache_path / DEFAULT_PAGE_CACHE_FILE
    state_path = cache_path / DEFAULT_STATE_FILE

    page_hashes: Dict[str, str] = load_json(page_hashes_path, default={})
    state: Dict[str, Any] = load_json(state_path, default={})
    source_key = str(Path(json_path).resolve())

    if source_name is None:
        source_name = Path(json_path).stem

    start_idx = 0
    if resume and state.get("source_key") == source_key:
        start_idx = int(state.get("last_page_index", 0))
        if start_idx > 0:
            print(f"[resume] Continuing from page index {start_idx}")

    # Working buffers
    embed_text_buf: List[str] = []
    meta_buf: List[Dict[str, Any]] = []

    def flush_embed_and_upsert():
        nonlocal embed_text_buf, meta_buf
        if not embed_text_buf:
            return

        vectors = embed_texts(embed_text_buf)

        points: List[PointStruct] = []
        for vec, meta in zip(vectors, meta_buf):
            points.append(
                PointStruct(
                    id=meta["point_id"],
                    vector=vec,
                    payload=meta["payload"],
                )
            )

        # Upsert in sub-batches
        for i in range(0, len(points), upsert_batch_size):
            batch = points[i:i + upsert_batch_size]
            client.upsert(collection_name=collection_name, points=batch)

        print(f"Upserted {len(points)} chunks.")
        embed_text_buf = []
        meta_buf = []

    total_pages = 0
    skipped_pages = 0
    total_chunks = 0

    for page_idx in range(start_idx, len(pages)):
        page = pages[page_idx]
        md = safe_get(page, "markdown", "").strip()
        if not md:
            continue

        url = canonicalize_url(safe_get(page, "url", ""))
        title = safe_get(page, "title", "")
        md_norm = normalize_markdown(md)

        page_hash = sha256_text(md_norm)
        prev_hash = page_hashes.get(url)

        if prev_hash == page_hash:
            skipped_pages += 1
            # update resume state anyway
            if resume and (page_idx % 50 == 0):
                state.update({"source_key": source_key, "last_page_index": page_idx})
                save_json(state_path, state)
            continue

        # Chunk the page
        chunks = chunk_markdown(md_norm, max_chars=chunk_size_chars, overlap_chars=chunk_overlap_chars)
        if not chunks:
            page_hashes[url] = page_hash
            continue

        # Build points for chunks
        for ci, ch in enumerate(chunks):
            chunk_text = ch["text"]
            section_heading = ch.get("section_heading", "")

            # Enrich embedding input with title + url (helps retrieval a lot)
            embed_input = f"Title: {title}\nURL: {url}\n{section_heading}\n\n{chunk_text}".strip()

            pid = stable_point_id(url, ci, chunk_text)

            payload = {
                "text": chunk_text,
                "url": url,
                "title": title,
                "topic": topic,
                "source_name": source_name,
                "source_file": str(Path(json_path).resolve()),
                "page_index": page_idx,
                "chunk_index": ci,
                "section_heading": section_heading,
                "page_hash": page_hash,
            }

            embed_text_buf.append(embed_input)
            meta_buf.append({"point_id": pid, "payload": payload})

            total_chunks += 1

            if len(embed_text_buf) >= embed_batch_size:
                flush_embed_and_upsert()

        # Mark page as updated in cache after preparing chunks
        page_hashes[url] = page_hash
        total_pages += 1

        # Periodically save caches/state and flush any remaining buffers
        if resume and (page_idx % 50 == 0):
            flush_embed_and_upsert()
            save_json(page_hashes_path, page_hashes)
            state.update({"source_key": source_key, "last_page_index": page_idx})
            save_json(state_path, state)
            print(f"[progress] pages processed={total_pages}, skipped={skipped_pages}, chunks={total_chunks}")

    # Final flush + save caches
    flush_embed_and_upsert()
    save_json(page_hashes_path, page_hashes)
    if resume:
        state.update({"source_key": source_key, "last_page_index": len(pages)})
        save_json(state_path, state)

    print("Done.")
    print(f"Pages processed (changed/new): {total_pages}")
    print(f"Pages skipped (unchanged):     {skipped_pages}")
    print(f"Total chunks upserted:         {total_chunks}")


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Ingest a crawl response JSON into Qdrant (improved).")
    parser.add_argument("json_file", type=str, help="Path to crawl response JSON")

    parser.add_argument("--collection", type=str, default=COLLECTION, help=f"Qdrant collection (default: {COLLECTION})")
    parser.add_argument("--topic", type=str, default=DEFAULT_TOPIC, help="Topic/category (default: cu_denver)")
    parser.add_argument("--source-name", type=str, default=None, help="Source name label (default: filename stem)")

    parser.add_argument("--chunk-size", type=int, default=2200, help="Max chunk size in characters (default: 2200)")
    parser.add_argument("--chunk-overlap", type=int, default=250, help="Chunk overlap in characters (default: 250)")

    parser.add_argument("--embed-batch", type=int, default=32, help="Texts per embedding request (default: 32)")
    parser.add_argument("--upsert-batch", type=int, default=128, help="Points per Qdrant upsert (default: 128)")

    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR, help="Cache dir for hashes/state")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume cursor")

    args = parser.parse_args()

    ingest_crawl_response(
        json_path=args.json_file,
        collection_name=args.collection,
        topic=args.topic,
        source_name=args.source_name,
        chunk_size_chars=args.chunk_size,
        chunk_overlap_chars=args.chunk_overlap,
        embed_batch_size=args.embed_batch,
        upsert_batch_size=args.upsert_batch,
        cache_dir=args.cache_dir,
        resume=(not args.no_resume),
    )


if __name__ == "__main__":
    main()