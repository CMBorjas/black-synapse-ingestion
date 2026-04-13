#!/usr/bin/env python3
"""
Ingest CU Denver crawl or faculty JSON into Qdrant using Azure OpenAI embeddings.

Supported input formats
-----------------------
crawl   -- [{markdown, url, title}, ...]         (e.g. cudenver_crawl_combined.json)
faculty -- [{text, metadata: {...}}, ...]         (e.g. cu_denver_faculty.json)

Format is auto-detected from the first record, or forced with --format.

Azure env vars required
-----------------------
AZURE_OPENAI_API_KEY
AZURE_OPENAI_ENDPOINT        e.g. https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT      deployment name, e.g. text-embedding-3-small
AZURE_OPENAI_API_VERSION     (optional, default: 2024-02-01)

Usage
-----
python qdrant/ingest_azure.py qdrant/cudenver_crawl_combined.json
python qdrant/ingest_azure.py qdrant/cu_denver_faculty.json
python qdrant/ingest_azure.py qdrant/cudenver_crawl_combined.json --collection cu_denver_azure
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
import random
import uuid
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (two levels up from this script)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
from typing import Any, Dict, List, Optional, Tuple

from openai import AzureOpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# ── Config ────────────────────────────────────────────────────────────
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
DEFAULT_COLLECTION = os.getenv("QDRANT_COLLECTION", "cu_denver_azure")
EMBED_DIM = 1536  # text-embedding-3-small

AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "text-embedding-3-small")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

DEFAULT_CACHE_DIR = "qdrant/.ingest_cache_azure"
QDRANT_ID_NAMESPACE = uuid.UUID("6ba7b811-9dad-11d1-80b4-00c04fd430c8")

CHUNK_SIZE_CHARS = 2200
CHUNK_OVERLAP_CHARS = 250
EMBED_BATCH_SIZE = 64   # Azure supports larger batches than Ollama
UPSERT_BATCH_SIZE = 128


# ── Utilities ─────────────────────────────────────────────────────────

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def stable_id(*parts: str) -> str:
    """Deterministic UUIDv5 from arbitrary string parts."""
    raw = "|".join(parts)
    return str(uuid.uuid5(QDRANT_ID_NAMESPACE, raw))


def canonicalize_url(url: str) -> str:
    if not url:
        return url
    url = url.split("#", 1)[0]
    if "?" in url:
        base, q = url.split("?", 1)
        skip = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "gclid", "fbclid"}
        kept = [p for p in q.split("&") if p.split("=", 1)[0].lower() not in skip]
        url = base + ("?" + "&".join(kept) if kept else "")
    if url.endswith("/") and len(url) > 8:
        url = url[:-1]
    return url


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


# ── Markdown chunking (reused from qdrant_ingestion.py) ──────────────

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)\s*$", re.MULTILINE)


def normalize_markdown(md: str) -> str:
    md = md.replace("\r\n", "\n").replace("\r", "\n")
    md = re.sub(r"\n{4,}", "\n\n\n", md)
    return md.strip()


def split_by_headings(md: str) -> List[Tuple[str, str]]:
    md = normalize_markdown(md)
    matches = list(_HEADING_RE.finditer(md))
    if not matches:
        return [("", md)]
    sections: List[Tuple[str, str]] = []
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
        sections.append((f"H{level}: {title}", body))
    return sections


def chunk_paragraphs(text: str, max_chars: int, overlap_chars: int) -> List[str]:
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
            cur, cur_len = [], 0

    for p in paras:
        p_len = len(p) + (2 if cur else 0)
        if cur_len + p_len <= max_chars:
            cur.append(p)
            cur_len += p_len
        else:
            flush()
            if len(p) > max_chars:
                start = 0
                while start < len(p):
                    chunks.append(p[start:start + max_chars])
                    start += max_chars - overlap_chars
            else:
                cur.append(p)
                cur_len = len(p)
    flush()

    if overlap_chars <= 0 or len(chunks) <= 1:
        return chunks
    overlapped = [chunks[0]]
    for ch in chunks[1:]:
        tail = overlapped[-1][-overlap_chars:]
        overlapped.append((tail + "\n\n" + ch).strip())
    return overlapped


def chunk_markdown(md: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for heading, body in split_by_headings(md):
        section_text = (f"{heading}\n\n{body}".strip() if heading else body).strip()
        if not section_text:
            continue
        for ch in chunk_paragraphs(section_text, CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS):
            out.append({"text": ch, "section_heading": heading})
    return out


# ── Azure embedding ───────────────────────────────────────────────────

def make_azure_client() -> AzureOpenAI:
    if not AZURE_API_KEY:
        sys.exit("AZURE_OPENAI_API_KEY is not set.")
    if not AZURE_ENDPOINT:
        sys.exit("AZURE_OPENAI_ENDPOINT is not set.")
    return AzureOpenAI(
        api_key=AZURE_API_KEY,
        azure_endpoint=AZURE_ENDPOINT,
        api_version=AZURE_API_VERSION,
    )


def embed_texts(
    texts: List[str],
    client: AzureOpenAI,
    max_retries: int = 5,
) -> List[List[float]]:
    last_err = None
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                model=AZURE_DEPLOYMENT,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                sleep_s = (2 ** attempt) + random.random()
                print(f"  [embed retry {attempt + 1}/{max_retries}] {e} → sleep {sleep_s:.1f}s")
                time.sleep(sleep_s)
    raise RuntimeError(f"Embedding failed after {max_retries} retries: {last_err}")


# ── Qdrant setup ──────────────────────────────────────────────────────

def ensure_collection(client: QdrantClient, collection_name: str) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if collection_name in existing:
        print(f"Using existing collection '{collection_name}'.")
        return
    print(f"Creating collection '{collection_name}' (dim={EMBED_DIM}, cosine).")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )


# ── Format detection ──────────────────────────────────────────────────

def detect_format(records: List[Dict]) -> str:
    """Return 'crawl' or 'faculty' based on first record keys."""
    if not records:
        sys.exit("JSON file is empty.")
    first = records[0]
    if "markdown" in first:
        return "crawl"
    if "text" in first and "metadata" in first:
        return "faculty"
    sys.exit(f"Cannot detect format. First record keys: {list(first.keys())}")


# ── Crawl ingestion ───────────────────────────────────────────────────

def ingest_crawl(
    records: List[Dict],
    source_path: str,
    collection_name: str,
    qdrant: QdrantClient,
    azure: AzureOpenAI,
    cache_dir: Path,
    resume: bool,
) -> None:
    page_hashes_path = cache_dir / "page_hashes.json"
    state_path = cache_dir / "ingest_state.json"
    page_hashes: Dict[str, str] = json.loads(page_hashes_path.read_text()) if page_hashes_path.exists() else {}
    state: Dict[str, Any] = json.loads(state_path.read_text()) if state_path.exists() else {}

    source_key = str(Path(source_path).resolve())
    start_idx = 0
    if resume and state.get("source_key") == source_key:
        start_idx = int(state.get("last_page_index", 0))
        if start_idx > 0:
            print(f"[resume] Continuing from page index {start_idx}")

    embed_buf: List[str] = []
    meta_buf: List[Dict] = []
    total_pages = skipped = total_chunks = 0

    def flush():
        nonlocal embed_buf, meta_buf
        if not embed_buf:
            return
        vectors = embed_texts(embed_buf, azure)
        points = [
            PointStruct(id=m["point_id"], vector=v, payload=m["payload"])
            for v, m in zip(vectors, meta_buf)
        ]
        for i in range(0, len(points), UPSERT_BATCH_SIZE):
            qdrant.upsert(collection_name=collection_name, points=points[i:i + UPSERT_BATCH_SIZE])
        print(f"  Upserted {len(points)} chunks.")
        embed_buf.clear()
        meta_buf.clear()

    for idx in range(start_idx, len(records)):
        page = records[idx]
        md = (page.get("markdown") or "").strip()
        if not md:
            continue

        url = canonicalize_url(page.get("url") or "")
        title = (page.get("title") or "").strip()
        md_norm = normalize_markdown(md)
        page_hash = sha256_text(md_norm)

        if page_hashes.get(url) == page_hash:
            skipped += 1
            continue

        chunks = chunk_markdown(md_norm)
        if not chunks:
            page_hashes[url] = page_hash
            continue

        for ci, ch in enumerate(chunks):
            embed_input = f"Title: {title}\nURL: {url}\n{ch['section_heading']}\n\n{ch['text']}".strip()
            embed_buf.append(embed_input)
            meta_buf.append({
                "point_id": stable_id(url, str(ci), ch["text"]),
                "payload": {
                    "text": ch["text"],
                    "url": url,
                    "title": title,
                    "source_name": Path(source_path).stem,
                    "chunk_index": ci,
                    "section_heading": ch["section_heading"],
                    "page_hash": page_hash,
                },
            })
            total_chunks += 1
            if len(embed_buf) >= EMBED_BATCH_SIZE:
                flush()

        page_hashes[url] = page_hash
        total_pages += 1

        if resume and idx % 50 == 0:
            flush()
            save_json(page_hashes_path, page_hashes)
            save_json(state_path, {"source_key": source_key, "last_page_index": idx})
            print(f"  [progress] pages={total_pages}, skipped={skipped}, chunks={total_chunks}")

    flush()
    save_json(page_hashes_path, page_hashes)
    if resume:
        save_json(state_path, {"source_key": source_key, "last_page_index": len(records)})

    print(f"\nDone (crawl).")
    print(f"  Pages processed : {total_pages}")
    print(f"  Pages skipped   : {skipped}  (unchanged)")
    print(f"  Chunks upserted : {total_chunks}")


# ── Faculty ingestion ─────────────────────────────────────────────────

def ingest_faculty(
    records: List[Dict],
    source_path: str,
    collection_name: str,
    qdrant: QdrantClient,
    azure: AzureOpenAI,
) -> None:
    """
    Each faculty record has a pre-built `text` sentence and structured metadata.
    No chunking needed — embed the text directly.
    """
    points_to_embed: List[Tuple[str, str, Dict]] = []  # (point_id, text, payload)

    for rec in records:
        text = (rec.get("text") or "").strip()
        if not text:
            continue
        meta = rec.get("metadata") or {}
        name = (meta.get("name") or "").strip()
        email = (meta.get("email") or "").strip()
        source_url = (meta.get("source_url") or "").strip()

        pid = stable_id(source_url, name, email, text)
        payload = {
            "text": text,
            "name": name,
            "title": meta.get("title") or "",
            "college": meta.get("college") or "",
            "department": meta.get("department") or "",
            "email": email,
            "phone": meta.get("phone") or "",
            "office": meta.get("office") or "",
            "research_areas": meta.get("research_areas") or "",
            "profile_url": meta.get("profile_url") or "",
            "source_url": source_url,
            "last_updated": meta.get("last_updated") or "",
            "source_name": Path(source_path).stem,
        }
        points_to_embed.append((pid, text, payload))

    print(f"Embedding {len(points_to_embed)} faculty records...")
    total_upserted = 0

    for i in range(0, len(points_to_embed), EMBED_BATCH_SIZE):
        batch = points_to_embed[i:i + EMBED_BATCH_SIZE]
        texts = [b[1] for b in batch]
        vectors = embed_texts(texts, azure)

        points = [
            PointStruct(id=pid, vector=vec, payload=payload)
            for (pid, _, payload), vec in zip(batch, vectors)
        ]
        for j in range(0, len(points), UPSERT_BATCH_SIZE):
            qdrant.upsert(collection_name=collection_name, points=points[j:j + UPSERT_BATCH_SIZE])

        total_upserted += len(points)
        print(f"  Upserted {total_upserted}/{len(points_to_embed)} records...")

    print(f"\nDone (faculty).")
    print(f"  Records upserted: {total_upserted}")


# ── CLI ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest crawl or faculty JSON into Qdrant via Azure OpenAI embeddings."
    )
    parser.add_argument("json_file", help="Path to input JSON file")
    parser.add_argument(
        "--collection", default=DEFAULT_COLLECTION,
        help=f"Qdrant collection name (default: {DEFAULT_COLLECTION})",
    )
    parser.add_argument(
        "--format", choices=["crawl", "faculty"],
        help="Input format — auto-detected if omitted",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Disable resume for crawl ingestion",
    )
    parser.add_argument(
        "--cache-dir", default=DEFAULT_CACHE_DIR,
        help=f"Cache directory for crawl resume state (default: {DEFAULT_CACHE_DIR})",
    )
    args = parser.parse_args()

    records: List[Dict] = load_json(args.json_file)
    if isinstance(records, dict) and "data" in records:
        records = records["data"]
    if not isinstance(records, list):
        sys.exit("Expected a JSON array at the top level.")

    fmt = args.format or detect_format(records)
    print(f"Format : {fmt}")
    print(f"Records: {len(records)}")
    print(f"Collection: {args.collection}")
    print(f"Azure deployment: {AZURE_DEPLOYMENT}\n")

    azure = make_azure_client()
    qdrant = QdrantClient(url=QDRANT_URL)
    ensure_collection(qdrant, args.collection)

    if fmt == "crawl":
        ingest_crawl(
            records=records,
            source_path=args.json_file,
            collection_name=args.collection,
            qdrant=qdrant,
            azure=azure,
            cache_dir=Path(args.cache_dir),
            resume=not args.no_resume,
        )
    else:
        ingest_faculty(
            records=records,
            source_path=args.json_file,
            collection_name=args.collection,
            qdrant=qdrant,
            azure=azure,
        )


if __name__ == "__main__":
    main()
