"""
AtlasAI Ingestion Pipeline

Core pipeline for processing documents through the ingestion workflow for the SPOT robot:
- Deduplication via content hashing
- Text chunking
- Embedding generation
- Vector storage in Qdrant
- Metadata tracking in Postgres
"""

import hashlib
import logging
import asyncio
import os
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json

# Removed openai dependency
import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import psycopg2
from psycopg2.extras import RealDictCursor
import tiktoken

from .utils import chunk_text, get_embedding, setup_logging

logger = logging.getLogger(__name__)

class IngestionPipeline:
    """Main pipeline class for document processing and ingestion."""
    
    def __init__(self):
        """Initialize the ingestion pipeline with database connections."""
        # Create an httpx client explicitly and pass it to OpenAI to avoid
        # compatibility issues between openai library's expected httpx kwargs
        # and the httpx version provided in the image.
        self.ollama_host = os.getenv("OLLAMA_URL", "http://ollama:11434")
        self.qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL"))
        self.postgres_url = os.getenv("POSTGRES_URL")
        
        # Initialize tokenizer for chunking
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Collection name for Qdrant
        self.collection_name = "atlasai_documents"
        self.chat_collection_name = "atlasai_chat_memory"
        self.profile_collection_name = "atlasai_profiles"
        
        #Embedding model and vector dimension (configurable via env)
        # Supported models:
        # - text-embedding-3-small => 1536 dims (default)
        # - text-embedding-3-large => 3072 dims
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "qwen2.5:7b")
        # Allow explicit override of embedding dimension if necessary
        model_dim_override = os.getenv("EMBEDDING_DIM")
        if model_dim_override:
            try:
                self.embedding_dim = int(model_dim_override)
            except ValueError:
                logger.warning("Invalid EMBEDDING_DIM value '%s', falling back to model default", model_dim_override)
                self.embedding_dim = 3584 # Default for qwen2.5:7b
        else:
            # Map common model names to dimensions
            if self.embedding_model == "nomic-embed-text":
                self.embedding_dim = 768
            else:
                # Default to qwen dims
                self.embedding_dim = 3584
        logger.info(f"Using embedding model: %s (dim=%d)", self.embedding_model, self.embedding_dim)

        
        # Initialize collections and database
        asyncio.create_task(self._initialize())
    
    async def _initialize(self):
        """Initialize database connections and create collections if needed."""
        try:
            # Create Qdrant collection if it doesn't exist
            await self._ensure_qdrant_collection()
            
            # Create Postgres tables if they don't exist
            await self._ensure_postgres_tables()
            
            logger.info("Pipeline initialization completed successfully")
        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}")
            raise
    
    async def _ensure_qdrant_collection(self):
        """Ensure Qdrant collection exists with proper configuration."""
        # Retry loop: Qdrant may not be ready when the worker container starts.
        max_attempts = 10
        delay = 1
        for attempt in range(1, max_attempts + 1):
            try:
                collections = self.qdrant_client.get_collections()
                collection_names = [col.name for col in collections.collections]

                if self.collection_name not in collection_names:
                    self.qdrant_client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(
                            size=self.embedding_dim,
                            distance=Distance.COSINE
                        )
                    )
                    logger.info(f"Created Qdrant collection: {self.collection_name}")
                else:
                    logger.info(f"Qdrant collection already exists: {self.collection_name}")

                if self.chat_collection_name not in collection_names:
                    self.qdrant_client.create_collection(
                        collection_name=self.chat_collection_name,
                        vectors_config=VectorParams(
                            size=self.embedding_dim,
                            distance=Distance.COSINE
                        )
                    )
                    logger.info(f"Created Qdrant chat collection: {self.chat_collection_name}")
                else:
                    logger.info(f"Qdrant chat collection already exists: {self.chat_collection_name}")

                if self.profile_collection_name not in collection_names:
                    self.qdrant_client.create_collection(
                        collection_name=self.profile_collection_name,
                        vectors_config=VectorParams(
                            size=self.embedding_dim,
                            distance=Distance.COSINE
                        )
                    )
                    logger.info(f"Created Qdrant profile collection: {self.profile_collection_name}")
                else:
                    logger.info(f"Qdrant profile collection already exists: {self.profile_collection_name}")

                # Success - exit retry loop
                return

            except Exception as e:
                logger.warning(f"Attempt {attempt}/{max_attempts} - failed to contact Qdrant: {e}")
                if attempt == max_attempts:
                    logger.error(f"Failed to ensure Qdrant collection after {max_attempts} attempts: {e}")
                    raise
                # Exponential backoff with cap
                await asyncio.sleep(delay)
                delay = min(delay * 2, 10)
    
    async def _ensure_postgres_tables(self):
        """Ensure Postgres tables exist with proper schema."""
        try:
            with psycopg2.connect(self.postgres_url) as conn:
                with conn.cursor() as cur:
                    # Create documents table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS documents (
                            doc_id VARCHAR(255) PRIMARY KEY,
                            source VARCHAR(100) NOT NULL,
                            title TEXT,
                            uri TEXT,
                            author VARCHAR(255),
                            created_at TIMESTAMP WITH TIME ZONE,
                            updated_at TIMESTAMP WITH TIME ZONE,
                            content_hash VARCHAR(64) UNIQUE,
                            processed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                            is_deleted BOOLEAN DEFAULT FALSE,
                            chunk_count INTEGER DEFAULT 0
                        )
                    """)
                    

                    
                    # Create ingestion_log table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS ingestion_log (
                            id SERIAL PRIMARY KEY,
                            doc_id VARCHAR(255) NOT NULL,
                            event_type VARCHAR(50) NOT NULL,
                            message TEXT,
                            timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                            metadata JSONB
                        )
                    """)

                    # Create chat_logs table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS chat_logs (
                            id SERIAL PRIMARY KEY,
                            session_id VARCHAR(255) NOT NULL,
                            role VARCHAR(50) NOT NULL,
                            message TEXT NOT NULL,
                            timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                            is_indexed BOOLEAN DEFAULT FALSE,
                            metadata JSONB
                        )
                    """)
                    
                    # Create indexes
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source)")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash)")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_processed_at ON documents(processed_at)")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_ingestion_log_doc_id ON ingestion_log(doc_id)")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_ingestion_log_timestamp ON ingestion_log(timestamp)")
                    
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_logs_session_id ON chat_logs(session_id)")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_logs_timestamp ON chat_logs(timestamp)")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_logs_is_indexed ON chat_logs(is_indexed)")
                    
                    conn.commit()
                    logger.info("Postgres tables ensured successfully")
        except Exception as e:
            logger.error(f"Failed to ensure Postgres tables: {e}")
            raise
    
    async def check_postgres_connection(self) -> bool:
        """Check if Postgres connection is healthy."""
        try:
            with psycopg2.connect(self.postgres_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    return True
        except Exception as e:
            logger.error(f"Postgres connection check failed: {e}")
            return False
    
    async def check_qdrant_connection(self) -> bool:
        """Check if Qdrant connection is healthy."""
        try:
            self.qdrant_client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Qdrant connection check failed: {e}")
            return False
    
    def _compute_content_hash(self, text: str) -> str:
        """Compute SHA-256 hash of document content for deduplication."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    async def _is_document_unchanged(self, doc_id: str, content_hash: str) -> bool:
        """Check if document content has changed since last processing."""
        try:
            with psycopg2.connect(self.postgres_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT content_hash FROM documents WHERE doc_id = %s",
                        (doc_id,)
                    )
                    result = cur.fetchone()
                    
                    if result and result[0] == content_hash:
                        return True
                    return False
        except Exception as e:
            logger.error(f"Failed to check document unchanged status: {e}")
            return False
    
    async def _log_ingestion_event(self, doc_id: str, event_type: str, message: str, metadata: Dict = None):
        """Log an ingestion event to the database."""
        try:
            with psycopg2.connect(self.postgres_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO ingestion_log (doc_id, event_type, message, metadata)
                        VALUES (%s, %s, %s, %s)
                    """, (doc_id, event_type, message, json.dumps(metadata or {})))
                    conn.commit()
        except Exception as e:
            logger.error(f"Failed to log ingestion event: {e}")
    
    async def process_document(self, document: Any, force_reindex: bool = False, collection_override: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a single document through the ingestion pipeline.
        
        Args:
            document: DocumentPayload object
            force_reindex: If True, process even if content hasn't changed
            collection_override: Optional name of collection to use instead of default
            
        Returns:
            Dict with success status, chunks processed, and any errors
        """
        try:
            # Compute content hash for deduplication
            content_hash = self._compute_content_hash(document.text)
            
            # Check if document is unchanged (unless force reindex)
            if not force_reindex and await self._is_document_unchanged(document.doc_id, content_hash):
                await self._log_ingestion_event(
                    document.doc_id, 
                    "skipped", 
                    "Document content unchanged, skipping processing"
                )
                return {
                    "success": True,
                    "chunks_processed": 0,
                    "message": "Document unchanged, skipped processing"
                }
            
            # Chunk the text
            chunks = chunk_text(document.text, self.tokenizer)
            logger.info(f"Chunked document {document.doc_id} into {len(chunks)} chunks")
            
            # Generate embeddings for all chunks (use configured model)
            chunk_texts = [chunk["text"] for chunk in chunks]
            embeddings = await get_embedding(chunk_texts, self.ollama_host, model=self.embedding_model)
            
            # Prepare points for Qdrant
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point = PointStruct(
                    id=f"{document.doc_id}_{i}",
                    vector=embedding,
                    payload={
                        "source": document.source,
                        "doc_id": document.doc_id,
                        "chunk_index": i,
                        "title": document.title,
                        "uri": document.uri,
                        "author": document.author,
                        "created_at": document.created_at,
                        "updated_at": document.updated_at,
                        "text": chunk["text"]
                    }
                )
                points.append(point)
            
            # Upsert to Qdrant
            target_collection = collection_override if collection_override else self.collection_name
            self.qdrant_client.upsert(
                collection_name=target_collection,
                points=points
            )
            
            # Update Postgres with document metadata
            await self._update_document_metadata(document, content_hash, len(chunks))
            
            # Log successful processing
            await self._log_ingestion_event(
                document.doc_id,
                "processed",
                f"Successfully processed {len(chunks)} chunks",
                {"chunks_processed": len(chunks), "content_hash": content_hash}
            )
            
            return {
                "success": True,
                "chunks_processed": len(chunks),
                "message": f"Successfully processed {len(chunks)} chunks"
            }
            
        except Exception as e:
            error_msg = f"Failed to process document {document.doc_id}: {str(e)}"
            logger.error(error_msg)
            
            # Log error
            await self._log_ingestion_event(
                document.doc_id,
                "error",
                error_msg,
                {"error": str(e)}
            )
            
            return {
                "success": False,
                "chunks_processed": 0,
                "error": error_msg
            }
    
    async def _update_document_metadata(self, document: Any, content_hash: str, chunk_count: int):
        """Update document metadata in Postgres."""
        try:
            with psycopg2.connect(self.postgres_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO documents (
                            doc_id, source, title, uri, author, created_at, updated_at, 
                            content_hash, chunk_count, is_deleted
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (doc_id) DO UPDATE SET
                            source = EXCLUDED.source,
                            title = EXCLUDED.title,
                            uri = EXCLUDED.uri,
                            author = EXCLUDED.author,
                            created_at = EXCLUDED.created_at,
                            updated_at = EXCLUDED.updated_at,
                            content_hash = EXCLUDED.content_hash,
                            chunk_count = EXCLUDED.chunk_count,
                            processed_at = NOW(),
                            is_deleted = FALSE
                    """, (
                        document.doc_id,
                        document.source,
                        document.title,
                        document.uri,
                        document.author,
                        document.created_at,
                        document.updated_at,
                        content_hash,
                        chunk_count,
                        False
                    ))
                    conn.commit()
        except Exception as e:
            logger.error(f"Failed to update document metadata: {e}")
            raise
    
    async def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve document metadata by ID."""
        try:
            with psycopg2.connect(self.postgres_url) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT doc_id, source, title, uri, author, created_at, updated_at
                        FROM documents WHERE doc_id = %s AND is_deleted = FALSE
                    """, (doc_id,))
                    result = cur.fetchone()
                    
                    if result:
                        return dict(result)
                    return None
        except Exception as e:
            logger.error(f"Failed to get document by ID: {e}")
            return None
    
    async def sync_source(self, source: str) -> Dict[str, Any]:
        """
        Perform full synchronization for a data source.
        
        This is a placeholder implementation. In a real system, this would:
        1. Connect to the source system
        2. Retrieve all documents
        3. Compare with existing documents in Postgres
        4. Process new/updated documents
        5. Mark deleted documents as deleted
        """
        try:
            # For now, just return a placeholder response
            await self._log_ingestion_event(
                source,
                "sync_started",
                f"Full sync started for source: {source}"
            )
            
            # Placeholder implementation
            documents_processed = 0
            documents_deleted = 0
            
            await self._log_ingestion_event(
                source,
                "sync_completed",
                f"Sync completed: {documents_processed} processed, {documents_deleted} deleted",
                {
                    "documents_processed": documents_processed,
                    "documents_deleted": documents_deleted
                }
            )
            
            return {
                "documents_processed": documents_processed,
                "documents_deleted": documents_deleted,
                "errors": []
            }
            
        except Exception as e:
            logger.error(f"Sync failed for source {source}: {e}")
            return {
                "documents_processed": 0,
                "documents_deleted": 0,
                "errors": [str(e)]
            }

    async def log_chat_message(self, session_id: str, role: str, message: str, meta: Dict = None) -> bool:
        """Log a chat message to Postgres."""
        try:
            with psycopg2.connect(self.postgres_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO chat_logs (session_id, role, message, metadata)
                        VALUES (%s, %s, %s, %s)
                    """, (session_id, role, message, json.dumps(meta or {})))
                    conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to log chat message: {e}")
            return False

    async def consolidate_chat_memory(self) -> Dict[str, Any]:
        """
        Consolidate chat logs into hierarchical memory (Day -> Hour -> Conversation).
        Fetches unindexed logs, pulls recent contextual history, and embeds them.
        """
        try:
            processed_count = 0
            sessions_processed = 0
            
            # Fetch unindexed logs
            with psycopg2.connect(self.postgres_url) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT id, session_id, role, message, timestamp 
                        FROM chat_logs 
                        WHERE is_indexed = FALSE
                        ORDER BY session_id, timestamp ASC
                    """)
                    unindexed_logs = cur.fetchall()
            
            if not unindexed_logs:
                return {"processed_count": 0, "sessions_processed": 0, "message": "No new logs to consolidate"}

            # Group logs hierarchically: session_id -> date -> hour -> logs
            sessions = {}
            for log in unindexed_logs:
                sid = log['session_id']
                if sid not in sessions:
                    sessions[sid] = []
                sessions[sid].append(log)
            
            ids_to_mark_indexed = []
            
            for sid, unindexed_session_logs in sessions.items():
                if not unindexed_session_logs:
                    continue
                    
                # We group the unindexed logs by Day and Hour to create chunks
                time_buckets = {}
                for log in unindexed_session_logs:
                    day = log['timestamp'].strftime('%Y-%m-%d')
                    hour = log['timestamp'].strftime('%H:00')
                    bucket_key = (day, hour)
                    if bucket_key not in time_buckets:
                        time_buckets[bucket_key] = []
                    time_buckets[bucket_key].append(log)
                    
                for (day, hour), bucket_logs in time_buckets.items():
                    # For this bucket of unindexed logs, fetch context from database 
                    # up to the latest timestamp in the bucket, for this session
                    latest_timestamp = bucket_logs[-1]['timestamp']
                    
                    with psycopg2.connect(self.postgres_url) as conn:
                        with conn.cursor(cursor_factory=RealDictCursor) as cur:
                            cur.execute("""
                                SELECT id, role, message, timestamp 
                                FROM chat_logs 
                                WHERE session_id = %s AND timestamp <= %s
                                ORDER BY timestamp DESC
                                LIMIT 15
                            """, (sid, latest_timestamp))
                            context_logs = cur.fetchall()
                            context_logs.reverse() # chronological order
                            
                    full_text = "\n".join([f"[{l['timestamp'].strftime('%Y-%m-%d %H:%M')}] {l['role'].title()}: {l['message']}" for l in context_logs])
                    
                    if not full_text.strip():
                        continue

                    # Chunk the text
                    chunks = chunk_text(full_text, self.tokenizer)
                    chunk_texts = [chunk["text"] for chunk in chunks]
                    
                    # Generate embeddings via Ollama
                    embeddings = await get_embedding(chunk_texts, self.ollama_host, model=self.embedding_model)
                    
                    # Prepare Qdrant points
                    points = []
                    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                        import uuid
                        point_id = str(uuid.uuid4())
                        points.append(PointStruct(
                            id=point_id,
                            vector=embedding,
                            payload={
                                "session_id": sid,
                                "day": day,
                                "hour": hour,
                                "text": chunk["text"],
                                "timestamp_start": context_logs[0]['timestamp'].isoformat(),
                                "timestamp_end": context_logs[-1]['timestamp'].isoformat(),
                                "log_ids": [l['id'] for l in bucket_logs], # Only track the newly indexed ones
                                "type": "chat_memory_hierarchy"
                            }
                        ))
                    
                    if points:
                        self.qdrant_client.upsert(
                            collection_name=self.chat_collection_name,
                            points=points
                        )
                    
                    processed_count += len(bucket_logs)
                    ids_to_mark_indexed.extend([l['id'] for l in bucket_logs])
                
                sessions_processed += 1

            # Mark logs as indexed
            if ids_to_mark_indexed:
                with psycopg2.connect(self.postgres_url) as conn:
                    with conn.cursor() as cur:
                        cur.execute("""
                            UPDATE chat_logs 
                            SET is_indexed = TRUE 
                            WHERE id = ANY(%s)
                        """, (ids_to_mark_indexed,))
                        conn.commit()

            return {
                "processed_count": processed_count, 
                "sessions_processed": sessions_processed,
                "message": f"Consolidated {processed_count} messages into hierarchical blocks from {sessions_processed} sessions"
            }

        except Exception as e:
            logger.error(f"Chat memory consolidation failed: {e}")
            return {"error": str(e)}
