"""
AtlasAI Worker

FastAPI application for processing and embedding data from various sources for the SPOT robot.
Handles ingestion, deduplication, chunking, and vector storage to power the robot's AI capabilities.
"""

import os
import logging
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from .pipeline import IngestionPipeline
from .utils import setup_logging
from .scraper import scrape_url
from .qr_analyzer import decode_qr_from_bytes, decode_qr_from_base64, classify_qr_content
import uuid
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AtlasAI Worker",
    description="ETL pipeline for processing and embedding data from various sources for the SPOT robot's AI system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ingestion pipeline
pipeline = IngestionPipeline()

# Pydantic models for API requests/responses
class DocumentPayload(BaseModel):
    """Unified document schema for ingestion."""
    doc_id: str = Field(..., description="Unique document identifier")
    source: str = Field(..., description="Data source (e.g., 'notion', 'gmail', 'slack')")
    title: str = Field(..., description="Document title")
    uri: str = Field(..., description="Document URI or URL")
    text: str = Field(..., description="Document content text")
    author: str = Field(..., description="Document author")
    created_at: str = Field(..., description="Creation timestamp (ISO format)")
    updated_at: str = Field(..., description="Last update timestamp (ISO format)")

class UserIngestRequest(BaseModel):
    """Schema for user profile ingestion request."""
    name: str = Field(..., description="Name of the user")
    url: str = Field(None, description="URL to scrape for user info")
    bio: str = Field(None, description="Directly provided biography or context")
    scraping_consent: bool = Field(False, description="Explicit user consent to scrape the provided URL")

class IngestionResponse(BaseModel):
    """Response model for ingestion operations."""
    success: bool
    message: str
    doc_id: str
    chunks_processed: int = 0
    error: str = None

class SyncResponse(BaseModel):
    """Response model for sync operations."""
    success: bool
    message: str
    documents_processed: int = 0
    documents_deleted: int = 0
    errors: List[str] = []

class QRCodeResult(BaseModel):
    """Result for a single decoded QR code value."""
    value: str
    content_type: str  # "url" or "text"
    ingested: bool
    doc_id: Optional[str] = None
    chunks_processed: int = 0
    error: Optional[str] = None

class QRAnalyzeResponse(BaseModel):
    """Response model for QR code analysis."""
    success: bool
    qr_codes_found: int
    results: List[QRCodeResult]
    message: str
class ChatLogPayload(BaseModel):
    """Schema for chat log ingestion."""
    session_id: str = Field(..., description="Unique session identifier")
    role: str = Field(..., description="Role of the message sender (user/assistant)")
    message: str = Field(..., description="Content of the message")
    timestamp: Optional[datetime] = Field(None, description="Timestamp of the message")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class ChatConsolidationResponse(BaseModel):
    """Response model for chat consolidation."""
    success: bool
    processed_count: int
    sessions_processed: int
    message: str
    error: str = None

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "AtlasAI Worker - SPOT Robot AI System", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check including database connections."""
    try:
        # Check database connections
        postgres_healthy = await pipeline.check_postgres_connection()
        qdrant_healthy = await pipeline.check_qdrant_connection()
        
        return {
            "status": "healthy" if postgres_healthy and qdrant_healthy else "unhealthy",
            "postgres": "connected" if postgres_healthy else "disconnected",
            "qdrant": "connected" if qdrant_healthy else "disconnected"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/ingest", response_model=IngestionResponse)
async def ingest_document(
    document: DocumentPayload,
    background_tasks: BackgroundTasks
):
    """
    Ingest a single document.
    
    Processes the document through the full pipeline:
    1. Validates payload
    2. Deduplicates via content hash
    3. Chunks text
    4. Generates embeddings
    5. Upserts to Qdrant
    6. Logs to Postgres
    """
    try:
        logger.info(f"Processing document: {document.doc_id} from {document.source}")
        
        # Process document through pipeline
        result = await pipeline.process_document(document)
        
        if result["success"]:
            logger.info(f"Successfully processed document {document.doc_id}: {result['chunks_processed']} chunks")
            return IngestionResponse(
                success=True,
                message="Document processed successfully",
                doc_id=document.doc_id,
                chunks_processed=result["chunks_processed"]
            )
        else:
            logger.error(f"Failed to process document {document.doc_id}: {result['error']}")
            return IngestionResponse(
                success=False,
                message="Document processing failed",
                doc_id=document.doc_id,
                error=result["error"]
            )
            
    except Exception as e:
        logger.error(f"Unexpected error processing document {document.doc_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/reindex", response_model=IngestionResponse)
async def reindex_document(
    doc_id: str,
    background_tasks: BackgroundTasks
):
    """
    Re-index an existing document.
    
    Re-processes a document that already exists in the system,
    useful for updating embeddings or fixing processing errors.
    """
    try:
        logger.info(f"Re-indexing document: {doc_id}")
        
        # Retrieve document from database
        document = await pipeline.get_document_by_id(doc_id)
        if not document:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
        
        # Process document through pipeline
        result = await pipeline.process_document(document, force_reindex=True)
        
        if result["success"]:
            logger.info(f"Successfully re-indexed document {doc_id}: {result['chunks_processed']} chunks")
            return IngestionResponse(
                success=True,
                message="Document re-indexed successfully",
                doc_id=doc_id,
                chunks_processed=result["chunks_processed"]
            )
        else:
            logger.error(f"Failed to re-index document {doc_id}: {result['error']}")
            return IngestionResponse(
                success=False,
                message="Document re-indexing failed",
                doc_id=doc_id,
                error=result["error"]
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error re-indexing document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Re-indexing failed: {str(e)}")

@app.post("/sync", response_model=SyncResponse)
async def sync_data_source(
    source: str,
    background_tasks: BackgroundTasks
):
    """
    Full synchronization for a data source.
    
    Performs a complete sync including:
    1. Processing all documents from the source
    2. Identifying and handling deletions
    3. Updating metadata
    """
    try:
        logger.info(f"Starting full sync for source: {source}")
        
        # Perform full synchronization
        result = await pipeline.sync_source(source)
        
        logger.info(f"Sync completed for {source}: {result['documents_processed']} processed, {result['documents_deleted']} deleted")
        return SyncResponse(
            success=True,
            message=f"Sync completed for {source}",
            documents_processed=result["documents_processed"],
            documents_deleted=result["documents_deleted"],
            errors=result.get("errors", [])
        )
        
    except Exception as e:
        logger.error(f"Sync failed for source {source}: {e}")
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")

@app.post("/ingest/user", response_model=IngestionResponse)
async def ingest_user_profile(
    request: UserIngestRequest,
    background_tasks: BackgroundTasks
):
    """
    Ingest a user profile.
    
    Combines provided bio and scraped content (if URL provided and consent given).
    """
    try:
        # Validate consent if URL is provided
        if request.url and not request.scraping_consent:
            raise HTTPException(
                status_code=400, 
                detail="scraping_consent must be True when providing a URL"
            )
            
        content_parts = []
        if request.bio:
            content_parts.append(f"Bio: {request.bio}")
            
        if request.url:
            logger.info(f"Scraping user profile from {request.url}")
            scraped_content = scrape_url(request.url)
            if scraped_content:
                content_parts.append(f"Scraped Content from {request.url}:\n{scraped_content}")
            else:
                logger.warning(f"Failed to scrape content from {request.url}")
                # We continue even if scraping fails, as long as we have valid request
        
        full_text = "\n\n".join(content_parts)
        
        if not full_text:
            raise HTTPException(status_code=400, detail="No content provided (bio or valid URL required)")
            
        doc_id = f"user_{request.name.lower().replace(' ', '_')}"
        now = datetime.utcnow().isoformat()
        
        document = DocumentPayload(
            doc_id=doc_id,
            source="user_profile",
            title=f"User Profile: {request.name}",
            uri=request.url or f"user://{doc_id}",
            text=full_text,
            author="system",
            created_at=now,
            updated_at=now
        )
        
        logger.info(f"Processing user profile: {doc_id}")
        result = await pipeline.process_document(document)
        
        if result["success"]:
            return IngestionResponse(
                success=True,
                message="User profile processed successfully",
                doc_id=doc_id,
                chunks_processed=result["chunks_processed"]
            )
        else:
            return IngestionResponse(
                success=False,
                message="User profile processing failed",
                doc_id=doc_id,
                error=result["error"]
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing user profile {request.name}: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

async def _ingest_qr_value(value: str, content_type: str) -> Dict[str, Any]:
    """Ingest a single decoded QR value through the pipeline."""
    now = datetime.utcnow().isoformat()
    doc_id = f"qr_{uuid.uuid4().hex[:12]}"

    if content_type == "url":
        text = scrape_url(value)
        if not text:
            return {"ingested": False, "doc_id": doc_id, "chunks_processed": 0,
                    "error": f"Failed to scrape URL: {value}"}
        title = f"QR Code URL: {value}"
        uri = value
        source = "qr_url"
    else:
        text = value
        title = f"QR Code Text: {value[:80]}"
        uri = f"qr://text/{doc_id}"
        source = "qr_text"

    document = DocumentPayload(
        doc_id=doc_id,
        source=source,
        title=title,
        uri=uri,
        text=text,
        author="qr_analyzer",
        created_at=now,
        updated_at=now,
    )

    result = await pipeline.process_document(document)
    return {
        "ingested": result["success"],
        "doc_id": doc_id,
        "chunks_processed": result.get("chunks_processed", 0),
        "error": result.get("error"),
    }


@app.post("/analyze/qr", response_model=QRAnalyzeResponse)
async def analyze_qr_code(
    file: Optional[UploadFile] = File(default=None),
    image_base64: Optional[str] = Form(default=None),
    ingest: bool = Form(default=True),
):
    """
    Analyze a QR code image and optionally ingest its content.

    Accepts either:
    - A binary image upload via `file` (multipart/form-data)
    - A base64-encoded image string via `image_base64`

    For each QR code found:
    - If the decoded value is a URL, it is scraped and embedded.
    - If it is plain text, it is embedded directly.

    Set `ingest=false` to decode without ingesting.
    """
    if file is None and not image_base64:
        raise HTTPException(status_code=400, detail="Provide either 'file' or 'image_base64'")

    # Decode QR codes
    try:
        if file is not None:
            image_bytes = await file.read()
            decoded_values = decode_qr_from_bytes(image_bytes)
        else:
            decoded_values = decode_qr_from_base64(image_base64)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(f"QR decode error: {exc}")
        raise HTTPException(status_code=500, detail=f"QR decoding failed: {str(exc)}")

    if not decoded_values:
        return QRAnalyzeResponse(
            success=True,
            qr_codes_found=0,
            results=[],
            message="No QR codes detected in the image",
        )

    results: List[QRCodeResult] = []
    for value in decoded_values:
        content_type = classify_qr_content(value)

        if ingest:
            ingest_result = await _ingest_qr_value(value, content_type)
            results.append(QRCodeResult(
                value=value,
                content_type=content_type,
                ingested=ingest_result["ingested"],
                doc_id=ingest_result["doc_id"],
                chunks_processed=ingest_result["chunks_processed"],
                error=ingest_result.get("error"),
            ))
        else:
            results.append(QRCodeResult(
                value=value,
                content_type=content_type,
                ingested=False,
            ))

    success = all(r.ingested for r in results) if ingest else True
    logger.info(f"QR analysis: {len(results)} code(s) found, ingest={ingest}")
    return QRAnalyzeResponse(
        success=success,
        qr_codes_found=len(results),
        results=results,
        message=f"Processed {len(results)} QR code(s)",
    )


@app.post("/chat/log", response_model=IngestionResponse)
async def log_chat(
    payload: ChatLogPayload,
    background_tasks: BackgroundTasks
):
    """
    Log a chat message to the database for future memory consolidation.
    """
    try:
        success = await pipeline.log_chat_message(
            session_id=payload.session_id,
            role=payload.role,
            message=payload.message,
            meta=payload.metadata
        )
        
        if success:
            return IngestionResponse(
                success=True,
                message="Chat message logged successfully",
                doc_id=payload.session_id # Using session_id as doc_id reference
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to log chat message")
            
    except Exception as e:
        logger.error(f"Error logging chat message: {e}")
        raise HTTPException(status_code=500, detail=f"Logging failed: {str(e)}")

@app.post("/chat/memory/consolidate", response_model=ChatConsolidationResponse)
async def consolidate_memory(
    background_tasks: BackgroundTasks
):
    """
    Trigger consolidation of chat logs into long-term vector memory.
    """
    try:
        result = await pipeline.consolidate_chat_memory()
        
        if "error" in result:
            return ChatConsolidationResponse(
                success=False,
                processed_count=0,
                sessions_processed=0,
                message="Consolidation failed",
                error=result["error"]
            )
            
        return ChatConsolidationResponse(
            success=True,
            processed_count=result["processed_count"],
            sessions_processed=result["sessions_processed"],
            message=result["message"]
        )
            
    except Exception as e:
        logger.error(f"Error eliminating chat memory: {e}")
        raise HTTPException(status_code=500, detail=f"Consolidation failed: {str(e)}")
if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", 8000))
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )
