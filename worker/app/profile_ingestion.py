import json
import asyncio
import argparse
import sys
import uuid
import datetime
from pathlib import Path

# Add current directory to path to import app modules
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
# Load env from project root
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

from app.pipeline import IngestionPipeline
from pydantic import BaseModel, Field

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

def generate_markdown_profile(data: dict) -> str:
    """
    Transforms the structured JSON profile data into a rich markdown format
    suitable for embedding and LLM retrieval.
    """
    lines = []
    
    # Header
    name = data.get("fullName", "Unknown Name")
    lines.append(f"# Professional Profile: {name}")
    lines.append("")
    
    # Bio
    bio = data.get("bio")
    if bio:
        lines.append("## Biography")
        lines.append(bio)
        lines.append("")
    
    # Career History
    career_history = data.get("careerHistory", [])
    if career_history:
        lines.append("## Career History")
        for role in career_history:
            title = role.get("title", "Unknown Role")
            company = role.get("company", "Unknown Company")
            dates = role.get("dates", "")
            desc = role.get("description", "")
            
            lines.append(f"### {title} at {company}")
            if dates:
                lines.append(f"**Dates:** {dates}")
            if desc:
                lines.append(desc)
            lines.append("")
            
    # Projects
    projects = data.get("projects", [])
    if projects:
        lines.append("## Projects")
        for project in projects:
            p_name = project.get("projectName", "Untitled Project")
            p_desc = project.get("description", "")
            techs = project.get("technologiesUsed", [])
            tech_list = ", ".join([t.get("value") for t in techs if t.get("value")])
            
            lines.append(f"### {p_name}")
            if p_desc:
                lines.append(p_desc)
            if tech_list:
                lines.append(f"**Technologies:** {tech_list}")
            citation = project.get("description_citation")
            if citation:
                lines.append(f"[More Info]({citation})")
            lines.append("")

    # News & Publications
    news = data.get("recentNewsAndPublications", [])
    if news:
        lines.append("## News and Publications")
        for item in news:
            title = item.get("title", "")
            source = item.get("source", "")
            date = item.get("publicationDate", "")
            url = item.get("url")
            
            header = f"### {title}"
            if source:
                header += f" ({source})"
            lines.append(header)
            
            if date:
                lines.append(f"**Date:** {date}")
            if url:
                lines.append(f"**Link:** {url}")
            lines.append("")
            
    return "\n".join(lines)

async def ingest_profile(json_path: str):
    print(f"Reading profile from {json_path}...")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {json_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON at {json_path}")
        return

    # Generate Markdown Content
    markdown_content = generate_markdown_profile(data)
    
    # Create Document Payload
    name = data.get("fullName", "Unknown")
    # Generate a deterministic ID based on the profile name or URL
    doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, data.get("fullName_citation", f"profile:{name}")))
    
    payload = DocumentPayload(
        doc_id=doc_id,
        source="professional_profile",
        title=f"{name} - Professional Profile",
        uri=data.get("fullName_citation", ""),
        text=markdown_content,
        author=name,
        created_at=datetime.datetime.utcnow().isoformat(),
        updated_at=datetime.datetime.utcnow().isoformat()
    )
    
    print("-" * 50)
    print(f"Generated Document Payload for {name}")
    print(f"ID: {payload.doc_id}")
    print(f"Text Length: {len(payload.text)} characters")
    print("-" * 50)
    
    # Initialize Pipeline
    print("Initializing Ingestion Pipeline...")
    pipeline = IngestionPipeline()
    
    # Process
    print("Ingesting document...")
    result = await pipeline.process_document(payload, force_reindex=True)
    
    if result["success"]:
        print(f"\n✅ SUCCESS: Profile ingested successfully.")
        print(f"Chunks processed: {result['chunks_processed']}")
    else:
        print(f"\n❌ FAILED: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a JSON profile into the system.")
    parser.add_argument("--file", type=str, default="worker/app/profile.json", help="Path to profile JSON file")
    args = parser.parse_args()
    
    asyncio.run(ingest_profile(args.file))
