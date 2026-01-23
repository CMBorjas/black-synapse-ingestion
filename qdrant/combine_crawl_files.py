#!/usr/bin/env python3
"""
Combine individual crawl JSON files into a single array format
expected by qdrant_ingestion.py
"""

import json
from pathlib import Path
from typing import Dict, Any, List

def combine_crawl_files(input_dir: str, output_file: str) -> None:
    """
    Combine all JSON files in input_dir into a single array.
    Each file should have: {markdown, metadata: {url or sourceURL, title}}
    Output format: [{markdown, url, title}, ...]
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Directory not found: {input_dir}")
    
    pages: List[Dict[str, Any]] = []
    
    json_files = sorted(input_path.glob("*.json"))
    print(f"Found {len(json_files)} JSON files")
    
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Extract markdown
            markdown = data.get("markdown", "").strip()
            if not markdown:
                continue
            
            # Extract url and title from metadata
            metadata = data.get("metadata", {})
            url = metadata.get("url") or metadata.get("sourceURL", "")
            title = metadata.get("title", "").strip()
            
            # Create page object in expected format
            page = {
                "markdown": markdown,
                "url": url,
                "title": title,
            }
            
            pages.append(page)
            
        except Exception as e:
            print(f"Warning: Error processing {json_file.name}: {e}")
            continue
    
    # Write combined file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)
    
    print(f"Combined {len(pages)} pages into {output_file}")
    print(f"Output file size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: combine_crawl_files.py <input_dir> <output_file>")
        sys.exit(1)
    
    combine_crawl_files(sys.argv[1], sys.argv[2])

