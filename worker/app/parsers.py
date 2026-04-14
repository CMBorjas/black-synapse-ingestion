import os
import logging
from typing import Optional
try:
    from markitdown import MarkItDown
except ImportError:
    MarkItDown = None

logger = logging.getLogger(__name__)

class DocumentParser:
    """Helper class to convert various document formats to Markdown using MarkItDown."""
    
    def __init__(self):
        if MarkItDown:
            self.md = MarkItDown()
        else:
            self.md = None
            logger.warning("MarkItDown not installed. Document conversion will not be available.")
            
        self.supported_extensions = {
            '.pdf', '.docx', '.xlsx', '.pptx', '.xls', 
            '.html', '.md', '.txt', '.csv', '.json',
            '.doc', '.ppt' # Added these as they are often requested
        }

    def parse_file(self, file_path: str) -> Optional[str]:
        """
        Convert a file to Markdown text.
        
        Args:
            file_path: Path to the file to be parsed.
            
        Returns:
            The extracted Markdown content, or None if parsing failed.
        """
        if not self.md:
            logger.error("MarkItDown is not initialized.")
            return None

        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.supported_extensions:
            logger.warning(f"Unsupported file extension: {ext}")
            # Fallback for text files if extension is unknown but might be text
            if ext == '':
                logger.info(f"Attempting to treat extensionless file {file_path} as text")
            else:
                return None

        try:
            logger.info(f"Converting {file_path} to Markdown...")
            result = self.md.convert(file_path)
            if result and hasattr(result, 'text_content'):
                return result.text_content
            return None
        except Exception as e:
            logger.error(f"Failed to parse file {file_path}: {e}")
            return None

    def is_supported(self, filename: str) -> bool:
        """Check if a filename's extension is supported."""
        ext = os.path.splitext(filename)[1].lower()
        return ext in self.supported_extensions
