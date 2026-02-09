from bs4 import BeautifulSoup
import logging
from typing import Optional
import os
try:
    from firecrawl import FirecrawlApp
except ImportError:
    FirecrawlApp = None

logger = logging.getLogger(__name__)

def scrape_url(url: str) -> Optional[str]:
    """
    Scrapes the content of a given URL.
    
    Tries to use Firecrawl if API key is present.
    Falls back to simple requests+BS4 scraping if Firecrawl fails or is not configured.
    
    Args:
        url: The URL to scrape.
        
    Returns:
        The extracted text content, or None if scraping failed.
    """
    
    # Define complex domains that require Firecrawl
    COMPLEX_DOMAINS = ['linkedin.com', 'twitter.com', 'x.com', 'facebook.com', 'instagram.com']
    is_complex_domain = any(domain in url.lower() for domain in COMPLEX_DOMAINS)

    # Try Firecrawl first
    firecrawl_key = os.getenv("FIRECRAWL_API_KEY")

    if firecrawl_key and FirecrawlApp:
        try:
            logger.info(f"Attempting to scrape {url} with Firecrawl")
            app = FirecrawlApp(api_key=firecrawl_key)
            scrape_result = app.scrape_url(url, params={'formats': ['markdown']})
            if scrape_result and 'markdown' in scrape_result:
                return scrape_result['markdown']
        except Exception as e:
            msg = f"Firecrawl scraping failed for {url}: {e}"
            logger.warning(msg)
            # For complex domains, we do NOT want to fall back to BS4 as it yields login pages/garbage
            if is_complex_domain:
                logger.error(f"Skipping fallback for complex domain {url}. Returning None.")
                return None
    
    if is_complex_domain:
        logger.warning(f"Complex domain {url} detected but Firecrawl not available/failed. Skipping local fallback.")
        return None

    # Fallback to local scraping
    return scrape_with_bs4(url)

def scrape_with_bs4(url: str) -> Optional[str]:
    """
    Scrapes the content of a given URL using BeautifulSoup.
    """
    try:
        import requests
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Get text
        text = soup.get_text()
        
        # Break into lines and remove leading/trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
        
    except Exception as e:
        logger.error(f"Failed to scrape URL {url}: {e}")
        return None
