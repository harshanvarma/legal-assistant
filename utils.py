import re
import time
import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import hashlib
import json
import os
from urllib.parse import urlparse, urljoin
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebScraper:
    def __init__(self, delay=1, max_retries=3):
        self.delay = delay
        self.max_retries = max_retries
        self.ua = UserAgent()
        self.session = requests.Session()
        
    def get_headers(self):
        return {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
    
    def scrape_page(self, url, timeout=30):
        """Scrape a single page with retry mechanism"""
        for attempt in range(self.max_retries):
            try:
                time.sleep(self.delay)
                response = self.session.get(
                    url, 
                    headers=self.get_headers(), 
                    timeout=timeout,
                    verify=False
                )
                response.raise_for_status()
                return BeautifulSoup(response.content, 'html.parser')
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {str(e)}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to scrape {url} after {self.max_retries} attempts")
                    return None
        return None

class TextProcessor:
    @staticmethod
    def clean_text(text):
        """Clean and normalize text"""
        if not text:
            return ""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove special characters but keep legal citations
        text = re.sub(r'[^\w\s\(\)\[\].,;:\-/]', '', text)
        return text
    
    @staticmethod
    def extract_legal_citations(text):
        """Extract Indian legal citations from text"""
        citations = []
        patterns = [
            r'\(\d{4}\)\s*\d+\s*SCC\s*\d+',  # Supreme Court Cases
            r'AIR\s*\d{4}\s*[A-Z]+\s*\d+',   # All India Reporter
            r'\(\d{4}\)\s*\d+\s*[A-Z]{2,}\s*\d+',  # High Court cases
            r'ILR\s*\d{4}\s*[A-Z]+\s*\d+',   # Indian Law Reports
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                citations.append(match.group().strip())
        
        return list(set(citations))  # Remove duplicates
    
    @staticmethod
    def extract_keywords(text, legal_domain=None):
        """Extract relevant keywords from legal text"""
        # Basic keyword extraction
        words = re.findall(r'\b[A-Za-z]{3,}\b', text.lower())
        
        # Filter out common legal stop words
        legal_stopwords = [
            'court', 'case', 'law', 'legal', 'section', 'act', 'rule',
            'order', 'judgment', 'decision', 'matter', 'application',
            'petition', 'appeal', 'revision', 'criminal', 'civil'
        ]
        
        keywords = [word for word in words if word not in legal_stopwords]
        
        # Get unique keywords with frequency > 1
        from collections import Counter
        word_freq = Counter(keywords)
        important_keywords = [word for word, freq in word_freq.items() if freq > 1]
        
        return important_keywords[:20]  # Return top 20 keywords

class CitationFormatter:
    @staticmethod
    def format_indian_citation(case_name, citation, court, year):
        """Format citation according to Indian legal citation standards"""
        formatted = f"{case_name}, {citation}"
        if court:
            formatted += f" ({court})"
        if year:
            formatted += f" [{year}]"
        return formatted
    
    @staticmethod
    def validate_citation(citation):
        """Validate if citation follows Indian legal citation patterns"""
        patterns = [
            r'\(\d{4}\)\s*\d+\s*SCC\s*\d+',
            r'AIR\s*\d{4}\s*[A-Z]+\s*\d+',
            r'\(\d{4}\)\s*\d+\s*[A-Z]{2,}\s*\d+',
        ]
        
        for pattern in patterns:
            if re.search(pattern, citation, re.IGNORECASE):
                return True
        return False

class CacheManager:
    def __init__(self, cache_dir='cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, data):
        """Generate cache key from data"""
        if isinstance(data, str):
            return hashlib.md5(data.encode()).hexdigest()
        return hashlib.md5(str(data).encode()).hexdigest()
    
    def get(self, key):
        """Get data from cache"""
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Check if cache is still valid (24 hours)
                    if time.time() - data.get('timestamp', 0) < 86400:
                        return data.get('content')
            except Exception as e:
                logger.warning(f"Error reading cache: {e}")
        return None
    
    def set(self, key, data):
        """Store data in cache"""
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        try:
            cache_data = {
                'content': data,
                'timestamp': time.time()
            }
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Error writing cache: {e}")

class URLValidator:
    @staticmethod
    def is_valid_url(url):
        """Check if URL is valid"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    @staticmethod
    def clean_url(url):
        """Clean and normalize URL"""
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        return url.strip()

class ResultsManager:
    def __init__(self, results_dir='results'):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
    
    def save_results(self, research_id, results):
        """Save research results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{research_id}_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return None
    
    def load_results(self, filepath):
        """Load research results from file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading results: {e}")
            return None