"""
Agent 2: Multi-Source Research Intelligence Engine
- Creates comprehensive keyword combinations using Indian legal terminology
- Executes searches across multiple Indian legal and academic platforms
- Implements intelligent web scraping with rate limiting
- Handles different website structures and content extraction methods
"""

import asyncio
import aiohttp
import time
from urllib.parse import quote_plus, urljoin
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
import logging
from utils import WebScraper, CacheManager, URLValidator
from config import Config


logger = logging.getLogger(__name__)

class ResearchEngine:
    def __init__(self):
        self.scraper = WebScraper(delay=Config.REQUEST_DELAY, max_retries=Config.MAX_RETRIES)
        self.cache = CacheManager()
        self.url_validator = URLValidator()
        self.session = aiohttp.ClientSession()
        
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()
    
    def generate_search_queries(self, research_requirements):
        """Generate comprehensive search queries for Indian legal research"""
        base_keywords = research_requirements.get('search_keywords', [])
        interpretation = research_requirements.get('interpretation', {})
        
        # Extract specific legal terms
        legal_questions = interpretation.get('legal_questions', [])
        domains = interpretation.get('domains', [])
        
        queries = []
        
        # 1. Direct keyword combinations
        for keyword in base_keywords[:10]:  # Limit to top 10
            queries.extend(self._create_keyword_variations(keyword))
        
        # 2. Legal domain specific queries
        for domain in domains:
            if domain.lower() in Config.LEGAL_KEYWORDS:
                domain_keywords = Config.LEGAL_KEYWORDS[domain.lower()]
                for base_keyword in base_keywords[:5]:
                    for domain_keyword in domain_keywords[:3]:
                        queries.append(f"{base_keyword} {domain_keyword}")
        
        # 3. Question-based queries
        for question in legal_questions:
            query = self._extract_query_from_question(question)
            if query:
                queries.append(query)
        
        # 4. Citation-based queries
        queries.extend(self._generate_citation_queries(base_keywords))
        
        # 5. Indian legal specific combinations
        queries.extend(self._generate_indian_legal_queries(base_keywords))
        
        # Remove duplicates and limit total queries
        unique_queries = list(set(queries))[:50]  # Limit to 50 queries
        
        return self._prioritize_queries(unique_queries, research_requirements)
    
    def _create_keyword_variations(self, keyword):
        """Create variations of a keyword for comprehensive search"""
        variations = [keyword]
        
        # Add common legal suffixes/prefixes
        legal_modifiers = [
            f"{keyword} case law",
            f"{keyword} Supreme Court",
            f"{keyword} High Court",
            f"{keyword} constitutional",
            f"{keyword} judgment",
            f"{keyword} precedent",
            f"recent {keyword}",
            f"{keyword} 2020-2024"  # Recent cases
        ]
        
        variations.extend(legal_modifiers)
        return variations
    
    def _extract_query_from_question(self, question):
        """Extract searchable query from legal question"""
        # Remove question words and focus on key terms
        stop_words = ['what', 'how', 'when', 'where', 'why', 'which', 'who', 'is', 'are', 'can', 'does']
        words = question.lower().split()
        
        key_words = [word for word in words if word not in stop_words and len(word) > 3]
        
        if len(key_words) >= 2:
            return ' '.join(key_words[:5])  # Take top 5 key words
        return None
    
    def _generate_citation_queries(self, keywords):
        """Generate queries focused on finding citations"""
        citation_queries = []
        
        for keyword in keywords[:5]:
            citation_queries.extend([
                f"{keyword} SCC",
                f"{keyword} AIR",
                f"{keyword} Supreme Court cases",
                f"{keyword} landmark judgment"
            ])
        
        return citation_queries
    
    def _generate_indian_legal_queries(self, keywords):
        """Generate India-specific legal queries"""
        indian_queries = []
        
        indian_terms = [
            "Indian Constitution",
            "Supreme Court of India",
            "Indian law",
            "constitutional law India",
            "Indian legal system"
        ]
        
        for keyword in keywords[:3]:
            for indian_term in indian_terms:
                indian_queries.append(f"{keyword} {indian_term}")
        
        return indian_queries
    
    def _prioritize_queries(self, queries, research_requirements):
        """Prioritize queries based on research requirements"""
        evidence_priorities = research_requirements.get('evidence_priorities', [])
        
        prioritized = []
        
        # High priority: case law queries
        for query in queries:
            if any(term in query.lower() for term in ['case law', 'supreme court', 'judgment', 'scc']):
                prioritized.append({'query': query, 'priority': 1, 'type': 'case_law'})
        
        # Medium priority: constitutional queries
        for query in queries:
            if any(term in query.lower() for term in ['constitutional', 'article', 'fundamental right']):
                prioritized.append({'query': query, 'priority': 2, 'type': 'constitutional'})
        
        # Lower priority: general queries
        for query in queries:
            if not any(existing['query'] == query for existing in prioritized):
                prioritized.append({'query': query, 'priority': 3, 'type': 'general'})
        
        return sorted(prioritized, key=lambda x: x['priority'])
    
    async def execute_multi_source_search(self, query_list):
        """Execute searches across multiple sources concurrently"""
        results = {
            'legal_sources': [],
            'academic_sources': [],
            'government_sources': [],
            'search_metadata': {
                'total_queries': len(query_list),
                'successful_searches': 0,
                'failed_searches': 0,
                'sources_searched': []
            }
        }
        
        # Create search tasks
        tasks = []
        
        for query_data in query_list[:20]:  # Limit concurrent searches
            query = query_data['query']
            
            # Search legal sources
            for source_name, base_url in Config.LEGAL_SOURCES.items():
                task = self._search_single_source(source_name, base_url, query, 'legal')
                tasks.append(task)
            
            # Search academic sources for high-priority queries
            if query_data['priority'] <= 2:
                for source_name, base_url in Config.ACADEMIC_SOURCES.items():
                    task = self._search_single_source(source_name, base_url, query, 'academic')
                    tasks.append(task)
        
        # Execute searches with concurrency limit
        semaphore = asyncio.Semaphore(Config.MAX_CONCURRENT_SEARCHES)
        
        async def bounded_search(task):
            async with semaphore:
                return await task
        
        search_results = await asyncio.gather(
            *[bounded_search(task) for task in tasks],
            return_exceptions=True
        )
        
        # Process results
        for result in search_results:
            if isinstance(result, Exception):
                results['search_metadata']['failed_searches'] += 1
                logger.warning(f"Search failed: {result}")
                continue
            
            if result and result.get('success'):
                results['search_metadata']['successful_searches'] += 1
                source_type = result.get('source_type')
                
                if source_type == 'legal':
                    results['legal_sources'].extend(result.get('documents', []))
                elif source_type == 'academic':
                    results['academic_sources'].extend(result.get('documents', []))
                elif source_type == 'government':
                    results['government_sources'].extend(result.get('documents', []))
        
        return results
    
    async def _search_single_source(self, source_name, base_url, query, source_type):
        """Search a single source for the given query"""
        try:
            # Check cache first
            cache_key = f"{source_name}_{query}"
            cached_result = self.cache.get(cache_key)
            if cached_result:
                return cached_result
            
            # Construct search URL
            search_url = self._construct_search_url(base_url, query, source_name)
            if not search_url:
                return None
            
            # Perform search
            documents = await self._scrape_search_results(search_url, source_name, source_type)
            
            result = {
                'success': True,
                'source_name': source_name,
                'source_type': source_type,
                'query': query,
                'documents': documents,
                'search_url': search_url
            }
            
            # Cache result
            self.cache.set(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error searching {source_name} for '{query}': {e}")
            return {
                'success': False,
                'source_name': source_name,
                'error': str(e),
                'query': query
            }
    
    def _construct_search_url(self, base_url, query, source_name):
        """Construct search URL for different sources"""
        encoded_query = quote_plus(query)
        
        url_patterns = {
            'indian_kanoon': f"{base_url}{encoded_query}",
            'live_law': f"{base_url}{encoded_query}",
            'bar_bench': f"{base_url}{encoded_query}",
            'legal_service_india': f"{base_url}{encoded_query}",
            'lawctopus': f"{base_url}{encoded_query}",
            'shodhganga': f"{base_url}{encoded_query}",
            'google_scholar': f"{base_url}{encoded_query}+india+law",
            'jstor': f"{base_url}{encoded_query}+law+india",
            'ssrn': f"{base_url}{encoded_query}+law"
        }
        
        return url_patterns.get(source_name, f"{base_url}?q={encoded_query}")
    
    async def _scrape_search_results(self, url, source_name, source_type):
        """Scrape search results from a specific source"""
        try:
            async with self.session.get(url, timeout=Config.TIMEOUT) as response:
                if response.status != 200:
                    logger.warning(f"HTTP {response.status} for {url}")
                    return []
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract documents based on source
                documents = self._extract_documents_by_source(soup, source_name, source_type)
                
                return documents[:10]  # Limit to top 10 results per source
                
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return []
    
    def _extract_documents_by_source(self, soup, source_name, source_type):
        """Extract documents based on specific source structure"""
        documents = []
        
        if source_name == 'indian_kanoon':
            documents = self._extract_indian_kanoon_results(soup)
        elif source_name == 'live_law':
            documents = self._extract_live_law_results(soup)
        elif source_name == 'bar_bench':
            documents = self._extract_bar_bench_results(soup)
        elif source_name == 'google_scholar':
            documents = self._extract_google_scholar_results(soup)
        elif source_name == 'shodhganga':
            documents = self._extract_shodhganga_results(soup)
        else:
            # Generic extraction for unknown sources
            documents = self._extract_generic_results(soup)
        
        # Add source information to each document
        for doc in documents:
            doc['source_name'] = source_name
            doc['source_type'] = source_type
            doc['extracted_at'] = time.time()
        
        return documents
    
    def _extract_indian_kanoon_results(self, soup):
        """Extract results from Indian Kanoon"""
        documents = []
        
        # Indian Kanoon specific selectors
        result_divs = soup.find_all('div', class_='result')
        if not result_divs:
            result_divs = soup.find_all('div', class_='results_middle')
        
        for div in result_divs[:10]:
            try:
                title_element = div.find('a')
                if title_element:
                    title = title_element.get_text(strip=True)
                    url = title_element.get('href', '')
                    
                    # Extract snippet
                    snippet_element = div.find('div', class_='search_result')
                    if not snippet_element:
                        snippet_element = div
                    
                    snippet = snippet_element.get_text(strip=True)
                    
                    # Extract citation if present
                    citation = self._extract_citation_from_text(snippet)
                    
                    document = {
                        'title': title,
                        'url': urljoin('https://indiankanoon.org', url) if url.startswith('/') else url,
                        'snippet': snippet[:500],  # Limit snippet length
                        'citation': citation,
                        'type': 'case_law'
                    }
                    
                    documents.append(document)
                    
            except Exception as e:
                logger.warning(f"Error extracting Indian Kanoon result: {e}")
                continue
        
        return documents
    
    def _extract_live_law_results(self, soup):
        """Extract results from Live Law"""
        documents = []
        
        # Live Law specific selectors
        articles = soup.find_all('article', class_='post')
        if not articles:
            articles = soup.find_all('div', class_='post')
        
        for article in articles[:10]:
            try:
                title_element = article.find('h2') or article.find('h3')
                if title_element:
                    title_link = title_element.find('a')
                    if title_link:
                        title = title_link.get_text(strip=True)
                        url = title_link.get('href', '')
                        
                        # Extract excerpt
                        excerpt_element = article.find('div', class_='excerpt') or article.find('p')
                        excerpt = excerpt_element.get_text(strip=True) if excerpt_element else ""
                        
                        document = {
                            'title': title,
                            'url': url,
                            'snippet': excerpt[:500],
                            'type': 'legal_news'
                        }
                        
                        documents.append(document)
                        
            except Exception as e:
                logger.warning(f"Error extracting Live Law result: {e}")
                continue
        
        return documents
    
    def _extract_bar_bench_results(self, soup):
        """Extract results from Bar & Bench"""
        documents = []
        
        # Bar & Bench specific selectors
        articles = soup.find_all('article') or soup.find_all('div', class_='article-item')
        
        for article in articles[:10]:
            try:
                title_element = article.find('h1') or article.find('h2') or article.find('h3')
                if title_element:
                    title_link = title_element.find('a')
                    if title_link:
                        title = title_link.get_text(strip=True)
                        url = title_link.get('href', '')
                        
                        # Extract summary
                        summary_element = article.find('div', class_='summary') or article.find('p')
                        summary = summary_element.get_text(strip=True) if summary_element else ""
                        
                        document = {
                            'title': title,
                            'url': url,
                            'snippet': summary[:500],
                            'type': 'legal_analysis'
                        }
                        
                        documents.append(document)
                        
            except Exception as e:
                logger.warning(f"Error extracting Bar & Bench result: {e}")
                continue
        
        return documents
    
    def _extract_google_scholar_results(self, soup):
        """Extract results from Google Scholar"""
        documents = []
        
        # Google Scholar specific selectors
        result_divs = soup.find_all('div', class_='gs_r')
        
        for div in result_divs[:10]:
            try:
                title_element = div.find('h3', class_='gs_rt')
                if title_element:
                    title_link = title_element.find('a')
                    title = title_link.get_text(strip=True) if title_link else title_element.get_text(strip=True)
                    url = title_link.get('href', '') if title_link else ""
                    
                    # Extract snippet
                    snippet_element = div.find('div', class_='gs_rs')
                    snippet = snippet_element.get_text(strip=True) if snippet_element else ""
                    
                    # Extract authors and publication info
                    authors_element = div.find('div', class_='gs_a')
                    authors = authors_element.get_text(strip=True) if authors_element else ""
                    
                    document = {
                        'title': title,
                        'url': url,
                        'snippet': snippet[:500],
                        'authors': authors,
                        'type': 'academic_paper'
                    }
                    
                    documents.append(document)
                    
            except Exception as e:
                logger.warning(f"Error extracting Google Scholar result: {e}")
                continue
        
        return documents
    
    def _extract_shodhganga_results(self, soup):
        """Extract results from Shodhganga"""
        documents = []
        
        # Shodhganga specific selectors
        result_items = soup.find_all('div', class_='search-result-item')
        if not result_items:
            result_items = soup.find_all('tr', class_='odd') + soup.find_all('tr', class_='even')
        
        for item in result_items[:10]:
            try:
                title_element = item.find('a', class_='title') or item.find('a')
                if title_element:
                    title = title_element.get_text(strip=True)
                    url = title_element.get('href', '')
                    
                    # Extract metadata
                    metadata_element = item.find('div', class_='metadata') or item.find('td')
                    metadata = metadata_element.get_text(strip=True) if metadata_element else ""
                    
                    document = {
                        'title': title,
                        'url': urljoin('https://shodhganga.inflibnet.ac.in', url) if url.startswith('/') else url,
                        'snippet': metadata[:500],
                        'type': 'thesis'
                    }
                    
                    documents.append(document)
                    
            except Exception as e:
                logger.warning(f"Error extracting Shodhganga result: {e}")
                continue
        
        return documents
    
    def _extract_generic_results(self, soup):
        """Generic extraction for unknown sources"""
        documents = []
        
        # Look for common patterns
        links = soup.find_all('a', href=True)
        
        for link in links[:20]:
            try:
                title = link.get_text(strip=True)
                url = link.get('href', '')
                
                # Skip if title is too short or URL looks like navigation
                if len(title) < 10 or any(nav in url.lower() for nav in ['menu', 'nav', 'footer', 'header']):
                    continue
                
                # Look for surrounding context
                parent = link.parent
                context = parent.get_text(strip=True) if parent else ""
                
                document = {
                    'title': title,
                    'url': url,
                    'snippet': context[:300],
                    'type': 'unknown'
                }
                
                documents.append(document)
                
                if len(documents) >= 5:  # Limit generic extraction
                    break
                    
            except Exception as e:
                continue
        
        return documents
    
    def _extract_citation_from_text(self, text):
        """Extract legal citation from text"""
        from utils import TextProcessor
        citations = TextProcessor.extract_legal_citations(text)
        return citations[0] if citations else None
    
    def filter_and_deduplicate_results(self, search_results):
        """Filter and deduplicate search results"""
        all_documents = []
        
        # Combine all documents
        for source_type in ['legal_sources', 'academic_sources', 'government_sources']:
            all_documents.extend(search_results.get(source_type, []))
        
        # Deduplicate by URL and title
        seen_urls = set()
        seen_titles = set()
        filtered_documents = []
        
        for doc in all_documents:
            url = doc.get('url', '')
            title = doc.get('title', '').lower()
            
            # Skip if URL or title already seen
            if url in seen_urls or title in seen_titles:
                continue
            
            # Skip if title or URL is too short
            if len(title) < 10 or len(url) < 10:
                continue
            
            seen_urls.add(url)
            seen_titles.add(title)
            filtered_documents.append(doc)
        
        return filtered_documents

# Example usage
if __name__ == "__main__":
    async def test_research_engine():
        async with ResearchEngine() as engine:
            # Test query generation
            sample_requirements = {
                'search_keywords': ['Article 21', 'privacy', 'digital rights'],
                'interpretation': {
                    'legal_questions': ['How does Article 21 protect digital privacy?'],
                    'domains': ['constitutional', 'technology'],
                    'evidence_types': ['case law', 'academic papers']
                }
            }
            
            queries = engine.generate_search_queries(sample_requirements)
            print(f"Generated {len(queries)} queries")
            
            # Test search execution
            results = await engine.execute_multi_source_search(queries[:5])  # Test with first 5 queries
            print(f"Search completed: {results['search_metadata']}")
    
    # Run test
    asyncio.run(test_research_engine())