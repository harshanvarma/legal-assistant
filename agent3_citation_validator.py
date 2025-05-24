"""
Agent 3: Citation Network & Authority Validator
- Follows citation trails from discovered sources
- Builds network of interconnected legal precedents and academic references
- Identifies frequently cited authorities and landmark judgments
- Evaluates source credibility and measures semantic relevance
"""

import re
import json
import asyncio
import networkx as nx
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datetime import datetime, timedelta
import logging
from utils import WebScraper, TextProcessor, CitationFormatter
from config import Config

logger = logging.getLogger(__name__)

class CitationValidator:
    def __init__(self):
        self.scraper = WebScraper()
        self.text_processor = TextProcessor()
        self.citation_formatter = CitationFormatter()
        self.citation_network = nx.DiGraph()  # Directed graph for citation relationships
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
    def build_citation_network(self, documents):
        """Build a network of citations from discovered documents"""
        try:
            citation_relationships = []
            all_citations = []
            
            # Extract citations from each document
            for doc in documents:
                doc_id = self._generate_doc_id(doc)
                doc_citations = self._extract_all_citations(doc)
                
                # Add document to network
                self.citation_network.add_node(doc_id, **doc)
                
                # Add citations and relationships
                for citation in doc_citations:
                    citation_id = self._normalize_citation(citation)
                    if citation_id:
                        # Add citation as node if not exists
                        if not self.citation_network.has_node(citation_id):
                            self.citation_network.add_node(citation_id, 
                                                         citation=citation,
                                                         type='citation',
                                                         authority_score=0)
                        
                        # Add edge from document to citation
                        self.citation_network.add_edge(doc_id, citation_id)
                        citation_relationships.append((doc_id, citation_id))
                        all_citations.append(citation)
            
            # Calculate citation frequencies
            citation_frequency = Counter(all_citations)
            
            # Update citation nodes with frequency information
            for citation, frequency in citation_frequency.items():
                citation_id = self._normalize_citation(citation)
                if citation_id and self.citation_network.has_node(citation_id):
                    self.citation_network.nodes[citation_id]['frequency'] = frequency
                    self.citation_network.nodes[citation_id]['authority_score'] = self._calculate_citation_authority(citation, frequency)
            
            network_metrics = {
                'total_documents': len([n for n in self.citation_network.nodes() if self.citation_network.nodes[n].get('type') != 'citation']),
                'total_citations': len([n for n in self.citation_network.nodes() if self.citation_network.nodes[n].get('type') == 'citation']),
                'citation_relationships': len(citation_relationships),
                'most_cited': self._get_most_cited_authorities(10),
                'citation_clusters': self._identify_citation_clusters()
            }
            
            return network_metrics
            
        except Exception as e:
            logger.error(f"Error building citation network: {e}")
            return None
    
    def _generate_doc_id(self, doc):
        """Generate unique document ID"""
        title = doc.get('title', '')
        url = doc.get('url', '')
        return f"doc_{hash(title + url) % 1000000}"
    
    def _extract_all_citations(self, doc):
        """Extract all citations from a document"""
        citations = []
        
        # Extract from title
        title_citations = self.text_processor.extract_legal_citations(doc.get('title', ''))
        citations.extend(title_citations)
        
        # Extract from snippet
        snippet_citations = self.text_processor.extract_legal_citations(doc.get('snippet', ''))
        citations.extend(snippet_citations)
        
        # Extract from existing citation field
        if doc.get('citation'):
            citations.append(doc['citation'])
        
        # Extract case names and normalize
        case_names = self._extract_case_names(doc.get('title', '') + ' ' + doc.get('snippet', ''))
        citations.extend(case_names)
        
        return list(set(citations))  # Remove duplicates
    
    def _extract_case_names(self, text):
        """Extract case names from text"""
        case_patterns = [
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+v\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # Case v. Case
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+vs\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # Case vs Case
            r'In\s+re\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # In re Case
        ]
        
        case_names = []
        for pattern in case_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                if pattern.startswith('In'):
                    case_names.append(f"In re {match.group(1)}")
                else:
                    case_names.append(f"{match.group(1)} v. {match.group(2)}")
        
        return case_names
    
    def _normalize_citation(self, citation):
        """Normalize citation for consistent identification"""
        if not citation:
            return None
        
        # Clean citation
        normalized = re.sub(r'\s+', ' ', citation.strip())
        normalized = re.sub(r'[^\w\s\(\)\[\].,:]', '', normalized)
        
        return f"cite_{hash(normalized) % 1000000}"
    
    def _calculate_citation_authority(self, citation, frequency):
        """Calculate authority score for a citation"""
        base_score = frequency * 2  # Base score from frequency
        
        # Boost score for Supreme Court cases
        if re.search(r'SCC|Supreme Court', citation, re.IGNORECASE):
            base_score *= 3
        
        # Boost for High Court cases
        elif re.search(r'High Court|HC', citation, re.IGNORECASE):
            base_score *= 2
        
        # Boost for landmark cases (based on year patterns)
        year_match = re.search(r'\b(19|20)\d{2}\b', citation)
        if year_match:
            year = int(year_match.group())
            # Older landmark cases get higher scores
            if year < 2000:
                base_score *= 1.5
        
        return min(base_score, 100)  # Cap at 100
    
    def _get_most_cited_authorities(self, limit=10):
        """Get most frequently cited authorities"""
        citation_nodes = [
            (node_id, data) for node_id, data in self.citation_network.nodes(data=True)
            if data.get('type') == 'citation'
        ]
        
        # Sort by frequency and authority score
        sorted_citations = sorted(
            citation_nodes,
            key=lambda x: (x[1].get('frequency', 0), x[1].get('authority_score', 0)),
            reverse=True
        )
        
        most_cited = []
        for node_id, data in sorted_citations[:limit]:
            most_cited.append({
                'citation': data.get('citation', ''),
                'frequency': data.get('frequency', 0),
                'authority_score': data.get('authority_score', 0),
                'in_degree': self.citation_network.in_degree(node_id)
            })
        
        return most_cited
    
    def _identify_citation_clusters(self):
        """Identify clusters of related citations"""
        try:
            # Use community detection algorithm
            undirected_graph = self.citation_network.to_undirected()
            
            if len(undirected_graph.nodes()) < 3:
                return []
            
            # Simple clustering based on connected components
            clusters = list(nx.connected_components(undirected_graph))
            
            cluster_info = []
            for i, cluster in enumerate(clusters):
                if len(cluster) > 2:  # Only consider clusters with more than 2 nodes
                    cluster_citations = [
                        self.citation_network.nodes[node].get('citation', node)
                        for node in cluster
                        if self.citation_network.nodes[node].get('type') == 'citation'
                    ]
                    
                    cluster_info.append({
                        'cluster_id': i,
                        'size': len(cluster),
                        'citations': cluster_citations[:5]  # Top 5 citations in cluster
                    })
            
            return cluster_info
            
        except Exception as e:
            logger.warning(f"Error identifying citation clusters: {e}")
            return []
    
    def validate_source_credibility(self, documents):
        """Validate and score source credibility"""
        credibility_scores = []
        
        for doc in documents:
            score = self._calculate_credibility_score(doc)
            credibility_scores.append({
                'document_id': self._generate_doc_id(doc),
                'title': doc.get('title', ''),
                'url': doc.get('url', ''),
                'credibility_score': score['total_score'],
                'credibility_breakdown': score['breakdown'],
                'credibility_level': score['level'],
                'recommendation': score['recommendation']
            })
        
        # Sort by credibility score
        credibility_scores.sort(key=lambda x: x['credibility_score'], reverse=True)
        
        return credibility_scores
    
    def _calculate_credibility_score(self, doc):
        """Calculate comprehensive credibility score for a document"""
        breakdown = {}
        total_score = 0
        
        # 1. Source authority (30 points)
        source_score = self._score_source_authority(doc)
        breakdown['source_authority'] = source_score
        total_score += source_score
        
        # 2. Court hierarchy (25 points)
        court_score = self._score_court_hierarchy(doc)
        breakdown['court_hierarchy'] = court_score
        total_score += court_score
        
        # 3. Citation quality (20 points)
        citation_score = self._score_citation_quality(doc)
        breakdown['citation_quality'] = citation_score
        total_score += citation_score
        
        # 4. Temporal relevance (15 points)
        temporal_score = self._score_temporal_relevance(doc)
        breakdown['temporal_relevance'] = temporal_score
        total_score += temporal_score
        
        # 5. Content quality indicators (10 points)
        content_score = self._score_content_quality(doc)
        breakdown['content_quality'] = content_score
        total_score += content_score
        
        # Determine credibility level
        if total_score >= 80:
            level = "Very High"
            recommendation = "Highly recommended for legal research"
        elif total_score >= 60:
            level = "High"
            recommendation = "Recommended for legal research"
        elif total_score >= 40:
            level = "Medium"
            recommendation = "Use with additional verification"
        elif total_score >= 20:
            level = "Low"
            recommendation = "Use with caution, verify independently"
        else:
            level = "Very Low"
            recommendation = "Not recommended for legal research"
        
        return {
            'total_score': total_score,
            'breakdown': breakdown,
            'level': level,
            'recommendation': recommendation
        }
    
    def _score_source_authority(self, doc):
        """Score based on source authority"""
        source_name = doc.get('source_name', '').lower()
        url = doc.get('url', '').lower()
        
        # Supreme Court and High Court websites - highest authority
        if any(domain in url for domain in ['sci.gov.in', 'hc.nic.in', 'hcmadras', 'bombayhighcourt']):
            return 30
        
        # Indian Kanoon - very high authority for case law
        if 'indiankanoon' in source_name or 'indiankanoon' in url:
            return 28
        
        # Manupatra, SCC Online - high authority legal databases
        if any(source in source_name for source in ['manupatra', 'scc']):
            return 26
        
        # Live Law, Bar & Bench - good legal news sources
        if any(source in source_name for source in ['livelaw', 'barandbench']):
            return 22
        
        # Academic institutions
        if any(domain in url for domain in ['.edu', '.ac.in', 'shodhganga']):
            return 20
        
        # Government websites
        if any(domain in url for domain in ['.gov.in', '.nic.in']):
            return 18
        
        # Legal service websites
        if 'legal' in url or 'law' in url:
            return 15
        
        return 10  # Default score
    
    def _score_court_hierarchy(self, doc):
        """Score based on court hierarchy"""
        text = (doc.get('title', '') + ' ' + doc.get('snippet', '')).lower()
        
        # Supreme Court cases - highest score
        if any(term in text for term in ['supreme court', 'sc ', 'scc']):
            return 25
        
        # High Court cases
        if any(term in text for term in ['high court', 'hc ', 'delhi hc', 'bombay hc']):
            return 20
        
        # Tribunal cases
        if any(term in text for term in ['tribunal', 'itat', 'nclat']):
            return 15
        
        # District court cases
        if any(term in text for term in ['district court', 'sessions court']):
            return 10
        
        # Constitutional benches, larger benches
        if any(term in text for term in ['constitutional bench', 'larger bench', 'full bench']):
            return 25
        
        return 5  # Default for unclear hierarchy
    
    def _score_citation_quality(self, doc):
        """Score based on citation quality and format"""
        citations = self._extract_all_citations(doc)
        
        if not citations:
            return 5  # Low score for no citations
        
        score = 0
        valid_citations = 0
        
        for citation in citations:
            if self.citation_formatter.validate_citation(citation):
                valid_citations += 1
                
                # Bonus for standard citation formats
                if re.search(r'\(\d{4}\)\s*\d+\s*SCC\s*\d+', citation):
                    score += 3  # SCC citations
                elif re.search(r'AIR\s*\d{4}', citation):
                    score += 2  # AIR citations
                else:
                    score += 1  # Other valid citations
        
        # Calculate final score based on citation quality
        if valid_citations >= 3:
            return min(score, 20)  # Cap at 20
        elif valid_citations >= 1:
            return min(score, 15)
        else:
            return 5
    
    def _score_temporal_relevance(self, doc):
        """Score based on temporal relevance"""
        text = doc.get('title', '') + ' ' + doc.get('snippet', '')
        
        # Extract years from text
        years = re.findall(r'\b(19|20)\d{2}\b', text)
        
        if not years:
            return 8  # Medium score if no year found
        
        current_year = datetime.now().year
        most_recent_year = max(int(year) for year in years)
        
        # Score based on recency
        year_diff = current_year - most_recent_year
        
        if year_diff <= 2:
            return 15  # Very recent
        elif year_diff <= 5:
            return 12  # Recent
        elif year_diff <= 10:
            return 10  # Moderately recent
        elif year_diff <= 20:
            return 8   # Older but still relevant
        else:
            return 5   # Very old
    
    def _score_content_quality(self, doc):
        """Score based on content quality indicators"""
        title = doc.get('title', '')
        snippet = doc.get('snippet', '')
        
        score = 0
        
        # Length indicators
        if len(title) > 20:
            score += 2
        if len(snippet) > 100:
            score += 2
        
        # Legal terminology presence
        legal_terms = ['judgment', 'order', 'petition', 'appeal', 'constitution', 'statute', 'precedent']
        legal_term_count = sum(1 for term in legal_terms if term.lower() in (title + snippet).lower())
        score += min(legal_term_count, 4)  # Up to 4 points
        
        # Proper noun indicators (case names, acts)
        proper_nouns = len(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', title + snippet))
        score += min(proper_nouns // 3, 2)  # Up to 2 points
        
        return min(score, 10)  # Cap at 10 points
    
    def measure_semantic_relevance(self, documents, research_angle):
        """Measure semantic relevance of documents to research angle"""
        try:
            # Prepare document texts
            doc_texts = []
            doc_metadata = []
            
            for doc in documents:
                text = doc.get('title', '') + ' ' + doc.get('snippet', '')
                doc_texts.append(text)
                doc_metadata.append({
                    'document_id': self._generate_doc_id(doc),
                    'title': doc.get('title', ''),
                    'url': doc.get('url', '')
                })
            
            if not doc_texts:
                return []
            
            # Add research angle to texts for comparison
            all_texts = doc_texts + [research_angle]
            
            # Calculate TF-IDF vectors
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            
            # Calculate cosine similarity with research angle
            research_vector = tfidf_matrix[-1]  # Last item is research angle
            doc_vectors = tfidf_matrix[:-1]
            
            similarities = cosine_similarity(doc_vectors, research_vector.reshape(1, -1)).flatten()
            
            # Create relevance scores
            relevance_scores = []
            for i, (similarity, metadata) in enumerate(zip(similarities, doc_metadata)):
                relevance_scores.append({
                    **metadata,
                    'semantic_similarity': float(similarity),
                    'relevance_level': self._categorize_relevance(similarity),
                    'key_terms': self._extract_matching_terms(doc_texts[i], research_angle)
                })
            
            # Sort by relevance
            relevance_scores.sort(key=lambda x: x['semantic_similarity'], reverse=True)
            
            return relevance_scores
            
        except Exception as e:
            logger.error(f"Error measuring semantic relevance: {e}")
            return []
    
    def _categorize_relevance(self, similarity_score):
        """Categorize relevance based on similarity score"""
        if similarity_score >= 0.7:
            return "Very High"
        elif similarity_score >= 0.5:
            return "High"
        elif similarity_score >= 0.3:
            return "Medium"
        elif similarity_score >= 0.1:
            return "Low"
        else:
            return "Very Low"
    
    def _extract_matching_terms(self, doc_text, research_angle):
        """Extract matching terms between document and research angle"""
        doc_words = set(re.findall(r'\b\w+\b', doc_text.lower()))
        research_words = set(re.findall(r'\b\w+\b', research_angle.lower()))
        
        # Find common meaningful terms (length > 3)
        matching_terms = [word for word in doc_words.intersection(research_words) if len(word) > 3]
        
        return matching_terms[:10]  # Return top 10 matching terms
    
    def follow_citation_trails(self, primary_citations, max_depth=2):
        """Follow citation trails to discover related cases"""
        citation_trails = {}
        
        for citation in primary_citations[:5]:  # Limit to top 5 primary citations
            trail = self._trace_citation_path(citation, max_depth)
            if trail:
                citation_trails[citation] = trail
        
        return citation_trails
    
    def _trace_citation_path(self, citation, max_depth):
        """Trace citation path using available online resources"""
        try:
            trail = {
                'primary_citation': citation,
                'related_cases': [],
                'citing_cases': [],
                'depth_explored': 0
            }
            
            # Search for the citation online to find related cases
            search_query = f'"{citation}" cited cases'
            
            # This would involve scraping legal databases
            # For now, return a placeholder structure
            trail['related_cases'] = self._find_related_cases(citation)
            trail['citing_cases'] = self._find_citing_cases(citation)
            trail['depth_explored'] = 1
            
            return trail
            
        except Exception as e:
            logger.error(f"Error tracing citation path for {citation}: {e}")
            return None
    
    def _find_related_cases(self, citation):
        """Find cases related to the given citation"""
        # Placeholder for related case discovery
        # In a full implementation, this would scrape legal databases
        return []
    
    def _find_citing_cases(self, citation):
        """Find cases that cite the given citation"""
        # Placeholder for citing case discovery
        # In a full implementation, this would scrape legal databases
        return []
    
    def generate_authority_ranking(self, documents):
        """Generate comprehensive authority ranking for documents"""
        rankings = []
        
        for doc in documents:
            # Get various scores
            credibility = self._calculate_credibility_score(doc)
            doc_id = self._generate_doc_id(doc)
            
            # Check if document is in citation network
            network_score = 0
            if self.citation_network.has_node(doc_id):
                network_score = self.citation_network.in_degree(doc_id) * 5
            
            # Calculate final authority score
            authority_score = (
                credibility['total_score'] * Config.SCORING_WEIGHTS['court_authority'] +
                network_score * Config.SCORING_WEIGHTS['citation_frequency']
            )
            
            rankings.append({
                'document_id': doc_id,
                'title': doc.get('title', ''),
                'url': doc.get('url', ''),
                'authority_score': authority_score,
                'credibility_level': credibility['level'],
                'network_score': network_score,
                'recommendation': credibility['recommendation']
            })
        
        # Sort by authority score
        rankings.sort(key=lambda x: x['authority_score'], reverse=True)
        
        return rankings

# Example usage and testing
if __name__ == "__main__":
    validator = CitationValidator()
    
    # Test with sample documents
    sample_documents = [
        {
            'title': 'K.S. Puttaswamy v. Union of India (2017) 10 SCC 1',
            'url': 'https://indiankanoon.org/doc/91938676/',
            'snippet': 'The Supreme Court in this landmark judgment recognized privacy as a fundamental right under Articles 14, 19, and 21 of the Constitution.',
            'source_name': 'indian_kanoon',
            'type': 'case_law'
        },
        {
            'title': 'Maneka Gandhi v. Union of India (1978) 1 SCC 248',
            'url': 'https://indiankanoon.org/doc/1766147/',
            'snippet': 'This case expanded the interpretation of Article 21 to include procedural due process.',
            'source_name': 'indian_kanoon',
            'type': 'case_law'
        }
    ]
    
    research_angle = "Exploring how Article 21 extends to digital privacy rights in the context of Aadhaar and data surveillance"
    
    # Test citation network building
    network_metrics = validator.build_citation_network(sample_documents)
    print("Citation Network Metrics:", json.dumps(network_metrics, indent=2))
    
    # Test credibility validation
    credibility_scores = validator.validate_source_credibility(sample_documents)
    print("Credibility Scores:", json.dumps(credibility_scores, indent=2))
    
    # Test semantic relevance
    relevance_scores = validator.measure_semantic_relevance(sample_documents, research_angle)
    print("Relevance Scores:", json.dumps(relevance_scores, indent=2))
    
    # Test authority ranking
    authority_rankings = validator.generate_authority_ranking(sample_documents)
    print("Authority Rankings:", json.dumps(authority_rankings, indent=2))