"""
Agent 4: Legal Research Curator
- Compiles and organizes findings by relevance and source type
- Generates proper legal citations (Indian legal citation format)
- Creates contextual summaries explaining relevance to research angle
- Produces structured research brief with organized supporting materials
"""

import json
import re
from collections import defaultdict
from datetime import datetime
import logging
from utils import CitationFormatter, TextProcessor
from config import Config

logger = logging.getLogger(__name__)

class ResearchCurator:
    def __init__(self):
        self.citation_formatter = CitationFormatter()
        self.text_processor = TextProcessor()
        
    def curate_research_findings(self, documents, credibility_scores, relevance_scores, network_metrics, research_angle):
        """Curate and organize all research findings into a comprehensive brief"""
        try:
            # Organize documents by type and relevance
            organized_docs = self._organize_documents_by_type(documents, credibility_scores, relevance_scores)
            
            # Generate research brief
            research_brief = {
                'research_summary': {
                    'research_angle': research_angle,
                    'total_sources_found': len(documents),
                    'high_relevance_sources': len([r for r in relevance_scores if r['relevance_level'] in ['Very High', 'High']]),
                    'high_credibility_sources': len([c for c in credibility_scores if c['credibility_level'] in ['Very High', 'High']]),
                    'generated_at': datetime.now().isoformat()
                },
                'key_findings': self._extract_key_findings(organized_docs, research_angle),
                'supporting_materials': self._create_supporting_materials_section(organized_docs),
                'citation_network_analysis': self._analyze_citation_network(network_metrics),
                'research_recommendations': self._generate_research_recommendations(organized_docs, relevance_scores),
                'complete_bibliography': self._generate_bibliography(organized_docs)
            }
            
            return research_brief
            
        except Exception as e:
            logger.error(f"Error curating research findings: {e}")
            return None
    
    def _organize_documents_by_type(self, documents, credibility_scores, relevance_scores):
        """Organize documents by type with scores"""
        # Create lookup dictionaries for scores
        credibility_lookup = {score['document_id']: score for score in credibility_scores}
        relevance_lookup = {score['document_id']: score for score in relevance_scores}
        
        organized = {
            'legal_judgments': [],
            'academic_papers': [],
            'legislative_materials': [],
            'policy_documents': [],\
            'legal_commentary': [],
            'comparative_sources': []
        }
        
        for doc in documents:
            doc_id = self._generate_doc_id(doc)
            
            # Enrich document with scores
            enriched_doc = {
                **doc,
                'document_id': doc_id,
                'credibility_info': credibility_lookup.get(doc_id, {}),
                'relevance_info': relevance_lookup.get(doc_id, {}),
                'formatted_citation': self._format_document_citation(doc),
                'contextual_summary': self._create_contextual_summary(doc),
                'key_extracts': self._extract_key_content(doc)
            }
            
            # Categorize by document type
            doc_type = doc.get('type', '').lower()
            
            if doc_type in ['case_law', 'judgment']:
                organized['legal_judgments'].append(enriched_doc)
            elif doc_type in ['academic_paper', 'thesis', 'research']:
                organized['academic_papers'].append(enriched_doc)
            elif doc_type in ['statute', 'act', 'amendment', 'legislative']:
                organized['legislative_materials'].append(enriched_doc)
            elif doc_type in ['policy', 'government', 'commission_report']:
                organized['policy_documents'].append(enriched_doc)
            elif doc_type in ['legal_news', 'legal_analysis', 'commentary']:
                organized['legal_commentary'].append(enriched_doc)
            elif doc_type in ['international', 'comparative']:
                organized['comparative_sources'].append(enriched_doc)
            else:
                # Categorize based on content analysis
                category = self._analyze_document_category(doc)
                organized[category].append(enriched_doc)
        
        # Sort each category by relevance and credibility
        for category in organized:
            organized[category] = self._sort_documents_by_priority(organized[category])
        
        return organized
    
    def _generate_doc_id(self, doc):
        """Generate unique document ID"""
        title = doc.get('title', '')
        url = doc.get('url', '')
        return f"doc_{hash(title + url) % 1000000}"
    
    def _format_document_citation(self, doc):
        """Format document citation according to Indian legal standards"""
        title = doc.get('title', '')
        url = doc.get('url', '')
        source_name = doc.get('source_name', '')
        doc_type = doc.get('type', '')
        
        # Extract year from title or content
        year_match = re.search(r'\b(19|20)\d{2}\b', title)
        year = year_match.group() if year_match else datetime.now().year
        
        # Format based on document type
        if doc_type == 'case_law':
            # Extract case citation if present
            existing_citation = self.text_processor.extract_legal_citations(title)
            if existing_citation:
                return existing_citation[0]
            else:
                return f"{title} [{year}] (Available at: {url})"
        
        elif doc_type == 'academic_paper':
            # Academic citation format
            authors = self._extract_authors(doc)
            if authors:
                return f"{authors}, '{title}' [{year}] (Available at: {url})"
            else:
                return f"'{title}' [{year}] (Available at: {url})"
        
        else:
            # Generic format
            return f"{title} [{year}] (Source: {source_name}, Available at: {url})"
    
    def _extract_authors(self, doc):
        """Extract author information from document"""
        authors = doc.get('authors', '')
        if authors:
            return authors
        
        # Try to extract from title or snippet
        snippet = doc.get('snippet', '')
        author_patterns = [
            r'by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),?\s+(?:et\s+al\.?)?'
        ]
        
        for pattern in author_patterns:
            match = re.search(pattern, snippet)
            if match:
                return match.group(1)
        
        return None
    
    def _create_contextual_summary(self, doc):
        """Create contextual summary explaining document's relevance"""
        title = doc.get('title', '')
        snippet = doc.get('snippet', '')
        doc_type = doc.get('type', '')
        
        # Extract key legal concepts
        legal_concepts = self._identify_legal_concepts(title + ' ' + snippet)
        
        summary_parts = []
        
        # Document type context
        type_descriptions = {
            'case_law': 'This legal judgment',
            'academic_paper': 'This academic research',
            'policy': 'This policy document',
            'legal_news': 'This legal analysis',
            'thesis': 'This scholarly thesis'
        }
        
        intro = type_descriptions.get(doc_type, 'This document')
        summary_parts.append(intro)
        
        # Key concepts
        if legal_concepts:
            concept_text = f"addresses {', '.join(legal_concepts[:3])}"
            summary_parts.append(concept_text)
        
        # Relevance context
        if len(snippet) > 50:
            key_phrase = snippet[:100] + "..."
            summary_parts.append(f"Key content: {key_phrase}")
        
        return ' '.join(summary_parts)
    
    def _identify_legal_concepts(self, text):
        """Identify legal concepts in text"""
        concepts = []
        
        # Constitutional concepts
        if re.search(r'article\s+\d+', text, re.IGNORECASE):
            concepts.append('constitutional provisions')
        
        # Rights concepts
        if any(term in text.lower() for term in ['fundamental right', 'privacy', 'liberty']):
            concepts.append('fundamental rights')
        
        # Legal processes
        if any(term in text.lower() for term in ['writ', 'petition', 'appeal']):
            concepts.append('legal procedures')
        
        # Specific legal areas
        legal_areas = {
            'criminal': ['criminal', 'penal', 'ipc'],
            'civil': ['civil', 'contract', 'tort'],
            'constitutional': ['constitutional', 'supreme court'],
            'administrative': ['administrative', 'government', 'public']
        }
        
        for area, keywords in legal_areas.items():
            if any(keyword in text.lower() for keyword in keywords):
                concepts.append(f"{area} law")
        
        return list(set(concepts))  # Remove duplicates
    
    def _extract_key_content(self, doc):
        """Extract key content snippets from document"""
        snippet = doc.get('snippet', '')
        title = doc.get('title', '')
        
        key_extracts = []
        
        # Extract sentences with legal terms
        sentences = re.split(r'[.!?]+', snippet)
        
        legal_keywords = ['court', 'judgment', 'held', 'law', 'constitutional', 'right', 'article', 'section']
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in legal_keywords):
                clean_sentence = sentence.strip()
                if len(clean_sentence) > 20:  # Meaningful length
                    key_extracts.append(clean_sentence)
        
        return key_extracts[:3]  # Top 3 key extracts
    
    def _analyze_document_category(self, doc):
        """Analyze document to determine category if not specified"""
        title = doc.get('title', '').lower()
        snippet = doc.get('snippet', '').lower()
        source_name = doc.get('source_name', '').lower()
        
        text = title + ' ' + snippet + ' ' + source_name
        
        # Category indicators
        if any(indicator in text for indicator in ['supreme court', 'high court', 'judgment', 'case']):
            return 'legal_judgments'
        elif any(indicator in text for indicator in ['research', 'study', 'analysis', 'journal']):
            return 'academic_papers'
        elif any(indicator in text for indicator in ['act', 'amendment', 'bill', 'statute']):
            return 'legislative_materials'
        elif any(indicator in text for indicator in ['policy', 'government', 'ministry', 'commission']):
            return 'policy_documents'
        elif any(indicator in text for indicator in ['news', 'editorial', 'opinion', 'commentary']):
            return 'legal_commentary'
        else:
            return 'legal_commentary'  # Default category
    
    def _sort_documents_by_priority(self, documents):
        """Sort documents by priority (relevance + credibility)"""
        def priority_score(doc):
            relevance_score = doc.get('relevance_info', {}).get('semantic_similarity', 0)
            credibility_score = doc.get('credibility_info', {}).get('credibility_score', 0) / 100  # Normalize to 0-1
            
            # Weighted combination
            return (relevance_score * 0.6) + (credibility_score * 0.4)
        
        return sorted(documents, key=priority_score, reverse=True)
    
    def _extract_key_findings(self, organized_docs, research_angle):
        """Extract key findings from organized documents"""
        key_findings = {
            'primary_legal_authorities': [],
            'constitutional_provisions': [],
            'recent_developments': [],
            'academic_insights': [],
            'policy_implications': []
        }
        
        # Extract primary legal authorities (top judgments)
        top_judgments = organized_docs['legal_judgments'][:3]
        for judgment in top_judgments:
            key_findings['primary_legal_authorities'].append({
                'title': judgment['title'],
                'citation': judgment['formatted_citation'],
                'relevance': judgment.get('relevance_info', {}).get('relevance_level', 'Unknown'),
                'key_principle': self._extract_legal_principle(judgment)
            })
        
        # Extract constitutional provisions
        constitutional_refs = self._find_constitutional_references(organized_docs)
        key_findings['constitutional_provisions'] = constitutional_refs
        
        # Extract recent developments
        recent_docs = self._find_recent_developments(organized_docs)
        key_findings['recent_developments'] = recent_docs
        
        # Extract academic insights
        top_academic = organized_docs['academic_papers'][:2]
        for paper in top_academic:
            key_findings['academic_insights'].append({
                'title': paper['title'],
                'authors': self._extract_authors(paper),
                'key_insight': self._extract_academic_insight(paper),
                'relevance_score': paper.get('relevance_info', {}).get('semantic_similarity', 0)
            })
        
        # Extract policy implications
        policy_docs = organized_docs['policy_documents'][:2]
        for policy in policy_docs:
            key_findings['policy_implications'].append({
                'document': policy['title'],
                'implication': self._extract_policy_implication(policy),
                'relevance': policy.get('relevance_info', {}).get('relevance_level', 'Unknown')
            })
        
        return key_findings
    
    def _extract_legal_principle(self, judgment):
        """Extract key legal principle from judgment"""
        snippet = judgment.get('snippet', '')
        key_extracts = judgment.get('key_extracts', [])
        
        # Look for common legal principle indicators
        principle_indicators = ['held that', 'ruled that', 'established that', 'principle', 'law laid down']
        
        for extract in key_extracts:
            for indicator in principle_indicators:
                if indicator in extract.lower():
                    return extract
        
        # Fallback: return first meaningful sentence
        if key_extracts:
            return key_extracts[0]
        
        return snippet[:200] + "..." if len(snippet) > 200 else snippet
    
    def _find_constitutional_references(self, organized_docs):
        """Find constitutional provisions mentioned across documents"""
        constitutional_refs = []
        all_docs = []
        
        # Collect all documents
        for category in organized_docs.values():
            all_docs.extend(category)
        
        # Extract constitutional references
        article_pattern = r'Article\s+(\d+)'
        part_pattern = r'Part\s+([IVX]+)'
        
        ref_count = defaultdict(int)
        ref_context = {}
        
        for doc in all_docs:
            text = doc.get('title', '') + ' ' + doc.get('snippet', '')
            
            # Find Article references
            articles = re.findall(article_pattern, text, re.IGNORECASE)
            for article in articles:
                ref_key = f"Article {article}"
                ref_count[ref_key] += 1
                if ref_key not in ref_context:
                    ref_context[ref_key] = self._extract_constitutional_context(text, ref_key)
            
            # Find Part references
            parts = re.findall(part_pattern, text, re.IGNORECASE)
            for part in parts:
                ref_key = f"Part {part}"
                ref_count[ref_key] += 1
                if ref_key not in ref_context:
                    ref_context[ref_key] = self._extract_constitutional_context(text, ref_key)
        
        # Create constitutional references list
        for ref, count in sorted(ref_count.items(), key=lambda x: x[1], reverse=True)[:5]:
            constitutional_refs.append({
                'provision': ref,
                'frequency': count,
                'context': ref_context.get(ref, ''),
                'importance': 'High' if count > 2 else 'Medium' if count > 1 else 'Low'
            })
        
        return constitutional_refs
    
    def _extract_constitutional_context(self, text, provision):
        """Extract context around constitutional provision"""
        # Find sentence containing the provision
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            if provision.lower() in sentence.lower():
                return sentence.strip()
        
        return f"Referenced in context of {provision}"
    
    def _find_recent_developments(self, organized_docs):
        """Find recent legal developments"""
        recent_developments = []
        current_year = datetime.now().year
        
        all_docs = []
        for category in organized_docs.values():
            all_docs.extend(category)
        
        # Filter for recent documents (last 3 years)
        for doc in all_docs:
            title = doc.get('title', '')
            year_match = re.search(r'\b(20\d{2})\b', title)
            
            if year_match:
                year = int(year_match.group())
                if current_year - year <= 3:  # Last 3 years
                    recent_developments.append({
                        'title': title,
                        'year': year,
                        'type': doc.get('type', 'Unknown'),
                        'significance': self._assess_development_significance(doc),
                        'relevance_score': doc.get('relevance_info', {}).get('semantic_similarity', 0)
                    })
        
        # Sort by relevance and recency
        recent_developments.sort(key=lambda x: (x['relevance_score'], x['year']), reverse=True)
        
        return recent_developments[:3]  # Top 3 recent developments
    
    def _assess_development_significance(self, doc):
        """Assess significance of legal development"""
        title = doc.get('title', '').lower()
        snippet = doc.get('snippet', '').lower()
        
        significance_indicators = {
            'landmark': ['landmark', 'historic', 'groundbreaking'],
            'important': ['important', 'significant', 'major'],
            'notable': ['notable', 'interesting', 'relevant']
        }
        
        text = title + ' ' + snippet
        
        for level, indicators in significance_indicators.items():
            if any(indicator in text for indicator in indicators):
                return level.capitalize()
        
        # Check court level as significance indicator
        if 'supreme court' in text:
            return 'Important'
        elif 'high court' in text:
            return 'Notable'
        
        return 'Standard'
    
    def _extract_academic_insight(self, paper):
        """Extract key academic insight from paper"""
        snippet = paper.get('snippet', '')
        key_extracts = paper.get('key_extracts', [])
        
        # Look for insight indicators
        insight_indicators = ['argues that', 'concludes that', 'suggests that', 'proposes that', 'findings show']
        
        for extract in key_extracts:
            for indicator in insight_indicators:
                if indicator in extract.lower():
                    return extract
        
        # Fallback: return most substantive extract
        if key_extracts:
            return max(key_extracts, key=len)
        
        return snippet[:150] + "..." if len(snippet) > 150 else snippet
    
    def _extract_policy_implication(self, policy_doc):
        """Extract policy implication from document"""
        snippet = policy_doc.get('snippet', '')
        key_extracts = policy_doc.get('key_extracts', [])
        
        # Look for implication indicators
        implication_indicators = ['recommends', 'suggests', 'proposes', 'policy should', 'government must']
        
        for extract in key_extracts:
            for indicator in implication_indicators:
                if indicator in extract.lower():
                    return extract
        
        return snippet[:150] + "..." if len(snippet) > 150 else snippet
    
    def _create_supporting_materials_section(self, organized_docs):
        """Create comprehensive supporting materials section"""
        supporting_materials = {}
        
        for category, docs in organized_docs.items():
            if docs:  # Only include categories with documents
                category_materials = []
                
                for doc in docs[:5]:  # Limit to top 5 per category
                    material = {
                        'title': doc['title'],
                        'citation': doc['formatted_citation'],
                        'relevance_score': doc.get('relevance_info', {}).get('semantic_similarity', 0),
                        'credibility_level': doc.get('credibility_info', {}).get('credibility_level', 'Unknown'),
                        'key_extract': doc.get('key_extracts', [''])[0] if doc.get('key_extracts') else '',
                        'context_summary': doc['contextual_summary'],
                        'url': doc.get('url', ''),
                        'recommendation': doc.get('credibility_info', {}).get('recommendation', 'Standard recommendation')
                    }
                    category_materials.append(material)
                
                # Sort by combined relevance and credibility
                category_materials.sort(key=lambda x: x['relevance_score'], reverse=True)
                supporting_materials[category] = category_materials
        
        return supporting_materials
    
    def _analyze_citation_network(self, network_metrics):
        """Analyze citation network for research brief"""
        if not network_metrics:
            return {
                'network_available': False,
                'message': 'Citation network analysis not available'
            }
        
        analysis = {
            'network_available': True,
            'network_size': {
                'total_documents': network_metrics.get('total_documents', 0),
                'total_citations': network_metrics.get('total_citations', 0),
                'citation_relationships': network_metrics.get('citation_relationships', 0)
            },
            'most_cited_authorities': network_metrics.get('most_cited', [])[:5],
            'citation_clusters': network_metrics.get('citation_clusters', []),
            'network_insights': self._generate_network_insights(network_metrics)
        }
        
        return analysis
    
    def _generate_network_insights(self, network_metrics):
        """Generate insights from citation network"""
        insights = []
        
        most_cited = network_metrics.get('most_cited', [])
        if most_cited:
            top_authority = most_cited[0]
            insights.append(f"Most frequently cited authority: {top_authority.get('citation', 'Unknown')}")
        
        clusters = network_metrics.get('citation_clusters', [])
        if clusters:
            insights.append(f"Identified {len(clusters)} clusters of related legal authorities")
        
        total_docs = network_metrics.get('total_documents', 0)
        total_citations = network_metrics.get('total_citations', 0)
        
        if total_docs > 0 and total_citations > 0:
            ratio = total_citations / total_docs
            if ratio > 3:
                insights.append("Rich citation network indicates comprehensive legal precedent coverage")
            elif ratio > 1:
                insights.append("Moderate citation network with good precedent coverage")
            else:
                insights.append("Limited citation network - additional research may be beneficial")
        
        return insights
    
    def _generate_research_recommendations(self, organized_docs, relevance_scores):
        """Generate research recommendations based on findings"""
        recommendations = {
            'priority_actions': [],
            'research_gaps': [],
            'additional_searches': [],
            'quality_assessment': {}
        }
        
        # Priority actions based on high-relevance sources
        high_relevance_docs = [r for r in relevance_scores if r['relevance_level'] in ['Very High', 'High']]
        
        if len(high_relevance_docs) >= 5:
            recommendations['priority_actions'].append("Strong foundation found - focus on detailed analysis of top sources")
        elif len(high_relevance_docs) >= 2:
            recommendations['priority_actions'].append("Good starting point - expand search for additional supporting materials")
        else:
            recommendations['priority_actions'].append("Limited highly relevant sources - consider broader search terms")
        
        # Identify research gaps
        category_counts = {category: len(docs) for category, docs in organized_docs.items() if docs}
        
        if category_counts.get('legal_judgments', 0) < 3:
            recommendations['research_gaps'].append("Insufficient case law - search for more judicial precedents")
        
        if category_counts.get('academic_papers', 0) < 2:
            recommendations['research_gaps'].append("Limited academic analysis - explore law journals and scholarly articles")
        
        if category_counts.get('policy_documents', 0) < 1:
            recommendations['research_gaps'].append("No policy documents found - consider government reports and policy papers")
        
        # Additional search suggestions
        low_coverage_areas = [cat for cat, count in category_counts.items() if count < 2]
        
        for area in low_coverage_areas:
            search_terms = self._suggest_search_terms_for_category(area)
            recommendations['additional_searches'].extend(search_terms)
        
        # Quality assessment
        total_sources = sum(category_counts.values())
        high_quality_sources = len([r for r in relevance_scores if r['relevance_level'] in ['Very High', 'High']])
        
        quality_ratio = high_quality_sources / total_sources if total_sources > 0 else 0
        
        recommendations['quality_assessment'] = {
            'total_sources': total_sources,
            'high_quality_sources': high_quality_sources,
            'quality_ratio': quality_ratio,
            'assessment': self._assess_research_quality(quality_ratio, total_sources)
        }
        
        return recommendations
    
    def _suggest_search_terms_for_category(self, category):
        """Suggest search terms for specific categories"""
        search_suggestions = {
            'legal_judgments': ['case law', 'Supreme Court judgment', 'High Court decision'],
            'academic_papers': ['law journal', 'legal research', 'scholarly article'],
            'legislative_materials': ['statute', 'act', 'amendment', 'legislation'],
            'policy_documents': ['government policy', 'law commission report', 'ministry guidelines'],
            'legal_commentary': ['legal analysis', 'expert opinion', 'case comment'],
            'comparative_sources': ['comparative law', 'international precedent', 'foreign jurisdiction']
        }
        
        return search_suggestions.get(category, ['additional research'])
    
    def _assess_research_quality(self, quality_ratio, total_sources):
        """Assess overall research quality"""
        if quality_ratio >= 0.7 and total_sources >= 10:
            return "Excellent - comprehensive and highly relevant research base"
        elif quality_ratio >= 0.5 and total_sources >= 5:
            return "Good - solid research foundation with room for expansion"
        elif quality_ratio >= 0.3 and total_sources >= 3:
            return "Fair - basic research completed, needs strengthening"
        else:
            return "Insufficient - requires significant additional research"
    
    def _generate_bibliography(self, organized_docs):
        """Generate complete bibliography in Indian legal citation format"""
        bibliography = {
            'cases': [],
            'statutes': [],
            'books_articles': [],
            'government_documents': [],
            'online_sources': []
        }
        
        all_docs = []
        for category in organized_docs.values():
            all_docs.extend(category)
        
        # Sort all documents by credibility and relevance
        all_docs.sort(key=lambda x: (
            x.get('credibility_info', {}).get('credibility_score', 0),
            x.get('relevance_info', {}).get('semantic_similarity', 0)
        ), reverse=True)
        
        for doc in all_docs:
            doc_type = doc.get('type', '').lower()
            citation_entry = {
                'citation': doc['formatted_citation'],
                'credibility_level': doc.get('credibility_info', {}).get('credibility_level', 'Unknown'),
                'relevance_level': doc.get('relevance_info', {}).get('relevance_level', 'Unknown'),
                'accessed_date': datetime.now().strftime("%d %B %Y")
            }
            
            if doc_type in ['case_law', 'judgment']:
                bibliography['cases'].append(citation_entry)
            elif doc_type in ['statute', 'act', 'legislative']:
                bibliography['statutes'].append(citation_entry)
            elif doc_type in ['academic_paper', 'thesis', 'book']:
                bibliography['books_articles'].append(citation_entry)
            elif doc_type in ['policy', 'government', 'commission_report']:
                bibliography['government_documents'].append(citation_entry)
            else:
                bibliography['online_sources'].append(citation_entry)
        
        return bibliography
    
    def generate_executive_summary(self, research_brief):
        """Generate executive summary of research findings"""
        summary = {
            'research_overview': {
                'research_angle': research_brief['research_summary']['research_angle'],
                'total_sources': research_brief['research_summary']['total_sources_found'],
                'quality_assessment': research_brief['research_recommendations']['quality_assessment']['assessment']
            },
            'key_legal_authorities': [],
            'primary_constitutional_provisions': [],
            'main_arguments': [],
            'research_strength': '',
            'next_steps': []
        }
        
        # Extract key legal authorities
        primary_authorities = research_brief['key_findings']['primary_legal_authorities'][:3]
        for authority in primary_authorities:
            summary['key_legal_authorities'].append({
                'case': authority['title'],
                'relevance': authority['relevance'],
                'principle': authority['key_principle'][:100] + "..."
            })
        
        # Extract constitutional provisions
        constitutional_provisions = research_brief['key_findings']['constitutional_provisions'][:3]
        for provision in constitutional_provisions:
            summary['primary_constitutional_provisions'].append({
                'provision': provision['provision'],
                'importance': provision['importance'],
                'context': provision['context'][:100] + "..."
            })
        
        # Main arguments (extracted from academic insights)
        academic_insights = research_brief['key_findings']['academic_insights'][:2]
        for insight in academic_insights:
            summary['main_arguments'].append(insight['key_insight'][:150] + "...")
        
        # Research strength assessment
        quality_ratio = research_brief['research_recommendations']['quality_assessment']['quality_ratio']
        if quality_ratio >= 0.7:
            summary['research_strength'] = "Strong research foundation with comprehensive coverage"
        elif quality_ratio >= 0.5:
            summary['research_strength'] = "Good research base with solid supporting materials"
        else:
            summary['research_strength'] = "Basic research completed, requires strengthening"
        
        # Next steps from recommendations
        priority_actions = research_brief['research_recommendations']['priority_actions'][:2]
        research_gaps = research_brief['research_recommendations']['research_gaps'][:2]
        
        summary['next_steps'].extend(priority_actions)
        summary['next_steps'].extend(research_gaps)
        
        return summary
    
    def generate_detailed_report(self, research_brief, format_type='markdown'):
        """Generate detailed research report in specified format"""
        if format_type == 'markdown':
            return self._generate_markdown_report(research_brief)
        elif format_type == 'json':
            return json.dumps(research_brief, indent=2, ensure_ascii=False)
        elif format_type == 'text':
            return self._generate_text_report(research_brief)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    def _generate_markdown_report(self, research_brief):
        """Generate markdown formatted research report"""
        md_content = []
        
        # Title and overview
        research_angle = research_brief['research_summary']['research_angle']
        md_content.append(f"# Legal Research Report: {research_angle}")
        md_content.append("")
        
        # Executive summary
        md_content.append("## Executive Summary")
        md_content.append("")
        summary = research_brief['research_summary']
        md_content.append(f"- **Research Topic**: {research_angle}")
        md_content.append(f"- **Total Sources Found**: {summary['total_sources_found']}")
        md_content.append(f"- **High Relevance Sources**: {summary['high_relevance_sources']}")
        md_content.append(f"- **High Credibility Sources**: {summary['high_credibility_sources']}")
        md_content.append(f"- **Generated**: {summary['generated_at']}")
        md_content.append("")
        
        # Key findings
        md_content.append("## Key Findings")
        md_content.append("")
        
        key_findings = research_brief['key_findings']
        
        # Primary legal authorities
        if key_findings['primary_legal_authorities']:
            md_content.append("### Primary Legal Authorities")
            md_content.append("")
            for i, authority in enumerate(key_findings['primary_legal_authorities'], 1):
                md_content.append(f"**{i}. {authority['title']}**")
                md_content.append(f"- Citation: {authority['citation']}")
                md_content.append(f"- Relevance: {authority['relevance']}")
                md_content.append(f"- Key Principle: {authority['key_principle']}")
                md_content.append("")
        
        # Constitutional provisions
        if key_findings['constitutional_provisions']:
            md_content.append("### Constitutional Provisions")
            md_content.append("")
            for provision in key_findings['constitutional_provisions']:
                md_content.append(f"**{provision['provision']}** ({provision['importance']} importance)")
                md_content.append(f"- Frequency: {provision['frequency']} references")
                md_content.append(f"- Context: {provision['context']}")
                md_content.append("")
        
        # Supporting materials
        md_content.append("## Supporting Materials")
        md_content.append("")
        
        supporting_materials = research_brief['supporting_materials']
        for category, materials in supporting_materials.items():
            if materials:
                category_name = category.replace('_', ' ').title()
                md_content.append(f"### {category_name} ({len(materials)} sources)")
                md_content.append("")
                
                for material in materials[:3]:  # Show top 3 per category
                    md_content.append(f"**{material['title']}**")
                    md_content.append(f"- Citation: {material['citation']}")
                    md_content.append(f"- Relevance Score: {material['relevance_score']:.2f}")
                    md_content.append(f"- Credibility: {material['credibility_level']}")
                    if material['key_extract']:
                        md_content.append(f"- Key Extract: {material['key_extract']}")
                    md_content.append(f"- URL: {material['url']}")
                    md_content.append("")
        
        # Research recommendations
        md_content.append("## Research Recommendations")
        md_content.append("")
        
        recommendations = research_brief['research_recommendations']
        
        if recommendations['priority_actions']:
            md_content.append("### Priority Actions")
            md_content.append("")
            for action in recommendations['priority_actions']:
                md_content.append(f"- {action}")
            md_content.append("")
        
        if recommendations['research_gaps']:
            md_content.append("### Research Gaps")
            md_content.append("")
            for gap in recommendations['research_gaps']:
                md_content.append(f"- {gap}")
            md_content.append("")
        
        # Quality assessment
        quality = recommendations['quality_assessment']
        md_content.append("### Quality Assessment")
        md_content.append("")
        md_content.append(f"- Total Sources: {quality['total_sources']}")
        md_content.append(f"- High Quality Sources: {quality['high_quality_sources']}")
        md_content.append(f"- Quality Ratio: {quality['quality_ratio']:.2f}")
        md_content.append(f"- Assessment: {quality['assessment']}")
        md_content.append("")
        
        # Bibliography
        md_content.append("## Complete Bibliography")
        md_content.append("")
        
        bibliography = research_brief['complete_bibliography']
        for category, citations in bibliography.items():
            if citations:
                category_name = category.replace('_', ' ').title()
                md_content.append(f"### {category_name}")
                md_content.append("")
                
                for citation in citations:
                    md_content.append(f"- {citation['citation']}")
                    md_content.append(f"  - Credibility: {citation['credibility_level']}")
                    md_content.append(f"  - Relevance: {citation['relevance_level']}")
                    md_content.append(f"  - Accessed: {citation['accessed_date']}")
                md_content.append("")
        
        return "\n".join(md_content)
    
    def _generate_text_report(self, research_brief):
        """Generate plain text research report"""
        text_content = []
        
        # Title
        research_angle = research_brief['research_summary']['research_angle']
        text_content.append("=" * 80)
        text_content.append(f"LEGAL RESEARCH REPORT: {research_angle.upper()}")
        text_content.append("=" * 80)
        text_content.append("")
        
        # Summary
        summary = research_brief['research_summary']
        text_content.append("RESEARCH SUMMARY")
        text_content.append("-" * 20)
        text_content.append(f"Research Topic: {research_angle}")
        text_content.append(f"Total Sources Found: {summary['total_sources_found']}")
        text_content.append(f"High Relevance Sources: {summary['high_relevance_sources']}")
        text_content.append(f"High Credibility Sources: {summary['high_credibility_sources']}")
        text_content.append(f"Generated: {summary['generated_at']}")
        text_content.append("")
        
        # Key findings
        text_content.append("KEY FINDINGS")
        text_content.append("-" * 15)
        
        key_findings = research_brief['key_findings']
        
        if key_findings['primary_legal_authorities']:
            text_content.append("Primary Legal Authorities:")
            for i, authority in enumerate(key_findings['primary_legal_authorities'], 1):
                text_content.append(f"{i}. {authority['title']}")
                text_content.append(f"   Citation: {authority['citation']}")
                text_content.append(f"   Relevance: {authority['relevance']}")
                text_content.append(f"   Key Principle: {authority['key_principle'][:200]}...")
                text_content.append("")
        
        # Recommendations
        text_content.append("RESEARCH RECOMMENDATIONS")
        text_content.append("-" * 25)
        
        recommendations = research_brief['research_recommendations']
        
        if recommendations['priority_actions']:
            text_content.append("Priority Actions:")
            for action in recommendations['priority_actions']:
                text_content.append(f"• {action}")
            text_content.append("")
        
        if recommendations['research_gaps']:
            text_content.append("Research Gaps:")
            for gap in recommendations['research_gaps']:
                text_content.append(f"• {gap}")
            text_content.append("")
        
        return "\n".join(text_content)
    
    def export_citations(self, research_brief, format_type='bibtex'):
        """Export citations in various academic formats"""
        bibliography = research_brief['complete_bibliography']
        
        if format_type == 'bibtex':
            return self._export_bibtex(bibliography)
        elif format_type == 'apa':
            return self._export_apa(bibliography)
        elif format_type == 'mla':
            return self._export_mla(bibliography)
        elif format_type == 'indian_legal':
            return self._export_indian_legal_format(bibliography)
        else:
            raise ValueError(f"Unsupported citation format: {format_type}")
    
    def _export_bibtex(self, bibliography):
        """Export citations in BibTeX format"""
        bibtex_entries = []
        
        for category, citations in bibliography.items():
            for i, citation in enumerate(citations):
                entry_type = self._get_bibtex_entry_type(category)
                entry_key = f"{category}_{i+1}"
                
                bibtex_entry = f"@{entry_type}{{{entry_key},\n"
                bibtex_entry += f"  title = {{{citation['citation']}}},\n"
                bibtex_entry += f"  note = {{Accessed: {citation['accessed_date']}}},\n"
                bibtex_entry += f"  year = {{2024}},\n"
                bibtex_entry += "}\n"
                
                bibtex_entries.append(bibtex_entry)
        
        return "\n".join(bibtex_entries)
    
    def _get_bibtex_entry_type(self, category):
        """Get appropriate BibTeX entry type for category"""
        entry_types = {
            'cases': 'misc',
            'statutes': 'misc',
            'books_articles': 'article',
            'government_documents': 'techreport',
            'online_sources': 'misc'
        }
        return entry_types.get(category, 'misc')
    
    def _export_apa(self, bibliography):
        """Export citations in APA format"""
        apa_citations = []
        
        for category, citations in bibliography.items():
            category_name = category.replace('_', ' ').title()
            apa_citations.append(f"{category_name}:")
            apa_citations.append("")
            
            for citation in citations:
                # Simple APA format - would need more sophisticated parsing for full APA
                apa_citations.append(f"{citation['citation']} Retrieved {citation['accessed_date']}")
            
            apa_citations.append("")
        
        return "\n".join(apa_citations)
    
    def _export_mla(self, bibliography):
        """Export citations in MLA format"""
        mla_citations = []
        
        for category, citations in bibliography.items():
            category_name = category.replace('_', ' ').title()
            mla_citations.append(f"{category_name}:")
            mla_citations.append("")
            
            for citation in citations:
                # Simple MLA format - would need more sophisticated parsing for full MLA
                mla_citations.append(f"{citation['citation']} Web. {citation['accessed_date']}")
            
            mla_citations.append("")
        
        return "\n".join(mla_citations)
    
    def _export_indian_legal_format(self, bibliography):
        """Export citations in standard Indian legal format"""
        indian_citations = []
        
        categories_order = ['cases', 'statutes', 'books_articles', 'government_documents', 'online_sources']
        
        for category in categories_order:
            if category in bibliography and bibliography[category]:
                category_names = {
                    'cases': 'CASES',
                    'statutes': 'STATUTES AND ACTS',
                    'books_articles': 'BOOKS AND ARTICLES',
                    'government_documents': 'GOVERNMENT DOCUMENTS',
                    'online_sources': 'ONLINE SOURCES'
                }
                
                indian_citations.append(category_names[category])
                indian_citations.append("=" * len(category_names[category]))
                indian_citations.append("")
                
                for citation in bibliography[category]:
                    indian_citations.append(citation['citation'])
                
                indian_citations.append("")
        
        return "\n".join(indian_citations)
    
    def generate_research_metrics(self, research_brief):
        """Generate comprehensive research metrics"""
        metrics = {
            'source_distribution': {},
            'quality_metrics': {},
            'coverage_analysis': {},
            'citation_analysis': {},
            'temporal_analysis': {}
        }
        
        supporting_materials = research_brief['supporting_materials']
        
        # Source distribution
        for category, materials in supporting_materials.items():
            metrics['source_distribution'][category] = len(materials)
        
        # Quality metrics
        recommendations = research_brief['research_recommendations']
        quality_assessment = recommendations['quality_assessment']
        
        metrics['quality_metrics'] = {
            'total_sources': quality_assessment['total_sources'],
            'high_quality_ratio': quality_assessment['quality_ratio'],
            'assessment_level': quality_assessment['assessment'],
            'research_gaps_count': len(recommendations['research_gaps'])
        }
        
        # Coverage analysis
        covered_categories = len([cat for cat, materials in supporting_materials.items() if materials])
        total_categories = len(supporting_materials)
        
        metrics['coverage_analysis'] = {
            'categories_covered': covered_categories,
            'total_categories': total_categories,
            'coverage_percentage': (covered_categories / total_categories) * 100 if total_categories > 0 else 0
        }
        
        # Citation analysis
        network_analysis = research_brief.get('citation_network_analysis', {})
        if network_analysis.get('network_available'):
            network_size = network_analysis['network_size']
            metrics['citation_analysis'] = {
                'total_citations': network_size['total_citations'],
                'citation_density': network_size['citation_relationships'] / network_size['total_documents'] if network_size['total_documents'] > 0 else 0,
                'most_cited_count': len(network_analysis.get('most_cited_authorities', [])),
                'clusters_found': len(network_analysis.get('citation_clusters', []))
            }
        
        # Temporal analysis
        recent_developments = research_brief['key_findings'].get('recent_developments', [])
        metrics['temporal_analysis'] = {
            'recent_developments_count': len(recent_developments),
            'average_recency_years': self._calculate_average_recency(recent_developments),
            'temporal_coverage': 'Good' if len(recent_developments) > 2 else 'Limited'
        }
        
        return metrics
    
    def _calculate_average_recency(self, recent_developments):
        """Calculate average recency of developments"""
        if not recent_developments:
            return 0
        
        current_year = datetime.now().year
        years = [dev.get('year', current_year) for dev in recent_developments]
        
        return sum(current_year - year for year in years) / len(years)

# Example usage and testing
if __name__ == "__main__":
    curator = ResearchCurator()
    
    # Test with sample data
    sample_documents = [
        {
            'title': 'K.S. Puttaswamy v. Union of India (2017) 10 SCC 1',
            'url': 'https://indiankanoon.org/doc/91938676/',
            'snippet': 'The Supreme Court in this landmark judgment recognized privacy as a fundamental right under Articles 14, 19, and 21 of the Constitution.',
            'source_name': 'indian_kanoon',
            'type': 'case_law'
        },
        {
            'title': 'Digital Privacy in India: A Constitutional Analysis',
            'url': 'https://example.com/paper1',
            'snippet': 'This paper argues that digital privacy rights must be protected under the constitutional framework of India.',
            'source_name': 'google_scholar',
            'type': 'academic_paper',
            'authors': 'Dr. Legal Scholar'
        }
    ]
    
    sample_credibility = [
        {
            'document_id': 'doc_123456',
            'credibility_score': 85,
            'credibility_level': 'Very High',
            'recommendation': 'Highly recommended for legal research'
        },
        {
            'document_id': 'doc_789012',
            'credibility_score': 75,
            'credibility_level': 'High',
            'recommendation': 'Recommended for legal research'
        }
    ]
    
    sample_relevance = [
        {
            'document_id': 'doc_123456',
            'semantic_similarity': 0.8,
            'relevance_level': 'Very High'
        },
        {
            'document_id': 'doc_789012',
            'semantic_similarity': 0.7,
            'relevance_level': 'High'
        }
    ]
    
    sample_network = {
        'total_documents': 2,
        'total_citations': 5,
        'citation_relationships': 3,
        'most_cited': [
            {'citation': 'Article 21', 'frequency': 5, 'authority_score': 95},
            {'citation': 'K.S. Puttaswamy case', 'frequency': 3, 'authority_score': 90}
        ],
        'citation_clusters': [
            {'cluster_id': 1, 'size': 3, 'citations': ['Article 21', 'Privacy', 'Digital Rights']}
        ]
    }
    
    research_angle = "Exploring how Article 21 extends to digital privacy rights in the context of Aadhaar and data surveillance"
    
    # Test curation
    research_brief = curator.curate_research_findings(
        sample_documents, sample_credibility, sample_relevance, 
        sample_network, research_angle
    )
    
    if research_brief:
        print("Research Brief Generated Successfully!")
        print("=" * 50)
        
        # Test executive summary
        executive_summary = curator.generate_executive_summary(research_brief)
        print("Executive Summary:")
        print(json.dumps(executive_summary, indent=2, ensure_ascii=False))
        print()
        
        # Test markdown report generation
        markdown_report = curator.generate_detailed_report(research_brief, 'markdown')
        print("Markdown Report (first 500 chars):")
        print(markdown_report[:500] + "...")
        print()
        
        # Test citation export
        indian_citations = curator.export_citations(research_brief, 'indian_legal')
        print("Indian Legal Citations:")
        print(indian_citations)
        print()
        
        # Test research metrics
        metrics = curator.generate_research_metrics(research_brief)
        print("Research Metrics:")
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        
    else:
        print("Failed to generate research brief")