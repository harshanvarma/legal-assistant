"""
Agent 1: Legal Document Analyzer & Research Interpreter
- Extracts key legal concepts, precedents, and arguments from base material
- Identifies legal domains, applicable statutes, and jurisdictional context
- Maps argument structure and logical flow
- Processes student's research direction and identifies specific legal questions
"""

import os
import re
import json
from dotenv import load_dotenv
from groq import Groq
from utils import TextProcessor, CitationFormatter
from config import Config
import logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class LegalAnalyzer:
    def __init__(self):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.text_processor = TextProcessor()
        self.citation_formatter = CitationFormatter()
        
    def analyze_base_document(self, document_text, document_url=None):
        """Analyze the base legal document or research paper"""
        try:
            # Extract basic information
            citations = self.text_processor.extract_legal_citations(document_text)
            keywords = self.text_processor.extract_keywords(document_text)
            
            # Use LLM to extract deeper legal concepts
            analysis_prompt = f"""
            Analyze this Indian legal document and extract the following information:
            
            Document Text: {document_text[:3000]}...
            
            Please provide a structured analysis including:
            1. Main legal concepts and principles discussed
            2. Key statutes, acts, and articles referenced
            3. Important case laws and precedents cited
            4. Legal domain(s) (Constitutional, Criminal, Civil, Commercial, etc.)
            5. Jurisdictional context (Supreme Court, High Court, etc.)
            6. Central legal arguments and reasoning
            7. Legal issues and questions raised
            8. Relevant Indian legal terminology used
            
            Format your response as JSON with these keys: legal_concepts, statutes_acts, case_laws, legal_domain, jurisdiction, arguments, legal_issues, terminology
            """
            
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": analysis_prompt}],
                model="llama-3.3-70b-versatile",
                stream=False,
            )
            
            # Parse LLM response
            llm_analysis = self._parse_llm_response(response.choices[0].message.content)
            
            # Combine extracted and analyzed information
            analysis = {
                'document_url': document_url,
                'extracted_citations': citations,
                'extracted_keywords': keywords,
                'llm_analysis': llm_analysis,
                'document_summary': self._generate_summary(document_text),
                'legal_complexity': self._assess_complexity(document_text),
                'timestamp': self._get_timestamp()
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing base document: {e}")
            return None
    
    def interpret_research_angle(self, research_angle, base_analysis):
        """Interpret the student's research angle and identify gaps"""
        try:
            interpretation_prompt = f"""
            Based on this base document analysis and the student's research angle, identify research gaps and requirements:
            
            Base Document Analysis: {json.dumps(base_analysis.get('llm_analysis', {}), indent=2)}
            
            Student's Research Angle: {research_angle}
            
            Please analyze and provide:
            1. Specific legal questions that need to be answered
            2. Types of supporting evidence needed (case law, academic papers, policy documents)
            3. Knowledge gaps between base material and research objective
            4. Relevant legal domains to explore
            5. Suggested search strategies and keywords
            6. Priority areas for research focus
            7. Potential counterarguments to address
            8. Related constitutional articles or statutory provisions to investigate
            
            Format as JSON with keys: legal_questions, evidence_types, knowledge_gaps, domains, search_strategies, priority_areas, counterarguments, related_provisions
            """
            
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": interpretation_prompt}],
                model="llama-3.3-70b-versatile",
                stream=False,
            )
            
            interpretation = self._parse_llm_response(response.choices[0].message.content)
            
            # Generate specific research requirements
            research_requirements = {
                'research_angle': research_angle,
                'interpretation': interpretation,
                'search_keywords': self._generate_search_keywords(interpretation, base_analysis),
                'evidence_priorities': self._prioritize_evidence_types(interpretation),
                'research_strategy': self._develop_research_strategy(interpretation),
                'timestamp': self._get_timestamp()
            }
            
            return research_requirements
            
        except Exception as e:
            logger.error(f"Error interpreting research angle: {e}")
            return None
    
    def identify_legal_context(self, base_analysis, research_requirements):
        """Identify comprehensive legal context for research"""
        try:
            context = {
                'primary_legal_domain': self._identify_primary_domain(base_analysis),
                'secondary_domains': self._identify_secondary_domains(base_analysis, research_requirements),
                'applicable_statutes': self._extract_applicable_statutes(base_analysis),
                'relevant_articles': self._extract_constitutional_articles(base_analysis),
                'jurisdictional_focus': self._determine_jurisdiction(base_analysis),
                'temporal_scope': self._determine_temporal_scope(research_requirements),
                'comparative_requirements': self._assess_comparative_needs(research_requirements)
            }
            
            return context
            
        except Exception as e:
            logger.error(f"Error identifying legal context: {e}")
            return None
    
    def _parse_llm_response(self, response_text):
        """Parse LLM response and extract JSON"""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # If no JSON found, create structured response from text
                return self._structure_text_response(response_text)
        except:
            return {'raw_response': response_text}
    
    def _structure_text_response(self, text):
        """Structure plain text response into categories"""
        structured = {}
        
        # Define patterns for different categories
        patterns = {
            'legal_concepts': r'(?:legal concepts?|principles?)[:\-]\s*(.+?)(?:\n\n|\d\.)',
            'statutes_acts': r'(?:statutes?|acts?)[:\-]\s*(.+?)(?:\n\n|\d\.)',
            'case_laws': r'(?:case laws?|precedents?)[:\-]\s*(.+?)(?:\n\n|\d\.)',
            'legal_domain': r'(?:legal domain|domain)[:\-]\s*(.+?)(?:\n\n|\d\.)',
            'arguments': r'(?:arguments?|reasoning)[:\-]\s*(.+?)(?:\n\n|\d\.)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                structured[key] = match.group(1).strip()
        
        return structured
    
    def _generate_summary(self, document_text):
        """Generate a concise summary of the document"""
        # Simple extractive summary - take first few sentences and key points
        sentences = document_text.split('.')[:5]
        return '. '.join(sentences) + '.'
    
    def _assess_complexity(self, document_text):
        """Assess the legal complexity of the document"""
        complexity_indicators = {
            'citations_count': len(self.text_processor.extract_legal_citations(document_text)),
            'legal_terms_count': len([word for word in document_text.split() if word in Config.LEGAL_KEYWORDS]),
            'document_length': len(document_text.split()),
            'statutory_references': len(re.findall(r'(?:Article|Section)\s+\d+', document_text))
        }
        
        # Simple complexity scoring
        score = (
            complexity_indicators['citations_count'] * 2 +
            complexity_indicators['legal_terms_count'] +
            complexity_indicators['statutory_references'] * 3
        ) / 10
        
        return {
            'score': min(score, 10),  # Cap at 10
            'indicators': complexity_indicators,
            'level': 'High' if score > 7 else 'Medium' if score > 4 else 'Low'
        }
    
    def _generate_search_keywords(self, interpretation, base_analysis):
        """Generate intelligent search keywords"""
        keywords = []
        
        # Extract keywords from interpretation - handle dict safely
        if isinstance(interpretation, dict):
            search_strategies = interpretation.get('search_strategies', [])
            if isinstance(search_strategies, list):
                keywords.extend([str(s) for s in search_strategies])
            elif isinstance(search_strategies, str):
                keywords.append(search_strategies)
        
        # Add keywords from base analysis - handle dict safely
        if isinstance(base_analysis, dict):
            extracted_keywords = base_analysis.get('extracted_keywords', [])
            if isinstance(extracted_keywords, list):
                keywords.extend([str(k) for k in extracted_keywords[:10]])
        
        # Add domain-specific keywords - handle dict safely
        if isinstance(base_analysis, dict):
            llm_analysis = base_analysis.get('llm_analysis', {})
            if isinstance(llm_analysis, dict):
                domain = llm_analysis.get('legal_domain', '')
                if isinstance(domain, str) and domain.lower() in Config.LEGAL_KEYWORDS:
                    domain_keywords = Config.LEGAL_KEYWORDS[domain.lower()]
                    if isinstance(domain_keywords, list):
                        keywords.extend([str(k) for k in domain_keywords])
        
        # Clean and return unique keywords
        clean_keywords = []
        for keyword in keywords:
            if keyword and isinstance(keyword, str) and len(keyword.strip()) > 0:
                clean_keywords.append(keyword.strip())
        
        return list(set(clean_keywords))  # Remove duplicates
    
    def _prioritize_evidence_types(self, interpretation):
        """Prioritize types of evidence needed"""
        evidence_types = interpretation.get('evidence_types', [])
        
        priority_map = {
            'case law': 1,
            'supreme court judgments': 1,
            'constitutional provisions': 2,
            'academic papers': 3,
            'policy documents': 4,
            'government reports': 4,
            'legal commentary': 5
        }
        
        prioritized = []
        for evidence in evidence_types:
            priority = priority_map.get(evidence.lower(), 6)
            prioritized.append({'type': evidence, 'priority': priority})
        
        return sorted(prioritized, key=lambda x: x['priority'])
    
    def _develop_research_strategy(self, interpretation):
        """Develop comprehensive research strategy"""
        return {
            'search_phases': [
                'Primary case law research',
                'Constitutional and statutory analysis',
                'Academic literature review',
                'Policy and government document analysis',
                'Comparative and international research'
            ],
            'search_order': self._determine_search_order(interpretation),
            'depth_levels': {
                'broad_search': 'Initial comprehensive search',
                'focused_search': 'Targeted specific queries',
                'citation_chase': 'Follow citation networks'
            }
        }
    
    def _identify_primary_domain(self, base_analysis):
        """Identify the primary legal domain"""
        domain = base_analysis.get('llm_analysis', {}).get('legal_domain', '')
        return domain if domain else 'Constitutional Law'  # Default
    
    def _identify_secondary_domains(self, base_analysis, research_requirements):
        """Identify secondary legal domains to explore"""
        domains = research_requirements.get('interpretation', {}).get('domains', [])
        return domains[:3]  # Top 3 secondary domains
    
    def _extract_applicable_statutes(self, base_analysis):
        """Extract applicable statutes and acts"""
        statutes = base_analysis.get('llm_analysis', {}).get('statutes_acts', [])
        if isinstance(statutes, str):
            statutes = [statutes]
        return statutes
    
    def _extract_constitutional_articles(self, base_analysis):
        """Extract relevant constitutional articles"""
        text = str(base_analysis)
        articles = re.findall(r'Article\s+(\d+)', text)
        return [f"Article {article}" for article in set(articles)]
    
    def _determine_jurisdiction(self, base_analysis):
        """Determine jurisdictional focus"""
        jurisdiction = base_analysis.get('llm_analysis', {}).get('jurisdiction', '')
        return jurisdiction if jurisdiction else 'Supreme Court'
    
    def _determine_temporal_scope(self, research_requirements):
        """Determine temporal scope for research"""
        return {
            'primary_period': 'Last 10 years',
            'landmark_cases': 'All time',
            'recent_developments': 'Last 2 years'
        }
    
    def _assess_comparative_needs(self, research_requirements):
        """Assess need for comparative legal research"""
        interpretation = research_requirements.get('interpretation', {})
        comparative_indicators = ['international', 'comparative', 'foreign', 'global']
        
        needs_comparative = any(
            indicator in str(interpretation).lower() 
            for indicator in comparative_indicators
        )
        
        return {
            'required': needs_comparative,
            'focus_areas': ['Constitutional law', 'Human rights', 'Privacy law'] if needs_comparative else []
        }
    
    def _determine_search_order(self, interpretation):
        """Determine optimal search order"""
        priorities = interpretation.get('priority_areas', [])
        return priorities[:5] if priorities else [
            'Primary case law',
            'Constitutional provisions',
            'Academic analysis',
            'Policy documents',
            'Recent developments'
        ]
    
    def _get_timestamp(self):
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()

# Example usage and testing
if __name__ == "__main__":
    analyzer = LegalAnalyzer()
    
    # Test with sample document
    sample_doc = """
    Article 21 of the Indian Constitution guarantees the right to life and personal liberty. 
    In Maneka Gandhi v. Union of India (1978) 1 SCC 248, the Supreme Court expanded the 
    interpretation of Article 21 to include the right to privacy. The Aadhaar case of 
    K.S. Puttaswamy v. Union of India (2017) 10 SCC 1 further established privacy as 
    a fundamental right under Articles 14, 19, and 21.
    """
    
    research_angle = "Exploring how Article 21 extends to digital privacy rights in the context of Aadhaar and data surveillance"
    
    base_analysis = analyzer.analyze_base_document(sample_doc)
    if base_analysis:
        print("Base Analysis:", json.dumps(base_analysis, indent=2))
        
        research_requirements = analyzer.interpret_research_angle(research_angle, base_analysis)
        if research_requirements:
            print("Research Requirements:", json.dumps(research_requirements, indent=2))