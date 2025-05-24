"""
Main Coordinator for Multi-Agent Legal Research Companion
Orchestrates all four agents to provide comprehensive legal research automation
"""

import asyncio
import json
import os
import sys
from datetime import datetime
import logging
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import all agents
from agent1_legal_analyzer import LegalAnalyzer
from agent2_research_engine import ResearchEngine
from agent3_citation_validator import CitationValidator
from agent4_research_curator import ResearchCurator
from utils import ResultsManager, CacheManager
from config import Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('legal_research.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class LegalResearchCompanion:
    """Main coordinator class that orchestrates all agents"""
    
    def __init__(self):
        self.analyzer = LegalAnalyzer()
        self.validator = CitationValidator()
        self.curator = ResearchCurator()
        self.results_manager = ResultsManager()
        self.cache_manager = CacheManager()
        
    async def conduct_research(self, base_material: str, research_angle: str, 
                             material_url: Optional[str] = None,
                             preferences: Optional[Dict] = None) -> Dict:
        """
        Main research orchestration method
        
        Args:
            base_material: Base research material text or URL
            research_angle: Student's specific research direction
            material_url: Optional URL of the base material
            preferences: Optional search preferences
            
        Returns:
            Comprehensive research results
        """
        try:
            research_id = self._generate_research_id()
            logger.info(f"Starting research session: {research_id}")
            
            # Phase 1: Analyze base material and research angle
            logger.info("Phase 1: Analyzing base material and research angle")
            base_analysis = await self._phase1_analysis(base_material, material_url, research_angle)
            
            if not base_analysis:
                raise Exception("Failed to analyze base material")
            
            # Phase 2: Execute multi-source research
            logger.info("Phase 2: Executing multi-source research")
            search_results = await self._phase2_research(base_analysis['research_requirements'])
            
            if not search_results:
                raise Exception("Failed to execute research searches")
            
            # Phase 3: Validate citations and build authority network
            logger.info("Phase 3: Validating citations and building authority network")
            validation_results = await self._phase3_validation(search_results)
            
            # Phase 4: Curate and organize final results
            logger.info("Phase 4: Curating and organizing results")
            final_results = await self._phase4_curation(
                search_results, validation_results, research_angle, base_analysis
            )
            
            # Generate comprehensive output
            research_output = {
                'research_id': research_id,
                'input_summary': {
                    'base_material_summary': base_analysis['base_analysis']['document_summary'],
                    'research_angle': research_angle,
                    'material_url': material_url,
                    'processing_timestamp': datetime.now().isoformat()
                },
                'research_brief': final_results['research_brief'],
                'executive_summary': final_results['executive_summary'],
                'detailed_findings': {
                    'base_analysis': base_analysis,
                    'search_results_summary': self._summarize_search_results(search_results),
                    'validation_summary': validation_results,
                    'supporting_materials': final_results['research_brief']['supporting_materials']
                },
                'recommendations': final_results['research_brief']['research_recommendations'],
                'complete_bibliography': final_results['research_brief']['complete_bibliography'],
                'research_metadata': {
                    'total_sources_found': len(search_results.get('filtered_documents', [])),
                    'high_quality_sources': len([
                        doc for doc in validation_results.get('credibility_scores', [])
                        if doc['credibility_level'] in ['Very High', 'High']
                    ]),
                    'processing_time': self._calculate_processing_time(research_id),
                    'confidence_score': self._calculate_confidence_score(final_results)
                }
            }
            
            # Save results
            saved_path = self.results_manager.save_results(research_id, research_output)
            research_output['saved_to'] = saved_path
            
            logger.info(f"Research completed successfully: {research_id}")
            return research_output
            
        except Exception as e:
            logger.error(f"Error in research process: {e}")
            return {
                'error': True,
                'message': str(e),
                'research_id': research_id if 'research_id' in locals() else 'unknown',
                'timestamp': datetime.now().isoformat()
            }
    
    def quick_research(self, research_query: str, max_sources: int = 20) -> Dict:
        """Quick research mode for simple queries"""
        try:
            research_id = self._generate_research_id()
            logger.info(f"Starting quick research: {research_id}")
            
            # Simple approach - treat query as both base material and research angle
            return asyncio.run(self.conduct_research(
                base_material=research_query,
                research_angle=research_query,
                preferences={'max_sources': max_sources, 'quick_mode': True}
            ))
            
        except Exception as e:
            logger.error(f"Quick research error: {e}")
            return {
                'error': True,
                'message': f"Quick research failed: {str(e)}",
                'timestamp': datetime.now().isoformat()
            }
    
    async def _phase1_analysis(self, base_material: str, material_url: Optional[str], 
                               research_angle: str) -> Dict:
        """Phase 1: Legal document analysis and research interpretation"""
        try:
            # Analyze base document
            base_analysis = self.analyzer.analyze_base_document(base_material, material_url)
            
            if not base_analysis:
                raise Exception("Failed to analyze base document")
            
            # Interpret research angle
            research_requirements = self.analyzer.interpret_research_angle(research_angle, base_analysis)
            
            if not research_requirements:
                raise Exception("Failed to interpret research angle")
            
            # Identify legal context
            legal_context = self.analyzer.identify_legal_context(base_analysis, research_requirements)
            
            return {
                'base_analysis': base_analysis,
                'research_requirements': research_requirements,
                'legal_context': legal_context,
                'phase_status': 'completed'
            }
            
        except Exception as e:
            logger.error(f"Phase 1 error: {e}")
            return None
    
    async def _phase2_research(self, research_requirements: Dict) -> Dict:
        """Phase 2: Multi-source research execution"""
        try:
            async with ResearchEngine() as research_engine:
                # Generate search queries
                query_list = research_engine.generate_search_queries(research_requirements)
                logger.info(f"Generated {len(query_list)} search queries")
                
                # Execute searches across multiple sources
                search_results = await research_engine.execute_multi_source_search(query_list)
                logger.info(f"Search completed: {search_results['search_metadata']}")
                
                # Filter and deduplicate results
                filtered_documents = research_engine.filter_and_deduplicate_results(search_results)
                logger.info(f"Filtered to {len(filtered_documents)} unique documents")
                
                return {
                    'raw_search_results': search_results,
                    'filtered_documents': filtered_documents,
                    'search_metadata': search_results['search_metadata'],
                    'query_list': query_list,
                    'phase_status': 'completed'
                }
                
        except Exception as e:
            logger.error(f"Phase 2 error: {e}")
            return None
    
    async def _phase3_validation(self, search_results: Dict) -> Dict:
        """Phase 3: Citation validation and authority assessment"""
        try:
            documents = search_results.get('filtered_documents', [])
            
            if not documents:
                logger.warning("No documents to validate")
                return {
                    'credibility_scores': [],
                    'relevance_scores': [],
                    'network_metrics': {},
                    'authority_rankings': [],
                    'phase_status': 'completed_with_warnings'
                }
            
            # Build citation network
            network_metrics = self.validator.build_citation_network(documents)
            logger.info(f"Built citation network: {network_metrics}")
            
            # Validate source credibility
            credibility_scores = self.validator.validate_source_credibility(documents)
            logger.info(f"Validated {len(credibility_scores)} sources for credibility")
            
            # Generate authority rankings
            authority_rankings = self.validator.generate_authority_ranking(documents)
            logger.info(f"Generated authority rankings for {len(authority_rankings)} documents")
            
            return {
                'credibility_scores': credibility_scores,
                'network_metrics': network_metrics,
                'authority_rankings': authority_rankings,
                'phase_status': 'completed'
            }
            
        except Exception as e:
            logger.error(f"Phase 3 error: {e}")
            return {
                'credibility_scores': [],
                'network_metrics': {},
                'authority_rankings': [],
                'phase_status': 'error',
                'error_message': str(e)
            }
    
    async def _phase4_curation(self, search_results: Dict, validation_results: Dict, 
                               research_angle: str, base_analysis: Dict) -> Dict:
        """Phase 4: Research curation and final organization"""
        try:
            documents = search_results.get('filtered_documents', [])
            credibility_scores = validation_results.get('credibility_scores', [])
            network_metrics = validation_results.get('network_metrics', {})
            
            # Calculate semantic relevance (this was missing from phase 3)
            relevance_scores = self.validator.measure_semantic_relevance(documents, research_angle)
            logger.info(f"Calculated semantic relevance for {len(relevance_scores)} documents")
            
            # Curate research findings
            research_brief = self.curator.curate_research_findings(
                documents, credibility_scores, relevance_scores, network_metrics, research_angle
            )
            
            if not research_brief:
                raise Exception("Failed to curate research findings")
            
            # Generate executive summary
            executive_summary = self.curator.generate_executive_summary(research_brief)
            
            return {
                'research_brief': research_brief,
                'executive_summary': executive_summary,
                'relevance_scores': relevance_scores,
                'phase_status': 'completed'
            }
            
        except Exception as e:
            logger.error(f"Phase 4 error: {e}")
            return {
                'research_brief': None,
                'executive_summary': None,
                'relevance_scores': [],
                'phase_status': 'error',
                'error_message': str(e)
            }
    
    def _generate_research_id(self) -> str:
        """Generate unique research session ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"research_{timestamp}_{hash(str(datetime.now())) % 10000}"
    
    def _summarize_search_results(self, search_results: Dict) -> Dict:
        """Summarize search results for detailed findings"""
        return {
            'total_sources_searched': len(Config.LEGAL_SOURCES) + len(Config.ACADEMIC_SOURCES),
            'successful_searches': search_results.get('search_metadata', {}).get('successful_searches', 0),
            'failed_searches': search_results.get('search_metadata', {}).get('failed_searches', 0),
            'documents_found': len(search_results.get('filtered_documents', [])),
            'source_breakdown': self._analyze_source_distribution(search_results.get('filtered_documents', []))
        }
    
    def _analyze_source_distribution(self, documents: List[Dict]) -> Dict:
        """Analyze distribution of documents by source"""
        source_counts = {}
        type_counts = {}
        
        for doc in documents:
            source = doc.get('source_name', 'Unknown')
            doc_type = doc.get('type', 'Unknown')
            
            source_counts[source] = source_counts.get(source, 0) + 1
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        
        return {
            'by_source': source_counts,
            'by_type': type_counts
        }
    
    def _calculate_processing_time(self, research_id: str) -> str:
        """Calculate processing time (placeholder - would need start time tracking)"""
        return "Processing time calculation not implemented"
    
    def _calculate_confidence_score(self, final_results: Dict) -> float:
        """Calculate confidence score for research results"""
        try:
            research_brief = final_results.get('research_brief', {})
            
            # Factors for confidence calculation
            quality_assessment = research_brief.get('research_recommendations', {}).get('quality_assessment', {})
            quality_ratio = quality_assessment.get('quality_ratio', 0)
            total_sources = quality_assessment.get('total_sources', 0)
            
            # Base confidence from quality ratio
            confidence = quality_ratio * 0.6
            
            # Boost for sufficient sources
            if total_sources >= 10:
                confidence += 0.2
            elif total_sources >= 5:
                confidence += 0.1
            
            # Boost for diverse source types
            supporting_materials = research_brief.get('supporting_materials', {})
            diverse_categories = len([cat for cat, docs in supporting_materials.items() if docs])
            
            if diverse_categories >= 4:
                confidence += 0.2
            elif diverse_categories >= 2:
                confidence += 0.1
            
            return min(confidence, 1.0)  # Cap at 1.0
            
        except Exception as e:
            logger.warning(f"Error calculating confidence score: {e}")
            return 0.5  # Default confidence
    
    def get_research_status(self, research_id: str) -> Dict:
        """Get status of ongoing research (placeholder for async tracking)"""
        return {
            'research_id': research_id,
            'status': 'completed',  # Placeholder
            'message': 'Research status tracking not implemented'
        }
    
    def list_previous_research(self, limit: int = 10) -> List[Dict]:
        """List previous research sessions"""
        try:
            import glob
            
            result_files = glob.glob(os.path.join(Config.RESULTS_DIR, "research_*.json"))
            result_files.sort(reverse=True)  # Most recent first
            
            previous_research = []
            
            for file_path in result_files[:limit]:
                try:
                    results = self.results_manager.load_results(file_path)
                    if results:
                        previous_research.append({
                            'research_id': results.get('research_id', 'Unknown'),
                            'research_angle': results.get('input_summary', {}).get('research_angle', 'Unknown'),
                            'timestamp': results.get('input_summary', {}).get('processing_timestamp', 'Unknown'),
                            'total_sources': results.get('research_metadata', {}).get('total_sources_found', 0),
                            'confidence_score': results.get('research_metadata', {}).get('confidence_score', 0),
                            'file_path': file_path
                        })
                except Exception as e:
                    logger.warning(f"Error loading {file_path}: {e}")
                    continue
            
            return previous_research
            
        except Exception as e:
            logger.error(f"Error listing previous research: {e}")
            return []

# CLI Interface
class CLIInterface:
    """Command Line Interface for the Legal Research Companion"""
    
    def __init__(self):
        self.companion = LegalResearchCompanion()
    
    def run_interactive_mode(self):
        """Run interactive CLI mode"""
        print("=" * 60)
        print("Multi-Agent Legal Research Companion for Indian Law")
        print("=" * 60)
        print()
        
        while True:
            try:
                print("\nChoose an option:")
                print("1. Conduct comprehensive research")
                print("2. Quick research query")
                print("3. View previous research")
                print("4. Help")
                print("5. Exit")
                
                choice = input("\nEnter your choice (1-5): ").strip()
                
                if choice == '1':
                    self._comprehensive_research()
                elif choice == '2':
                    self._quick_research()
                elif choice == '3':
                    self._view_previous_research()
                elif choice == '4':
                    self._show_help()
                elif choice == '5':
                    print("Thank you for using Legal Research Companion!")
                    break
                else:
                    print("Invalid choice. Please enter 1-5.")
                    
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {e}")
                print("Please try again.")
    
    def _comprehensive_research(self):
        """Handle comprehensive research workflow"""
        print("\n" + "=" * 50)
        print("COMPREHENSIVE LEGAL RESEARCH")
        print("=" * 50)
        
        # Get base material
        print("\n1. Base Material:")
        print("   You can provide either:")
        print("   - Text content of your base paper/case")
        print("   - URL of the document")
        
        base_input = input("\nEnter base material (text or URL): ").strip()
        
        if not base_input:
            print("Base material is required.")
            return
        
        # Determine if input is URL or text
        material_url = None
        if base_input.startswith(('http://', 'https://')):
            material_url = base_input
            base_material = f"Document from URL: {base_input}"
        else:
            base_material = base_input
        
        # Get research angle
        print("\n2. Research Angle:")
        print("   Describe your specific research direction or thesis")
        
        research_angle = input("Enter your research angle: ").strip()
        
        if not research_angle:
            print("Research angle is required.")
            return
        
        # Optional preferences
        print("\n3. Optional Preferences:")
        preferences = {}
        
        jurisdiction = input("Preferred jurisdiction (e.g., Supreme Court, Delhi HC) [Enter to skip]: ").strip()
        if jurisdiction:
            preferences['jurisdiction'] = jurisdiction
        
        time_period = input("Time period focus (e.g., last 5 years) [Enter to skip]: ").strip()
        if time_period:
            preferences['time_period'] = time_period
        
        # Execute research
        print("\n" + "-" * 50)
        print("STARTING RESEARCH...")
        print("-" * 50)
        print("This may take several minutes. Please wait...")
        
        try:
            results = asyncio.run(self.companion.conduct_research(
                base_material=base_material,
                research_angle=research_angle,
                material_url=material_url,
                preferences=preferences if preferences else None
            ))
            
            if results.get('error'):
                print(f"\nResearch failed: {results.get('message')}")
                return
            
            self._display_research_results(results)
            
        except Exception as e:
            print(f"\nResearch failed with error: {e}")
    
    def _quick_research(self):
        """Handle quick research workflow"""
        print("\n" + "=" * 40)
        print("QUICK LEGAL RESEARCH")
        print("=" * 40)
        
        query = input("\nEnter your legal research query: ").strip()
        
        if not query:
            print("Query is required.")
            return
        
        max_sources = input("Maximum sources to find (default 20): ").strip()
        try:
            max_sources = int(max_sources) if max_sources else 20
        except ValueError:
            max_sources = 20
        
        print(f"\nSearching for up to {max_sources} sources...")
        print("This may take a few minutes...")
        
        try:
            results = self.companion.quick_research(query, max_sources)
            
            if results.get('error'):
                print(f"\nQuick research failed: {results.get('message')}")
                return
            
            self._display_research_results(results, quick_mode=True)
            
        except Exception as e:
            print(f"\nQuick research failed: {e}")
    
    def _display_research_results(self, results: Dict, quick_mode: bool = False):
        """Display research results"""
        print("\n" + "=" * 60)
        print("RESEARCH RESULTS")
        print("=" * 60)
        
        # Basic information
        research_id = results.get('research_id', 'Unknown')
        print(f"Research ID: {research_id}")
        
        metadata = results.get('research_metadata', {})
        print(f"Total Sources Found: {metadata.get('total_sources_found', 0)}")
        print(f"High Quality Sources: {metadata.get('high_quality_sources', 0)}")
        print(f"Confidence Score: {metadata.get('confidence_score', 0):.2f}")
        
        if results.get('saved_to'):
            print(f"Results saved to: {results['saved_to']}")
        
        # Executive Summary
        exec_summary = results.get('executive_summary', {})
        if exec_summary:
            print("\n" + "-" * 40)
            print("EXECUTIVE SUMMARY")
            print("-" * 40)
            
            overview = exec_summary.get('research_overview', {})
            print(f"Research Topic: {overview.get('research_angle', 'N/A')}")
            print(f"Quality Assessment: {overview.get('quality_assessment', 'N/A')}")
            
            # Key authorities
            authorities = exec_summary.get('key_legal_authorities', [])
            if authorities:
                print(f"\nKey Legal Authorities ({len(authorities)}):")
                for i, auth in enumerate(authorities[:3], 1):
                    print(f"  {i}. {auth.get('case', 'Unknown')} - {auth.get('relevance', 'Unknown')} relevance")
        
        # Quick view of top sources
        research_brief = results.get('research_brief', {})
        supporting_materials = research_brief.get('supporting_materials', {})
        
        if supporting_materials:
            print("\n" + "-" * 40)
            print("TOP SUPPORTING MATERIALS")
            print("-" * 40)
            
            for category, materials in supporting_materials.items():
                if materials:
                    category_name = category.replace('_', ' ').title()
                    print(f"\n{category_name} ({len(materials)} sources):")
                    
                    for material in materials[:3]:  # Show top 3 per category
                        title = material.get('title', 'Untitled')[:80]
                        relevance = material.get('relevance_score', 0)
                        credibility = material.get('credibility_level', 'Unknown')
                        
                        print(f"  • {title}")
                        print(f"    Relevance: {relevance:.2f} | Credibility: {credibility}")
        
        # Recommendations
        recommendations = results.get('recommendations', {})
        if recommendations:
            print("\n" + "-" * 40)
            print("RESEARCH RECOMMENDATIONS")
            print("-" * 40)
            
            priority_actions = recommendations.get('priority_actions', [])
            if priority_actions:
                print("Priority Actions:")
                for action in priority_actions:
                    print(f"  • {action}")
            
            research_gaps = recommendations.get('research_gaps', [])
            if research_gaps:
                print("\nResearch Gaps:")
                for gap in research_gaps:
                    print(f"  • {gap}")
        
        # Ask if user wants to see full results
        if not quick_mode:
            show_full = input("\nWould you like to see the complete research brief? (y/n): ").strip().lower()
            if show_full == 'y':
                self._show_full_brief(research_brief)
    
    def _show_full_brief(self, research_brief: Dict):
        """Show complete research brief"""
        print("\n" + "=" * 60)
        print("COMPLETE RESEARCH BRIEF")
        print("=" * 60)
        
        # Key findings
        key_findings = research_brief.get('key_findings', {})
        
        print("\nKEY FINDINGS:")
        print("-" * 20)
        
        # Primary authorities
        primary_authorities = key_findings.get('primary_legal_authorities', [])
        if primary_authorities:
            print("\nPrimary Legal Authorities:")
            for i, auth in enumerate(primary_authorities, 1):
                print(f"\n{i}. {auth.get('title', 'Unknown')}")
                print(f"   Citation: {auth.get('citation', 'Not available')}")
                print(f"   Relevance: {auth.get('relevance', 'Unknown')}")
                print(f"   Key Principle: {auth.get('key_principle', 'Not available')[:200]}...")
        
        # Constitutional provisions
        constitutional = key_findings.get('constitutional_provisions', [])
        if constitutional:
            print(f"\nConstitutional Provisions ({len(constitutional)}):")
            for prov in constitutional:
                print(f"  • {prov.get('provision', 'Unknown')} - {prov.get('importance', 'Unknown')} importance")
                print(f"    Context: {prov.get('context', 'Not available')[:150]}...")
        
        # Complete bibliography
        bibliography = research_brief.get('complete_bibliography', {})
        if bibliography:
            print("\n" + "-" * 40)
            print("COMPLETE BIBLIOGRAPHY")
            print("-" * 40)
            
            for category, citations in bibliography.items():
                if citations:
                    category_name = category.replace('_', ' ').title()
                    print(f"\n{category_name}:")
                    for citation in citations[:5]:  # Show first 5 per category
                        print(f"  • {citation.get('citation', 'Unknown')}")
                        print(f"    Credibility: {citation.get('credibility_level', 'Unknown')} | "
                              f"Relevance: {citation.get('relevance_level', 'Unknown')}")
    
    def _view_previous_research(self):
        """View previous research sessions"""
        print("\n" + "=" * 40)
        print("PREVIOUS RESEARCH SESSIONS")
        print("=" * 40)
        
        previous = self.companion.list_previous_research()
        
        if not previous:
            print("\nNo previous research sessions found.")
            return
        
        print(f"\nFound {len(previous)} previous sessions:")
        
        for i, research in enumerate(previous, 1):
            print(f"\n{i}. Research ID: {research.get('research_id', 'Unknown')}")
            print(f"   Topic: {research.get('research_angle', 'Unknown')[:60]}...")
            print(f"   Date: {research.get('timestamp', 'Unknown')}")
            print(f"   Sources: {research.get('total_sources', 0)} | "
                  f"Confidence: {research.get('confidence_score', 0):.2f}")
        
        # Allow user to load a previous session
        try:
            choice = input("\nEnter number to view details (or Enter to go back): ").strip()
            if choice:
                index = int(choice) - 1
                if 0 <= index < len(previous):
                    file_path = previous[index]['file_path']
                    results = self.companion.results_manager.load_results(file_path)
                    if results:
                        self._display_research_results(results)
                else:
                    print("Invalid selection.")
        except ValueError:
            print("Invalid number.")
    
    def _show_help(self):
        """Show help information"""
        print("\n" + "=" * 50)
        print("HELP - LEGAL RESEARCH COMPANION")
        print("=" * 50)
        
        help_text = """
1. COMPREHENSIVE RESEARCH:
   - Analyzes your base legal document or paper
   - Interprets your specific research angle
   - Searches multiple legal databases and academic sources
   - Validates citations and builds authority networks
   - Provides organized research brief with recommendations

2. QUICK RESEARCH:
   - Simplified research for quick queries
   - Searches for relevant legal materials
   - Provides basic results and recommendations

3. SUPPORTED SOURCES:
   - Indian Kanoon (case law database)
   - Live Law (legal news and analysis)
   - Bar & Bench (legal commentary)
   - Google Scholar (academic papers)
   - Shodhganga (thesis repository)
   - Supreme Court and High Court websites
   - Government legal documents

4. SEARCH CAPABILITIES:
   - Constitutional law research
   - Case law and precedent analysis
   - Academic literature review  
   - Policy document analysis
   - Citation network mapping
   - Authority validation

5. OUTPUT FORMATS:
   - Executive summary
   - Detailed research brief
   - Organized bibliography
   - Research recommendations
   - Quality assessments

6. TIPS FOR BEST RESULTS:
   - Provide clear, specific research angles
   - Include relevant constitutional articles or statutes
   - Be specific about legal domains of interest
   - Use proper legal terminology where possible

For technical support, check the logs at: legal_research.log
        """
        
        print(help_text)

# Main execution
def main():
    """Main function"""
    # Check if GROQ API key is set
    if not os.environ.get("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY environment variable not set.")
        print("Please set your Groq API key:")
        print("export GROQ_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    # Check if running in interactive mode or with arguments
    if len(sys.argv) > 1:
        # Command line mode
        if sys.argv[1] == '--help' or sys.argv[1] == '-h':
            print("Legal Research Companion - Command Line Usage:")
            print("python main.py                    # Interactive mode")
            print("python main.py --help            # Show this help")
            print("python main.py --quick 'query'   # Quick research mode")
        elif sys.argv[1] == '--quick' and len(sys.argv) > 2:
            # Quick research from command line
            companion = LegalResearchCompanion()
            query = ' '.join(sys.argv[2:])
            results = companion.quick_research(query)
            print(json.dumps(results, indent=2))
        else:
            print("Invalid arguments. Use --help for usage information.")
    else:
        # Interactive mode
        cli = CLIInterface()
        cli.run_interactive_mode()

if __name__ == "__main__":
    main()