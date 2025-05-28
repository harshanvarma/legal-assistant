"""
Debug test for Legal Research Companion
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging to see what's happening
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

def test_environment():
    """Test environment setup"""
    print("=" * 50)
    print("ENVIRONMENT TEST")
    print("=" * 50)
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check API key
    api_key = os.environ.get("GROQ_API_KEY")
    if api_key:
        print(f"✅ GROQ_API_KEY is set (length: {len(api_key)})")
    else:
        print("❌ GROQ_API_KEY is not set")
        return False
    
    # Check required modules
    required_modules = ['groq', 'requests', 'beautifulsoup4', 'nltk', 'sklearn', 'networkx']
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module} is available")
        except ImportError as e:
            print(f"❌ {module} is missing: {e}")
    
    return True

def test_groq_api():
    """Test Groq API connection"""
    print("\n" + "=" * 50)
    print("GROQ API TEST")
    print("=" * 50)
    
    try:
        from groq import Groq
        
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        
        print("Sending test request to Groq...")
        response = client.chat.completions.create(
            messages=[
                {"role": "user", "content": "Say 'API test successful' if you can read this."}
            ],
            model="llama-3.3-70b-versatile",
            stream=False,
        )
        
        result = response.choices[0].message.content
        print(f"✅ Groq API Response: {result}")
        return True
        
    except Exception as e:
        print(f"❌ Groq API Error: {e}")
        return False

def test_imports():
    """Test all project imports"""
    print("\n" + "=" * 50)
    print("IMPORT TEST")
    print("=" * 50)
    
    imports_to_test = [
        ('config', 'Config'),
        ('utils', 'WebScraper'),
        ('agent1_legal_analyzer', 'LegalAnalyzer'),
        ('agent2_research_engine', 'ResearchEngine'),
        ('agent3_citation_validator', 'CitationValidator'),
        ('agent4_research_curator', 'ResearchCurator'),
        ('main', 'LegalResearchCompanion')
    ]
    
    for module_name, class_name in imports_to_test:
        try:
            module = __import__(module_name)
            class_obj = getattr(module, class_name)
            print(f"✅ {module_name}.{class_name} imported successfully")
        except Exception as e:
            print(f"❌ {module_name}.{class_name} failed: {e}")
            return False
    
    return True

def test_quick_research():
    """Test quick research functionality"""
    print("\n" + "=" * 50)
    print("QUICK RESEARCH TEST")
    print("=" * 50)
    
    try:
        from main import LegalResearchCompanion
        
        print("Creating LegalResearchCompanion instance...")
        companion = LegalResearchCompanion()
        
        print("Starting quick research test...")
        query = "Article 21 privacy rights"
        
        print(f"Query: {query}")
        print("This may take a few minutes...")
        
        results = companion.quick_research(query, max_sources=5)
        
        if results.get('error'):
            print(f"❌ Research failed: {results.get('message')}")
            return False
        else:
            print("✅ Research completed successfully!")
            print(f"Research ID: {results.get('research_id')}")
            print(f"Total sources: {results.get('research_metadata', {}).get('total_sources_found', 0)}")
            return True
            
    except Exception as e:
        print(f"❌ Quick research test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🔍 Legal Research Companion - Debug Test")
    print("This will help identify why the system isn't working.\n")
    
    # Test environment
    if not test_environment():
        print("\n❌ Environment test failed. Fix the issues above and try again.")
        return
    
    # Test API
    if not test_groq_api():
        print("\n❌ API test failed. Check your API key and internet connection.")
        return
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Check if all files are present and correct.")
        return
    
    # Test quick research
    print("\n🚀 All basic tests passed! Now testing quick research...")
    if test_quick_research():
        print("\n🎉 All tests passed! The system should be working correctly.")
    else:
        print("\n❌ Research test failed. Check the error messages above.")

if __name__ == "__main__":
    main()