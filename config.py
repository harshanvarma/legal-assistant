import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    
    # Search Configuration
    MAX_CONCURRENT_SEARCHES = 5
    REQUEST_DELAY = 1  # seconds between requests
    MAX_RETRIES = 3
    TIMEOUT = 30
    
    # Indian Legal Sources
    LEGAL_SOURCES = {
        'indian_kanoon': 'https://indiankanoon.org/search/?formInput=',
        'manupatra': 'https://www.manupatrafast.com/',
        'scc_online': 'https://www.scconline.com/',
        'live_law': 'https://www.livelaw.in/search?q=',
        'supreme_court': 'https://main.sci.gov.in/',
        'bar_bench': 'https://www.barandbench.com/search?q=',
        'legal_service_india': 'http://www.legalserviceindia.com/search/searchresult.php?search=',
        'lawctopus': 'https://www.lawctopus.com/?s='
    }
    
    # Academic Sources
    ACADEMIC_SOURCES = {
        'shodhganga': 'https://shodhganga.inflibnet.ac.in/simple-search?query=',
        'jstor': 'https://www.jstor.org/action/doBasicSearch?Query=',
        'google_scholar': 'https://scholar.google.com/scholar?q=',
        'ssrn': 'https://www.ssrn.com/index.cfm/en/janda/job-market/job-market-search/?search_type=basic&search_query='
    }
    
    # Government Sources
    GOVT_SOURCES = {
        'legislative': 'https://legislative.gov.in/',
        'law_commission': 'https://lawcommissionofindia.nic.in/',
        'delhi_hc': 'https://www.delhihighcourt.nic.in/',
        'bombay_hc': 'https://bombayhighcourt.nic.in/',
        'calcutta_hc': 'https://www.calcuttahighcourt.gov.in/',
        'madras_hc': 'https://www.hcmadras.tn.nic.in/'
    }
    
    # Indian Legal Keywords and Patterns
    LEGAL_KEYWORDS = {
        'constitutional': ['Article', 'Constitution', 'Fundamental Rights', 'Directive Principles', 'Amendment'],
        'criminal': ['IPC', 'CrPC', 'POCSO', 'Dowry', 'Domestic Violence', 'Section'],
        'civil': ['CPC', 'Contract Act', 'Transfer of Property', 'Partnership', 'Companies Act'],
        'family': ['Hindu Marriage Act', 'Muslim Personal Law', 'Guardianship', 'Maintenance', 'Divorce'],
        'commercial': ['Arbitration', 'SEBI', 'Competition Act', 'Insolvency', 'Banking'],
        'administrative': ['Service Law', 'Writ Petition', 'Mandamus', 'Certiorari', 'Prohibition'],
        'taxation': ['Income Tax', 'GST', 'Service Tax', 'Customs', 'ITAT'],
        'labor': ['Industrial Disputes', 'Minimum Wages', 'Provident Fund', 'Gratuity', 'Bonus']
    }
    
    # Indian Courts Hierarchy
    COURT_HIERARCHY = {
        'supreme_court': {'weight': 10, 'abbreviation': 'SC'},
        'high_court': {'weight': 8, 'abbreviation': 'HC'},
        'district_court': {'weight': 6, 'abbreviation': 'DC'},
        'tribunal': {'weight': 7, 'abbreviation': 'Tribunal'},
        'commission': {'weight': 5, 'abbreviation': 'Commission'}
    }
    
    # Citation Patterns
    CITATION_PATTERNS = {
        'supreme_court': r'\(\d{4}\)\s*\d+\s*SCC\s*\d+',
        'high_court': r'\(\d{4}\)\s*\d+\s*[A-Z]+\s*\d+',
        'air': r'AIR\s*\d{4}\s*[A-Z]+\s*\d+',
        'indian_kanoon': r'https://indiankanoon\.org/doc/\d+/'
    }
    
    # Relevance Scoring Weights
    SCORING_WEIGHTS = {
        'keyword_match': 0.3,
        'court_authority': 0.25,
        'citation_frequency': 0.15,
        'temporal_relevance': 0.15,
        'semantic_similarity': 0.15
    }
    
    # File Paths
    DATA_DIR = 'data'
    CACHE_DIR = 'cache'
    RESULTS_DIR = 'results'
    
    # Create directories if they don't exist
    for directory in [DATA_DIR, CACHE_DIR, RESULTS_DIR]:
        os.makedirs(directory, exist_ok=True)