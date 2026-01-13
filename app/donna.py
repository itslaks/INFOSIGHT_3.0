import os
import re
import json
import random
import warnings
import tempfile
import hashlib
import subprocess
import time
import platform
import signal
import atexit
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, quote_plus, urljoin
from collections import defaultdict, Counter
import sqlite3
from functools import lru_cache
import base64

import requests
from bs4 import BeautifulSoup
from flask import Flask, Blueprint, request, jsonify, send_file, render_template, Response, g
from flask_cors import CORS
from utils.security import rate_limit_strict, validate_request, InputValidator

# Configure logging
logger = logging.getLogger(__name__)

# Optional: reportlab for PDF
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image as RLImage, KeepTogether
    from reportlab.lib.colors import HexColor, colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    from reportlab.pdfgen import canvas
    from reportlab.graphics.shapes import Drawing, Rect, String
    from reportlab.graphics import renderPDF
    REPORTLAB_AVAILABLE = True
except:
    REPORTLAB_AVAILABLE = False

# LangChain / LLM libs
try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_ollama import ChatOllama
    LLM_LIBS_AVAILABLE = True
except:
    LLM_LIBS_AVAILABLE = False

# Use centralized LLM router
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.llm_router import generate_text
    LLM_ROUTER_AVAILABLE = True
    logger.info("✓ LLM router available for DONNA AI")
except ImportError as e:
    LLM_ROUTER_AVAILABLE = False
    logger.warning(f"⚠️ LLM router not available: {e}")
    def generate_text(*args, **kwargs):
        return {"response": "", "model": "none", "source": "none"}

# Local LLM fallback utility
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.local_llm_utils import generate_with_ollama, check_ollama_available
    from utils.llm_logger import log_llm_status, log_llm_request, log_llm_success, log_llm_error, log_llm_fallback, log_processing_step
    LOCAL_LLM_AVAILABLE = True
except ImportError as e:
    LOCAL_LLM_AVAILABLE = False
    logger.warning(f"⚠️ Local LLM utilities not available for DONNA AI: {e}")
    # Create dummy functions
    def log_llm_status(*args, **kwargs): return (False, False)
    def log_llm_request(*args, **kwargs): pass
    def log_llm_success(*args, **kwargs): pass
    def log_llm_error(*args, **kwargs): pass
    def log_llm_fallback(*args, **kwargs): pass
    def log_processing_step(*args, **kwargs): pass

warnings.filterwarnings("ignore")

# ==================== CONFIGURATION ====================
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import Config
    # Removed - using centralized router
    OLLAMA_BASE_URL = Config.OLLAMA_BASE_URL
    OLLAMA_MODEL = Config.OLLAMA_MODEL
except (ImportError, AttributeError):
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    # Removed - using centralized router
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:3b-instruct")

# Groq initialization removed - using centralized router
GROQ_CONFIGURED = LLM_ROUTER_AVAILABLE

# Rate limiting
LAST_API_CALL = {}
MIN_API_INTERVAL = 0.5

# Advanced User Agents
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15",
]

# Enhanced Search Engines (Clearnet + Tor Gateways)
# Categorized by web layer
SURFACE_WEB_ENGINES = [
    "https://www.google.com/search?q={query}",
    "https://www.bing.com/search?q={query}",
    "https://duckduckgo.com/html/?q={query}",
    "https://www.startpage.com/do/search?q={query}",
    "https://yandex.com/search/?text={query}",
]

DEEP_WEB_ENGINES = [
    "https://www.google.com/search?q={query}+filetype:pdf",
    "https://www.google.com/search?q={query}+site:pastebin.com",
    "https://www.google.com/search?q={query}+site:github.com",
    "https://www.google.com/search?q={query}+site:stackoverflow.com",
    "https://www.google.com/search?q={query}+inurl:database",
    "https://www.google.com/search?q={query}+inurl:admin",
]

DARK_WEB_ENGINES = [
    "https://ahmia.fi/search/?q={query}",
    "https://tor.link/search?q={query}",
    "https://onion.live/search?q={query}",
    "https://duckduckgo.com/html/?q={query}+site:.onion",
    "https://www.startpage.com/do/search?q={query}+.onion",
    "https://www.bing.com/search?q={query}+onion",
]

# Combined for backward compatibility
SEARCH_ENGINES = SURFACE_WEB_ENGINES + DEEP_WEB_ENGINES + DARK_WEB_ENGINES

# Cache & Database
REQUEST_CACHE = {}
MAX_CACHE_SIZE = 10000
DB_PATH = os.path.join(tempfile.gettempdir(), "donna_osint.db")

# Tor
TOR_AVAILABLE = False

# Investigation History
INVESTIGATION_HISTORY = []

# ==================== DATABASE SETUP ====================
def init_database():
    """Initialize comprehensive SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Results table
    c.execute('''CREATE TABLE IF NOT EXISTS results
                (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                 investigation_id TEXT,
                 query TEXT, 
                 url TEXT UNIQUE, 
                 title TEXT, 
                 content TEXT, 
                 source TEXT, 
                 scraped_date TIMESTAMP, 
                 relevance_score REAL,
                 metadata TEXT)''')
    
    # Artifacts table
    c.execute('''CREATE TABLE IF NOT EXISTS artifacts
                (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                 investigation_id TEXT,
                 result_id INTEGER, 
                 artifact_type TEXT, 
                 value TEXT, 
                 confidence REAL,
                 context TEXT,
                 discovered_at TIMESTAMP,
                 FOREIGN KEY(result_id) REFERENCES results(id))''')
    
    # Investigations table
    c.execute('''CREATE TABLE IF NOT EXISTS investigations
                (id TEXT PRIMARY KEY, 
                 query TEXT, 
                 refined_query TEXT, 
                 start_time TIMESTAMP, 
                 end_time TIMESTAMP, 
                 total_results INTEGER, 
                 total_scraped INTEGER,
                 artifacts_found INTEGER, 
                 threat_level TEXT,
                 threat_score REAL,
                 status TEXT,
                 report TEXT)''')
    
    # Analytics table
    c.execute('''CREATE TABLE IF NOT EXISTS analytics
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 event_type TEXT,
                 event_data TEXT,
                 timestamp TIMESTAMP)''')
    
    conn.commit()
    conn.close()
    logger.info("[DB] Database initialized with 4 tables")

def save_investigation(inv_id: str, data: Dict):
    """Save complete investigation to database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''INSERT OR REPLACE INTO investigations 
                    (id, query, refined_query, start_time, end_time, total_results, 
                     total_scraped, artifacts_found, threat_level, threat_score, status, report)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (inv_id, data.get('query'), data.get('refined_query'),
                   data.get('start_time'), data.get('end_time'),
                   data.get('total_results', 0), data.get('total_scraped', 0),
                   data.get('artifacts_found', 0), data.get('threat_level'),
                   data.get('threat_score'), data.get('status'), data.get('report')))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"[DB] Error saving investigation: {e}")

def get_investigation_history(limit: int = 50) -> List[Dict]:
    """Retrieve investigation history"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''SELECT id, query, threat_level, artifacts_found, start_time, status
                    FROM investigations ORDER BY start_time DESC LIMIT ?''', (limit,))
        rows = c.fetchall()
        conn.close()
        
        return [{
            'id': r[0], 'query': r[1], 'threat_level': r[2],
            'artifacts': r[3], 'timestamp': r[4], 'status': r[5]
        } for r in rows]
    except:
        return []

# ==================== TOR MANAGEMENT ====================
def check_tor():
    """Check Tor connectivity"""
    try:
        response = requests.get(
            "https://check.torproject.org/api/ip",
            proxies={"http": "socks5h://127.0.0.1:9050", "https": "socks5h://127.0.0.1:9050"},
            timeout=10, verify=False
        )
        return response.json().get('IsTor', False)
    except:
        return False

def get_tor_proxies():
    return {"http": "socks5h://127.0.0.1:9050", "https": "socks5h://127.0.0.1:9050"}

def start_tor_service():
    global TOR_AVAILABLE
    logger.info("\n[TOR] Checking Tor connectivity...")
    if check_tor():
        logger.info("[TOR] ✓ Tor SOCKS proxy active on port 9050")
        TOR_AVAILABLE = True
        return True
    logger.warning("[TOR] ⚠ Tor not available - using clearnet gateway mode")
    TOR_AVAILABLE = False
    return False

# ==================== SECURITY & UTILITIES ====================
def get_headers():
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

def sanitize_url(url: str) -> str:
    try:
        parsed = urlparse(url)
        return url if parsed.scheme in ['http', 'https'] else ""
    except:
        return ""

def hash_url(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()

def generate_investigation_id(query: str) -> str:
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
    return f"INV-{timestamp}-{query_hash}"

# ==================== OLLAMA LLM ====================
def check_ollama_connection(url: str) -> bool:
    try:
        return requests.get(f"{url}/api/tags", timeout=5).status_code == 200
    except:
        return False

def get_ollama_url() -> str:
    for url in [OLLAMA_BASE_URL, "http://localhost:11434", "http://127.0.0.1:11434"]:
        if check_ollama_connection(url):
            return url
    raise ConnectionError("Ollama not running")

def rate_limit_check(key: str):
    global LAST_API_CALL
    if key in LAST_API_CALL:
        elapsed = time.time() - LAST_API_CALL[key]
        if elapsed < MIN_API_INTERVAL:
            time.sleep(MIN_API_INTERVAL - elapsed)
    LAST_API_CALL[key] = time.time()

def get_ollama_llm():
    if not LLM_LIBS_AVAILABLE:
        raise RuntimeError("LangChain not available")
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=get_ollama_url(),
        temperature=0.3,
        num_ctx=8192,
        request_timeout=120
    )

# ==================== ADVANCED QUERY PROCESSING ====================
def refine_query_advanced(user_input: str) -> Dict:
    stop_words = {'the', 'a', 'an', 'in', 'on', 'at', 'for', 'to', 'of', 'and', 'or', 'how', 'what', 'find', 'search'}
    words = user_input.lower().split()
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    
    # Generate variations
    variations = []
    if len(keywords) > 2:
        variations.append(' '.join(keywords[:3]))
        variations.append(' '.join(keywords[-3:]))
    variations.append(' '.join(keywords[:5]))
    
    return {
        "original": user_input,
        "refined": ' '.join(keywords[:6]) if keywords else user_input,
        "keywords": keywords[:10],
        "variations": list(set(variations))[:3],
        "length": len(keywords)
    }

def refine_query_with_ollama(llm, user_input: str) -> Dict:
    try:
        rate_limit_check("refine_query")
        
        system = """You are a dark web search optimizer. Analyze the query and provide:
1. Refined keywords (3-6 words max)
2. Three search variations
3. Brief search strategy

JSON format only:
{"refined": "keywords here", "variations": ["var1", "var2", "var3"], "strategy": "strategy description"}"""
        
        prompt = ChatPromptTemplate([("system", system), ("user", "Query: {query}")])
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"query": user_input})
        
        data = json.loads(response.strip())
        return {
            "original": user_input,
            "refined": data.get("refined", user_input),
            "variations": data.get("variations", []),
            "strategy": data.get("strategy", ""),
            "keywords": data.get("refined", user_input).split()
        }
    except Exception as e:
        logger.warning(f"[OLLAMA] Refinement fallback: {e}")
        return refine_query_advanced(user_input)

# ==================== ADVANCED SEARCH ====================
def fetch_search_results_advanced(endpoint: str, query: str) -> List[Dict]:
    url = endpoint.format(query=quote_plus(query))
    cache_key = hash_url(url)
    
    if cache_key in REQUEST_CACHE:
        return REQUEST_CACHE[cache_key]
    
    use_tor = '.onion' in endpoint and TOR_AVAILABLE
    proxies = get_tor_proxies() if use_tor else None
    
    for timeout in [25, 20, 15]:
        try:
            response = requests.get(
                url,
                headers=get_headers(),
                proxies=proxies,
                timeout=timeout,
                verify=not use_tor,
                allow_redirects=True
            )
            
            if response.status_code == 200 and len(response.text) > 500:
                soup = BeautifulSoup(response.text, "html.parser")
                links = []
                seen = set()
                
                # Extract all links
                for a in soup.find_all('a', href=True):
                    href = a.get('href', '')
                    title = a.get_text(strip=True)[:200]
                    
                    # Handle relative URLs
                    if href.startswith('/'):
                        parsed = urlparse(endpoint)
                        href = f"{parsed.scheme}://{parsed.netloc}{href}"
                    
                    # Determine web layer type
                    def classify_url(url_str):
                        """Classify URL as dark, deep, or surface web"""
                        url_lower = url_str.lower()
                        
                        # Dark web (.onion)
                        if '.onion' in url_lower:
                            return "dark"
                        
                        # Deep web indicators
                        deep_indicators = [
                            '/login', '/admin', '/database', '/api/', '/private',
                            'pastebin.com', 'github.com/gist', 'hastebin.com',
                            'filetype:pdf', 'inurl:database', 'inurl:admin',
                            'stackoverflow.com/questions', 'reddit.com/r/',
                            'facebook.com/groups', 'linkedin.com/groups'
                        ]
                        if any(indicator in url_lower for indicator in deep_indicators):
                            return "deep"
                        
                        # Surface web (default)
                        return "surface"
                    
                    # .onion pattern detection (Dark Web)
                    onion_pattern = r'https?://[a-z2-7]{16,56}\.onion[^\s"\'<>]*'
                    onion_matches = re.findall(onion_pattern, href + ' ' + response.text[:20000])
                    
                    for link in onion_matches:
                        clean = sanitize_url(link)
                        if clean and clean not in seen:
                            seen.add(clean)
                            links.append({
                                "title": title or clean,
                                "link": clean,
                                "source": urlparse(endpoint).netloc,
                                "type": "dark",
                                "web_layer": "dark"
                            })
                    
                    # Regular links with classification
                    if href and ('http://' in href or 'https://' in href):
                        clean = sanitize_url(href)
                        if clean and clean not in seen and len(links) < 100:
                            seen.add(clean)
                            web_layer = classify_url(clean)
                            links.append({
                                "title": title or clean,
                                "link": clean,
                                "source": urlparse(endpoint).netloc,
                                "type": "dark" if web_layer == "dark" else ("deep" if web_layer == "deep" else "clearnet"),
                                "web_layer": web_layer
                            })
                
                if len(REQUEST_CACHE) < MAX_CACHE_SIZE:
                    REQUEST_CACHE[cache_key] = links[:80]
                
                return links[:80]
        except:
            continue
    
    return []

def get_search_results_distributed(query: str, variations: List[str], max_workers: int = 8) -> List[Dict]:
    """Enhanced distributed search with web layer categorization"""
    all_results = []
    seen = set()
    
    queries = [query] + variations[:3]
    
    # Search across all web layers
    all_engines = SURFACE_WEB_ENGINES + DEEP_WEB_ENGINES + DARK_WEB_ENGINES
    logger.info(f"[SEARCH] Searching {len(all_engines)} engines ({len(SURFACE_WEB_ENGINES)} surface, {len(DEEP_WEB_ENGINES)} deep, {len(DARK_WEB_ENGINES)} dark) with {len(queries)} query variations...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for q in queries:
            # Surface web search
            for engine in SURFACE_WEB_ENGINES:
                futures.append(executor.submit(fetch_search_results_advanced, engine, q))
            # Deep web search
            for engine in DEEP_WEB_ENGINES:
                futures.append(executor.submit(fetch_search_results_advanced, engine, q))
            # Dark web search
            for engine in DARK_WEB_ENGINES:
                futures.append(executor.submit(fetch_search_results_advanced, engine, q))
        
        for future in as_completed(futures):
            try:
                results = future.result()
                for res in results:
                    link = res.get('link')
                    if link and link not in seen:
                        seen.add(link)
                        # Ensure web_layer is set
                        if 'web_layer' not in res:
                            if '.onion' in link.lower():
                                res['web_layer'] = 'dark'
                            else:
                                deep_indicators = ['/login', '/admin', '/database', '/api/', 'pastebin', 'github.com/gist']
                                res['web_layer'] = 'deep' if any(ind in link.lower() for ind in deep_indicators) else 'surface'
                        all_results.append(res)
            except Exception as e:
                logger.debug(f"[SEARCH] Engine error: {str(e)[:100]}")
                pass
    
    # Count by web layer
    layer_counts = {"dark": 0, "deep": 0, "surface": 0}
    for res in all_results:
        layer = res.get('web_layer', 'surface')
        layer_counts[layer] = layer_counts.get(layer, 0) + 1
    
    logger.info(f"[SEARCH] Found {len(all_results)} unique results (Dark: {layer_counts['dark']}, Deep: {layer_counts['deep']}, Surface: {layer_counts['surface']})")
    return all_results

# ==================== ADVANCED SCRAPING ====================
def extract_text_advanced(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    
    # Remove unwanted elements
    for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 
                     'iframe', 'noscript', 'meta', 'link', 'button']):
        tag.decompose()
    
    # Extract main content
    main_content = soup.find(['main', 'article', 'div.content', 'div.post'])
    text = main_content.get_text(separator=' ', strip=True) if main_content else soup.get_text(separator=' ', strip=True)
    
    # Clean text
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    return text[:15000]

def scrape_url_advanced(url_data: Dict, inv_id: str) -> Tuple[str, str, Dict]:
    url = url_data.get('link')
    cache_key = hash_url(url)
    
    if cache_key in REQUEST_CACHE and isinstance(REQUEST_CACHE[cache_key], dict):
        cached = REQUEST_CACHE[cache_key]
        if cached.get('content'):
            return url, cached['content'], cached.get('metadata', {})
    
    # Determine web layer and scraping method
    web_layer = url_data.get('web_layer', 'surface')
    if '.onion' in url.lower():
        web_layer = 'dark'
    elif not web_layer or web_layer == 'clearnet':
        # Re-classify if needed
        deep_indicators = ['/login', '/admin', '/database', '/api/', 'pastebin', 'github.com/gist']
        if any(ind in url.lower() for ind in deep_indicators):
            web_layer = 'deep'
        else:
            web_layer = 'surface'
    
    use_tor = (web_layer == 'dark' and TOR_AVAILABLE)
    proxies = get_tor_proxies() if use_tor else None
    
    metadata = {
        "status_code": None,
        "content_type": None,
        "content_length": 0,
        "load_time": 0,
        "success": False,
        "web_layer": web_layer,
        "scrape_type": "dark" if web_layer == "dark" else ("deep" if web_layer == "deep" else "surface")
    }
    
    for timeout in [25, 20, 15]:
        try:
            start = time.time()
            response = requests.get(
                url,
                headers=get_headers(),
                proxies=proxies,
                timeout=timeout,
                verify=not use_tor,
                allow_redirects=True
            )
            
            metadata['load_time'] = time.time() - start
            metadata['status_code'] = response.status_code
            metadata['content_type'] = response.headers.get('content-type', '')
            metadata['content_length'] = len(response.text)
            
            if response.status_code == 200 and len(response.text) > 200:
                text = extract_text_advanced(response.text)
                
                if len(text) > 200:
                    metadata['success'] = True
                    
                    cache_entry = {
                        'content': text,
                        'metadata': metadata,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    if len(REQUEST_CACHE) < MAX_CACHE_SIZE:
                        REQUEST_CACHE[cache_key] = cache_entry
                    
                    logger.debug(f"[SCRAPE] ✓ {urlparse(url).netloc} ({metadata['content_length']} bytes)")
                    return url, text, metadata
        except:
            continue
    
    return url, "", metadata

def scrape_batch(urls: List[Dict], inv_id: str, max_workers: int = 8) -> Dict[str, Dict]:
    """Enhanced batch scraping with web layer categorization"""
    results = {}
    failed = []
    scrape_stats = {
        "dark": {"total": 0, "success": 0, "failed": 0},
        "deep": {"total": 0, "success": 0, "failed": 0},
        "surface": {"total": 0, "success": 0, "failed": 0}
    }
    
    logger.info(f"[SCRAPE] Starting batch scrape of {len(urls)} URLs...")
    
    # Categorize URLs by web layer
    for url_data in urls:
        url = url_data.get('link', '')
        web_layer = url_data.get('web_layer', 'surface')
        if '.onion' in url.lower():
            web_layer = 'dark'
        elif not web_layer or web_layer == 'clearnet':
            deep_indicators = ['/login', '/admin', '/database', '/api/', 'pastebin', 'github.com/gist']
            web_layer = 'deep' if any(ind in url.lower() for ind in deep_indicators) else 'surface'
        scrape_stats[web_layer]["total"] += 1
    
    logger.info(f"[SCRAPE] Breakdown: {scrape_stats['dark']['total']} dark, {scrape_stats['deep']['total']} deep, {scrape_stats['surface']['total']} surface")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(scrape_url_advanced, url, inv_id): url for url in urls}
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            try:
                url, content, metadata = future.result()
                web_layer = metadata.get('web_layer', 'surface')
                
                if content and len(content) > 200:
                    results[url] = {
                        "content": content,
                        "metadata": metadata,
                        "web_layer": web_layer,
                        "scrape_type": metadata.get('scrape_type', web_layer)
                    }
                    scrape_stats[web_layer]["success"] += 1
                    logger.debug(f"[SCRAPE] [{completed}/{len(urls)}] {web_layer.upper()} Success ({len(results)} total)")
                else:
                    failed.append(url)
                    scrape_stats[web_layer]["failed"] += 1
            except Exception as e:
                url_data = futures[future]
                failed.append(url_data.get('link', 'unknown'))
                logger.warning(f"[SCRAPE] Failed: {str(e)[:100]}")
    
    logger.info(f"[SCRAPE] Completed: {len(results)} succeeded, {len(failed)} failed")
    logger.info(f"[SCRAPE] Stats - Dark: {scrape_stats['dark']['success']}/{scrape_stats['dark']['total']}, "
                f"Deep: {scrape_stats['deep']['success']}/{scrape_stats['deep']['total']}, "
                f"Surface: {scrape_stats['surface']['success']}/{scrape_stats['surface']['total']}")
    
    return results

# ==================== COMPREHENSIVE ARTIFACT EXTRACTION ====================
def extract_artifacts_comprehensive(content: Dict[str, str]) -> Dict:
    artifacts = {
        "emails": [],
        "domains": [],
        "ip_addresses": [],
        "urls": [],
        "phone_numbers": [],
        "crypto_addresses": {
            "bitcoin": [],
            "ethereum": [],
            "monero": [],
            "litecoin": [],
            "dogecoin": [],
            "ripple": []
        },
        "hashes": {
            "md5": [],
            "sha1": [],
            "sha256": [],
            "sha512": []
        },
        "social_media": {
            "twitter": [],
            "telegram": [],
            "discord": [],
            "reddit": []
        },
        "passwords": [],
        "api_keys": [],
        "private_keys": []
    }
    
    text = " ".join(content.values())
    
    # Emails with enhanced detection
    emails = set(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
    artifacts["emails"] = [{"value": e, "confidence": 0.95, "context": "email"} 
                           for e in list(emails)[:50]]
    
    # Domains
    domains = set(re.findall(r'\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,}\b', text.lower()))
    artifacts["domains"] = [{"value": d, "confidence": 0.9, "context": "domain"} 
                           for d in list(domains)[:50] if '.' in d and len(d) > 3]
    
    # Bitcoin addresses
    btc = set(re.findall(r'\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b', text))
    artifacts["crypto_addresses"]["bitcoin"] = [{"value": b, "confidence": 0.98} for b in list(btc)[:30]]
    
    # Ethereum addresses
    eth = set(re.findall(r'\b0x[a-fA-F0-9]{40}\b', text))
    artifacts["crypto_addresses"]["ethereum"] = [{"value": e, "confidence": 0.98} for e in list(eth)[:30]]
    
    # Monero addresses
    xmr = set(re.findall(r'\b4[0-9AB][1-9A-HJ-NP-Za-km-z]{93}\b', text))
    artifacts["crypto_addresses"]["monero"] = [{"value": x, "confidence": 0.97} for x in list(xmr)[:20]]
    
    # Litecoin
    ltc = set(re.findall(r'\b[LM3][a-km-zA-HJ-NP-Z1-9]{26,33}\b', text))
    artifacts["crypto_addresses"]["litecoin"] = [{"value": l, "confidence": 0.95} for l in list(ltc)[:20]]
    
    # Dogecoin
    doge = set(re.findall(r'\bD{1}[5-9A-HJ-NP-U]{1}[1-9A-HJ-NP-Za-km-z]{32}\b', text))
    artifacts["crypto_addresses"]["dogecoin"] = [{"value": d, "confidence": 0.94} for d in list(doge)[:15]]
    
    # IP Addresses
    ips = set(re.findall(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', text))
    valid_ips = [ip for ip in ips if all(0 <= int(octet) <= 255 for octet in ip.split('.'))]
    artifacts["ip_addresses"] = [{"value": ip, "confidence": 0.99} for ip in valid_ips[:50]]
    
    # URLs
    urls = set(re.findall(r'https?://[^\s<>"]+|www\.[^\s<>"]+', text))
    artifacts["urls"] = [{"value": u[:300], "confidence": 0.95} for u in list(urls)[:40]]
    
    # Phone numbers
    phones = set(re.findall(r'\+?1?\d{9,15}', text))
    artifacts["phone_numbers"] = [{"value": p, "confidence": 0.80} for p in list(phones)[:25]]
    
    # Hashes
    artifacts["hashes"]["md5"] = [{"value": h, "confidence": 0.90} 
                                  for h in list(set(re.findall(r'\b[a-fA-F0-9]{32}\b', text)))[:25]]
    artifacts["hashes"]["sha1"] = [{"value": h, "confidence": 0.92} 
                                   for h in list(set(re.findall(r'\b[a-fA-F0-9]{40}\b', text)))[:25]]
    artifacts["hashes"]["sha256"] = [{"value": h, "confidence": 0.95} 
                                     for h in list(set(re.findall(r'\b[a-fA-F0-9]{64}\b', text)))[:25]]
    artifacts["hashes"]["sha512"] = [{"value": h, "confidence": 0.96} 
                                     for h in list(set(re.findall(r'\b[a-fA-F0-9]{128}\b', text)))[:15]]
    
    # Social Media
    twitter = set(re.findall(r'@[A-Za-z0-9_]{1,15}', text))
    artifacts["social_media"]["twitter"] = [{"value": t, "confidence": 0.85} for t in list(twitter)[:20]]
    
    telegram = set(re.findall(r't\.me/[A-Za-z0-9_]{5,32}', text))
    artifacts["social_media"]["telegram"] = [{"value": t, "confidence": 0.90} for t in list(telegram)[:15]]
    
    # API Keys (generic pattern)
    api_keys = set(re.findall(r'[A-Za-z0-9_-]{20,}', text))
    suspected_keys = [k for k in api_keys if any(x in k.lower() for x in ['key', 'token', 'secret'])]
    artifacts["api_keys"] = [{"value": k[:50], "confidence": 0.70} for k in suspected_keys[:10]]
    
    # Private keys (PEM format detection)
    private_keys = re.findall(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----', text)
    if private_keys:
        artifacts["private_keys"] = [{"value": "PRIVATE KEY DETECTED", "confidence": 0.99, 
                                     "count": len(private_keys)}]
    
    return artifacts

def consolidate_artifacts(all_artifacts: List[Dict]) -> Dict:
    consolidated = {
        "emails": [],
        "domains": [],
        "ip_addresses": [],
        "urls": [],
        "phone_numbers": [],
        "crypto_addresses": {},
        "hashes": {},
        "social_media": {},
        "api_keys": [],
        "private_keys": [],
        "summary": {}
    }
    
    seen = defaultdict(set)
    
    for artifacts in all_artifacts:
        # Emails
        for email_obj in artifacts.get('emails', []):
            email = email_obj['value']
            if email not in seen["emails"]:
                seen["emails"].add(email)
                consolidated['emails'].append(email_obj)
        
        # Domains
        for domain_obj in artifacts.get('domains', []):
            domain = domain_obj['value']
            if domain not in seen["domains"]:
                seen["domains"].add(domain)
                consolidated['domains'].append(domain_obj)
        
        # IPs
        for ip_obj in artifacts.get('ip_addresses', []):
            ip = ip_obj['value']
            if ip not in seen["ips"]:
                seen["ips"].add(ip)
                consolidated['ip_addresses'].append(ip_obj)
        
        # Crypto
        for crypto_type in ['bitcoin', 'ethereum', 'monero', 'litecoin', 'dogecoin', 'ripple']:
            if crypto_type not in consolidated['crypto_addresses']:
                consolidated['crypto_addresses'][crypto_type] = []
            for addr_obj in artifacts.get('crypto_addresses', {}).get(crypto_type, []):
                if addr_obj['value'] not in seen[f"crypto_{crypto_type}"]:
                    seen[f"crypto_{crypto_type}"].add(addr_obj['value'])
                    consolidated['crypto_addresses'][crypto_type].append(addr_obj)
        
        # Hashes
        for hash_type in ['md5', 'sha1', 'sha256', 'sha512']:
            if hash_type not in consolidated['hashes']:
                consolidated['hashes'][hash_type] = []
            for hash_obj in artifacts.get('hashes', {}).get(hash_type, []):
                if hash_obj['value'] not in seen[f"hash_{hash_type}"]:
                    seen[f"hash_{hash_type}"].add(hash_obj['value'])
                    consolidated['hashes'][hash_type].append(hash_obj)
        
        # Social Media
        for social_type in ['twitter', 'telegram', 'discord', 'reddit']:
            if social_type not in consolidated['social_media']:
                consolidated['social_media'][social_type] = []
            for social_obj in artifacts.get('social_media', {}).get(social_type, []):
                if social_obj['value'] not in seen[f"social_{social_type}"]:
                    seen[f"social_{social_type}"].add(social_obj['value'])
                    consolidated['social_media'][social_type].append(social_obj)
        
        # URLs
        for url_obj in artifacts.get('urls', []):
            if url_obj['value'] not in seen["urls"]:
                seen["urls"].add(url_obj['value'])
                consolidated['urls'].append(url_obj)
        
        # API Keys
        for key_obj in artifacts.get('api_keys', []):
            if key_obj['value'] not in seen["api_keys"]:
                seen["api_keys"].add(key_obj['value'])
                consolidated['api_keys'].append(key_obj)
        
        # Private Keys
        consolidated['private_keys'].extend(artifacts.get('private_keys', []))
    
    # Calculate summary
    total_crypto = sum(len(addrs) for addrs in consolidated['crypto_addresses'].values())
    total_hashes = sum(len(hashes) for hashes in consolidated['hashes'].values())
    total_social = sum(len(social) for social in consolidated['social_media'].values())
    
    consolidated['summary'] = {
        'total_emails': len(consolidated['emails']),
        'total_domains': len(consolidated['domains']),
        'total_ips': len(consolidated['ip_addresses']),
        'total_urls': len(consolidated['urls']),
        'total_phones': len(consolidated['phone_numbers']),
        'total_crypto': total_crypto,
        'total_hashes': total_hashes,
        'total_social': total_social,
        'total_api_keys': len(consolidated['api_keys']),
        'total_private_keys': len(consolidated['private_keys']),
        'total_artifacts': (
            len(consolidated['emails']) + len(consolidated['domains']) + 
            len(consolidated['ip_addresses']) + total_crypto + total_hashes +
            total_social + len(consolidated['urls']) + len(consolidated['phone_numbers']) +
            len(consolidated['api_keys']) + len(consolidated['private_keys'])
        )
    }
    
    return consolidated

# ==================== THREAT ANALYSIS ====================
def analyze_threat_level(artifacts: Dict, content: str) -> Dict:
    threat_score = 0.0
    indicators = []
    risk_factors = []
    
    # Cryptocurrency activity
    total_crypto = artifacts.get('summary', {}).get('total_crypto', 0)
    if total_crypto > 0:
        threat_score += min(total_crypto * 0.04, 0.25)
        indicators.append(f"Cryptocurrency wallets detected: {total_crypto} addresses")
        if total_crypto > 20:
            risk_factors.append("High volume cryptocurrency activity")
    
    # IP addresses
    total_ips = len(artifacts.get('ip_addresses', []))
    if total_ips > 10:
        threat_score += 0.15
        indicators.append(f"Multiple IP addresses identified: {total_ips}")
        if total_ips > 30:
            risk_factors.append("Potential botnet or distributed network")
    
    # Email addresses
    total_emails = len(artifacts.get('emails', []))
    if total_emails > 15:
        threat_score += 0.10
        indicators.append(f"Large number of email addresses: {total_emails}")
    
    # Private keys detected
    if artifacts.get('private_keys'):
        threat_score += 0.30
        indicators.append("CRITICAL: Private cryptographic keys detected")
        risk_factors.append("Potential credential exposure")
    
    # API keys detected
    if len(artifacts.get('api_keys', [])) > 0:
        threat_score += 0.20
        indicators.append(f"API keys/tokens detected: {len(artifacts['api_keys'])}")
        risk_factors.append("Possible unauthorized API access")
    
    # Suspicious keywords
    suspicious_keywords = [
        'malware', 'exploit', 'ransomware', 'phishing', 'breach', 
        'vulnerability', 'zero-day', 'botnet', 'ddos', 'backdoor',
        'trojan', 'keylogger', 'credential', 'dump', 'leak',
        'hack', 'crack', 'bypass', 'injection', 'shell'
    ]
    keyword_count = sum(1 for kw in suspicious_keywords if kw in content.lower())
    
    if keyword_count > 3:
        threat_score += min(keyword_count * 0.08, 0.25)
        indicators.append(f"Suspicious security keywords found: {keyword_count}")
        if keyword_count > 10:
            risk_factors.append("High concentration of threat-related terminology")
    
    # Hash analysis
    total_hashes = artifacts.get('summary', {}).get('total_hashes', 0)
    if total_hashes > 20:
        threat_score += 0.12
        indicators.append(f"File hashes detected: {total_hashes}")
        risk_factors.append("Possible malware samples or file distribution")
    
    # Social media presence
    total_social = artifacts.get('summary', {}).get('total_social', 0)
    if total_social > 10:
        threat_score += 0.08
        indicators.append(f"Social media accounts identified: {total_social}")
    
    threat_score = min(threat_score, 1.0)
    
    # Determine threat level
    if threat_score >= 0.75:
        level = "CRITICAL"
        color = "#DC2626"
    elif threat_score >= 0.55:
        level = "HIGH"
        color = "#EF4444"
    elif threat_score >= 0.35:
        level = "MEDIUM"
        color = "#F59E0B"
    else:
        level = "LOW"
        color = "#10B981"
    
    return {
        "score": threat_score,
        "level": level,
        "color": color,
        "indicators": indicators,
        "risk_factors": risk_factors,
        "severity_description": get_severity_description(level)
    }

def get_severity_description(level: str) -> str:
    descriptions = {
        "CRITICAL": "Immediate action required. High-risk indicators detected including exposed credentials or significant security vulnerabilities.",
        "HIGH": "Significant threats identified. Comprehensive investigation and mitigation measures recommended.",
        "MEDIUM": "Moderate risk level detected. Continuous monitoring and assessment advised.",
        "LOW": "Minimal threat indicators present. Routine observation and standard security protocols sufficient."
    }
    return descriptions.get(level, "Unknown severity level")

# ==================== ADVANCED REPORT GENERATION ====================
def generate_intelligence_report_advanced(llm, query: str, sources: Dict, 
                                         artifacts: Dict, threat: Dict) -> str:
    """Generate intelligence report using Groq with Ollama fallback"""
    
    # Try LLM router first if available
    if GROQ_CONFIGURED and LLM_ROUTER_AVAILABLE:
        try:
            rate_limit_check("generate_report")
            
            sources_summary = "\n".join([f"Source {i+1}: {url[:100]} - {text[:250]}..." 
                                        for i, (url, text) in enumerate(list(sources.items())[:8])])
            
            artifacts_summary = f"""
Emails: {len(artifacts.get('emails', []))}
Domains: {len(artifacts.get('domains', []))}
IP Addresses: {len(artifacts.get('ip_addresses', []))}
Cryptocurrency Wallets: {artifacts.get('summary', {}).get('total_crypto', 0)}
File Hashes: {artifacts.get('summary', {}).get('total_hashes', 0)}
Social Media: {artifacts.get('summary', {}).get('total_social', 0)}
API Keys: {len(artifacts.get('api_keys', []))}
Private Keys: {len(artifacts.get('private_keys', []))}
"""
            
            system_prompt = """You are an elite OSINT analyst specializing in dark web intelligence.

Generate a COMPREHENSIVE, PROFESSIONAL intelligence report with these sections:

EXECUTIVE SUMMARY
Write a clear 4-5 sentence overview of findings, threat level, and key discoveries.

INVESTIGATION SCOPE
- Query analyzed
- Sources examined
- Time period

KEY FINDINGS
List 8-12 significant discoveries with specific data points and evidence.

ARTIFACT INTELLIGENCE
Organize all discovered artifacts by category with analysis.

THREAT ASSESSMENT
- Current threat level and justification
- Risk indicators with severity
- Attack vectors identified
- Operational security assessment

INFRASTRUCTURE ANALYSIS
- Network topology
- Hosting patterns
- Geographic indicators
- Technical sophistication

ACTOR PROFILING
If applicable, assess:
- Technical capabilities
- Operational patterns
- Geographic/linguistic indicators
- Potential motivations

RECOMMENDATIONS
Provide 5-8 actionable recommendations:
- Immediate actions
- Investigation leads
- Defensive measures
- Monitoring strategies

CONFIDENCE ASSESSMENT
- Overall confidence level
- Data quality assessment
- Known limitations

Use clear paragraphs and bullet points. DO NOT use markdown symbols like ** or *. Be specific and data-driven."""
            
            user_prompt = f"""Query: {query}

SOURCES ANALYZED:
{sources_summary}

ARTIFACTS DISCOVERED:
{artifacts_summary}

THREAT LEVEL: {threat['level']}
THREAT SCORE: {threat['score']:.0%}

Generate the comprehensive intelligence report now."""
            
            result = generate_text(
                prompt=user_prompt,
                app_name="donna",
                task_type="osint",
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=8192
            )
            
            report = result.get("response", "").strip()
            if report:
                # Clean markdown artifacts
                report = re.sub(r'\*\*(.*?)\*\*', r'\1', report)
                report = re.sub(r'\*(.*?)\*', r'\1', report)
                
                if len(report) > 500:
                    logger.info(f"✓ Report generated using {result.get('model', 'unknown')} ({result.get('source', 'unknown')})")
                    return report
        except Exception as llm_error:
            log_llm_error("DONNA AI", "cloud", llm_error, fallback=True)
            logger.warning(f"⚠ LLM error: {llm_error}. Router will handle fallback automatically.")
    
    # Fallback to LangChain Ollama if available
    if llm:
        try:
            rate_limit_check("generate_report")
            
            sources_summary = "\n".join([f"Source {i+1}: {url[:100]} - {text[:250]}..." 
                                        for i, (url, text) in enumerate(list(sources.items())[:8])])
            
            artifacts_summary = f"""
Emails: {len(artifacts.get('emails', []))}
Domains: {len(artifacts.get('domains', []))}
IP Addresses: {len(artifacts.get('ip_addresses', []))}
Cryptocurrency Wallets: {artifacts.get('summary', {}).get('total_crypto', 0)}
File Hashes: {artifacts.get('summary', {}).get('total_hashes', 0)}
Social Media: {artifacts.get('summary', {}).get('total_social', 0)}
API Keys: {len(artifacts.get('api_keys', []))}
Private Keys: {len(artifacts.get('private_keys', []))}
"""
            
            system = """You are an elite OSINT analyst specializing in dark web intelligence.

Generate a COMPREHENSIVE, PROFESSIONAL intelligence report with these sections:

EXECUTIVE SUMMARY
Write a clear 4-5 sentence overview of findings, threat level, and key discoveries.

INVESTIGATION SCOPE
- Query analyzed
- Sources examined
- Time period

KEY FINDINGS
List 8-12 significant discoveries with specific data points and evidence.

ARTIFACT INTELLIGENCE
Organize all discovered artifacts by category with analysis.

THREAT ASSESSMENT
- Current threat level and justification
- Risk indicators with severity
- Attack vectors identified
- Operational security assessment

INFRASTRUCTURE ANALYSIS
- Network topology
- Hosting patterns
- Geographic indicators
- Technical sophistication

ACTOR PROFILING
If applicable, assess:
- Technical capabilities
- Operational patterns
- Geographic/linguistic indicators
- Potential motivations

RECOMMENDATIONS
Provide 5-8 actionable recommendations:
- Immediate actions
- Investigation leads
- Defensive measures
- Monitoring strategies

CONFIDENCE ASSESSMENT
- Overall confidence level
- Data quality assessment
- Known limitations

Use clear paragraphs and bullet points. DO NOT use markdown symbols like ** or *. Be specific and data-driven."""
        
            prompt = ChatPromptTemplate([
                ("system", system),
                ("user", """Query: {query}

SOURCES ANALYZED:
{sources}

ARTIFACTS DISCOVERED:
{artifacts}

THREAT LEVEL: {threat_level}
THREAT SCORE: {threat_score}

Generate the comprehensive intelligence report now.""")
            ])
            
            chain = prompt | llm | StrOutputParser()
            report = chain.invoke({
                "query": query,
                "sources": sources_summary,
                "artifacts": artifacts_summary,
                "threat_level": threat['level'],
                "threat_score": f"{threat['score']:.0%}"
            })
            
            # Clean markdown artifacts
            report = re.sub(r'\*\*(.*?)\*\*', r'\1', report)
            report = re.sub(r'\*(.*?)\*', r'\1', report)
            
            if len(report) > 500:
                logger.info("✓ Report generated using Ollama")
                return report
        except Exception as e:
            logger.warning(f"[REPORT] Ollama error: {e}")
            # Final fallback to local LLM utility
            if LOCAL_LLM_AVAILABLE:
                try:
                    local_result, success = generate_with_ollama(
                        f"Generate an intelligence report for query: {query}. Sources: {len(sources)}. Artifacts: {artifacts_summary}. Threat: {threat['level']}",
                        system_prompt="You are an elite OSINT analyst. Generate a comprehensive intelligence report.",
                        temperature=0.7,
                        max_tokens=4096
                    )
                    if success and local_result:
                        logger.info("✓ Report generated using local LLM utility")
                        return local_result.strip()
                except Exception as local_error:
                    logger.warning(f"[REPORT] Local LLM utility also failed: {local_error}")
    
    # Final fallback to simple report
    logger.info("Using simple report generation")
    return generate_intelligence_report_simple(query, sources, artifacts, threat)

def generate_intelligence_report_simple(query: str, sources: Dict, 
                                       artifacts: Dict, threat: Dict) -> str:
    now = datetime.now()
    
    report = f"""INTELLIGENCE REPORT: {query.upper()}
Generated: {now.strftime('%B %d, %Y at %H:%M:%S UTC')}
Investigation ID: {generate_investigation_id(query)}

{'='*80}

EXECUTIVE SUMMARY

This investigation analyzed {len(sources)} sources across the dark web and clearnet in response to the query: "{query}". 

The analysis extracted {artifacts.get('summary', {}).get('total_artifacts', 0)} intelligence artifacts including {len(artifacts.get('emails', []))} email addresses, {len(artifacts.get('domains', []))} domains, {artifacts.get('summary', {}).get('total_crypto', 0)} cryptocurrency wallets, and {artifacts.get('summary', {}).get('total_hashes', 0)} file hashes.

Current Threat Assessment: {threat.get('level', 'UNKNOWN')} (Confidence Score: {threat.get('score', 0):.0%})

{threat.get('severity_description', '')}

{'='*80}

INVESTIGATION SCOPE

- Primary Query: {query}
- Investigation Date: {now.strftime('%Y-%m-%d')}
- Sources Analyzed: {len(sources)}
- Total Artifacts Extracted: {artifacts.get('summary', {}).get('total_artifacts', 0)}
- Processing Duration: ~{len(sources) * 2} seconds
- Analysis Method: Multi-engine dark web search with AI-powered extraction

{'='*80}

KEY FINDINGS
"""
    
    # Add threat indicators
    if threat.get('indicators'):
        report += "\nTHREAT INDICATORS:\n\n"
        for i, ind in enumerate(threat['indicators'], 1):
            report += f"  {i}. {ind}\n"
        report += "\n"
    
    # Add risk factors
    if threat.get('risk_factors'):
        report += "RISK FACTORS IDENTIFIED:\n\n"
        for i, risk in enumerate(threat['risk_factors'], 1):
            report += f"  {i}. {risk}\n"
        report += "\n"
    
    report += f"""
{'='*80}

EXTRACTED INTELLIGENCE ARTIFACTS

"""
    
    # Emails
    if artifacts.get('emails'):
        report += f"EMAIL ADDRESSES ({len(artifacts['emails'])} discovered)\n\n"
        for i, email_obj in enumerate(artifacts['emails'][:15], 1):
            email = email_obj['value']
            conf = email_obj.get('confidence', 0.9)
            report += f"  {i}. {email} (Confidence: {conf:.0%})\n"
        if len(artifacts['emails']) > 15:
            report += f"  ... and {len(artifacts['emails']) - 15} more\n"
        report += "\n"
    
    # Domains
    if artifacts.get('domains'):
        report += f"DOMAIN NAMES ({len(artifacts['domains'])} discovered)\n\n"
        for i, domain_obj in enumerate(artifacts['domains'][:15], 1):
            domain = domain_obj['value']
            conf = domain_obj.get('confidence', 0.9)
            report += f"  {i}. {domain} (Confidence: {conf:.0%})\n"
        if len(artifacts['domains']) > 15:
            report += f"  ... and {len(artifacts['domains']) - 15} more\n"
        report += "\n"
    
    # IP Addresses
    if artifacts.get('ip_addresses'):
        report += f"IP ADDRESSES ({len(artifacts['ip_addresses'])} discovered)\n\n"
        for i, ip_obj in enumerate(artifacts['ip_addresses'][:12], 1):
            ip = ip_obj['value']
            conf = ip_obj.get('confidence', 0.99)
            report += f"  {i}. {ip} (Confidence: {conf:.0%})\n"
        if len(artifacts['ip_addresses']) > 12:
            report += f"  ... and {len(artifacts['ip_addresses']) - 12} more\n"
        report += "\n"
    
    # Cryptocurrency wallets
    for crypto_type in ['bitcoin', 'ethereum', 'monero', 'litecoin']:
        addrs = artifacts.get('crypto_addresses', {}).get(crypto_type, [])
        if addrs:
            report += f"{crypto_type.upper()} WALLET ADDRESSES ({len(addrs)} discovered)\n\n"
            for i, addr_obj in enumerate(addrs[:10], 1):
                addr = addr_obj['value']
                conf = addr_obj.get('confidence', 0.95)
                report += f"  {i}. {addr} (Confidence: {conf:.0%})\n"
            if len(addrs) > 10:
                report += f"  ... and {len(addrs) - 10} more\n"
            report += "\n"
    
    # File Hashes
    for hash_type in ['md5', 'sha256', 'sha512']:
        hashes = artifacts.get('hashes', {}).get(hash_type, [])
        if hashes:
            report += f"{hash_type.upper()} HASHES ({len(hashes)} discovered)\n\n"
            for i, hash_obj in enumerate(hashes[:8], 1):
                h = hash_obj['value']
                conf = hash_obj.get('confidence', 0.9)
                report += f"  {i}. {h} (Confidence: {conf:.0%})\n"
            if len(hashes) > 8:
                report += f"  ... and {len(hashes) - 8} more\n"
            report += "\n"
    
    # Social Media
    for social_type in ['twitter', 'telegram']:
        social = artifacts.get('social_media', {}).get(social_type, [])
        if social:
            report += f"{social_type.upper()} ACCOUNTS ({len(social)} discovered)\n\n"
            for i, social_obj in enumerate(social[:10], 1):
                s = social_obj['value']
                conf = social_obj.get('confidence', 0.85)
                report += f"  {i}. {s} (Confidence: {conf:.0%})\n"
            report += "\n"
    
    # API Keys
    if artifacts.get('api_keys'):
        report += f"API KEYS/TOKENS ({len(artifacts['api_keys'])} suspected)\n\n"
        report += "  WARNING: Potential API keys or authentication tokens detected.\n"
        report += "  Manual verification required for accuracy.\n\n"
    
    # Private Keys
    if artifacts.get('private_keys'):
        report += "PRIVATE CRYPTOGRAPHIC KEYS DETECTED\n\n"
        report += "  CRITICAL: Private key material found in sources.\n"
        report += "  This represents a severe security exposure.\n\n"
    
    report += f"""
{'='*80}

THREAT ANALYSIS

Threat Level: {threat.get('level', 'UNKNOWN')}
Confidence Score: {threat.get('score', 0):.0%}
Severity: {threat.get('severity_description', 'Not assessed')}

ASSESSMENT:

The investigation indicates a {threat.get('level', 'UNKNOWN')} threat level based on the discovered artifacts and content analysis. This assessment considers:

- Volume and type of artifacts discovered
- Presence of high-risk indicators (credentials, keys, malware indicators)
- Concentration of suspicious terminology
- Infrastructure complexity

DETAILED RISK ANALYSIS:
"""
    
    if threat.get('risk_factors'):
        for i, risk in enumerate(threat['risk_factors'], 1):
            report += f"\n  {i}. {risk}"
    else:
        report += "\n  No specific high-risk factors identified at this time."
    
    report += f"""

{'='*80}

RECOMMENDATIONS

IMMEDIATE ACTIONS:

  1. Cross-reference all discovered artifacts with existing threat intelligence databases
  2. Monitor identified cryptocurrency wallets for transaction activity
  3. Investigate domain registration and hosting information for infrastructure mapping
  4. Check IP addresses against known threat actor infrastructure
  5. Set up continuous monitoring alerts for discovered email addresses and domains

INVESTIGATION LEADS:

  1. Perform deeper analysis on high-confidence artifacts
  2. Correlate discovered indicators across multiple investigations
  3. Research associated infrastructure and hosting providers
  4. Examine temporal patterns in artifact discovery
  5. Identify potential relationships between discovered entities

DEFENSIVE MEASURES:

  1. Update security controls with discovered indicators of compromise
  2. Implement network-level blocking for malicious IP addresses
  3. Monitor for attempted use of discovered credentials
  4. Review and enhance authentication mechanisms
  5. Conduct security awareness training on identified threats

MONITORING STRATEGIES:

  1. Establish automated scanning for new instances of discovered artifacts
  2. Monitor dark web forums and marketplaces for related activity
  3. Track cryptocurrency wallet transactions
  4. Set up alerts for domain and infrastructure changes
  5. Maintain ongoing threat intelligence collection

{'='*80}

CONFIDENCE ASSESSMENT

Overall Confidence: {"HIGH" if threat.get('score', 0) > 0.6 else "MEDIUM" if threat.get('score', 0) > 0.3 else "LOW"}

DATA QUALITY: The artifacts extracted have been assigned individual confidence scores based on pattern matching accuracy and context. High-confidence artifacts (>90%) are considered reliable for immediate action.

LIMITATIONS:

  • Search engine coverage may not include all dark web resources
  • Tor network access limitations may restrict .onion site accessibility
  • Automated extraction may produce false positives requiring manual verification
  • Temporal limitations: data represents a point-in-time snapshot
  • Language barriers may affect non-English content analysis

VERIFICATION RECOMMENDED: All critical findings should be independently verified before taking action.

{'='*80}

INVESTIGATION METADATA

Report Generated: {now.isoformat()}
Investigation ID: {generate_investigation_id(query)}
DONNA AI Version: 3.0 Enhanced
Analysis Engine: Advanced Multi-Source OSINT
Report Format: Comprehensive Intelligence Assessment

{'='*80}

END OF REPORT

This document contains sensitive intelligence information and should be handled according to your organization's data classification policies.

DONNA AI - Dark Web OSINT Intelligence Platform
For authorized cybersecurity research and intelligence gathering only
© {now.year} All Rights Reserved
"""
    
    return report

# ==================== PDF HELPER FUNCTIONS ====================
def clean_text_for_pdf(text):

    """Remove problematic characters for PDF generation"""
    if not text:
        return ""
    
    # Properly handle unicode
    text = str(text)

    # Replace problematic characters
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2013', '-').replace('\u2014', '-')
    text = text.replace('\u2022', '*')
    text = text.replace('\xa0', ' ')

    # Remove any remaining non-ASCII safely
    text = ''.join(char if ord(char) < 128 else ' ' for char in text)

    # Remove markdown-style formatting
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'`(.*?)`', r'\1', text)

    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def create_threat_gauge(threat_score, threat_level):
    """Create a visual threat gauge for PDF"""
    if not REPORTLAB_AVAILABLE:
        return None
    
    try:
        d = Drawing(400, 60)
        
        # Background bar
        d.add(Rect(0, 20, 400, 20, fillColor=HexColor('#E5E7EB'), strokeColor=None))
        
        # Threat level colors
        color_map = {
            'LOW': HexColor('#10B981'),
            'MEDIUM': HexColor('#F59E0B'),
            'HIGH': HexColor('#EF4444'),
            'CRITICAL': HexColor('#DC2626')
        }
        
        fill_color = color_map.get(threat_level, HexColor('#94A3B8'))
        fill_width = threat_score * 400
        
        # Filled bar
        d.add(Rect(0, 20, fill_width, 20, fillColor=fill_color, strokeColor=None))
        
        # Labels
        d.add(String(200, 5, f'Threat Score: {threat_score:.1%}', 
                    textAnchor='middle', fontSize=12, fillColor=colors.black))
        d.add(String(200, 45, threat_level, 
                    textAnchor='middle', fontSize=14, fillColor=fill_color, 
                    fontName='Helvetica-Bold'))
        
        return d
    except Exception as e:
        logger.error(f"[PDF] Error creating threat gauge: {e}")
        return None

def create_pdf_header_footer(canvas_obj, doc):
    """Add professional header and footer to each page with watermark"""
    try:
        canvas_obj.saveState()
        
        # LARGE WATERMARK IN CENTER
        canvas_obj.setFont('Helvetica-Bold', 80)
        canvas_obj.setFillColorRGB(0.95, 0.95, 0.95, alpha=0.1)  # Very light gray
        canvas_obj.saveState()
        canvas_obj.translate(A4[0]/2, A4[1]/2)  # Center of page
        canvas_obj.rotate(45)  # Diagonal
        canvas_obj.drawCentredString(0, 0, 'DONNA AI')
        canvas_obj.restoreState()
        
        # Smaller secondary watermark
        canvas_obj.setFont('Helvetica', 24)
        canvas_obj.setFillColorRGB(0.9, 0.9, 0.9, alpha=0.08)
        canvas_obj.saveState()
        canvas_obj.translate(A4[0]/2, A4[1]/2 - 80)
        canvas_obj.rotate(45)
        canvas_obj.drawCentredString(0, 0, 'CONFIDENTIAL OSINT REPORT')
        canvas_obj.restoreState()
        
        # Header
        canvas_obj.setStrokeColor(HexColor('#FF6B35'))
        canvas_obj.setLineWidth(2)
        canvas_obj.line(0.75*inch, doc.height + 1.2*inch, 
                    doc.width + 0.75*inch, doc.height + 1.2*inch)
        
        canvas_obj.setFont('Helvetica-Bold', 10)
        canvas_obj.setFillColor(HexColor('#FF6B35'))
        canvas_obj.drawString(0.75*inch, doc.height + 1.3*inch, "DONNA AI - OSINT Intelligence Report")
        
        canvas_obj.setFont('Helvetica', 8)
        canvas_obj.setFillColor(HexColor('#64748B'))
        canvas_obj.drawRightString(doc.width + 0.75*inch, doc.height + 1.3*inch, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        # Footer
        canvas_obj.setStrokeColor(HexColor('#E2E8F0'))
        canvas_obj.setLineWidth(1)
        canvas_obj.line(0.75*inch, 0.6*inch, doc.width + 0.75*inch, 0.6*inch)
        
        canvas_obj.setFont('Helvetica', 7)
        canvas_obj.setFillColor(HexColor('#94A3B8'))
        canvas_obj.drawString(0.75*inch, 0.45*inch, "CONFIDENTIAL - For Authorized Use Only")
        
        canvas_obj.setFont('Helvetica-Bold', 7)
        page_num = canvas_obj.getPageNumber()
        canvas_obj.drawRightString(doc.width + 0.75*inch, 0.45*inch, f"Page {page_num}")
        
        # Bottom watermark
        canvas_obj.setFont('Helvetica', 6)
        canvas_obj.setFillColor(HexColor('#CBD5E1'))
        canvas_obj.drawCentredString(A4[0]/2, 0.25*inch, 
            f"DONNA AI Watermark | Report ID: {hashlib.md5(str(datetime.now()).encode()).hexdigest()[:12]}")
        
        canvas_obj.restoreState()
    except Exception as e:
        logger.error(f"[PDF] Error in header/footer: {e}")

# ==================== ADVANCED PDF GENERATION ====================
def generate_advanced_pdf(report: str, query: str, artifacts: Dict, threat: Dict, stats: Dict, filename: str) -> str:
    if not REPORTLAB_AVAILABLE:
        txt_file = filename.replace('.pdf', '.txt')
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(report)
        return txt_file
    
    try:
        # Validate inputs
        if not report or len(report) < 100:
            raise ValueError("Report content too short")
        
        # Sanitize filename
        filename = filename.replace('\\', '/').replace('//', '/')
        filename = os.path.abspath(filename)
        
        # Clean all text inputs
        report = clean_text_for_pdf(report)
        query = clean_text_for_pdf(query)
        
        if not filename.endswith('.pdf'):
            filename += '.pdf'
        
        
        # Ensure directory exists and file is writable
        os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
        
        # Create PDF with proper page size
        doc = SimpleDocTemplate(
            filename, 
            pagesize=A4,
            topMargin=1.2*inch,  
            bottomMargin=0.9*inch,
            leftMargin=0.75*inch, 
            rightMargin=0.75*inch,
            title="DONNA AI Intelligence Report",
            author="DONNA AI - Lucifer",
            subject="OSINT Investigation Report"
        )
        
        styles = getSampleStyleSheet()
        story = []
        
        # Enhanced professional styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=26,
            textColor=HexColor('#1E293B'),
            spaceAfter=10,
            spaceBefore=8,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold',
            leading=32
        )
        
        heading1_style = ParagraphStyle(
            'CustomHeading1',
            parent=styles['Heading1'],
            fontSize=14,
            textColor=HexColor('#FF6B35'),
            spaceAfter=8,
            spaceBefore=16,
            fontName='Helvetica-Bold',
            leading=18,
            borderColor=HexColor('#FF6B35'),
            borderWidth=2,
            borderPadding=6,
            backColor=HexColor('#FEF3C7')
        )
        
        heading2_style = ParagraphStyle(
            'CustomHeading2',
            parent=styles['Heading2'],
            fontSize=12,
            textColor=HexColor('#1E293B'),
            spaceAfter=6,
            spaceBefore=10,
            fontName='Helvetica-Bold',
            leading=16,
            leftIndent=10
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=10,
            textColor=HexColor('#334155'),
            spaceAfter=8,
            alignment=TA_JUSTIFY,
            leading=14,
            leftIndent=5,
            rightIndent=5
        )
        
        bullet_style = ParagraphStyle(
            'CustomBullet',
            parent=styles['Normal'],
            fontSize=9,
            textColor=HexColor('#475569'),
            spaceAfter=5,
            leftIndent=25,
            bulletIndent=15,
            leading=13
        )
        
        # Add logo if exists
        logo_path = os.path.join('static', 'images', 'donna_ai.png')
        if os.path.exists(logo_path):
            try:
                img = RLImage(logo_path, width=1.2*inch, height=1.2*inch)
                img.hAlign = 'CENTER'
                story.append(img)
                story.append(Spacer(1, 0.2*inch))
            except:
                pass
        
        # Title Page
        story.append(Paragraph("DONNA AI", title_style))
        story.append(Paragraph(
            "Advanced Dark Web OSINT Intelligence Platform v3.0",
            ParagraphStyle('subtitle', parent=styles['Normal'], 
                         fontSize=11, alignment=TA_CENTER, 
                         textColor=HexColor('#94A3B8'), spaceAfter=6)
        ))
        story.append(Spacer(1, 0.3*inch))
        
        # Report metadata table
        now = datetime.now()
        meta_data = [
            ['Investigation Query', clean_text_for_pdf(query)[:80]],
            ['Report Generated', now.strftime('%B %d, %Y at %H:%M:%S UTC')],
            ['Investigation ID', generate_investigation_id(query)],
            ['Threat Level', threat.get('level', 'UNKNOWN')],
            ['Threat Score', f"{threat.get('score', 0):.1%}"],
            ['Total Artifacts', str(artifacts.get('summary', {}).get('total_artifacts', 0))],
            ['Sources Analyzed', str(stats.get('successfully_scraped', 0))],
            ['Analysis Duration', f"{stats.get('investigation_duration_seconds', 0):.1f}s"],
        ]
        
        meta_table = Table(meta_data, colWidths=[2.2*inch, 4*inch])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), HexColor('#1E293B')),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
            ('TEXTCOLOR', (1, 0), (1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0, 0), (-1, -1), [HexColor('#F5F5F5'), colors.white]),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Threat Assessment Visualization
        story.append(Paragraph("Threat Assessment Visualization", heading2_style))
        story.append(Spacer(1, 0.1*inch))
        threat_gauge = create_threat_gauge(threat.get('score', 0), threat.get('level', 'UNKNOWN'))
        if threat_gauge:
            story.append(threat_gauge)
        story.append(Spacer(1, 0.3*inch))
        
        # Process report content with improved error handling
        lines = clean_text_for_pdf(report).split('\n')
        current_section = []
        
        for line in lines:
            line = line.strip()
            
            if not line or line == '=' * 80:
                continue
            
            try:
                # Main headings (ALL CAPS)
                if line.isupper() and len(line) > 5 and not line.startswith('  '):
                    if current_section:
                        story.extend(current_section)
                        current_section = []
                    story.append(Spacer(1, 0.15*inch))
                    story.append(Paragraph(clean_text_for_pdf(line[:100]), heading1_style))
                    continue
                
                # Sub headings
                if (line.endswith(':') or line.istitle()) and not line.startswith('  '):
                    story.append(Paragraph(clean_text_for_pdf(line[:120]), heading2_style))
                    continue
                
                # Bullet points
                if re.match(r'^\s+\d+\.', line) or re.match(r'^\s+[•\-]', line):
                    clean_line = re.sub(r'^\s+\d+\.\s*', '• ', line)
                    clean_line = re.sub(r'^\s+[•\-]\s*', '• ', clean_line)
                    # Limit bullet length to prevent overflow
                    clean_line = clean_text_for_pdf(clean_line[:300])
                    story.append(Paragraph(clean_line, bullet_style))
                    continue
                
                # Regular paragraphs - split if too long
                if line:
                    clean_line = clean_text_for_pdf(line)
                    # Split very long lines into chunks
                    max_len = 400  # Reduced to prevent overflow
                    if len(clean_line) > max_len:
                        words = clean_line.split()
                        current_chunk = []
                        current_length = 0
                        
                        for word in words:
                            if current_length + len(word) + 1 <= max_len:
                                current_chunk.append(word)
                                current_length += len(word) + 1
                            else:
                                if current_chunk:
                                    story.append(Paragraph(' '.join(current_chunk), body_style))
                                current_chunk = [word]
                                current_length = len(word)
                        
                        if current_chunk:
                            story.append(Paragraph(' '.join(current_chunk), body_style))
                    else:
                        if clean_line.strip():
                            story.append(Paragraph(clean_line, body_style))

            except Exception as e:
                # Skip problematic lines
                logger.warning(f"[PDF] Skipping line due to error: {str(e)[:50]}")
                continue
        
        # Add artifacts appendix
        story.append(PageBreak())
        story.append(Paragraph("APPENDIX: DETAILED ARTIFACTS", title_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Artifacts summary table
        summary_data = [
            ['Artifact Type', 'Count'],
            ['Email Addresses', str(len(artifacts.get('emails', [])))],
            ['Domain Names', str(len(artifacts.get('domains', [])))],
            ['IP Addresses', str(len(artifacts.get('ip_addresses', [])))],
            ['Cryptocurrency Wallets', str(artifacts.get('summary', {}).get('total_crypto', 0))],
            ['File Hashes', str(artifacts.get('summary', {}).get('total_hashes', 0))],
            ['Social Media Accounts', str(artifacts.get('summary', {}).get('total_social', 0))],
            ['URLs', str(len(artifacts.get('urls', [])))],
            ['Phone Numbers', str(len(artifacts.get('phone_numbers', [])))],
        ]
        
        summary_table = Table(summary_data, colWidths=[3.5*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#FF6B35')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, HexColor('#F5F5F5')]),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Detailed artifact tables with improved formatting
        if artifacts.get('emails') and len(artifacts['emails']) > 0:
            story.append(Paragraph("Email Addresses", heading2_style))
            email_data = [['Email Address', 'Conf.']]
            for email_obj in artifacts['emails'][:25]:
                email_val = clean_text_for_pdf(email_obj['value'][:50])
                email_data.append([
                    Paragraph(email_val, ParagraphStyle('tiny', fontSize=7, fontName='Courier', leading=10, textColor=HexColor('#334155'))),
                    f"{email_obj['confidence']:.0%}"
                ])
            
            email_table = Table(email_data, colWidths=[4.5*inch, 0.8*inch])
            email_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#06B6D4')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTSIZE', (0, 0), (-1, -1), 7),
                ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, HexColor('#F0F9FF')]),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('LEFTPADDING', (0, 0), (-1, -1), 4),
                ('RIGHTPADDING', (0, 0), (-1, -1), 4),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))

            story.append(KeepTogether([
                Paragraph("Email Addresses", heading2_style),
                email_table
            ]))
            story.append(Spacer(1, 0.2*inch))
        
        # Bitcoin wallets
        btc_wallets = artifacts.get('crypto_addresses', {}).get('bitcoin', [])
        if btc_wallets and len(btc_wallets) > 0:
            story.append(Paragraph("Bitcoin Wallet Addresses", heading2_style))
            btc_data = [['Bitcoin Address', 'Conf.']]
            for addr in btc_wallets[:20]:
                addr_val = clean_text_for_pdf(addr['value'][:45])
                btc_data.append([
                    Paragraph(addr_val,
                             ParagraphStyle('tiny', fontSize=6, fontName='Courier',
                                          leading=9, textColor=HexColor('#334155'))),
                    f"{addr['confidence']:.0%}"
                ])
            
            btc_table = Table(btc_data, colWidths=[4.5*inch, 0.8*inch])
            btc_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#F59E0B')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 7),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, HexColor('#FFFBEB')]),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('LEFTPADDING', (0, 0), (-1, -1), 4),
                ('RIGHTPADDING', (0, 0), (-1, -1), 4),
                ('TOPPADDING', (0, 0), (-1, -1), 3),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ]))
            
            story.append(KeepTogether([
                Paragraph("Bitcoin Wallet Addresses", heading2_style),
                btc_table
            ]))
            story.append(Spacer(1, 0.2*inch))
        
        # Ethereum wallets
        eth_wallets = artifacts.get('crypto_addresses', {}).get('ethereum', [])
        if eth_wallets and len(eth_wallets) > 0:
            story.append(Paragraph("Ethereum Wallet Addresses", heading2_style))
            eth_data = [['Ethereum Address', 'Conf.']]
            for addr in eth_wallets[:20]:
                addr_val = clean_text_for_pdf(addr['value'][:45])
                eth_data.append([
                    Paragraph(addr_val,
                             ParagraphStyle('tiny', fontSize=6, fontName='Courier',
                                          leading=9, textColor=HexColor('#334155'))),
                    f"{addr['confidence']:.0%}"
                ])
            
            eth_table = Table(eth_data, colWidths=[4.5*inch, 0.8*inch])
            eth_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#8B5CF6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 7),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, HexColor('#F5F3FF')]),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('LEFTPADDING', (0, 0), (-1, -1), 4),
                ('RIGHTPADDING', (0, 0), (-1, -1), 4),
                ('TOPPADDING', (0, 0), (-1, -1), 3),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ]))
            story.append(KeepTogether([
                Paragraph("Ethereum Wallet Addresses", heading2_style),
                eth_table
            ]))
            story.append(Spacer(1, 0.2*inch))
        
        # IP Addresses
        if artifacts.get('ip_addresses') and len(artifacts['ip_addresses']) > 0:
            story.append(Paragraph("IP Addresses", heading2_style))
            ip_data = [['IP Address', 'Confidence']]
            for ip_obj in artifacts['ip_addresses'][:30]:
                ip_data.append([
                    clean_text_for_pdf(ip_obj['value']),
                    f"{ip_obj['confidence']:.0%}"
                ])
            
            ip_table = Table(ip_data, colWidths=[3.5*inch, 1.3*inch])
            ip_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#10B981')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, HexColor('#ECFDF5')]),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ]))
            story.append(KeepTogether([
                Paragraph("IP Addresses", heading2_style),
                ip_table
            ]))
            story.append(Spacer(1, 0.2*inch))
        
        # Domains
        if artifacts.get('domains') and len(artifacts['domains']) > 0:
            story.append(Paragraph("Domain Names", heading2_style))
            domain_data = [['Domain Name', 'Conf.']]
            for domain_obj in artifacts['domains'][:30]:
                domain_val = clean_text_for_pdf(domain_obj['value'][:60])
                domain_data.append([
                    domain_val,
                    f"{domain_obj['confidence']:.0%}"
                ])
            
            domain_table = Table(domain_data, colWidths=[4.5*inch, 0.8*inch])
            domain_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#3B82F6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 7),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, HexColor('#EFF6FF')]),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ]))

            story.append(KeepTogether([
                Paragraph("Domain Names", heading2_style),
                domain_table
            ]))
            story.append(Spacer(1, 0.2*inch))
        
        # File Hashes (SHA256)
        sha256_hashes = artifacts.get('hashes', {}).get('sha256', [])
        if sha256_hashes and len(sha256_hashes) > 0:
            story.append(Paragraph("SHA256 File Hashes", heading2_style))
            hash_data = [['SHA256 Hash', 'Conf.']]
            for hash_obj in sha256_hashes[:15]:
                hash_val = clean_text_for_pdf(hash_obj['value'][:50])
                hash_data.append([
                    Paragraph(hash_val,
                             ParagraphStyle('tiny', fontSize=6, fontName='Courier',
                                          leading=8, textColor=HexColor('#334155'))),
                    f"{hash_obj['confidence']:.0%}"
                ])
            
            hash_table = Table(hash_data, colWidths=[4.5*inch, 0.8*inch])
            hash_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#EF4444')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 7),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, HexColor('#FEF2F2')]),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('LEFTPADDING', (0, 0), (-1, -1), 4),
                ('RIGHTPADDING', (0, 0), (-1, -1), 4),
                ('TOPPADDING', (0, 0), (-1, -1), 3),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ]))
            story.append(KeepTogether([
                Paragraph("SHA256 File Hashes", heading2_style),
                hash_table
            ]))
            story.append(Spacer(1, 0.2*inch))
        
        # Footer with watermark
        story.append(PageBreak())
        story.append(Spacer(1, 2*inch))
        
        footer_lines = [
            "<b>CONFIDENTIAL INTELLIGENCE DOCUMENT</b>",
            "",
            "This report contains sensitive OSINT intelligence information.",
            "Generated by DONNA AI - Dark Web OSINT Intelligence Platform v3.0",
            "",
            f'<font color="#FF6B35"><b>© {datetime.now().year} DONNA AI - All Rights Reserved</b></font>',
            "",
            f'<font size="8" color="#999999">Report ID: {hashlib.md5(query.encode()).hexdigest()[:16]}</font>',
            f'<font size="8" color="#999999">Generated: {datetime.now().isoformat()}</font>',
            f'<font size="8" color="#999999">⚠ For authorized cybersecurity research only</font>',
            "",
            f'<font size="10" color="#FF6B35"><b>[DONNA AI WATERMARK - CLASSIFIED]</b></font>'
        ]
        
        for line in footer_lines:
            story.append(Paragraph(line, ParagraphStyle('footer', 
                parent=styles['Normal'], fontSize=9, alignment=TA_CENTER,
                textColor=HexColor('#64748B'))))
        
        # Build PDF with header/footer
        logger.info(f"[PDF] Building PDF document...")
        try:
            doc.build(story, onFirstPage=create_pdf_header_footer, onLaterPages=create_pdf_header_footer)
            
            # Verify PDF was created and is not empty
            if os.path.exists(filename) and os.path.getsize(filename) > 1000:
                logger.info(f"[PDF] ✓ Generated: {filename} ({os.path.getsize(filename)} bytes)")
                return filename
            else:
                raise ValueError("PDF file is empty or too small")
                
        except Exception as build_error:
            logger.error(f"[PDF] Build error: {build_error}")
            # Try simple build without complex elements
            simple_story = [
                Paragraph("DONNA AI Intelligence Report", title_style),
                Spacer(1, 0.3*inch),
                Paragraph(clean_text_for_pdf(report[:5000]), body_style)
            ]
            doc.build(simple_story, onFirstPage=create_pdf_header_footer)
            return filename
        
    except Exception as e:
        logger.error(f"[PDF] Critical error: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to text file
        txt_file = filename.replace('.pdf', '.txt')
        try:
            with open(txt_file, 'w', encoding='utf-8', errors='ignore') as f:
                f.write(f"DONNA AI Intelligence Report\n")
                f.write(f"{'='*80}\n\n")
                f.write(report)
            logger.info(f"[PDF] Fallback: Generated text file instead: {txt_file}")
            return txt_file
        except Exception as txt_error:
            logger.error(f"[PDF] Even text file failed: {txt_error}")
            raise

# ==================== BLUEPRINT ====================
donna = Blueprint('donna', __name__, template_folder='templates')
CORS(donna, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"]}})

@donna.route('/')
def index():
    return render_template('donna.html')

@donna.route('/health', methods=['GET'])
def health():
    ollama_status = False
    try:
        ollama_url = get_ollama_url()
        ollama_status = True
    except Exception:
        ollama_status = False
    
    # Check local LLM availability
    local_llm_status = False
    if LOCAL_LLM_AVAILABLE:
        try:
            local_llm_status = check_ollama_available()
        except Exception:
            local_llm_status = False
    
    return jsonify({
        "status": "OPERATIONAL",
        "version": "3.0-ULTIMATE-ENHANCED",
        "timestamp": datetime.now().isoformat(),
        "systems": {
            "llm_router": {
                "available": GROQ_CONFIGURED,
                "model": "llama3-70b-8192" if GROQ_CONFIGURED else "N/A"
            },
            "ollama": {
                "available": ollama_status,
                "model": OLLAMA_MODEL if ollama_status else "N/A",
                "url": OLLAMA_BASE_URL if ollama_status else "N/A"
            },
            "local_llm_utility": {
                "available": local_llm_status,
                "fallback_enabled": LOCAL_LLM_AVAILABLE
            },
            "tor": {
                "available": TOR_AVAILABLE,
                "running": check_tor() if TOR_AVAILABLE else False
            },
            "database": {
                "available": os.path.exists(DB_PATH),
                "path": DB_PATH
            },
            "pdf_export": REPORTLAB_AVAILABLE,
            "langchain": LLM_LIBS_AVAILABLE
        },
        "configuration": {
            "search_engines": {
                "total": len(SEARCH_ENGINES),
                "surface_web": len(SURFACE_WEB_ENGINES),
                "deep_web": len(DEEP_WEB_ENGINES),
                "dark_web": len(DARK_WEB_ENGINES)
            },
            "cache_size": len(REQUEST_CACHE),
            "max_cache": MAX_CACHE_SIZE,
            "cache_usage_percent": f"{(len(REQUEST_CACHE)/MAX_CACHE_SIZE)*100:.1f}%"
        }
    })

@donna.route('/search', methods=['POST'])
@rate_limit_strict(requests_per_minute=3, requests_per_hour=30)  # Very strict for resource-intensive dark web searches
@validate_request({
    "query": {
        "type": "string",
        "required": True,
        "max_length": 500,
        "min_length": 2
    },
    "threads": {
        "type": "int",
        "required": False,
        "min_value": 1,
        "max_value": 12
    },
    "scrape_limit": {
        "type": "int",
        "required": False,
        "min_value": 1,
        "max_value": 100
    }
}, strict=True)
def search_endpoint():
    """
    Dark web search endpoint
    OWASP: Rate limited, input validated, schema-based validation
    """
    try:
        # Get validated data from request context
        data = g.validated_data
        query = InputValidator.validate_string(
            data.get('query'), 'query', max_length=500, required=True
        )
        threads = min(data.get('threads', 8), 12)
        scrape_limit = data.get('scrape_limit', 20)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"[INVESTIGATION] Starting: {query}")
        logger.info(f"{'='*80}\n")
        
        investigation_start = datetime.now()
        inv_id = generate_investigation_id(query)
        
        # Initialize LLM
        llm = None
        try:
            llm = get_ollama_llm()
            logger.info("[✓] Ollama LLM initialized")
        except Exception as e:
            logger.warning(f"[!] Ollama unavailable: {e}")
        
        # Step 1: Query refinement (try Groq via LLM router first, then Ollama, then simple)
        logger.info("\n[STEP 1/7] Refining query...")
        query_analysis = None
        
        # Try LLM router first for query refinement
        if GROQ_CONFIGURED and LLM_ROUTER_AVAILABLE:
            try:
                rate_limit_check("refine_query_groq")
                groq_prompt = f"""You are a dark web search optimizer. Analyze this query and provide:
1. Refined keywords (3-6 words max)
2. Three search variations
3. Brief search strategy

Query: {query}

Respond in JSON format:
{{"refined": "keywords here", "variations": ["var1", "var2", "var3"], "strategy": "strategy description"}}"""
                
                result = generate_text(
                    prompt=groq_prompt,
                    app_name="donna",
                    task_type="osint",
                    system_prompt="You are a dark web search optimizer. Analyze queries and provide refined keywords, search variations, and strategies in JSON format.",
                    temperature=0.3,
                    max_tokens=500
                )
                
                response_text = result.get("response", "").strip()
                if response_text:
                    data = json.loads(response_text)
                    query_analysis = {
                        "original": query,
                        "refined": data.get("refined", query),
                        "variations": data.get("variations", []),
                        "strategy": data.get("strategy", ""),
                        "keywords": data.get("refined", query).split(),
                        "model_used": "llama-3.3-70b-versatile"
                    }
                    logger.info("✓ Query refined using LLM Router")
            except Exception as llm_error:
                error_str = str(llm_error).lower()
                logger.warning(f"⚠ LLM query refinement failed: {llm_error}")
                
                # Fallback to local LLM
                if LOCAL_LLM_AVAILABLE and any(keyword in error_str for keyword in [
                    "resource exhausted", "quota", "rate limit", "429", 
                    "503", "500", "timeout", "unavailable", "error"
                ]):
                    try:
                        local_result, success = generate_with_ollama(
                            f"Refine this dark web search query: {query}. Provide refined keywords and 3 variations.",
                            system_prompt="You are a dark web search optimizer.",
                            temperature=0.3,
                            max_tokens=300
                        )
                        if success and local_result:
                            # Try to parse JSON from local result
                            try:
                                data = json.loads(local_result.strip())
                                query_analysis = {
                                    "original": query,
                                    "refined": data.get("refined", query),
                                    "variations": data.get("variations", []),
                                    "keywords": data.get("refined", query).split(),
                                    "model_used": "local-ollama"
                                }
                                logger.info("✓ Query refined using local Ollama")
                            except:
                                pass
                    except:
                        pass
        
        # Fallback to LangChain Ollama or simple refinement
        if not query_analysis:
            query_analysis = refine_query_with_ollama(llm, query) if llm else refine_query_advanced(query)
            if not query_analysis.get('model_used'):
                query_analysis['model_used'] = "ollama" if llm else "simple"
        refined_query = query_analysis.get('refined', query)
        variations = query_analysis.get('variations', [])
        logger.info(f"[✓] Refined: {refined_query}")
        logger.info(f"[✓] Variations: {', '.join(variations)}")
        
        # Step 2: Distributed search across all web layers
        logger.info(f"\n[STEP 2/7] Searching across {len(SURFACE_WEB_ENGINES)} surface, {len(DEEP_WEB_ENGINES)} deep, {len(DARK_WEB_ENGINES)} dark web engines...")
        search_results = get_search_results_distributed(refined_query, variations, threads)
        
        if not search_results:
            return jsonify({
                "success": False,
                "error": "No results found. Try different keywords or check connectivity."
            }), 404
        
        logger.info(f"[✓] Found {len(search_results)} results")
        
        # Step 3: Filter results
        logger.info(f"\n[STEP 3/7] Filtering and ranking...")
        filtered_results = search_results[:scrape_limit]
        logger.info(f"[✓] Filtered to {len(filtered_results)} top results")
        
        # Step 4: Batch scraping with categorization
        logger.info(f"\n[STEP 4/7] Scraping {len(filtered_results)} sources (categorized by web layer)...")
        scraped_results = scrape_batch(filtered_results, inv_id, threads)
        logger.info(f"[✓] Successfully scraped {len(scraped_results)} sources")
        
        if not scraped_results:
            return jsonify({
                "success": False,
                "error": "Failed to scrape results. Try again later."
            }), 500
        
        # Extract content and categorize by web layer
        scraped_content = {}
        scrape_breakdown = {"dark": [], "deep": [], "surface": []}
        
        for url, result_data in scraped_results.items():
            if isinstance(result_data, dict):
                content = result_data.get("content", "")
                web_layer = result_data.get("web_layer", "surface")
            else:
                # Backward compatibility
                content = result_data
                web_layer = "dark" if ".onion" in url.lower() else "surface"
            
            if content and len(content) > 200:
                scraped_content[url] = content
                scrape_breakdown[web_layer].append(url)
        
        logger.info(f"[✓] Scrape breakdown: {len(scrape_breakdown['dark'])} dark, "
                   f"{len(scrape_breakdown['deep'])} deep, {len(scrape_breakdown['surface'])} surface")
        
        # Step 5: Extract artifacts
        logger.info(f"\n[STEP 5/7] Extracting intelligence artifacts...")
        # Handle both old format (dict of strings) and new format (dict of dicts)
        content_dict = {}
        for url, result_data in scraped_results.items():
            if isinstance(result_data, dict):
                content_dict[url] = result_data.get("content", "")
            else:
                content_dict[url] = result_data
        
        all_artifacts = [extract_artifacts_comprehensive({url: content}) 
                        for url, content in content_dict.items()]
        consolidated_artifacts = consolidate_artifacts(all_artifacts)
        total_artifacts = consolidated_artifacts.get('summary', {}).get('total_artifacts', 0)
        logger.info(f"[✓] Extracted {total_artifacts} unique artifacts")
        
        # Step 6: Threat analysis
        logger.info(f"\n[STEP 6/7] Performing threat analysis...")
        content_text = " ".join(content_dict.values())
        threat_analysis = analyze_threat_level(consolidated_artifacts, content_text)
        logger.info(f"[✓] Threat Level: {threat_analysis['level']} (Score: {threat_analysis['score']:.2f})")
        
        # Step 7: Generate report
        logger.info(f"\n[STEP 7/7] Generating intelligence report...")
        intelligence_report = generate_intelligence_report_advanced(
            llm, query, content_dict, consolidated_artifacts, threat_analysis
        )
        logger.info(f"[✓] Report generated ({len(intelligence_report)} characters)")
        
        # Calculate statistics
        investigation_end = datetime.now()
        duration = (investigation_end - investigation_start).total_seconds()
        
        stats = {
            "total_search_results": len(search_results),
            "filtered_results": len(filtered_results),
            "successfully_scraped": len(content_dict),
            "artifacts_extracted": total_artifacts,
            "investigation_duration_seconds": duration,
            "threat_level": threat_analysis['level'],
            "threat_score": threat_analysis['score'],
            "refined_query": refined_query,
            "average_scrape_time": duration / max(len(content_dict), 1),
            "scrape_breakdown": {
                "dark_web": len(scrape_breakdown['dark']),
                "deep_web": len(scrape_breakdown['deep']),
                "surface_web": len(scrape_breakdown['surface'])
            }
        }
        
        # Save to database
        save_investigation(inv_id, {
            'query': query,
            'refined_query': refined_query,
            'start_time': investigation_start,
            'end_time': investigation_end,
            'total_results': len(search_results),
            'total_scraped': len(content_dict),
            'artifacts_found': total_artifacts,
            'threat_level': threat_analysis['level'],
            'threat_score': threat_analysis['score'],
            'status': 'COMPLETED',
            'report': intelligence_report
        })
        
        logger.info(f"\n{'='*80}")
        logger.info(f"[INVESTIGATION] COMPLETED in {duration:.1f}s")
        logger.info(f"{'='*80}\n")
        
        # Prepare sources with web layer information
        sources_with_metadata = []
        for url in content_dict.keys():
            source_info = {"url": url}
            if url in scrape_breakdown['dark']:
                source_info["web_layer"] = "dark"
                source_info["type"] = "Dark Web (.onion)"
            elif url in scrape_breakdown['deep']:
                source_info["web_layer"] = "deep"
                source_info["type"] = "Deep Web"
            else:
                source_info["web_layer"] = "surface"
                source_info["type"] = "Surface Web"
            
            # Get metadata if available
            if url in scraped_results and isinstance(scraped_results[url], dict):
                source_info["metadata"] = scraped_results[url].get("metadata", {})
            
            sources_with_metadata.append(source_info)
        
        return jsonify({
            "success": True,
            "investigation_id": inv_id,
            "query_analysis": query_analysis,
            "statistics": stats,
            "artifacts": consolidated_artifacts,
            "threat_analysis": threat_analysis,
            "report": intelligence_report,
            "sources": list(content_dict.keys()),
            "sources_metadata": sources_with_metadata,
            "scrape_breakdown": scrape_breakdown,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"\n[ERROR] Investigation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"Investigation failed: {str(e)}"
        }), 500

@donna.route('/export-pdf', methods=['POST'])
def export_pdf():
    try:
        data = request.get_json(force=True) or {}
        report = data.get('report', '')
        query = data.get('query', 'Investigation')
        artifacts = data.get('artifacts', {})
        threat = data.get('threat_analysis', {})
        stats = data.get('statistics', {})
        
        if not report:
            return jsonify({"error": "No report content"}), 400
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"DONNA_Intelligence_{timestamp}.pdf"
        filepath = os.path.join(tempfile.gettempdir(), filename)
        
        logger.info(f"[PDF] Generating PDF report: {filename}")
        filepath = generate_advanced_pdf(report, query, artifacts, threat, stats, filepath)
        
        return send_file(
            filepath,
            mimetype='application/pdf' if filepath.endswith('.pdf') else 'text/plain',
            as_attachment=True,
            download_name=os.path.basename(filepath)
        )
    except Exception as e:
        logger.error(f"[PDF] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@donna.route('/history', methods=['GET'])
def get_history():
    """Get investigation history"""
    try:
        limit = int(request.args.get('limit', 50))
        history = get_investigation_history(limit)
        return jsonify({
            "success": True,
            "history": history,
            "count": len(history)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@donna.route('/investigation/<inv_id>', methods=['GET'])
def get_investigation(inv_id):
    """Get specific investigation details"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('SELECT * FROM investigations WHERE id = ?', (inv_id,))
        row = c.fetchone()
        conn.close()
        
        if row:
            return jsonify({
                "success": True,
                "investigation": {
                    "id": row[0],
                    "query": row[1],
                    "refined_query": row[2],
                    "start_time": row[3],
                    "end_time": row[4],
                    "total_results": row[5],
                    "total_scraped": row[6],
                    "artifacts_found": row[7],
                    "threat_level": row[8],
                    "threat_score": row[9],
                    "status": row[10],
                    "report": row[11]
                }
            })
        else:
            return jsonify({"success": False, "error": "Investigation not found"}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@donna.route('/clear-cache', methods=['POST'])
def clear_cache():
    size = len(REQUEST_CACHE)
    REQUEST_CACHE.clear()
    return jsonify({
        "success": True,
        "cleared_items": size,
        "message": f"Cleared {size} cached items"
    })

@donna.route('/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        c.execute('SELECT COUNT(*) FROM investigations')
        total_investigations = c.fetchone()[0]
        
        c.execute('SELECT COUNT(*) FROM artifacts')
        total_artifacts = c.fetchone()[0]
        
        c.execute('SELECT threat_level, COUNT(*) FROM investigations GROUP BY threat_level')
        threat_distribution = {row[0]: row[1] for row in c.fetchall()}
        
        conn.close()
        
        return jsonify({
            "success": True,
            "stats": {
                "total_investigations": total_investigations,
                "total_artifacts_db": total_artifacts,
                "cache_size": len(REQUEST_CACHE),
                "cache_max": MAX_CACHE_SIZE,
                "threat_distribution": threat_distribution
            }
        })
    except:
        return jsonify({"success": True, "stats": {"total_investigations": 0}})

@donna.route('/system-info', methods=['GET'])
def system_info():
    try:
        ollama_ok = check_ollama_connection(OLLAMA_BASE_URL)
    except:
        ollama_ok = False
    
    return jsonify({
        "system": {
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "processor": platform.processor() or "Unknown"
        },
        "services": {
            "ollama": {
                "available": ollama_ok,
                "model": OLLAMA_MODEL,
                "url": OLLAMA_BASE_URL
            },
            "tor": {
                "available": TOR_AVAILABLE,
                "status": "CONNECTED" if TOR_AVAILABLE else "DISCONNECTED"
            },
            "pdf_generation": REPORTLAB_AVAILABLE,
            "langchain": LLM_LIBS_AVAILABLE
        },
        "configuration": {
            "search_engines": len(SEARCH_ENGINES),
            "max_cache_size": MAX_CACHE_SIZE,
            "cache_current_size": len(REQUEST_CACHE),
            "max_workers": 12
        },
        "database": {
            "path": DB_PATH,
            "available": os.path.exists(DB_PATH)
        }
    })

# Blueprint is registered in server.py
# This module should only define the blueprint, not run the app