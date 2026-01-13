import json
import os
import sys
import re
import logging
from io import StringIO
from flask import Blueprint, render_template, request, jsonify, g
from utils.security import rate_limit_api, validate_request, InputValidator
# Groq import removed - using centralized router
from fuzzywuzzy import fuzz
from difflib import SequenceMatcher
import time
import markdown2
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib
import threading
import concurrent.futures
from functools import lru_cache
import asyncio

# Local LLM fallback
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.local_llm_utils import generate_with_ollama, check_ollama_available
    from utils.llm_logger import log_llm_status, log_llm_request, log_llm_success, log_llm_error, log_llm_fallback, log_processing_step
    LOCAL_LLM_AVAILABLE = True
except ImportError as e:
    LOCAL_LLM_AVAILABLE = False
    # Use logging instead of print for better control
    import logging
    logging.getLogger(__name__).warning(f"Local LLM utilities not available for CyberSentry AI: {e}")
    # Create dummy functions
    def log_llm_status(*args, **kwargs): return (False, False)
    def log_llm_request(*args, **kwargs): pass
    def log_llm_success(*args, **kwargs): pass
    def log_llm_error(*args, **kwargs): pass
    def log_llm_fallback(*args, **kwargs): pass
    def log_processing_step(*args, **kwargs): pass

# Create a blueprint
cybersentry_ai = Blueprint('cybersentry_ai', __name__, template_folder='templates')

# Configure logger
_logger = logging.getLogger(__name__)

# Load responses from JSON file with caching
_responses_cache = None
_cache_time = None

def load_responses():
    """Load responses with caching mechanism"""
    global _responses_cache, _cache_time
    
    # Cache for 5 minutes
    if _responses_cache and _cache_time and (time.time() - _cache_time < 300):
        return _responses_cache
    
    try:
        # Path relative to project root
        try:
            from utils.paths import get_data_path
            responses_path = str(get_data_path('responses.json'))
        except ImportError:
            project_root = os.path.dirname(os.path.dirname(__file__))
            responses_path = os.path.join(project_root, 'data', 'responses.json')
        with open(responses_path, 'r', encoding='utf-8') as file:
            _responses_cache = json.load(file)
            _cache_time = time.time()
            return _responses_cache
    except FileNotFoundError:
        import logging
        logging.getLogger(__name__).warning("Warning: responses.json not found. Creating empty response list.")
        return []
    except json.JSONDecodeError as e:
        import logging
        logging.getLogger(__name__).error(f"Error parsing responses.json: {e}")
        return []
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error loading responses: {e}")
        return []

responses = load_responses()
if not responses:
    _logger.warning("⚠️ No responses loaded - JSON database is empty. AI fallback will be used.")

# Log LLM status at startup
try:
    log_llm_status("CyberSentry AI")
except:
    pass

# Use centralized LLM router
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.llm_router import generate_text
    LLM_ROUTER_AVAILABLE = True
except ImportError as e:
    LLM_ROUTER_AVAILABLE = False
    _logger.error(f"✗ LLM router not available: {e}")
    def generate_text(*args, **kwargs):
        return {"response": "", "model": "none", "source": "none"}

# Cache LLM responses
@lru_cache(maxsize=100)
def get_cached_response(query_hash):
    """Cache responses to avoid redundant API calls"""
    pass

def capture_output(func):
    """Decorator to capture stdout output and preserve model_used"""
    def wrapper(*args, **kwargs):
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            result = func(*args, **kwargs)
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
            
            # Extract model_used if present
            model_used = getattr(result, '_model_used', 'groq') if result else 'groq'
            
            # Return (result, output, model_used)
            return result, output, model_used
        except Exception as e:
            sys.stdout = old_stdout
            import logging
            logging.getLogger(__name__).error(f"Error in capture_output: {e}")
            return None, f"Error: {str(e)}", None
    return wrapper

def normalize_text(text):
    """Normalize text for better matching"""
    text = text.lower().strip()
    text = ' '.join(text.split())
    text = re.sub(r'[?.!,;]+$', '', text)
    return text

def calculate_similarity_score(query, question):
    """Calculate multiple similarity scores and return weighted average"""
    query_norm = normalize_text(query)
    question_norm = normalize_text(question)
    
    token_score = fuzz.token_set_ratio(query_norm, question_norm)
    partial_score = fuzz.partial_ratio(query_norm, question_norm)
    sort_score = fuzz.token_sort_ratio(query_norm, question_norm)
    seq_score = SequenceMatcher(None, query_norm, question_norm).ratio() * 100
    
    weighted_score = (token_score * 0.4 + partial_score * 0.2 + 
                     sort_score * 0.3 + seq_score * 0.1)
    
    return weighted_score

def extract_keywords(text):
    """Extract important keywords from text"""
    stop_words = {'what', 'is', 'the', 'a', 'an', 'how', 'does', 'in', 'of', 
                  'to', 'for', 'and', 'or', 'can', 'you', 'me', 'explain', 
                  'define', 'describe', 'tell'}
    
    words = normalize_text(text).split()
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    return keywords

def keyword_match_score(query, question):
    """Calculate score based on keyword matching"""
    query_keywords = set(extract_keywords(query))
    question_keywords = set(extract_keywords(question))
    
    if not query_keywords or not question_keywords:
        return 0
    
    intersection = query_keywords.intersection(question_keywords)
    union = query_keywords.union(question_keywords)
    
    return (len(intersection) / len(union)) * 100 if union else 0

def is_valid_answer(answer):
    """Validate if answer is meaningful (not just a single word like 'groq')"""
    if not answer:
        return False
    
    # Convert to string if needed and strip whitespace
    if not isinstance(answer, str):
        answer = str(answer)
    answer = answer.strip()
    
    # Empty after stripping
    if not answer:
        return False
    
    # Check if answer is just a single word (common invalid answers) - CHECK THIS FIRST
    invalid_words = ['groq', 'local', 'json', 'ai', 'llm', 'ollama', 'error', 'none', 'null', 'undefined', 'empty']
    words = answer.lower().strip().split()
    
    # Single word check - most important
    if len(words) == 1:
        if words[0] in invalid_words:
            _logger.warning(f"Answer is invalid single word: '{answer}'")
            return False
        # Also reject if it's too short (less than 3 chars) and not a valid word
        if len(words[0]) < 3:
            _logger.warning(f"Answer is too short single word: '{answer}'")
            return False
    
    # Check if answer contains only model names or technical terms without context
    if len(words) <= 2 and all(word in invalid_words for word in words):
        _logger.warning(f"Answer contains only invalid words: '{answer}'")
        return False
    
    # Check if answer is too short overall (less than 10 characters) - but allow if it's a meaningful phrase
    if len(answer) < 10:
        # Allow short answers if they have multiple words or are not in invalid list
        if len(words) > 1:
            # Multiple words but short - might be valid (e.g., "Yes, it is")
            return True
        else:
            # Single word and short - invalid
            _logger.warning(f"Answer too short: '{answer}' (length: {len(answer)})")
            return False
    
    return True

def format_ai_response(text):
    """Format AI response with markdown and enhanced styling"""
    # Convert markdown to HTML
    html = markdown2.markdown(text, extras=['fenced-code-blocks', 'tables', 'break-on-newline'])
    
    # Add custom formatting
    html = re.sub(r'\*\*(.*?)\*\*', r'<strong class="highlight">\1</strong>', html)
    html = re.sub(r'#{1,6}\s+(.*?)(?:\n|$)', r'<h3 class="section-header">\1</h3>', html)
    
    return html

@capture_output
def fuzzy_match(query, responses, threshold=70):
    """Enhanced fuzzy matching with multiple scoring methods"""
    query_clean = normalize_text(query)
    
    if not query_clean:
        _logger.warning("Empty query after normalization")
        return None
    
    if not responses:
        _logger.warning("No responses available for matching")
        return None
    
    best_match = None
    best_score = 0
    all_scores = []
    
    log_processing_step("CyberSentry AI", "fuzzy_match", "processing", f"Query: '{query[:50]}...'")
    _logger.debug(f"Searching for: '{query}' in {len(responses)} responses")
    
    for response in responses:
        if not isinstance(response, dict):
            continue
        if 'question' not in response or 'answer' not in response:
            continue
        
        question = response['question']
        if not isinstance(question, str):
            continue
            
        similarity_score = calculate_similarity_score(query, question)
        keyword_score = keyword_match_score(query, question)
        combined_score = similarity_score * 0.7 + keyword_score * 0.3
        
        all_scores.append({
            'question': question,
            'similarity': similarity_score,
            'keyword': keyword_score,
            'combined': combined_score
        })
        
        if combined_score > best_score:
            best_score = combined_score
            best_match = response
    
    all_scores.sort(key=lambda x: x['combined'], reverse=True)
    top_matches = all_scores[:3]
    _logger.debug("Top 3 Matches:")
    for i, score_data in enumerate(top_matches, 1):
        _logger.debug(f"{i}. '{score_data['question']}' - Combined: {score_data['combined']:.2f} | Similarity: {score_data['similarity']:.2f} | Keyword: {score_data['keyword']:.2f}")
    
    if best_score >= threshold and best_match:
        answer = best_match.get('answer')
        if isinstance(answer, dict):
            # Convert dict answer to string
            answer_parts = []
            for key, value in answer.items():
                if isinstance(value, dict):
                    if 'command' in value and 'description' in value:
                        answer_parts.append(f"{key}: {value.get('command', '')} - {value.get('description', '')}")
                    else:
                        answer_parts.append(f"{key}: {str(value)}")
                else:
                    answer_parts.append(f"{key}: {value}")
            answer = '\n'.join(answer_parts)
        elif not isinstance(answer, str):
            answer = str(answer) if answer else None
            
        if answer:
            # Validate answer before returning
            _logger.debug(f"[VALIDATION] Checking answer: '{answer}' (type: {type(answer)}, length: {len(str(answer))})")
            if is_valid_answer(answer):
                log_processing_step("CyberSentry AI", "fuzzy_match", "success", f"Match found (score: {best_score:.2f})")
                _logger.info(f"Match found with score: {best_score:.2f} - Question: '{best_match['question']}' - Answer: '{answer[:50]}...'")
                return answer
            else:
                _logger.warning(f"[VALIDATION FAILED] Match found but answer is invalid: '{answer}' - treating as no match")
                log_processing_step("CyberSentry AI", "fuzzy_match", "error", f"Invalid answer: '{answer}'")
                return None
        else:
            _logger.warning(f"Match found but answer is empty")
            return None
    else:
        log_processing_step("CyberSentry AI", "fuzzy_match", "error", f"No match (best score: {best_score:.2f}, threshold: {threshold})")
        _logger.warning(f"No match found (best score: {best_score:.2f}, threshold: {threshold})")
        return None

@capture_output
def get_groq_response(query):
    """Get response using centralized LLM router. Returns (answer, output, model_used)"""
    if not LLM_ROUTER_AVAILABLE:
        _logger.error("✗ LLM router not available")
        return None
    
    try:
        system_prompt = """You are CyberSentry AI, an advanced cybersecurity assistant specializing in:
- Ethical hacking and penetration testing
- Network security and vulnerability assessment
- Security tools (Nmap, Metasploit, Wireshark, Burp Suite, etc.)
- Threat analysis and mitigation strategies
- Secure coding practices
- Compliance and security frameworks

FORMAT YOUR RESPONSE WITH:
- Clear section headings using **bold text**
- Bullet points for lists
- Code blocks for commands (use ```language ```)
- Step-by-step numbered instructions when applicable
- Key terms in **bold**
- Important warnings or notes highlighted

Provide clear, educational, and actionable information while adhering to ethical and legal standards."""
        
        log_processing_step("CyberSentry AI", "llm_request", "processing", f"Query: '{query[:50]}...'")
        log_llm_request("CyberSentry AI", "cloud", len(query))
        start_time = time.time()
        
        result = generate_text(
            prompt=query,
            app_name="cybersentry_ai",
            task_type="security_analysis",
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=2048
        )
        
        latency_ms = (time.time() - start_time) * 1000
        response_text = result.get("response", "")
        source = result.get("source", "none")
        model_used = result.get("model", "unknown")
        
        if response_text:
            log_llm_success("CyberSentry AI", source, len(response_text), latency_ms)
            log_processing_step("CyberSentry AI", "llm_request", "success", f"Response received ({len(response_text)} chars, source: {source})")
            _logger.info(f"✓ LLM response received ({len(response_text)} chars, {model_used}, {source})")
            formatted_response = format_ai_response(response_text.strip())
            formatted_response._model_used = model_used
            return formatted_response
        else:
            log_llm_error("CyberSentry AI", source, Exception("Empty response"), fallback=False)
            _logger.warning("✗ Empty response from LLM")
            return None
            
    except Exception as e:
        _logger.error(f"✗ Error fetching response from LLM router: {e}", exc_info=True)
        log_llm_error("CyberSentry AI", "router", e, fallback=False)
        return None

@capture_output
def get_local_llm_response(query):
    """Get response from local LLM using centralized router. Returns (answer, output, model_used)"""
    if not LLM_ROUTER_AVAILABLE:
        _logger.debug("LLM router not available")
        return None
    
    try:
        system_prompt = """You are CyberSentry AI, an advanced cybersecurity assistant specializing in:
- Ethical hacking and penetration testing
- Network security and vulnerability assessment
- Security tools (Nmap, Metasploit, Wireshark, Burp Suite, etc.)
- Threat analysis and mitigation strategies
- Secure coding practices
- Compliance and security frameworks

FORMAT YOUR RESPONSE WITH:
- Clear section headings using **bold text**
- Bullet points for lists
- Code blocks for commands (use ```language ```)
- Step-by-step numbered instructions when applicable
- Key terms in **bold**
- Important warnings or notes highlighted

Provide clear, educational, and actionable information while adhering to ethical and legal standards."""
        
        log_processing_step("CyberSentry AI", "local_llm_request", "processing", f"Query: '{query[:50]}...'")
        _logger.info("[LOCAL LLM] Trying local LLM first...")
        
        log_llm_request("CyberSentry AI", "local", len(query))
        start_time = time.time()
        
        result = generate_text(
            prompt=query,
            app_name="cybersentry_ai",
            task_type="security_analysis",
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=2048,
            prefer_local=True
        )
        
        latency_ms = (time.time() - start_time) * 1000
        local_result = result.get("response", "")
        source = result.get("source", "none")
        model_used = result.get("model", "unknown")
        
        if local_result and local_result.strip():
            log_llm_success("CyberSentry AI", "local", len(local_result), latency_ms)
            log_processing_step("CyberSentry AI", "local_llm_request", "success", f"Response received ({len(local_result)} chars)")
            _logger.info(f"✓ Local LLM response received ({len(local_result)} chars)")
            formatted_response = format_ai_response(local_result.strip())
            formatted_response._model_used = "local"
            return formatted_response
        else:
            log_llm_error("CyberSentry AI", "local", Exception("Local LLM returned empty"), fallback=True)
            _logger.warning("✗ Local LLM returned empty or failed")
            return None
            
    except Exception as e:
        log_llm_error("CyberSentry AI", "local", e, fallback=True)
        _logger.error(f"✗ Error with local LLM: {e}", exc_info=True)
        return None

def get_local_fallback(full_prompt):
    """Helper function to get local LLM fallback"""
    log_llm_fallback("CyberSentry AI", "cloud", "local")
    _logger.warning("⚠ Falling back to local Ollama...")
    
    log_llm_request("CyberSentry AI", "local", len(full_prompt))
    start_time = time.time()
    local_result, success = generate_with_ollama(
        full_prompt,
        system_prompt="You are CyberSentry AI, an expert cybersecurity assistant. Provide accurate security guidance.",
        temperature=0.7,
        max_tokens=2048
    )
    latency_ms = (time.time() - start_time) * 1000
    
    if success and local_result:
        log_llm_success("CyberSentry AI", "local", len(local_result), latency_ms)
        _logger.info("✓ Successfully used local Ollama model")
        formatted_response = format_ai_response(local_result.strip())
        formatted_response._model_used = "local"
        return formatted_response
    else:
        log_llm_error("CyberSentry AI", "local", Exception("Local LLM returned empty"), fallback=False)
        _logger.error("✗ Local LLM fallback also failed")
        return None

@cybersentry_ai.route('/')
def index():
    """Render the main chat interface"""
    return render_template('cybersentry_AI.html')

@cybersentry_ai.route('/ask', methods=['POST'])
@rate_limit_api(requests_per_minute=20, requests_per_hour=200)
@validate_request({
    "question": {
        "type": "string",
        "required": True,
        "max_length": 2000
    },
    "force_source": {
        "type": "string",
        "required": False,
        "max_length": 10,
        "allowed_values": ["json", "ai", None]
    }
}, strict=True)
def ask():
    """
    Handle question requests with enhanced processing
    OWASP: Rate limited, input validated
    """
    try:
        # Get validated data from request context
        data = g.validated_data
        question = InputValidator.validate_string(
            data.get('question'), 'question', max_length=2000, required=True
        )
        force_source = data.get('force_source', None)
        
        _logger.info(f"{'='*60}")
        _logger.info(f"[{time.strftime('%H:%M:%S')}] New Question: {question}")
        if force_source:
            _logger.info(f"[REGENERATE] Forcing source: {force_source}")
        _logger.info('='*60)
        
        # Handle regeneration with forced source
        if force_source == 'ai':
            log_processing_step("CyberSentry AI", "forced_ai", "processing", f"Question: '{question[:50]}...'")
            # Use LLM router with automatic fallback (tries Groq first, then local)
            system_prompt = """You are CyberSentry AI, an advanced cybersecurity assistant specializing in:
- Ethical hacking and penetration testing
- Network security and vulnerability assessment
- Security tools (Nmap, Metasploit, Wireshark, Burp Suite, etc.)
- Threat analysis and mitigation strategies
- Secure coding practices
- Compliance and security frameworks

FORMAT YOUR RESPONSE WITH:
- Clear section headings using **bold text**
- Bullet points for lists
- Code blocks for commands (use ```language ```)
- Step-by-step numbered instructions when applicable
- Key terms in **bold**
- Important warnings or notes highlighted

Provide clear, educational, and actionable information while adhering to ethical and legal standards."""
            
            if LLM_ROUTER_AVAILABLE:
                log_llm_request("CyberSentry AI", "auto", len(question))
                start_time = time.time()
                result = generate_text(
                    prompt=question,
                    app_name="cybersentry_ai",
                    task_type="security_analysis",
                    system_prompt=system_prompt,
                    temperature=0.7,
                    max_tokens=2048
                )
                latency_ms = (time.time() - start_time) * 1000
                ai_answer = result.get("response", "")
                source = result.get("source", "none")
                model_used = result.get("model", "unknown")
                
                if ai_answer:
                    log_llm_success("CyberSentry AI", source, len(ai_answer), latency_ms)
                    formatted_answer = format_ai_response(ai_answer.strip())
                    log_processing_step("CyberSentry AI", "forced_ai", "success", f"Response generated (model: {model_used}, source: {source})")
                    return jsonify({
                        'answer': formatted_answer,
                        'source': 'AI',
                        'terminal_output': '',
                        'confidence': 'medium',
                        'can_regenerate': True,
                        'model_used': model_used
                    })
            
            log_processing_step("CyberSentry AI", "forced_ai", "error", "No response generated from LLM router")
        
        elif force_source == 'json':
            log_processing_step("CyberSentry AI", "forced_json", "processing", f"Question: '{question[:50]}...'")
            result = fuzzy_match(question, responses, threshold=60)
            # fuzzy_match has @capture_output decorator, returns (answer, output, model_used)
            if isinstance(result, tuple) and len(result) >= 3:
                answer, json_output, _ = result
            elif isinstance(result, tuple) and len(result) >= 1:
                answer = result[0] if len(result) > 0 else None
                json_output = result[1] if len(result) > 1 else ""
            else:
                answer = result
                json_output = ""
            
            # Ensure answer is a string, not a tuple or other type
            if isinstance(answer, tuple):
                _logger.warning(f"[FORCED JSON] Answer is a tuple, extracting first element: {answer}")
                answer = answer[0] if len(answer) > 0 else None
            if answer is not None and not isinstance(answer, str):
                _logger.warning(f"[FORCED JSON] Answer is not a string (type: {type(answer)}), converting: {answer}")
                answer = str(answer) if answer else None
            
            # Validate JSON answer even when forced
            if answer and isinstance(answer, str) and is_valid_answer(answer):
                log_processing_step("CyberSentry AI", "forced_json", "success", "Match found")
                return jsonify({
                    'answer': answer,
                    'source': 'JSON',
                    'terminal_output': json_output,
                    'confidence': 'high',
                    'can_regenerate': True,
                    'model_used': 'json'
                })
            elif answer:
                # Invalid answer from JSON, log warning but still return it (user forced JSON)
                # Ensure answer is string
                if isinstance(answer, tuple):
                    answer = answer[0] if len(answer) > 0 else ''
                if not isinstance(answer, str):
                    answer = str(answer) if answer else ''
                answer_str = str(answer) if not isinstance(answer, str) else answer
                _logger.warning(f"[FORCED JSON] Invalid answer found: '{answer_str}' - but user forced JSON source")
                log_processing_step("CyberSentry AI", "forced_json", "warning", f"Invalid answer: '{answer_str}'")
                # Still return it since user forced JSON, but log the issue
                return jsonify({
                    'answer': answer_str,
                    'source': 'JSON',
                    'terminal_output': json_output,
                    'confidence': 'low',
                    'can_regenerate': True,
                    'model_used': 'json'
                })
            else:
                log_processing_step("CyberSentry AI", "forced_json", "error", "No match found")
        
        # Normal flow: Try JSON first
        log_processing_step("CyberSentry AI", "normal_flow", "processing", f"Question: '{question[:50]}...'")
        result = fuzzy_match(question, responses, threshold=70)
        # fuzzy_match has @capture_output decorator, returns (answer, output, model_used)
        if isinstance(result, tuple) and len(result) >= 3:
            answer, json_output, _ = result
        elif isinstance(result, tuple) and len(result) >= 1:
            answer = result[0] if len(result) > 0 else None
            json_output = result[1] if len(result) > 1 else ""
        else:
            answer = result
            json_output = ""
        
        # Ensure answer is a string, not a tuple or other type
        if isinstance(answer, tuple):
            _logger.warning(f"[NORMAL FLOW] Answer is a tuple, extracting first element: {answer}")
            answer = answer[0] if len(answer) > 0 else None
        if answer is not None and not isinstance(answer, str):
            _logger.warning(f"[NORMAL FLOW] Answer is not a string (type: {type(answer)}), converting: {answer}")
            answer = str(answer) if answer else None
        
        # Validate JSON answer - reject if invalid (e.g., single word "groq")
        _logger.debug(f"[NORMAL FLOW] JSON result: answer='{answer}', type={type(answer)}")
        if answer:
            _logger.debug(f"[VALIDATION] Validating JSON answer: '{answer}'")
            is_valid = is_valid_answer(answer)
            _logger.debug(f"[VALIDATION] Result: {is_valid}")
            
        if answer and isinstance(answer, str) and is_valid_answer(answer):
            log_processing_step("CyberSentry AI", "normal_flow", "success", "JSON match found")
            answer_preview = answer[:50] if len(answer) > 50 else answer
            _logger.info(f"[RESULT] Using JSON database response: '{answer_preview}...'")
            
            # Threat analysis - ensure answer is string
            threat_data = threat_analyzer.analyze_threat_level(question, str(answer))
            
            # Save to database
            user_id = request.remote_addr or 'anonymous'
            try:
                with sqlite3.connect(db_manager.db_name) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO conversations 
                        (user_id, question, answer, source, model_used, confidence, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (user_id, question, str(answer), 'JSON', 'json', 'high', datetime.now().isoformat()))
                    conn.commit()
            except Exception as e:
                _logger.warning(f"Failed to save conversation: {e}")
            
            return jsonify({
                'answer': str(answer),  # Ensure answer is string
                'source': 'JSON',
                'terminal_output': json_output,
                'confidence': 'high',
                'can_regenerate': True,
                'model_used': 'json',
                'threat_analysis': threat_data
            })
        elif answer:
            # Invalid answer from JSON (e.g., just "groq"), skip to Local LLM
            # Ensure answer is string for logging
            answer_str = str(answer) if not isinstance(answer, str) else answer
            _logger.warning(f"[JSON REJECTED] Invalid answer found: '{answer_str}' (length: {len(answer_str)}) - skipping JSON and using Local LLM")
            log_processing_step("CyberSentry AI", "normal_flow", "processing", f"JSON answer invalid: '{answer_str}' - trying Local LLM")
            # Continue to Local LLM below
        
        # Step 2: Use LLM router (tries Groq first, automatically falls back to local)
        log_processing_step("CyberSentry AI", "normal_flow", "processing", "Trying LLM router (Groq -> Ollama fallback)...")
        _logger.info("[STEP 2] Using LLM router with automatic fallback...")
        
        if LLM_ROUTER_AVAILABLE:
            system_prompt = """You are CyberSentry AI, an advanced cybersecurity assistant specializing in:
- Ethical hacking and penetration testing
- Network security and vulnerability assessment
- Security tools (Nmap, Metasploit, Wireshark, Burp Suite, etc.)
- Threat analysis and mitigation strategies
- Secure coding practices
- Compliance and security frameworks

FORMAT YOUR RESPONSE WITH:
- Clear section headings using **bold text**
- Bullet points for lists
- Code blocks for commands (use ```language ```)
- Step-by-step numbered instructions when applicable
- Key terms in **bold**
- Important warnings or notes highlighted

Provide clear, educational, and actionable information while adhering to ethical and legal standards."""
            
            log_llm_request("CyberSentry AI", "auto", len(question))
            start_time = time.time()
            result = generate_text(
                prompt=question,
                app_name="cybersentry_ai",
                task_type="security_analysis",
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=2048
            )
            latency_ms = (time.time() - start_time) * 1000
            ai_answer = result.get("response", "")
            source = result.get("source", "none")
            model_used = result.get("model", "unknown")
            
            if ai_answer:
                log_llm_success("CyberSentry AI", source, len(ai_answer), latency_ms)
                formatted_answer = format_ai_response(ai_answer.strip())
                log_processing_step("CyberSentry AI", "normal_flow", "success", f"LLM response generated (model: {model_used}, source: {source})")
                _logger.info(f"[RESULT] Using LLM response (source: {source}, model: {model_used})")
                
                # Threat analysis
                threat_data = threat_analyzer.analyze_threat_level(question, ai_answer)
                
                # Save to database
                user_id = request.remote_addr or 'anonymous'
                try:
                    with sqlite3.connect(db_manager.db_name) as conn:
                        cursor = conn.cursor()
                        cursor.execute('''
                            INSERT INTO conversations 
                            (user_id, question, answer, source, model_used, confidence, timestamp, response_time)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (user_id, question, ai_answer, 'AI', model_used, 'medium', datetime.now().isoformat(), latency_ms))
                        conn.commit()
                except Exception as e:
                    _logger.warning(f"Failed to save conversation: {e}")
                
                return jsonify({
                    'answer': formatted_answer,
                    'source': 'AI',
                    'terminal_output': '',
                    'confidence': 'medium',
                    'can_regenerate': True,
                    'model_used': model_used,
                    'threat_analysis': threat_data
                })
        
        # Final fallback
        _logger.info("[FALLBACK] Using default response")
        fallback_answer = format_ai_response("""I don't have specific information about that topic in my knowledge base. 

**🔒 Security Best Practices:**
- Keep systems and software updated
- Use strong, unique passwords with MFA
- Implement network segmentation
- Regular security audits and monitoring
- Follow the principle of least privilege

**💡 Try asking about:**
- Common security tools (Nmap, Wireshark, Metasploit)
- Attack types (DDoS, SQL injection, XSS)
- Security concepts (encryption, firewalls, VPNs)
- Penetration testing methodologies""")
        
        return jsonify({
            'answer': fallback_answer,
            'source': 'Fallback',
            'terminal_output': '',
            'confidence': 'low',
            'can_regenerate': False,
            'model_used': 'none'
        })
        
    except ValueError as e:
        error_msg = f"Validation error: {str(e)}"
        _logger.error(f"\n[VALIDATION ERROR] {error_msg}")
        log_processing_step("CyberSentry AI", "ask", "error", f"Validation: {error_msg}")
        return jsonify({
            'error': error_msg,
            'terminal_output': ''
        }), 400
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        _logger.error(f"\n[ERROR] {error_msg}", exc_info=True)
        log_processing_step("CyberSentry AI", "ask", "error", f"Exception: {error_msg}")
        return jsonify({
            'error': error_msg,
            'terminal_output': ''
        }), 500

@cybersentry_ai.route('/reload-responses', methods=['POST'])
@rate_limit_api(requests_per_minute=5, requests_per_hour=20)  # Strict limit for admin operations
def reload_responses():
    """
    Reload responses from JSON file
    OWASP: Rate limited
    """
    """Endpoint to reload responses.json without restarting server"""
    global responses
    responses = load_responses()
    return jsonify({
        'message': f'Responses reloaded successfully. Total responses: {len(responses)}'
    })

@cybersentry_ai.route('/stats', methods=['GET'])
def stats():
    """Get statistics about the response database"""
    return jsonify({
        'total_responses': len(responses),
        'categories': list(set(r.get('category', 'uncategorized') for r in responses if isinstance(r, dict)))
    })

# Database Manager for Advanced Features
class CyberSentryDatabase:
    """Database manager for conversation history, threat intelligence, and analytics."""
    
    def __init__(self, db_name='cybersentry_ai.db'):
        self.db_name = db_name
        self.init_db()
    
    def init_db(self):
        """Initialize database with proper error handling"""
        try:
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()
                
                # Conversation history
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS conversations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT,
                        question TEXT,
                        answer TEXT,
                        source TEXT,
                        model_used TEXT,
                        confidence TEXT,
                        timestamp DATETIME,
                        response_time REAL
                    )
                ''')
                
                # Threat intelligence cache
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS threat_intel (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        query_hash TEXT UNIQUE,
                        query TEXT,
                        threat_level TEXT,
                        risk_score INTEGER,
                        indicators TEXT,
                        recommendations TEXT,
                        timestamp DATETIME
                    )
                ''')
                
                # Security analytics
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS analytics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_type TEXT,
                        event_data TEXT,
                        timestamp DATETIME
                    )
                ''')
                
                # User feedback
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS feedback (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        conversation_id INTEGER,
                        helpful BOOLEAN,
                        feedback_text TEXT,
                        timestamp DATETIME
                    )
                ''')
                
                # Create indexes
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_id ON conversations(user_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON conversations(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_query_hash ON threat_intel(query_hash)')
                
                conn.commit()
            _logger.info("✓ CyberSentry AI database initialized")
        except sqlite3.Error as e:
            _logger.error(f"⚠ Database initialization error: {e}")

# Initialize database
db_manager = CyberSentryDatabase()

# Threat Intelligence Analyzer
class ThreatIntelligenceAnalyzer:
    """Advanced threat intelligence and security analysis."""
    
    def __init__(self):
        self.threat_keywords = {
            'critical': ['exploit', 'vulnerability', 'breach', 'attack', 'malware', 'ransomware', 'phishing', 'ddos'],
            'high': ['security', 'threat', 'risk', 'compromise', 'intrusion', 'backdoor', 'trojan'],
            'medium': ['warning', 'alert', 'suspicious', 'anomaly', 'unusual'],
            'low': ['information', 'guide', 'tutorial', 'explanation']
        }
    
    def analyze_threat_level(self, query, answer):
        """Analyze threat level based on query and answer content."""
        query_lower = query.lower() if isinstance(query, str) else str(query).lower()
        # Ensure answer is a string
        if isinstance(answer, tuple):
            answer = answer[0] if len(answer) > 0 else ''
        if not isinstance(answer, str):
            answer = str(answer) if answer else ''
        answer_lower = answer.lower() if answer else ''
        combined = query_lower + ' ' + answer_lower
        
        threat_score = 0
        indicators = []
        
        for level, keywords in self.threat_keywords.items():
            matches = [kw for kw in keywords if kw in combined]
            if matches:
                if level == 'critical':
                    threat_score += 30
                elif level == 'high':
                    threat_score += 20
                elif level == 'medium':
                    threat_score += 10
                indicators.extend(matches)
        
        if threat_score >= 30:
            threat_level = 'CRITICAL'
        elif threat_score >= 20:
            threat_level = 'HIGH'
        elif threat_score >= 10:
            threat_level = 'MEDIUM'
        else:
            threat_level = 'LOW'
        
        return {
            'threat_level': threat_level,
            'risk_score': min(threat_score, 100),
            'indicators': list(set(indicators)),
            'recommendations': self._generate_recommendations(threat_level)
        }
    
    def _generate_recommendations(self, threat_level):
        """Generate security recommendations based on threat level."""
        recommendations = {
            'CRITICAL': [
                'Immediate action required',
                'Isolate affected systems',
                'Notify security team',
                'Review security logs',
                'Implement emergency patches'
            ],
            'HIGH': [
                'Review security configurations',
                'Update security policies',
                'Monitor network traffic',
                'Conduct security audit'
            ],
            'MEDIUM': [
                'Review security best practices',
                'Update documentation',
                'Schedule security review'
            ],
            'LOW': [
                'Continue monitoring',
                'Maintain security hygiene'
            ]
        }
        return recommendations.get(threat_level, [])

# Initialize threat analyzer
threat_analyzer = ThreatIntelligenceAnalyzer()

@cybersentry_ai.route('/threat-analysis', methods=['POST'])
@rate_limit_api(requests_per_minute=10, requests_per_hour=100)
@validate_request({
    "query": {
        "type": "string",
        "required": True,
        "max_length": 2000
    }
}, strict=True)
def threat_analysis():
    """Advanced threat intelligence analysis endpoint."""
    try:
        data = g.validated_data
        query = InputValidator.validate_string(data.get('query'), 'query', max_length=2000, required=True)
        
        # Check cache
        query_hash = hashlib.md5(query.encode()).hexdigest()
        with sqlite3.connect(db_manager.db_name) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM threat_intel WHERE query_hash = ?', (query_hash,))
            cached = cursor.fetchone()
            
            if cached:
                return jsonify({
                    'threat_level': cached['threat_level'],
                    'risk_score': cached['risk_score'],
                    'indicators': json.loads(cached['indicators']),
                    'recommendations': json.loads(cached['recommendations']),
                    'cached': True
                })
        
        # Get answer first
        result = fuzzy_match(query, responses, threshold=70)
        if isinstance(result, tuple):
            answer = result[0] if len(result) > 0 else None
        else:
            answer = result
        
        if not answer or not is_valid_answer(answer):
            # Try LLM
            if LLM_ROUTER_AVAILABLE:
                system_prompt = "You are a cybersecurity expert. Provide a brief security analysis."
                result = generate_text(
                    prompt=query,
                    app_name="cybersentry_ai",
                    task_type="security_analysis",
                    system_prompt=system_prompt,
                    temperature=0.7,
                    max_tokens=1024
                )
                answer = result.get("response", "")
        
        # Analyze threat
        analysis = threat_analyzer.analyze_threat_level(query, answer or '')
        
        # Cache result
        with sqlite3.connect(db_manager.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO threat_intel 
                (query_hash, query, threat_level, risk_score, indicators, recommendations, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                query_hash, query, analysis['threat_level'], analysis['risk_score'],
                json.dumps(analysis['indicators']), json.dumps(analysis['recommendations']),
                datetime.now().isoformat()
            ))
            conn.commit()
        
        return jsonify({
            **analysis,
            'cached': False
        })
    except Exception as e:
        _logger.error(f"Threat analysis error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@cybersentry_ai.route('/analytics', methods=['GET'])
def get_analytics():
    """Get security analytics and statistics."""
    try:
        with sqlite3.connect(db_manager.db_name) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Conversation stats
            cursor.execute('SELECT COUNT(*) as total FROM conversations')
            total_conversations = cursor.fetchone()['total']
            
            # Source distribution
            cursor.execute('''
                SELECT source, COUNT(*) as count 
                FROM conversations 
                GROUP BY source
            ''')
            source_dist = {row['source']: row['count'] for row in cursor.fetchall()}
            
            # Threat level distribution
            cursor.execute('''
                SELECT threat_level, COUNT(*) as count 
                FROM threat_intel 
                GROUP BY threat_level
            ''')
            threat_dist = {row['threat_level']: row['count'] for row in cursor.fetchall()}
            
            # Recent activity
            cursor.execute('''
                SELECT COUNT(*) as count 
                FROM conversations 
                WHERE timestamp > datetime('now', '-24 hours')
            ''')
            recent_activity = cursor.fetchone()['count']
            
            return jsonify({
                'total_conversations': total_conversations,
                'source_distribution': source_dist,
                'threat_distribution': threat_dist,
                'recent_activity_24h': recent_activity
            })
    except Exception as e:
        _logger.error(f"Analytics error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@cybersentry_ai.route('/execute-code', methods=['POST'])
@rate_limit_api(requests_per_minute=5, requests_per_hour=20)
@validate_request({
    "code": {
        "type": "string",
        "required": True,
        "max_length": 5000
    },
    "language": {
        "type": "string",
        "required": True,
        "allowed_values": ["python", "bash", "javascript"]
    }
}, strict=True)
def execute_code():
    """
    Execute code snippets in sandboxed environment
    WARNING: This is for demonstration only - use proper sandboxing in production
    """
    try:
        data = g.validated_data
        code = data.get('code')
        language = data.get('language')
        
        # For security, only allow specific safe operations
        # In production, use Docker containers or proper sandboxing
        
        if language == 'python':
            # Simple safe execution (very limited)
            import subprocess
            import tempfile
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            try:
                result = subprocess.run(
                    ['python3', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                output = result.stdout if result.returncode == 0 else result.stderr
                return jsonify({
                    'success': result.returncode == 0,
                    'output': output,
                    'language': language
                })
            finally:
                import os
                os.unlink(temp_file)
        
        return jsonify({
            'success': False,
            'output': f'Language {language} not supported yet',
            'language': language
        })
        
    except Exception as e:
        _logger.error(f"Code execution error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'output': str(e),
            'language': language
        }), 500

@cybersentry_ai.route('/history', methods=['GET'])
def get_history():
    """Get conversation history."""
    try:
        user_id = request.args.get('user_id', 'default')
        limit = request.args.get('limit', 50, type=int)
        
        with sqlite3.connect(db_manager.db_name) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM conversations 
                WHERE user_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (user_id, limit))
            
            history = [dict(row) for row in cursor.fetchall()]
            return jsonify({'history': history, 'count': len(history)})
    except Exception as e:
        _logger.error(f"History error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@cybersentry_ai.route('/ask-all', methods=['POST'])
@rate_limit_api(requests_per_minute=10, requests_per_hour=50)
@validate_request({
    "question": {
        "type": "string",
        "required": True,
        "max_length": 2000
    }
}, strict=True)
def ask_all():
    """
    Get responses from all sources in PARALLEL for faster results
    """
    try:
        data = g.validated_data
        question = InputValidator.validate_string(
            data.get('question'), 'question', max_length=2000, required=True
        )
        
        _logger.info(f"[ASK-ALL] Parallel processing for: '{question[:50]}...'")
        
        results = {
            'json': None,
            'cloud_llm': None,
            'local_llm': None
        }
        
        # Use ThreadPoolExecutor for parallel execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all tasks simultaneously
            future_json = executor.submit(get_json_response, question)
            future_cloud = executor.submit(get_cloud_llm_response_fast, question)
            future_local = executor.submit(get_local_llm_response_fast, question)
            
            # Wait for all with timeout
            try:
                results['json'] = future_json.result(timeout=5)
            except Exception as e:
                _logger.error(f"JSON error: {e}")
                results['json'] = {'answer': 'Timeout', 'status': 'error'}
            
            try:
                results['cloud_llm'] = future_cloud.result(timeout=10)
            except Exception as e:
                _logger.error(f"Cloud LLM error: {e}")
                results['cloud_llm'] = {'answer': 'Timeout', 'status': 'error'}
            
            try:
                results['local_llm'] = future_local.result(timeout=15)  # Increased timeout for local LLM
            except concurrent.futures.TimeoutError:
                _logger.warning("Local LLM timeout - may still be processing")
                results['local_llm'] = {'answer': 'Processing timeout - Local LLM may still be running', 'status': 'timeout'}
            except Exception as e:
                _logger.error(f"Local LLM error: {e}", exc_info=True)
                results['local_llm'] = {'answer': f'Error: {str(e)}', 'status': 'error'}
        
        return jsonify({
            'question': question,
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        _logger.error(f"Ask-all error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

def get_json_response(question):
    """Fast JSON lookup"""
    try:
        json_result = fuzzy_match(question, responses, threshold=60)
        # fuzzy_match returns (answer, output, model_used) because of @capture_output
        if isinstance(json_result, tuple):
             # Try to extract the answer part
            json_answer = json_result[0] if len(json_result) > 0 else None
        else:
            json_answer = json_result
        
        if json_answer and isinstance(json_answer, str) and is_valid_answer(json_answer):
            return {
                'answer': json_answer,
                'source': 'JSON',
                'model_used': 'json',
                'confidence': 'high',
                'status': 'success'
            }
        else:
            return {
                'answer': 'No match found in knowledge base',
                'source': 'JSON',
                'status': 'no_match'
            }
    except Exception as e:
        return {'answer': f'Error: {str(e)}', 'status': 'error'}

def get_cloud_llm_response_fast(question):
    """Fast cloud LLM with reduced tokens"""
    if not LLM_ROUTER_AVAILABLE:
        return {'answer': 'Unavailable', 'status': 'unavailable'}
    
    try:
        start_time = time.time()
        result = generate_text(
            prompt=question,
            app_name="cybersentry_ai",
            task_type="security_analysis",
            system_prompt="You are CyberSentry AI. Provide concise security guidance.",
            temperature=0.7,
            max_tokens=1024,  # Reduced from 2048
            prefer_local=False
        )
        latency_ms = (time.time() - start_time) * 1000
        
        cloud_answer = result.get("response", "")
        if cloud_answer and result.get("source") != "local":
            return {
                'answer': format_ai_response(cloud_answer.strip()),
                'source': 'Cloud LLM',
                'model_used': result.get("model", "unknown"),
                'status': 'success',
                'latency_ms': latency_ms
            }
        return {'answer': 'Unavailable', 'status': 'unavailable'}
    except Exception as e:
        return {'answer': f'Error: {str(e)}', 'status': 'error'}

def get_local_llm_response_fast(question):
    """Fast local LLM with reduced tokens - ensures local execution"""
    if not LLM_ROUTER_AVAILABLE:
        return {'answer': 'LLM router unavailable', 'status': 'unavailable'}
    
    try:
        _logger.info("[LOCAL LLM] Attempting local LLM generation...")
        start_time = time.time()
        
        # Force local execution by explicitly checking and using local fallback if needed
        result = generate_text(
            prompt=question,
            app_name="cybersentry_ai",
            task_type="security_analysis",
            system_prompt="You are CyberSentry AI. Provide concise security guidance.",
            temperature=0.7,
            max_tokens=1024,  # Reduced from 2048
            prefer_local=True
        )
        
        latency_ms = (time.time() - start_time) * 1000
        source = result.get("source", "none")
        local_answer = result.get("response", "")
        
        _logger.info(f"[LOCAL LLM] Result - source: {source}, has_answer: {bool(local_answer)}, latency: {latency_ms:.0f}ms")
        
        # Accept if source is local OR if we got a response (router may not always set source correctly)
        if local_answer and (source == "local" or source == "ollama" or len(local_answer) > 10):
            return {
                'answer': format_ai_response(local_answer.strip()),
                'source': 'Local LLM',
                'model_used': result.get("model", "local-unknown"),
                'status': 'success',
                'latency_ms': latency_ms
            }
        
        # If no response but source indicates local was attempted, return timeout
        if source in ["local", "ollama"]:
            return {'answer': 'Local LLM processing - response pending', 'status': 'processing'}
        
        return {'answer': 'Local LLM unavailable or not responding', 'status': 'unavailable'}
    except Exception as e:
        _logger.error(f"[LOCAL LLM] Exception: {e}", exc_info=True)
        return {'answer': f'Error: {str(e)}', 'status': 'error'}

# Blueprint is registered in server.py
# This function is kept for backward compatibility but does nothing
def init_app(app):
    """Legacy function - blueprint is registered in server.py"""
    pass