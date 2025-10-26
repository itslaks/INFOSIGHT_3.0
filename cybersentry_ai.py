import json
import sys
import re
from io import StringIO
from flask import Blueprint, render_template, request, jsonify
import google.generativeai as genai
from fuzzywuzzy import fuzz
from difflib import SequenceMatcher
import time
import markdown2

# Create a blueprint
cybersentry_ai = Blueprint('cybersentry_ai', __name__, template_folder='templates')

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
        with open('responses.json', 'r', encoding='utf-8') as file:
            _responses_cache = json.load(file)
            _cache_time = time.time()
            return _responses_cache
    except FileNotFoundError:
        print("Warning: responses.json not found. Creating empty response list.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing responses.json: {e}")
        return []
    except Exception as e:
        print(f"Error loading responses: {e}")
        return []

responses = load_responses()

# Configure Gemini API
API_KEY = 'AIzaSyAL0EJGSp-g7rhuBwQpgk8T95llLa6kq1c'  # Replace with your API key
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

def capture_output(func):
    """Decorator to capture stdout output"""
    def wrapper(*args, **kwargs):
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            result = func(*args, **kwargs)
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
            return result, output
        except Exception as e:
            sys.stdout = old_stdout
            print(f"Error in capture_output: {e}")
            return None, f"Error: {str(e)}"
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
        return None
    
    best_match = None
    best_score = 0
    all_scores = []
    
    print(f"\n=== Searching for: '{query}' ===")
    
    for response in responses:
        if 'question' not in response or 'answer' not in response:
            continue
        
        question = response['question']
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
    print("\nTop 3 Matches:")
    for i, score_data in enumerate(all_scores[:3], 1):
        print(f"{i}. '{score_data['question']}'")
        print(f"   Combined: {score_data['combined']:.2f} | "
              f"Similarity: {score_data['similarity']:.2f} | "
              f"Keyword: {score_data['keyword']:.2f}")
    
    if best_score >= threshold:
        print(f"\n✓ Match found with score: {best_score:.2f}")
        print(f"  Question: '{best_match['question']}'")
        return best_match.get('answer')
    else:
        print(f"\n✗ No match found (best score: {best_score:.2f}, threshold: {threshold})")
        return None

@capture_output
def get_gemini_response(query):
    """Get response from Gemini AI with improved formatting context"""
    try:
        context = """You are CyberSentry AI, an advanced cybersecurity assistant specializing in:
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
        
        full_prompt = f"{context}\n\nUser Question: {query}\n\nAssistant:"
        
        print(f"\n=== Querying Gemini AI ===")
        print(f"Query: '{query}'")
        
        response = model.generate_content(full_prompt)
        
        if response and response.text:
            print(f"✓ Gemini response received ({len(response.text)} chars)")
            formatted_response = format_ai_response(response.text.strip())
            return formatted_response
        else:
            print("✗ Empty response from Gemini")
            return None
            
    except Exception as e:
        print(f"✗ Error fetching response from Gemini API: {e}")
        return None

@cybersentry_ai.route('/')
def index():
    """Render the main chat interface"""
    return render_template('cybersentry_AI.html')

@cybersentry_ai.route('/ask', methods=['POST'])
def ask():
    """Handle question requests with enhanced processing"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        force_source = data.get('force_source', None)  # 'json' or 'ai'
        
        if not question:
            return jsonify({
                'error': 'Question cannot be empty',
                'terminal_output': ''
            }), 400
        
        print(f"\n{'='*60}")
        print(f"[{time.strftime('%H:%M:%S')}] New Question: {question}")
        if force_source:
            print(f"[REGENERATE] Forcing source: {force_source}")
        print('='*60)
        
        # Handle regeneration with forced source
        if force_source == 'ai':
            gemini_answer, gemini_output = get_gemini_response(question)
            if gemini_answer:
                return jsonify({
                    'answer': gemini_answer,
                    'source': 'AI',
                    'terminal_output': gemini_output,
                    'confidence': 'medium',
                    'can_regenerate': True
                })
        
        elif force_source == 'json':
            answer, json_output = fuzzy_match(question, responses, threshold=60)
            if answer:
                return jsonify({
                    'answer': answer,
                    'source': 'JSON',
                    'terminal_output': json_output,
                    'confidence': 'high',
                    'can_regenerate': True
                })
        
        # Normal flow: Try JSON first
        answer, json_output = fuzzy_match(question, responses, threshold=70)
        
        if answer:
            print(f"\n[RESULT] Using JSON database response")
            return jsonify({
                'answer': answer,
                'source': 'JSON',
                'terminal_output': json_output,
                'confidence': 'high',
                'can_regenerate': True
            })
        
        # Fallback to Gemini
        print("\n[FALLBACK] Trying Gemini AI...")
        gemini_answer, gemini_output = get_gemini_response(question)
        
        if gemini_answer:
            print(f"[RESULT] Using Gemini AI response")
            return jsonify({
                'answer': gemini_answer,
                'source': 'AI',
                'terminal_output': gemini_output,
                'confidence': 'medium',
                'can_regenerate': True
            })
        
        # Final fallback
        print("\n[FALLBACK] Using default response")
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
            'can_regenerate': False
        })
        
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        print(f"\n[ERROR] {error_msg}")
        return jsonify({
            'error': error_msg,
            'terminal_output': ''
        }), 500

@cybersentry_ai.route('/reload-responses', methods=['POST'])
def reload_responses():
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

def init_app(app):
    """Register blueprint with Flask app"""
    app.register_blueprint(cybersentry_ai, url_prefix='/cybersentry_ai')
    print(f"CyberSentry AI Blueprint registered with {len(responses)} responses loaded.")