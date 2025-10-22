from flask import Blueprint, request, jsonify, render_template
from flask_cors import CORS
import requests
import base64
import google.generativeai as genai
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from collections import deque
import threading
import time
import hashlib
import json
from functools import wraps

import warnings
import os
import logging

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

# Suppress specific warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', module='tensorflow')

# Configure logging to suppress TensorFlow info messages
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('tf_keras').setLevel(logging.ERROR)



# Create blueprint
infosight_ai = Blueprint('infosight_ai', __name__, template_folder='templates')
logger = logging.getLogger(__name__)
CORS(infosight_ai)

# API Configuration with validation
print("\n" + "="*60)
print("🚀 INFOSIGHT AI Pro - Initializing")
print("="*60)

GEMINI_API_KEY = 'AIzaSyCMwpK-6Dr9X_MpcCyRR1PJcixg4pW55e8'
HF_API_TOKEN = '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'

# Configure and validate Gemini
GEMINI_CONFIGURED = False
try:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
    GEMINI_CONFIGURED = True
    print("✓ Gemini API: CONFIGURED for INFOSIGHT AI")
except Exception as e:
    print(f"✗ Gemini API: CONFIGURATION FAILED - {str(e)}")

# Validate Hugging Face Token
HF_CONFIGURED = False
try:
    # First, check if token format is valid
    if not HF_API_TOKEN or HF_API_TOKEN.startswith('your_') or len(HF_API_TOKEN) < 20:
        print("✗ Hugging Face API: INVALID TOKEN FORMAT")
    else:
        # Test the token with a simple API call
        test_headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        
        # Try whoami endpoint first
        test_response = requests.get(
            "https://huggingface.co/api/whoami", 
            headers=test_headers, 
            timeout=10
        )
        
        if test_response.status_code == 200:
            HF_CONFIGURED = True
            hf_user = test_response.json().get('name', 'Unknown')
            print(f"✓ Hugging Face API: CONFIGURED for INFOSIGHT AI (User: {hf_user})")
        elif test_response.status_code == 401:
            
            
            # Try to use anyway for inference API (sometimes whoami fails but inference works)
            print("  Attempting to use token for inference anyway...")
            HF_CONFIGURED = True  # Set to True to allow trying
        else:
            print(f"✗ Hugging Face API: CONFIGURATION FAILED - HTTP {test_response.status_code}")
            
except requests.exceptions.Timeout:
    print("✗ Hugging Face API: CONNECTION TIMEOUT")
    HF_CONFIGURED = True  # Allow to proceed, might work for inference
except requests.exceptions.ConnectionError:
    print("✗ Hugging Face API: CONNECTION ERROR - Check your internet connection")
except Exception as e:
    print(f"✗ Hugging Face API: CONFIGURATION FAILED - {str(e)}")

print("="*60 + "\n")

# Safety settings for Gemini
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

class CacheManager:
    """Simple in-memory cache with TTL."""
    def __init__(self, ttl_seconds=3600):
        self.cache = {}
        self.ttl = ttl_seconds
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl:
                    logger.info(f"Cache hit for key: {key[:20]}...")
                    return value
                else:
                    del self.cache[key]
        return None

    def set(self, key, value):
        with self.lock:
            self.cache[key] = (value, time.time())
            # Keep cache size under control
            if len(self.cache) > 100:
                oldest_key = min(self.cache.items(), key=lambda x: x[1][1])[0]
                del self.cache[oldest_key]

    def clear(self):
        with self.lock:
            self.cache.clear()

class RateLimiter:
    """Advanced rate limiting with per-IP tracking."""
    def __init__(self, max_requests=15, time_window=60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = {}
        self.lock = threading.Lock()

    def can_proceed(self, identifier="global"):
        """Check if a request can proceed for a given identifier."""
        now = datetime.now()
        
        with self.lock:
            if identifier not in self.requests:
                self.requests[identifier] = deque()
            
            request_queue = self.requests[identifier]
            
            # Remove old requests
            while request_queue and request_queue[0] < now - timedelta(seconds=self.time_window):
                request_queue.popleft()
            
            if len(request_queue) < self.max_requests:
                request_queue.append(now)
                return True
            
            return False

    def wait_time(self, identifier="global"):
        """Calculate wait time for next request."""
        if identifier not in self.requests or not self.requests[identifier]:
            return 0
        
        now = datetime.now()
        oldest_request = self.requests[identifier][0]
        wait = (oldest_request + timedelta(seconds=self.time_window) - now).total_seconds()
        return max(0, wait)

    def reset(self, identifier=None):
        """Reset rate limits."""
        with self.lock:
            if identifier:
                self.requests.pop(identifier, None)
            else:
                self.requests.clear()


class AIGenerator:
    """Enhanced AI content generator with caching and error recovery."""
    
    def __init__(self):
        if not GEMINI_CONFIGURED:
            logger.warning("Gemini API not properly configured")
            self.gemini_model = None
        else:
            self.gemini_model = genai.GenerativeModel(
                'gemini-2.0-flash-exp',
                safety_settings=SAFETY_SETTINGS,
                generation_config={
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 4096,
                }
            )
        
        # Image models with fallback chain
        self.image_models = [
            "black-forest-labs/FLUX.1-dev",
            "stabilityai/stable-diffusion-xl-base-1.0",
            "SG161222/Realistic_Vision_V5.1_noVAE",
            "prompthero/openjourney-v4"
        ]
        
        self.rate_limiter = RateLimiter(max_requests=20, time_window=60)
        self.cache = CacheManager(ttl_seconds=3600)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"AIGenerator initialized (Gemini: {GEMINI_CONFIGURED}, HF: {HF_CONFIGURED})")

    def _create_cache_key(self, prompt, content_type):
        """Create a cache key from prompt and content type."""
        return hashlib.md5(f"{content_type}:{prompt}".encode()).hexdigest()

    def _sanitize_prompt(self, prompt):
        """Clean and validate prompt."""
        prompt = prompt.strip()
        
        # Remove excessive whitespace
        prompt = re.sub(r'\s+', ' ', prompt)
        
        # Limit length
        if len(prompt) > 2000:
            prompt = prompt[:2000]
        
        return prompt

    def format_text_content(self, text):
        text = re.sub(r'\*+', '', text)
        text = re.sub(r'#+\s*', '', text)
        text = re.sub(r'_+', '', text)
        
        # Split into lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Organize content into structured sections
        formatted_output = {
            'sections': []
        }
        
        current_section = {
            'heading': '',
            'paragraphs': []
        }
        current_paragraph = []
        
        for line in lines:
            # Check if line is a heading (ends with colon or is short and capitalized)
            if line.endswith(':') or (len(line.split()) <= 6 and line[0].isupper()):
                # Save previous section if it has content
                if current_paragraph:
                    current_section['paragraphs'].append(' '.join(current_paragraph))
                    current_paragraph = []
                if current_section['heading'] or current_section['paragraphs']:
                    formatted_output['sections'].append(current_section)
                
                # Start new section
                current_section = {
                    'heading': line.rstrip(':'),
                    'paragraphs': []
                }
            else:
                # Add to current paragraph
                current_paragraph.append(line)
                
                # End paragraph on sentence endings
                if line.endswith(('.', '!', '?')) and len(' '.join(current_paragraph).split()) > 20:
                    current_section['paragraphs'].append(' '.join(current_paragraph))
                    current_paragraph = []
        
        # Save remaining content
        if current_paragraph:
            current_section['paragraphs'].append(' '.join(current_paragraph))
        if current_section['heading'] or current_section['paragraphs']:
            formatted_output['sections'].append(current_section)
        
        # Convert to HTML with styling
        html_output = '<div class="formatted-content">'
        
        for idx, section in enumerate(formatted_output['sections']):
            html_output += f'<div class="content-section" data-section="{idx}">'
            
            if section['heading']:
                html_output += f'<h3 class="section-heading">{section["heading"]}</h3>'
            
            for para in section['paragraphs']:
                html_output += f'<p class="section-paragraph">{para}</p>'
            
            html_output += '</div>'
        
        html_output += '</div>'
        
        return html_output

    def generate_text(self, prompt, use_cache=True):
        """Generate text with caching and error handling."""
        try:
            if not self.gemini_model:
                raise ValueError("Gemini API not configured")
                
            prompt = self._sanitize_prompt(prompt)
            
            # Check cache
            if use_cache:
                cache_key = self._create_cache_key(prompt, 'text')
                cached = self.cache.get(cache_key)
                if cached:
                    return cached
            
            # Rate limiting
            if not self.rate_limiter.can_proceed('text'):
                wait_time = self.rate_limiter.wait_time('text')
                raise ValueError(f"Rate limit exceeded. Please wait {wait_time:.0f} seconds.")
            
            # Enhanced prompt engineering
            enhanced_prompt = f"""
You are an expert content writer. Create high-quality, accurate, and engaging content about: {prompt}

Requirements:
- Be informative and well-structured
- Use clear, professional language
- Include relevant details and examples
- Make it engaging and easy to read
- Keep the tone appropriate for the topic

Provide a comprehensive response without using markdown formatting or special characters.
"""
            
            logger.info(f"Generating text for prompt: {prompt[:50]}...")
            
            response = self.gemini_model.generate_content(enhanced_prompt)
            
            if not response or not response.text:
                raise ValueError("No text generated from AI model")
            
            formatted_text = self.format_text_content(response.text)
            
            # Cache the result
            if use_cache:
                self.cache.set(cache_key, formatted_text)
            
            logger.info("Text generation successful")
            return formatted_text
            
        except Exception as e:
            logger.error(f"Text generation error: {str(e)}", exc_info=True)
            raise ValueError(f"Text generation failed: {str(e)}")

    def generate_image(self, prompt, use_cache=True):
        """Generate image with model fallback chain."""
        try:
            prompt = self._sanitize_prompt(prompt)
            
            # Check cache
            if use_cache:
                cache_key = self._create_cache_key(prompt, 'image')
                cached = self.cache.get(cache_key)
                if cached:
                    return cached
            
            # Rate limiting
            if not self.rate_limiter.can_proceed('image'):
                wait_time = self.rate_limiter.wait_time('image')
                raise ValueError(f"Rate limit exceeded. Please wait {wait_time:.0f} seconds.")
            
            # Enhanced prompt for better image quality
            enhanced_prompt = (
                f"{prompt}, "
                "professional photography, 8K UHD, highly detailed, "
                "perfect composition, dramatic lighting, sharp focus, "
                "photorealistic, masterpiece quality"
            )
            
            negative_prompt = (
                "blurry, distorted, low quality, draft, ugly, deformed, "
                "bad anatomy, watermark, text, signature, logo, "
                "low resolution, pixelated, grainy"
            )
            
            headers = {
                "Authorization": f"Bearer {HF_API_TOKEN}",
                "Content-Type": "application/json"
            }
            
            logger.info(f"Generating image for prompt: {prompt[:50]}...")
            
            # Try each model in sequence
            for model_idx, model_name in enumerate(self.image_models):
                try:
                    api_url = f"https://api-inference.huggingface.co/models/{model_name}"
                    
                    payload = {
                        "inputs": enhanced_prompt,
                        "parameters": {
                            "negative_prompt": negative_prompt,
                            "num_inference_steps": 50 if model_idx == 0 else 40,
                            "guidance_scale": 7.5,
                        }
                    }
                    
                    logger.info(f"Trying model {model_idx + 1}/{len(self.image_models)}: {model_name}")
                    
                    response = requests.post(
                        api_url,
                        headers=headers,
                        json=payload,
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        image_bytes = response.content
                        logger.info(f"Image generated successfully with model: {model_name}")
                        
                        # Cache the result
                        if use_cache:
                            self.cache.set(cache_key, image_bytes)
                        
                        return image_bytes
                    
                    elif response.status_code == 503:
                        logger.warning(f"Model {model_name} is loading, trying next...")
                        continue
                    else:
                        logger.warning(f"Model {model_name} failed with status {response.status_code}")
                        continue
                        
                except requests.exceptions.Timeout:
                    logger.warning(f"Timeout for model {model_name}, trying next...")
                    continue
                except Exception as e:
                    logger.warning(f"Error with model {model_name}: {str(e)}")
                    continue
            
            raise ValueError("All image generation models failed. Please try again later.")
            
        except Exception as e:
            logger.error(f"Image generation error: {str(e)}", exc_info=True)
            raise ValueError(f"Image generation failed: {str(e)}")

    def generate_both(self, prompt, use_cache=True):
        """Generate both text and image in parallel with error handling."""
        prompt = self._sanitize_prompt(prompt)
        
        logger.info(f"Starting parallel generation for: {prompt[:50]}...")
        
        text_result = None
        image_result = None
        text_error = None
        image_error = None
        
        # Submit both tasks to executor
        text_future = self.executor.submit(self._safe_generate_text, prompt, use_cache)
        image_future = self.executor.submit(self._safe_generate_image, prompt, use_cache)
        
        # Wait for both with timeout
        try:
            text_result, text_error = text_future.result(timeout=90)
        except Exception as e:
            text_error = f"Text generation timeout: {str(e)}"
            logger.error(text_error)
        
        try:
            image_result, image_error = image_future.result(timeout=90)
        except Exception as e:
            image_error = f"Image generation timeout: {str(e)}"
            logger.error(image_error)
        
        # Return results with error info
        return text_result, image_result, text_error, image_error

    def _safe_generate_text(self, prompt, use_cache):
        """Wrapper for text generation that catches exceptions."""
        try:
            result = self.generate_text(prompt, use_cache)
            return result, None
        except Exception as e:
            return None, str(e)

    def _safe_generate_image(self, prompt, use_cache):
        """Wrapper for image generation that catches exceptions."""
        try:
            result = self.generate_image(prompt, use_cache)
            return result, None
        except Exception as e:
            return None, str(e)

    def cleanup(self):
        """Clean up resources."""
        self.executor.shutdown(wait=False)
        self.cache.clear()
        self.rate_limiter.reset()


# Initialize generator
generator = AIGenerator()


# Decorator for error handling
def handle_errors(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValueError as e:
            logger.warning(f"Validation error in {f.__name__}: {str(e)}")
            return jsonify({'error': str(e), 'type': 'validation'}), 400
        except requests.exceptions.RequestException as e:
            logger.error(f"API request error in {f.__name__}: {str(e)}")
            return jsonify({'error': 'External API error. Please try again.', 'type': 'api'}), 503
        except Exception as e:
            logger.error(f"Unexpected error in {f.__name__}: {str(e)}", exc_info=True)
            return jsonify({'error': 'An unexpected error occurred', 'type': 'internal'}), 500
    return wrapper


# Decorator for request validation
def validate_request(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 415
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Empty request body'}), 400
        
        if 'prompt' not in data:
            return jsonify({'error': 'Missing required field: prompt'}), 400
        
        prompt = data['prompt']
        if not isinstance(prompt, str):
            return jsonify({'error': 'Prompt must be a string'}), 400
        
        prompt = prompt.strip()
        if not prompt:
            return jsonify({'error': 'Prompt cannot be empty'}), 400
        
        if len(prompt) > 2000:
            return jsonify({'error': 'Prompt too long (max 2000 characters)'}), 400
        
        return f(*args, **kwargs)
    return wrapper


# Routes
@infosight_ai.route('/')
def index():
    """Serve the main application page."""
    return render_template('infosight_ai.html')


@infosight_ai.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'cache_size': len(generator.cache.cache),
        'version': '2.0.0'
    })


@infosight_ai.route('/api-status', methods=['GET'])
def api_status():
    """Check API key status via HTTP endpoint."""
    status = {
        'timestamp': datetime.now().isoformat(),
        'gemini': {
            'configured': bool(GEMINI_API_KEY and GEMINI_API_KEY != 'your_gemini_api_key_here'),
            'working': False,
            'error': None
        },
        'huggingface': {
            'configured': bool(HF_API_TOKEN and HF_API_TOKEN != 'your_hf_token_here'),
            'working': False,
            'error': None
        }
    }
    
    # Test Gemini API
    if status['gemini']['configured'] and generator.gemini_model:
        try:
            test_response = generator.gemini_model.generate_content("Test")
            if test_response and test_response.text:
                status['gemini']['working'] = True
        except Exception as e:
            status['gemini']['error'] = str(e)
    
    # Test Hugging Face API
    if status['huggingface']['configured']:
        try:
            headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
            response = requests.get("https://huggingface.co/api/whoami", headers=headers, timeout=10)
            if response.status_code == 200:
                status['huggingface']['working'] = True
                status['huggingface']['user'] = response.json().get('name', 'Unknown')
            else:
                status['huggingface']['error'] = f"HTTP {response.status_code}"
        except Exception as e:
            status['huggingface']['error'] = str(e)
    
    return jsonify(status)


@infosight_ai.route('/generate-text', methods=['POST'])
@handle_errors
@validate_request
def generate_text_endpoint():
    """Generate text content."""
    data = request.get_json()
    prompt = data['prompt'].strip()
    use_cache = data.get('use_cache', True)
    
    logger.info(f"Text generation request: {prompt[:50]}...")
    start_time = time.time()
    
    text = generator.generate_text(prompt, use_cache=use_cache)
    
    elapsed = time.time() - start_time
    logger.info(f"Text generation completed in {elapsed:.2f}s")
    
    return jsonify({
        'text': text,
        'cached': generator.cache.get(generator._create_cache_key(prompt, 'text')) is not None,
        'generation_time': elapsed,
        'word_count': len(text.split()),
        'char_count': len(text)
    })


@infosight_ai.route('/generate-image', methods=['POST'])
@handle_errors
@validate_request
def generate_image_endpoint():
    """Generate image content."""
    data = request.get_json()
    prompt = data['prompt'].strip()
    use_cache = data.get('use_cache', True)
    
    logger.info(f"Image generation request: {prompt[:50]}...")
    start_time = time.time()
    
    image_bytes = generator.generate_image(prompt, use_cache=use_cache)
    
    if not image_bytes:
        return jsonify({'error': 'Image generation failed'}), 500
    
    elapsed = time.time() - start_time
    logger.info(f"Image generation completed in {elapsed:.2f}s")
    
    # Encode to base64
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    return jsonify({
        'image_url': f"data:image/png;base64,{image_base64}",
        'cached': generator.cache.get(generator._create_cache_key(prompt, 'image')) is not None,
        'generation_time': elapsed,
        'image_size': len(image_bytes)
    })


@infosight_ai.route('/generate-both', methods=['POST'])
@handle_errors
@validate_request
def generate_both_endpoint():
    """Generate both text and image content."""
    data = request.get_json()
    prompt = data['prompt'].strip()
    use_cache = data.get('use_cache', True)
    
    logger.info(f"Combined generation request: {prompt[:50]}...")
    start_time = time.time()
    
    text, image_bytes, text_error, image_error = generator.generate_both(prompt, use_cache)
    
    elapsed = time.time() - start_time
    logger.info(f"Combined generation completed in {elapsed:.2f}s")
    
    response = {
        'generation_time': elapsed
    }
    
    # Handle text result
    if text:
        response['text'] = text
        response['text_cached'] = generator.cache.get(generator._create_cache_key(prompt, 'text')) is not None
        response['word_count'] = len(text.split())
    elif text_error:
        response['text_error'] = text_error
        response['text'] = f"Text generation failed: {text_error}"
    
    # Handle image result
    if image_bytes:
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        response['image_url'] = f"data:image/png;base64,{image_base64}"
        response['image_cached'] = generator.cache.get(generator._create_cache_key(prompt, 'image')) is not None
        response['image_size'] = len(image_bytes)
    elif image_error:
        response['image_error'] = image_error
    
    return jsonify(response)


@infosight_ai.route('/enhance-prompt', methods=['POST'])
@handle_errors
@validate_request
def enhance_prompt_endpoint():
    """Enhance a user's prompt using AI."""
    data = request.get_json()
    prompt = data['prompt'].strip()
    
    try:
        enhancement_prompt = f"""
Enhance this prompt to make it more detailed and effective for AI generation: "{prompt}"

Provide an improved version that:
- Adds relevant descriptive details
- Specifies quality and style
- Maintains the original intent
- Is concise but comprehensive

Return only the enhanced prompt without explanation.
"""
        
        if not generator.gemini_model:
            return jsonify({'error': 'Gemini API not configured'}), 503
        
        response = generator.gemini_model.generate_content(enhancement_prompt)
        enhanced = response.text.strip()
        
        return jsonify({
            'original': prompt,
            'enhanced': enhanced,
            'improvement': len(enhanced) - len(prompt)
        })
        
    except Exception as e:
        logger.error(f"Prompt enhancement error: {str(e)}")
        return jsonify({'error': 'Enhancement failed'}), 500


@infosight_ai.route('/stats', methods=['GET'])
def get_stats():
    """Get generation statistics."""
    return jsonify({
        'cache_size': len(generator.cache.cache),
        'cache_ttl': generator.cache.ttl,
        'rate_limit_window': generator.rate_limiter.time_window,
        'rate_limit_max': generator.rate_limiter.max_requests,
        'timestamp': datetime.now().isoformat()
    })


@infosight_ai.route('/clear-cache', methods=['POST'])
def clear_cache():
    """Clear the generation cache."""
    generator.cache.clear()
    logger.info("Cache cleared")
    return jsonify({'message': 'Cache cleared successfully'})


@infosight_ai.route('/reset-rate-limit', methods=['POST'])
def reset_rate_limit():
    """Reset rate limiting (for debugging)."""
    generator.rate_limiter.reset()
    logger.info("Rate limits reset")
    return jsonify({'message': 'Rate limits reset successfully'})


# Error handlers
@infosight_ai.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@infosight_ai.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed'}), 405


@infosight_ai.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500


# Cleanup on shutdown
import atexit

@atexit.register
def cleanup():
    """Clean up resources on shutdown."""
    logger.info("Cleaning up INFOSIGHT AI resources...")
    generator.cleanup()
    logger.info("Cleanup complete")


# Production configuration
if __name__ != '__main__':
    # Configure logging for production
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('infosight_ai.log'),
            logging.StreamHandler()
        ]
    )
    
    logger.info("INFOSIGHT AI Pro initialized successfully")