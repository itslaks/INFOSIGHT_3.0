from flask import Blueprint, request, jsonify, render_template, g
from flask_cors import CORS
from utils.security import rate_limit_api, validate_request as validate_request_central, InputValidator
import requests
import base64
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
import sqlite3
from pathlib import Path

import warnings
import os

# Local LLM fallback
try:
    import sys
    from pathlib import Path
    # Add parent directory to path for utils
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.local_llm_utils import generate_with_ollama, check_ollama_available
    from utils.llm_logger import log_llm_status, log_llm_request, log_llm_success, log_llm_error, log_llm_fallback, log_processing_step
    LOCAL_LLM_AVAILABLE = True
except ImportError as e:
    LOCAL_LLM_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ Local LLM utilities not available: {e}")
    # Create dummy functions
    def log_llm_status(*args, **kwargs): return (False, False)
    def log_llm_request(*args, **kwargs): pass
    def log_llm_success(*args, **kwargs): pass
    def log_llm_error(*args, **kwargs): pass
    def log_llm_fallback(*args, **kwargs): pass
    def log_processing_step(*args, **kwargs): pass

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
logger.info("="*60)
logger.info("🚀 INFOSIGHT AI Pro - Initializing")
logger.info("="*60)

# Log LLM status at startup
try:
    log_llm_status("InfoSight AI")
except:
    pass

# Load Hugging Face token safely with reload capability
HF_API_TOKEN = None

def reload_hf_token():
    """Reload HF token from config or environment variables."""
    global HF_API_TOKEN
    token = None
    
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from config import Config
        token = getattr(Config, "HF_API_TOKEN", None)
    except Exception:
        pass
    
    if not token:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except:
            pass
        token = os.getenv("HF_API_TOKEN") or os.getenv("HUGGINGFACE_API_TOKEN")
    
    # Normalize token
    if token:
        token = token.strip()
        if token.startswith("hf_") and len(token) > 20:
            HF_API_TOKEN = token
            return True
    
    HF_API_TOKEN = None
    return False

# Initial load
reload_hf_token()

# Log status
if HF_API_TOKEN and HF_API_TOKEN.startswith("hf_") and len(HF_API_TOKEN) > 20:
    masked = HF_API_TOKEN[:8] + "..." + HF_API_TOKEN[-4:]
    logger.info(f"✓ HF_API_TOKEN loaded correctly ({masked})")
else:
    logger.error("✗ HF_API_TOKEN missing or invalid")
    HF_API_TOKEN = None

# Use centralized LLM router (replaces Gemini)
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.llm_router import generate_text
    LLM_ROUTER_AVAILABLE = True
    logger.info("✓ LLM router available for InfoSight AI")
except ImportError as e:
    LLM_ROUTER_AVAILABLE = False
    logger.warning(f"⚠ LLM router not available: {e}")
    def generate_text(*args, **kwargs):
        return {"response": "", "model": "none", "source": "none"}

GEMINI_CONFIGURED = LLM_ROUTER_AVAILABLE  # Keep for backward compatibility
gemini_model = None  # Deprecated

# Validate Hugging Face Token
HF_CONFIGURED = False

if not HF_API_TOKEN:
    logger.error("✗ Hugging Face API: TOKEN NOT SET")
else:
    try:
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        logger.info("Testing HF_API_TOKEN with Hugging Face API...")

        r = requests.get(
            "https://huggingface.co/api/whoami-v2",
            headers=headers,
            timeout=10
        )

        if r.status_code == 200:
            HF_CONFIGURED = True
            user = r.json().get("name", "Unknown")
            logger.info(f"✓ Hugging Face API CONFIGURED (User: {user})")
        else:
            logger.error(f"✗ Hugging Face API INVALID TOKEN (HTTP {r.status_code})")
            logger.error(r.text)
            HF_API_TOKEN = None

    except Exception as e:
        logger.error(f"✗ Hugging Face API validation failed: {e}")
        HF_API_TOKEN = None

logger.info("="*60)

# Safety settings removed - Groq handles content moderation automatically

def check_image_generation_config():
    """Verify image generation is properly configured."""
    if not HF_API_TOKEN:
        logger.error("⚠ HF_API_TOKEN not configured")
        return False

    try:
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        r = requests.get("https://huggingface.co/api/whoami-v2", headers=headers, timeout=10)

        if r.status_code == 200:
            logger.info(f"✓ HF API ready for {r.json().get('name','Unknown')}")
            return True
        else:
            logger.error(f"✗ HF API token rejected (HTTP {r.status_code})")
            return False

    except Exception as e:
        logger.error(f"✗ HF API validation error: {e}")
        return False

# Check image generation configuration at startup
logger.info("="*60)
logger.info("INFOSIGHT AI - Image Generation Configuration")
logger.info("="*60)

if check_image_generation_config():
    logger.info("✓ Image generation ready")
else:
    logger.warning("⚠ Image generation may not work properly")

logger.info("="*60)

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
    """Enhanced AI content generator with caching, error recovery, and advanced features."""
    
    def __init__(self):
        # Gemini models removed - using centralized router
        self.gemini_model = None
        self.gemini_pro_model = None
        
        # Image models with fallback chain - Updated to 2026 models (for backward compatibility)
        self.image_models = [
            "stabilityai/stable-diffusion-3.5-large",  # Most reliable
            "black-forest-labs/FLUX.1-schnell",  # Fast and reliable
            "stabilityai/sdxl-turbo",  # High quality alternative
            "runwayml/stable-diffusion-v1-5",  # Stable fallback
            "CompVis/stable-diffusion-v1-4"  # Final fallback
        ]
        
        # Content style templates
        self.style_templates = {
            'professional': {
                'tone': 'professional and authoritative',
                'style': 'formal, clear, and structured',
                'length': 'comprehensive'
            },
            'casual': {
                'tone': 'friendly and conversational',
                'style': 'relaxed, engaging, and easy-going',
                'length': 'moderate'
            },
            'academic': {
                'tone': 'scholarly and analytical',
                'style': 'precise, well-researched, and citation-ready',
                'length': 'detailed'
            },
            'creative': {
                'tone': 'imaginative and expressive',
                'style': 'vivid, descriptive, and engaging',
                'length': 'elaborate'
            },
            'technical': {
                'tone': 'precise and informative',
                'style': 'clear, structured, and code-friendly',
                'length': 'comprehensive'
            },
            'marketing': {
                'tone': 'persuasive and compelling',
                'style': 'engaging, benefit-focused, and action-oriented',
                'length': 'concise'
            }
        }
        
        self.rate_limiter = RateLimiter(max_requests=30, time_window=60)  # Increased rate limit
        self.cache = CacheManager(ttl_seconds=7200)  # 2 hours cache
        self.executor = ThreadPoolExecutor(max_workers=8)  # Increased workers
        
        logger.info(f"AIGenerator initialized (LLM Router: {LLM_ROUTER_AVAILABLE}, HF: {HF_CONFIGURED})")

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

    def generate_text(self, prompt, use_cache=True, style='professional', variations=1, length='medium', **kwargs):
        """Generate text with caching, error handling, local LLM fallback, and style support."""
        model_used = "llm_router"  # Track which model was used
        
        try:
            if not LLM_ROUTER_AVAILABLE:
                raise ValueError("LLM router not available")
                
            prompt = self._sanitize_prompt(prompt)
            
            # Check cache
            cache_key = self._create_cache_key(f"{prompt}:{style}:{length}", 'text')
            if use_cache:
                cached = self.cache.get(cache_key)
                if cached:
                    return cached, model_used
            
            # Rate limiting
            if not self.rate_limiter.can_proceed('text'):
                wait_time = self.rate_limiter.wait_time('text')
                raise ValueError(f"Rate limit exceeded. Please wait {wait_time:.0f} seconds.")
            
            # Get style template
            style_config = self.style_templates.get(style, self.style_templates['professional'])
            
            # Length mapping
            length_map = {
                'short': '2-3 paragraphs, concise',
                'medium': '4-6 paragraphs, moderate detail',
                'long': '8-12 paragraphs, comprehensive',
                'extended': '15+ paragraphs, in-depth analysis'
            }
            length_desc = length_map.get(length, length_map['medium'])
            
            # Enhanced prompt engineering with style and length
            enhanced_prompt = f"""
You are an expert content writer using advanced AI. Create high-quality, accurate, and engaging content about: {prompt}

Style Requirements:
- Tone: {style_config['tone']}
- Writing Style: {style_config['style']}
- Length: {length_desc}

Content Requirements:
- Be informative and well-structured
- Use clear, appropriate language for the style
- Include relevant details, examples, and insights
- Make it engaging and easy to read
- Ensure accuracy and factual correctness
- Use proper formatting with clear sections

Provide a comprehensive response without using markdown formatting or special characters. Structure with clear headings and paragraphs.
"""
            
            logger.info(f"Generating text for prompt: {prompt[:50]}...")
            
            # Use LLM router (handles Groq + local fallback automatically)
            result = generate_text(
                prompt=enhanced_prompt,
                app_name="infosight_ai",
                task_type="chat",
                system_prompt="You are an expert content writer. Create high-quality, accurate, and engaging content.",
                temperature=0.7,
                max_tokens=8192
            )
            
            response_text = result.get("response", "")
            model_used = result.get("model", "unknown")
            source = result.get("source", "unknown")
            
            if response_text:
                formatted_text = self.format_text_content(response_text)
                logger.info(f"✓ Content generated using {model_used} ({source})")
            else:
                raise ValueError("No text generated from LLM router")
            
            # Cache the result
            if use_cache:
                self.cache.set(cache_key, formatted_text)
            
            logger.info(f"Text generation successful (model: {model_used})")
            return formatted_text, model_used
            
        except Exception as e:
            logger.error(f"Text generation error: {str(e)}", exc_info=True)
            # Provide helpful error message
            error_msg = str(e).lower()
            if 'quota' in error_msg or 'rate limit' in error_msg:
                raise ValueError("API quota exceeded. Please try again later or use local LLM fallback.")
            elif 'timeout' in error_msg:
                raise ValueError("Request timed out. Please try again with a shorter prompt.")
            elif 'invalid' in error_msg or 'bad request' in error_msg:
                raise ValueError("Invalid request. Please check your prompt and try again.")
            else:
                raise ValueError(f"Text generation failed: {str(e)}")

    def generate_image(self, prompt, use_cache=True):
        """Generate contextually relevant, high-quality images with smart prompt analysis."""
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
            
            logger.info(f"Generating image: {prompt[:50]}...")
            
            # SMART PROMPT ANALYSIS - Detect if prompt needs conversion
            prompt_lower = prompt.lower()
            
            # Detect abstract/technical concepts that need visual representation
            abstract_concepts = {
                # AI/ML Terms
                'rnn': 'diagram of recurrent neural network architecture with nodes and connections',
                'neural network': 'diagram of artificial neural network layers with interconnected nodes',
                'cnn': 'convolutional neural network architecture diagram',
                'lstm': 'long short-term memory network diagram with gates',
                'transformer': 'transformer architecture diagram with attention mechanism',
                'machine learning': 'machine learning workflow diagram with data and models',
                'deep learning': 'deep neural network layers visualization',
                'ai': 'artificial intelligence concept with neural networks and data',
                
                # Programming Concepts
                'algorithm': 'flowchart diagram of algorithm steps',
                'binary tree': 'binary tree data structure diagram',
                'graph': 'graph data structure with nodes and edges',
                'api': 'API architecture diagram with client and server',
                'database': 'database schema diagram with tables and relationships',
                
                # General Abstract Concepts
                'blockchain': 'blockchain technology diagram with connected blocks',
                'cloud computing': 'cloud computing architecture diagram',
                'iot': 'Internet of Things network diagram with connected devices',
                'cybersecurity': 'cybersecurity concept with shield and network protection',
            }
            
            # Check if prompt is asking a question (what is, explain, how does)
            is_question = any(phrase in prompt_lower for phrase in [
                'what is', 'what are', 'explain', 'how does', 'how do',
                'tell me about', 'describe', 'definition of'
            ])
            
            # Convert abstract concept to visual prompt
            visual_prompt = prompt
            for concept, visual_desc in abstract_concepts.items():
                if concept in prompt_lower:
                    visual_prompt = visual_desc
                    logger.info(f"Converting abstract concept '{concept}' to visual: '{visual_desc}'")
                    break
            
            # If it's a question but not matched, try to extract the subject
            if is_question and visual_prompt == prompt:
                # Extract subject from question
                words = prompt_lower.replace('what is', '').replace('what are', '').replace('explain', '').strip()
                words = words.replace('?', '').strip()
                
                # If it seems technical/abstract, make it a diagram
                if len(words.split()) <= 4:  # Short technical term
                    visual_prompt = f"infographic diagram explaining {words}, educational illustration, clean design"
                else:
                    visual_prompt = words
            
            # NOW BUILD ENHANCED PROMPT based on detected category
            if 'diagram' in visual_prompt or 'architecture' in visual_prompt or 'flowchart' in visual_prompt:
                # Technical diagram style
                enhanced_prompt = (
                    f"{visual_prompt}, "
                    "professional infographic, clean design, educational illustration, "
                    "technical diagram, clear labels, modern style, "
                    "high contrast, vector art style, white background"
                )
                negative_aspects = (
                    "photo, photograph, realistic, blurry, cluttered, "
                    "messy, handwritten, low quality"
                )
            
            elif any(word in prompt_lower for word in ['person', 'man', 'woman', 'portrait', 'face', 'people']):
                # Portrait/person style
                enhanced_prompt = (
                    f"{visual_prompt}, "
                    "photorealistic portrait, highly detailed facial features, "
                    "professional photography, 85mm lens, natural lighting, "
                    "sharp focus, realistic skin texture, National Geographic style"
                )
                negative_aspects = (
                    "blurry, distorted, deformed, ugly, bad anatomy, "
                    "poorly drawn, low quality, cartoon"
                )
            
            elif any(word in prompt_lower for word in ['landscape', 'nature', 'scenery', 'mountains', 'city']):
                # Landscape style
                enhanced_prompt = (
                    f"{visual_prompt}, "
                    "landscape photography, ultra detailed, dramatic lighting, "
                    "golden hour, wide angle, professional nature photography, 8K HDR"
                )
                negative_aspects = "blurry, low quality, overexposed, amateur"
            
            elif any(word in prompt_lower for word in ['logo', 'icon', 'symbol', 'badge']):
                # Icon/logo style
                enhanced_prompt = (
                    f"{visual_prompt}, "
                    "professional logo design, minimalist, clean, vector graphic, "
                    "modern, simple, memorable design"
                )
                negative_aspects = "complex, cluttered, photo, realistic, blurry"
            
            else:
                # General high quality
                enhanced_prompt = (
                    f"{visual_prompt}, "
                    "highly detailed, professional quality, 8K resolution, "
                    "sharp focus, perfect composition, masterpiece"
                )
                negative_aspects = (
                    "blurry, low quality, distorted, deformed, ugly, "
                    "bad anatomy, poorly drawn, amateur, low resolution"
                )
            
            # Method 1: Pollinations.AI (Primary)
            try:
                logger.info("Trying Pollinations.AI...")
                
                import urllib.parse
                encoded_prompt = urllib.parse.quote(enhanced_prompt)
                
                pollinations_url = (
                    f"https://image.pollinations.ai/prompt/{encoded_prompt}"
                    f"?width=1024&height=1024"
                    f"&model=flux"
                    f"&nologo=true"
                    f"&enhance=true"
                )
                
                response = requests.get(pollinations_url, timeout=90)
                
                if response.status_code == 200 and len(response.content) > 5000:
                    image_bytes = response.content
                    logger.info(f"✓ Image generated with Pollinations.AI ({len(image_bytes)} bytes)")
                    
                    if use_cache:
                        self.cache.set(cache_key, image_bytes)
                    
                    return image_bytes
                else:
                    logger.warning(f"Pollinations.AI failed: HTTP {response.status_code}")
            
            except Exception as e:
                logger.warning(f"Pollinations.AI error: {str(e)}")
            
            # Method 2: Segmind (Fallback)
            try:
                logger.info("Trying Segmind...")
                
                segmind_url = "https://api.segmind.com/v1/sd1.5-txt2img"
                payload = {
                    "prompt": enhanced_prompt,
                    "negative_prompt": negative_aspects,
                    "samples": 1,
                    "scheduler": "DPM++ 2M Karras",
                    "num_inference_steps": 25,
                    "guidance_scale": 7.5,
                    "seed": -1,
                    "img_width": 1024,
                    "img_height": 1024
                }
                
                response = requests.post(segmind_url, json=payload, timeout=90)
                
                if response.status_code == 200 and len(response.content) > 5000:
                    image_bytes = response.content
                    logger.info(f"✓ Image generated with Segmind ({len(image_bytes)} bytes)")
                    
                    if use_cache:
                        self.cache.set(cache_key, image_bytes)
                    
                    return image_bytes
            
            except Exception as e:
                logger.warning(f"Segmind error: {str(e)}")
            
            # Method 3: DeepAI (Fallback)
            try:
                logger.info("Trying DeepAI...")
                
                deepai_url = "https://api.deepai.org/api/text2img"
                payload = {
                    'text': enhanced_prompt,
                    'grid_size': 1,
                    'width': 1024,
                    'height': 1024
                }
                
                response = requests.post(deepai_url, data=payload, timeout=90)
                
                if response.status_code == 200:
                    data = response.json()
                    if 'output_url' in data:
                        img_response = requests.get(data['output_url'], timeout=30)
                        if img_response.status_code == 200:
                            image_bytes = img_response.content
                            logger.info(f"✓ Image generated with DeepAI ({len(image_bytes)} bytes)")
                            
                            if use_cache:
                                self.cache.set(cache_key, image_bytes)
                            
                            return image_bytes
            
            except Exception as e:
                logger.warning(f"DeepAI error: {str(e)}")
            
            raise ValueError("All image services unavailable. Please try again in 1-2 minutes.")
            
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
        model_used = "llm_router"  # Track which model was used
        
        # Submit both tasks to executor
        text_future = self.executor.submit(self._safe_generate_text, prompt, use_cache)
        image_future = self.executor.submit(self._safe_generate_image, prompt, use_cache)
        
        # Wait for both with timeout
        model_used = "llm_router"
        try:
            text_result_data, text_error = text_future.result(timeout=90)
            if text_result_data and isinstance(text_result_data, tuple):
                text_result, model_used = text_result_data
            else:
                text_result = text_result_data
        except Exception as e:
            text_error = f"Text generation timeout: {str(e)}"
            text_result = None
            logger.error(text_error)
        
        try:
            image_result, image_error = image_future.result(timeout=90)
        except Exception as e:
            image_error = f"Image generation timeout: {str(e)}"
            logger.error(image_error)
        
        # Return results with error info and model used
        return text_result, image_result, text_error, image_error, model_used

    def _safe_generate_text(self, prompt, use_cache):
        """Wrapper for text generation that catches exceptions."""
        try:
            result = self.generate_text(prompt, use_cache)
            # Handle new return format (text, model_used) or old format (text)
            if isinstance(result, tuple):
                text, model_used = result
                return (text, model_used), None
            else:
                return (result, "llm_router"), None
        except Exception as e:
            return None, str(e)

    def _safe_generate_image(self, prompt, use_cache):
        """Wrapper for image generation that catches exceptions."""
        try:
            result = self.generate_image(prompt, use_cache)
            return result, None
        except Exception as e:
            return None, str(e)

    def generate_multiple_images(self, prompts, use_cache=True):
        """Generate multiple images in parallel with error handling."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self.generate_image, prompt, use_cache): prompt 
                for prompt in prompts
            }
            
            for future in as_completed(futures):
                prompt = futures[future]
                try:
                    image_bytes = future.result(timeout=180)
                    results.append({
                        'prompt': prompt,
                        'image': image_bytes,
                        'success': True
                    })
                except Exception as e:
                    results.append({
                        'prompt': prompt,
                        'error': str(e),
                        'success': False
                    })
        
        return results

    def cleanup(self):
        """Clean up resources."""
        self.executor.shutdown(wait=False)
        self.cache.clear()
        self.rate_limiter.reset()


# Database Manager for History and Favorites
class InfosightDatabase:
    """Database manager for storing generation history and favorites."""
    
    def __init__(self, db_name='infosight_ai.db'):
        self.db_name = db_name
        self.init_db()
    
    def init_db(self):
        """Initialize database with proper error handling"""
        try:
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()
                
                # Generation history table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS generations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT,
                        prompt TEXT,
                        content_type TEXT,
                        content_data TEXT,
                        image_url TEXT,
                        model_used TEXT,
                        style TEXT,
                        length TEXT,
                        generation_time REAL,
                        word_count INTEGER,
                        timestamp DATETIME,
                        cached BOOLEAN
                    )
                ''')
                
                # Favorites table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS favorites (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT,
                        prompt TEXT,
                        content_type TEXT,
                        content_data TEXT,
                        image_url TEXT,
                        timestamp DATETIME,
                        UNIQUE(user_id, prompt, content_type)
                    )
                ''')
                
                # Analytics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS analytics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT,
                        event_type TEXT,
                        event_data TEXT,
                        timestamp DATETIME
                    )
                ''')
                
                # Create indexes
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_id ON generations(user_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON generations(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_content_type ON generations(content_type)')
                
                conn.commit()
            logger.info("✓ InfoSight AI database initialized")
        except sqlite3.Error as e:
            logger.error(f"⚠ Database initialization error: {e}")
    
    def save_generation(self, user_id, prompt, content_type, content_data=None, image_url=None, 
                       model_used=None, style=None, length=None, generation_time=0, word_count=0, cached=False):
        """Save a generation to history"""
        try:
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO generations 
                    (user_id, prompt, content_type, content_data, image_url, model_used, 
                     style, length, generation_time, word_count, timestamp, cached)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (user_id, prompt, content_type, content_data, image_url, model_used,
                      style, length, generation_time, word_count, datetime.now().isoformat(), cached))
                conn.commit()
                return cursor.lastrowid
        except sqlite3.Error as e:
            logger.error(f"⚠ Database save error: {e}")
            return None
    
    def get_history(self, user_id, content_type=None, limit=50):
        """Get user generation history"""
        try:
            with sqlite3.connect(self.db_name) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                if content_type:
                    cursor.execute('''
                        SELECT * FROM generations 
                        WHERE user_id = ? AND content_type = ?
                        ORDER BY timestamp DESC LIMIT ?
                    ''', (user_id, content_type, limit))
                else:
                    cursor.execute('''
                        SELECT * FROM generations 
                        WHERE user_id = ?
                        ORDER BY timestamp DESC LIMIT ?
                    ''', (user_id, limit))
                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logger.error(f"⚠ Database query error: {e}")
            return []
    
    def add_favorite(self, user_id, prompt, content_type, content_data=None, image_url=None):
        """Add to favorites"""
        try:
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO favorites (user_id, prompt, content_type, content_data, image_url, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (user_id, prompt, content_type, content_data, image_url, datetime.now().isoformat()))
                conn.commit()
                return True
        except sqlite3.IntegrityError:
            return False
        except sqlite3.Error as e:
            logger.error(f"⚠ Database favorite error: {e}")
            return False
    
    def get_favorites(self, user_id, content_type=None):
        """Get user favorites"""
        try:
            with sqlite3.connect(self.db_name) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                if content_type:
                    cursor.execute('''
                        SELECT * FROM favorites 
                        WHERE user_id = ? AND content_type = ?
                        ORDER BY timestamp DESC
                    ''', (user_id, content_type))
                else:
                    cursor.execute('''
                        SELECT * FROM favorites 
                        WHERE user_id = ?
                        ORDER BY timestamp DESC
                    ''', (user_id,))
                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logger.error(f"⚠ Database favorites error: {e}")
            return []


# Initialize database
db_manager = InfosightDatabase()

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
        'gemini': {  # Keep field name for backward compatibility (LLM Router)
            'configured': LLM_ROUTER_AVAILABLE,
            'working': False,
            'error': None,
            'source': None
        },
        'local_llm': {
            'available': False,
            'url': None,
            'error': None
        },
        'huggingface': {
            'configured': bool(HF_API_TOKEN and HF_API_TOKEN != 'your_hf_token_here' and not HF_API_TOKEN.startswith('your_')),
            'working': False,
            'error': None,
            'token_format_valid': bool(HF_API_TOKEN and HF_API_TOKEN.startswith('hf_')) if HF_API_TOKEN else False
        }
    }
    
    # Test LLM router (cloud + local fallback)
    if LLM_ROUTER_AVAILABLE:
        try:
            test_result = generate_text(
                prompt="Test",
                app_name="infosight_ai",
                task_type="chat",
                max_tokens=10
            )
            if test_result.get("response"):
                status['gemini']['working'] = True
                status['gemini']['source'] = test_result.get("source", "unknown")
        except Exception as e:
            status['gemini']['error'] = str(e)
    
    # Check local LLM availability
    try:
        from utils.local_llm_utils import check_ollama_available, LOCAL_LLM_BASE_URL
        local_available = check_ollama_available(retries=1, delay=1.0)
        status['local_llm']['available'] = local_available
        status['local_llm']['url'] = LOCAL_LLM_BASE_URL
        if not local_available:
            status['local_llm']['error'] = f"Local LLM server not responding at {LOCAL_LLM_BASE_URL}"
    except Exception as e:
        status['local_llm']['error'] = str(e)
    
    # Test Hugging Face API
    if HF_API_TOKEN:
        try:
            headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
            r = requests.get("https://huggingface.co/api/whoami-v2", headers=headers, timeout=10)

            if r.status_code == 200:
                status['huggingface']['working'] = True
                status['huggingface']['user'] = r.json().get("name","Unknown")
            else:
                status['huggingface']['error'] = f"HTTP {r.status_code} - Invalid token"

        except Exception as e:
            status['huggingface']['error'] = str(e)
    else:
        status['huggingface']['error'] = "HF_API_TOKEN not configured"
    
    return jsonify(status)


@infosight_ai.route('/generate-text', methods=['POST'])
@rate_limit_api(requests_per_minute=10, requests_per_hour=100)  # Rate limit for AI generation
@validate_request_central({
    "prompt": {
        "type": "string",
        "required": True,
        "max_length": 2000
    },
    "use_cache": {
        "type": "bool",
        "required": False
    },
    "style": {
        "type": "string",
        "required": False,
        "max_length": 50,
        "allowed_values": ['professional', 'casual', 'formal', 'creative', 'technical']
    },
    "length": {
        "type": "string",
        "required": False,
        "max_length": 20,
        "allowed_values": ['short', 'medium', 'long']
    },
    "variations": {
        "type": "int",
        "required": False,
        "min_value": 1,
        "max_value": 5
    }
}, strict=True)
@handle_errors
def generate_text_endpoint():
    """
    Generate text content with advanced options
    OWASP: Rate limited, input validated, schema-based validation
    """
    # Get validated data from request context
    data = g.validated_data
    prompt = InputValidator.validate_string(
        data.get('prompt'), 'prompt', max_length=2000, required=True
    )
    use_cache = data.get('use_cache', True)
    style = data.get('style', 'professional')
    length = data.get('length', 'medium')
    variations = data.get('variations', 1)
    
    logger.info(f"Text generation request: {prompt[:50]}... (style: {style}, length: {length})")
    start_time = time.time()
    
    # Generate variations if requested
    if variations > 1 and variations <= 5:
        results = []
        for i in range(variations):
            result = generator.generate_text(
                prompt, 
                use_cache=False,  # Don't cache variations
                style=style,
                length=length,
                variations=1
            )
            if isinstance(result, tuple):
                text, model_used = result
            else:
                text = result
                model_used = "llm_router"
            results.append({
                'text': text,
                'variation': i + 1
            })
        
        elapsed = time.time() - start_time
        return jsonify({
            'variations': results,
            'model_used': model_used,
            'generation_time': elapsed,
            'style': style,
            'length': length
        })
    else:
        result = generator.generate_text(prompt, use_cache=use_cache, style=style, length=length)
        # Handle new return format (text, model_used) or old format (text)
        if isinstance(result, tuple):
            text, model_used = result
        else:
            text = result
            model_used = "llm_router"
        
        elapsed = time.time() - start_time
        logger.info(f"Text generation completed in {elapsed:.2f}s (model: {model_used})")
        
        # Save to database
        user_id = request.remote_addr or 'anonymous'
        word_count = len(text.split())
        cached = generator.cache.get(generator._create_cache_key(f"{prompt}:{style}:{length}", 'text')) is not None
        db_manager.save_generation(
            user_id=user_id,
            prompt=prompt,
            content_type='text',
            content_data=text,
            model_used=model_used,
            style=style,
            length=length,
            generation_time=elapsed,
            word_count=word_count,
            cached=cached
        )
        
        return jsonify({
            'text': text,
            'model_used': model_used,
            'cached': cached,
            'generation_time': elapsed,
            'word_count': word_count,
            'char_count': len(text),
            'style': style,
            'length': length
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
    
    try:
        image_bytes = generator.generate_image(prompt, use_cache=use_cache)
        
        if not image_bytes:
            return jsonify({'error': 'Image generation failed - no image data returned'}), 500
        
        elapsed = time.time() - start_time
        logger.info(f"Image generation completed in {elapsed:.2f}s ({len(image_bytes)} bytes)")
        
        # Encode to base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        image_url = f"data:image/png;base64,{image_base64}"
        cached = generator.cache.get(generator._create_cache_key(prompt, 'image')) is not None
        
        # Save to database
        user_id = request.remote_addr or 'anonymous'
        db_manager.save_generation(
            user_id=user_id,
            prompt=prompt,
            content_type='image',
            image_url=image_url,
            model_used='huggingface',
            generation_time=elapsed,
            cached=cached
        )
        
        return jsonify({
            'image_url': image_url,
            'cached': cached,
            'generation_time': elapsed,
            'image_size': len(image_bytes),
            'model_used': 'huggingface'
        })
    except ValueError as e:
        # Handle validation errors with helpful messages
        logger.warning(f"Image generation validation error: {str(e)}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Image generation error: {str(e)}", exc_info=True)
        return jsonify({'error': f'Image generation failed: {str(e)}'}), 500


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
    
    result = generator.generate_both(prompt, use_cache)
    
    # Handle new return format with model_used
    if len(result) == 5:
        text, image_bytes, text_error, image_error, model_used = result
    else:
        text, image_bytes, text_error, image_error = result
        model_used = "llm_router"
    
    elapsed = time.time() - start_time
    logger.info(f"Combined generation completed in {elapsed:.2f}s (model: {model_used})")
    
    response = {
        'generation_time': elapsed,
        'model_used': model_used  # 'llm_router' or 'local'
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
    """Enhance a user's prompt using LLM router (Groq with Ollama fallback)."""
    data = request.get_json()
    prompt = data['prompt'].strip()
    style = data.get('style', 'professional')
    
    try:
        enhancement_prompt = f"""
You are an expert prompt engineer using advanced AI. Enhance this prompt to make it more detailed and effective for AI content generation:

Original Prompt: "{prompt}"

Style Context: {style}

Provide an improved version that:
- Adds relevant descriptive details and context
- Specifies quality, style, and output requirements
- Maintains the original intent and core message
- Is concise but comprehensive
- Includes specific examples or use cases where relevant
- Defines clear success criteria

Return ONLY the enhanced prompt without any explanation, preamble, or additional text.
"""
        
        if not LLM_ROUTER_AVAILABLE:
            return jsonify({'error': 'LLM router not available'}), 503
        
        try:
            result = generate_text(
                prompt=enhancement_prompt,
                app_name="infosight_ai",
                task_type="prompt_optimization",
                system_prompt="You are an expert prompt engineer. Enhance prompts to make them more detailed and effective for AI content generation.",
                temperature=0.7,
                max_tokens=2048
            )
            
            enhanced = result.get("response", "").strip()
            model_used = result.get("model", "unknown")
            
            if not enhanced:
                raise ValueError("No enhanced prompt generated")
            
            # Clean up any extra text
            if enhanced.startswith('Enhanced Prompt:'):
                enhanced = enhanced.replace('Enhanced Prompt:', '').strip()
            if enhanced.startswith('Here is the enhanced prompt:'):
                enhanced = enhanced.replace('Here is the enhanced prompt:', '').strip()
        except Exception as llm_error:
            logger.error(f"LLM router error: {llm_error}")
            raise ValueError("LLM router failed")
        
        return jsonify({
            'original': prompt,
            'enhanced': enhanced,
            'improvement': len(enhanced) - len(prompt),
            'improvement_percentage': round(((len(enhanced) - len(prompt)) / len(prompt) * 100) if prompt else 0, 2),
            'model_used': model_used,
            'style': style
        })
        
    except Exception as e:
        logger.error(f"Prompt enhancement error: {str(e)}")
        return jsonify({'error': f'Enhancement failed: {str(e)}'}), 500


@infosight_ai.route('/stats', methods=['GET'])
def get_stats():
    """Get generation statistics."""
    return jsonify({
        'cache_size': len(generator.cache.cache),
        'cache_ttl': generator.cache.ttl,
        'rate_limit_window': generator.rate_limiter.time_window,
        'rate_limit_max': generator.rate_limiter.max_requests,
        'available_styles': list(generator.style_templates.keys()),
        'image_models': generator.image_models,
        'llm_router_available': LLM_ROUTER_AVAILABLE,
        'timestamp': datetime.now().isoformat()
    })

@infosight_ai.route('/styles', methods=['GET'])
def get_styles():
    """Get available content styles."""
    return jsonify({
        'styles': generator.style_templates,
        'count': len(generator.style_templates)
    })

@infosight_ai.route('/batch-generate', methods=['POST'])
@handle_errors
def batch_generate():
    """Generate multiple content pieces in parallel."""
    data = request.get_json()
    prompts = data.get('prompts', [])
    style = data.get('style', 'professional')
    length = data.get('length', 'medium')
    
    if not prompts or len(prompts) > 10:
        return jsonify({'error': 'Provide 1-10 prompts'}), 400
    
    logger.info(f"Batch generation request: {len(prompts)} prompts")
    start_time = time.time()
    
    # Generate in parallel
    futures = []
    for prompt in prompts:
        future = generator.executor.submit(
            generator.generate_text,
            prompt,
            use_cache=True,
            style=style,
            length=length
        )
        futures.append((prompt, future))
    
    results = []
    for prompt, future in futures:
        try:
            result = future.result(timeout=120)
            if isinstance(result, tuple):
                text, model_used = result
            else:
                text = result
                model_used = "llm_router"
            results.append({
                'prompt': prompt,
                'text': text,
                'model_used': model_used,
                'word_count': len(text.split())
            })
        except Exception as e:
            results.append({
                'prompt': prompt,
                'error': str(e)
            })
    
    elapsed = time.time() - start_time
    return jsonify({
        'results': results,
        'total_time': elapsed,
        'average_time': elapsed / len(prompts) if prompts else 0,
        'success_count': len([r for r in results if 'text' in r])
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

@infosight_ai.route('/history', methods=['GET'])
def get_history():
    """Get user generation history"""
    user_id = request.args.get('user_id', request.remote_addr or 'anonymous')
    content_type = request.args.get('content_type', None)
    limit = request.args.get('limit', 50, type=int)
    
    history = db_manager.get_history(user_id, content_type, limit)
    return jsonify({
        'history': history,
        'count': len(history)
    })

@infosight_ai.route('/favorites', methods=['POST'])
@handle_errors
def add_favorite():
    """Add generation to favorites"""
    data = request.get_json()
    user_id = data.get('user_id', request.remote_addr or 'anonymous')
    prompt = data.get('prompt', '')
    content_type = data.get('content_type', 'text')
    content_data = data.get('content_data', None)
    image_url = data.get('image_url', None)
    
    success = db_manager.add_favorite(user_id, prompt, content_type, content_data, image_url)
    return jsonify({
        'success': success,
        'message': 'Added to favorites!' if success else 'Already in favorites'
    })

@infosight_ai.route('/favorites', methods=['GET'])
def get_favorites():
    """Get user favorites"""
    user_id = request.args.get('user_id', request.remote_addr or 'anonymous')
    content_type = request.args.get('content_type', None)
    
    favorites = db_manager.get_favorites(user_id, content_type)
    return jsonify({
        'favorites': favorites,
        'count': len(favorites)
    })


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