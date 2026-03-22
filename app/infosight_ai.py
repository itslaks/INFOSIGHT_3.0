#infosight_ai.py
import sys
import os
from pathlib import Path

# Add project root to sys.path to allow importing from utils, core, etc.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
except Exception:
    pass
# =========================
# Hugging Face Token Loader
# =========================

import os
import requests

HF_API_TOKEN = None
HF_CONFIGURED = False


def _is_valid_hf_token(token: str) -> bool:
    return bool(token and token.startswith("hf_") and len(token) > 20)


def reload_hf_token():
    """Reload HF token from config or environment variables."""
    global HF_API_TOKEN

    token = None

    # 1. Try loading from Config
    try:
        import sys
        from pathlib import Path

        project_root = Path(__file__).resolve().parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from config import Config
        token = getattr(Config, "HF_API_TOKEN", None)

    except ImportError:
        pass

    # 2. Fallback to environment
    if not token:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        token = (
            os.getenv("HF_API_TOKEN")
            or os.getenv("HUGGINGFACE_API_TOKEN")
        )

    # 3. Normalize and validate
    if token:
        token = token.strip()

    if _is_valid_hf_token(token):
        HF_API_TOKEN = token
        return True

    HF_API_TOKEN = None
    return False


# Initial load
reload_hf_token()

# Log status
if HF_API_TOKEN:
    masked = HF_API_TOKEN[:8] + "..." + HF_API_TOKEN[-4:]
    logger.info(f"✓ HF_API_TOKEN loaded ({masked})")
else:
    logger.error("✗ HF_API_TOKEN missing or invalid")


# =========================
# Validate Hugging Face Token
# =========================

if HF_API_TOKEN:
    try:
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        logger.info("Validating HF_API_TOKEN...")

        r = requests.get(
            "https://huggingface.co/api/whoami-v2",
            headers=headers,
            timeout=10
        )

        if r.status_code == 200 and r.json().get("name"):
            HF_CONFIGURED = True
            user = r.json().get("name", "Unknown")
            logger.info(f"✓ Hugging Face API configured (User: {user})")
        else:
            logger.error(f"✗ Invalid HF token (HTTP {r.status_code})")
            HF_API_TOKEN = None

    except requests.RequestException as e:
        logger.error(f"✗ HF validation request failed: {e}")
        HF_API_TOKEN = None
else:
    logger.error("✗ Hugging Face API: TOKEN NOT SET")


# =========================
# LLM Router
# =========================

try:
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from core.llm_router import generate_text

    LLM_ROUTER_AVAILABLE = True
    logger.info("✓ LLM router available")

except ImportError as e:
    LLM_ROUTER_AVAILABLE = False
    logger.warning(f"LLM router not available: {e}")

    def generate_text(*args, **kwargs):
        return {"response": "", "model": "none", "source": "none"}


GEMINI_CONFIGURED = LLM_ROUTER_AVAILABLE
gemini_model = None

logger.info("=" * 60)

# Safety settings removed - Groq handles content moderation automatically

def check_image_generation_config():
    """Verify image generation is properly configured."""
    if not HF_API_TOKEN:
        logger.error("⚠ HF_API_TOKEN not configured")
        return False

    try:
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        r = requests.get("https://huggingface.co/api/whoami-v2", headers=headers, timeout=10)

        if r.status_code == 200 and r.json().get("name"):
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







    def generate_image(self, prompt, use_cache=True, width=1024, height=1024):
        """
        Generate images with maximum accuracy using multiple providers:
        1. HuggingFace AI (PRIMARY - most accurate for any concept)
        2. Unsplash Stock (SECONDARY - relevant, high-quality photos)
        3. Pexels Stock (TERTIARY - additional stock fallback)
        4. Pixabay Stock (QUATERNARY - free stock photos)
        5. Wikipedia (QUINARY - named entities only)
        
        Required .env variables:
        - HF_API_TOKEN (HuggingFace)
        - UNSPLASH_ACCESS_KEY (Unsplash)
        - PEXELS_API_KEY (Pexels)
        - PIXABAY_API_KEY (Pixabay)
        """
        try:
            # Load API keys from environment
            HF_API_TOKEN = os.getenv('HF_API_TOKEN')
            UNSPLASH_ACCESS_KEY = os.getenv('UNSPLASH_ACCESS_KEY')
            PEXELS_API_KEY = os.getenv('PEXELS_API_KEY')
            
            prompt = self._sanitize_prompt(prompt)
            cache_key = self._create_cache_key(prompt, 'image')
            
            if use_cache:
                cached = self.cache.get(cache_key)
                if cached:
                    logger.info("Returning cached image")
                    return cached

            if not self.rate_limiter.can_proceed('image'):
                wait_time = self.rate_limiter.wait_time('image')
                raise ValueError(f"Rate limit exceeded. Please wait {wait_time:.0f} seconds.")

            logger.info(f"Generating image: {prompt[:50]}...")

            # Clamp dimensions
            width = min(max(int(width), 512), 1024)
            height = min(max(int(height), 512), 1024)
            
            prompt_lower = prompt.lower()

            # ═══════════════════════════════════════════════════════════════════════
            # CONTENT TYPE DETECTION
            # ═══════════════════════════════════════════════════════════════════════
            
            is_question = any(q in prompt_lower for q in [
                'what is', 'what are', 'explain', 'how does', 'how do',
                'tell me about', 'describe', 'definition of', 'concept of'
            ])
            
            is_diagram = any(k in prompt_lower for k in [
                'diagram', 'architecture', 'flowchart', 'infographic', 'schematic',
                'blueprint', 'chart', 'graph', 'visualization', 'structure'
            ])
            
            is_portrait = any(w in prompt_lower for w in [
                'person', 'man', 'woman', 'portrait', 'face', 'people', 
                'selfie', 'headshot', 'photo of'
            ])
            
            is_landscape = any(w in prompt_lower for w in [
                'landscape', 'nature', 'scenery', 'mountains', 'city', 'forest',
                'ocean', 'sky', 'sunset', 'sunrise', 'beach', 'vista', 'view'
            ])
            
            is_food = any(w in prompt_lower for w in [
                'food', 'cake', 'muffin', 'pizza', 'burger', 'coffee', 'drink',
                'meal', 'recipe', 'cooking', 'dessert', 'bread', 'fruit', 'dish'
            ])
            
            is_vehicle = any(w in prompt_lower for w in [
                'car', 'bike', 'motorcycle', 'truck', 'vehicle', 'racing',
                'formula', 'motogp', 'race', 'auto', 'transport', 'f1'
            ])
            
            is_character = any(w in prompt_lower for w in [
                'hulk', 'superhero', 'character', 'hero', 'villain', 'comic',
                'marvel', 'dc', 'anime', 'cartoon', 'mascot'
            ])
            
            is_tech = any(w in prompt_lower for w in [
                'technology', 'computer', 'robot', 'ai', 'circuit', 'digital',
                'cyber', 'futuristic', 'machine', 'device'
            ])
            
            is_space = any(w in prompt_lower for w in [
                'space', 'rocket', 'spaceship', 'satellite', 'planet', 'galaxy',
                'astronaut', 'cosmos', 'spacecraft', 'launch', 'nasa'
            ])
            
            is_animal = any(w in prompt_lower for w in [
                'animal', 'dog', 'cat', 'bird', 'lion', 'tiger', 'elephant',
                'wildlife', 'pet', 'creature', 'beast'
            ])
            
            is_business = any(w in prompt_lower for w in [
                'business', 'office', 'meeting', 'corporate', 'work',
                'professional', 'conference', 'team', 'entrepreneur'
            ])
            
            is_sports = any(w in prompt_lower for w in [
                'sport', 'sports', 'football', 'basketball', 'tennis', 'cricket',
                'athlete', 'game', 'match', 'tournament', 'championship'
            ])
            
            # Detect named entities
            known_entities = [
                'elon musk', 'steve jobs', 'bill gates', 'cristiano ronaldo',
                'lionel messi', 'lebron james', 'taylor swift', 'beyonce',
                'nasa', 'spacex', 'tesla', 'apple', 'google', 'microsoft'
            ]
            
            original_words = prompt.strip().split()
            is_named_entity = (
                any(entity in prompt_lower for entity in known_entities) or
                (len(original_words) <= 4 and 
                sum(1 for w in original_words if w and w[0].isupper() and 
                    w.lower() not in {'a','an','the','and','or','of','in','on','at'}) >= 1
                and not is_food and not is_landscape and not is_animal)
            )
            
            logger.info(f"Content type - diagram={is_diagram}, vehicle={is_vehicle}, "
                    f"character={is_character}, space={is_space}, sports={is_sports}, "
                    f"named_entity={is_named_entity}")

            # ═══════════════════════════════════════════════════════════════════════
            # BUILD ENHANCED AI PROMPT
            # ═══════════════════════════════════════════════════════════════════════
            
            # Concept mapping for educational queries
            abstract_concepts = {
                'llm': 'large language model architecture diagram with transformer layers and attention mechanisms',
                'neural network': 'artificial neural network diagram with interconnected nodes and layers',
                'blockchain': 'blockchain technology diagram showing connected blocks with cryptographic hashes',
                'photosynthesis': 'photosynthesis process diagram in plant leaf showing light and dark reactions',
                'mitosis': 'cell division stages diagram showing prophase metaphase anaphase telophase',
                'algorithm': 'algorithm flowchart diagram with decision nodes and process steps',
                'database': 'database schema diagram with tables and relationships',
            }
            
            visual_prompt = prompt
            for concept, visual_desc in abstract_concepts.items():
                if concept in prompt_lower:
                    visual_prompt = visual_desc
                    logger.info(f"Concept '{concept}' → '{visual_desc}'")
                    break
            
            # Enhance prompt based on content type
            negative_prompt = "blurry, low quality, distorted, bad anatomy, watermark, text, signature, ugly"
            
            if is_diagram:
                ai_prompt = (f"{visual_prompt}, professional infographic, clean modern design, "
                            "educational illustration with clear labels, high contrast, "
                            "minimalist style, vector graphics, white background")
                negative_prompt = "photograph, photo, realistic, blurry, dark, messy"
            
            elif is_character:
                ai_prompt = (f"{visual_prompt}, detailed character illustration, dynamic pose, "
                            "vibrant colors, comic book style, high quality digital art, "
                            "dramatic lighting, heroic composition, 4k artwork, full body")
                negative_prompt += ", photograph, realistic photo, blurry, low detail"
            
            elif is_vehicle or 'motogp' in prompt_lower or 'racing' in prompt_lower:
                ai_prompt = (f"{visual_prompt}, professional motorsports photography, dynamic action shot, "
                            "motion blur background, sharp focus on vehicle, dramatic lighting, "
                            "racing livery details, high speed capture, 4k resolution, track racing")
                negative_prompt += ", static, boring, low quality, toy, parked"
            
            elif is_space or 'rocket' in prompt_lower:
                ai_prompt = (f"{visual_prompt}, cinematic space scene, dramatic rocket launch, "
                            "epic composition, volumetric lighting, nasa style photography, "
                            "detailed spacecraft, atmospheric effects, 4k quality, powerful engines")
                negative_prompt += ", cartoon, illustration, low quality, toy"
            
            elif is_sports:
                ai_prompt = (f"{visual_prompt}, professional sports action photography, dynamic movement, "
                            "stadium lighting, athletic performance, dramatic moment, "
                            "sharp focus, intense competition, 4k quality")
                negative_prompt += ", static, posed, boring, low quality"
            
            elif is_portrait or is_named_entity:
                ai_prompt = (f"{visual_prompt}, professional portrait photography, photorealistic, "
                            "studio lighting, 85mm lens, sharp focus, high detail, perfect composition")
                negative_prompt += ", cartoon, illustration, anime, 3d render"
            
            elif is_landscape:
                ai_prompt = (f"{visual_prompt}, stunning landscape photography, golden hour lighting, "
                            "ultra detailed, wide angle, 8K HDR, professional nature photography")
                negative_prompt += ", cartoon, indoor, people, low quality"
            
            elif is_food:
                ai_prompt = (f"{visual_prompt}, professional food photography, appetizing, "
                            "studio lighting, shallow depth of field, culinary art, vibrant colors")
                negative_prompt += ", unappetizing, dark, messy"
            
            elif is_tech:
                ai_prompt = (f"{visual_prompt}, futuristic technology, sleek design, neon accents, "
                            "high-tech aesthetic, professional 3d render, detailed, modern")
                negative_prompt += ", outdated, low-tech, blurry"
            
            elif is_animal:
                ai_prompt = (f"{visual_prompt}, wildlife photography, natural habitat, "
                            "sharp focus, National Geographic style, beautiful lighting")
                negative_prompt += ", cartoon, cage, zoo, unnatural"
            
            elif is_business:
                ai_prompt = (f"{visual_prompt}, professional corporate photography, modern office, "
                            "business environment, clean composition, professional lighting")
                negative_prompt += ", casual, messy, unprofessional"
            
            else:
                ai_prompt = (f"{visual_prompt}, highly detailed, professional quality, "
                            "sharp focus, perfect composition, photorealistic, 8k")

            # ═══════════════════════════════════════════════════════════════════════
            # HELPER FUNCTIONS
            # ═══════════════════════════════════════════════════════════════════════
            
            import urllib.parse as _urlparse
            
            def _try_huggingface_2026(prompt_text, neg_text, w, h):
                """
                Use HuggingFace NEW router (router.huggingface.co) with working 2026 models.
                Old api-inference.huggingface.co models are deprecated (410).
                """
                if not HF_API_TOKEN:
                    logger.warning("HF_API_TOKEN not set in .env file")
                    return None

                # New Inference Providers router endpoint (replaces deprecated api-inference)
                working_models = [
                    "black-forest-labs/FLUX.1-dev",      # Best quality, fal-ai/replicate provider
                    "black-forest-labs/FLUX.1-schnell",  # Fast version
                    "stabilityai/stable-diffusion-xl-base-1.0",  # SDXL via hf-inference
                    "Kwai-Kolors/Kolors",                # Alternative high quality
                ]

                headers = {
                    "Authorization": f"Bearer {HF_API_TOKEN}",
                    "Content-Type": "application/json",
                    "User-Agent": "InfoSightAI/3.0",
                }

                for model_id in working_models:
                    try:
                        logger.info(f"Trying HF router model: {model_id}")

                        is_schnell = "schnell" in model_id.lower()

                        payload = {
                            "inputs": prompt_text[:500],
                            "parameters": {
                                "num_inference_steps": 4 if is_schnell else 20,
                                "guidance_scale": 0.0 if is_schnell else 3.5,
                                "width": min(w, 1024),
                                "height": min(h, 1024),
                            }
                        }

                        # Use new router endpoint
                        url = f"https://router.huggingface.co/hf-inference/models/{model_id}"

                        response = requests.post(
                            url,
                            headers=headers,
                            json=payload,
                            timeout=90
                        )

                        if response.status_code == 200:
                            content_type = response.headers.get("Content-Type", "")
                            if "image" in content_type and len(response.content) > 5000:
                                logger.info(f"✓ HF router SUCCESS: {model_id} ({len(response.content):,} bytes)")
                                self.cache.set(cache_key + "_source", f"huggingface:{model_id}")
                                return response.content

                        elif response.status_code == 503:
                            logger.info(f"HF model {model_id} loading (503), trying next...")
                            continue
                        elif response.status_code == 401:
                            logger.error("HF 401 - Invalid token. Check HF_API_TOKEN in .env")
                            return None
                        elif response.status_code == 429:
                            logger.warning("HF rate limit reached")
                            break
                        elif response.status_code == 422:
                            logger.warning(f"HF {model_id}: params not supported (422), trying next...")
                            # Retry without extra params
                            payload_simple = {"inputs": prompt_text[:500]}
                            r2 = requests.post(url, headers=headers, json=payload_simple, timeout=90)
                            if r2.status_code == 200 and "image" in r2.headers.get("Content-Type","") and len(r2.content) > 5000:
                                logger.info(f"✓ HF router SUCCESS (simple): {model_id}")
                                self.cache.set(cache_key + "_source", f"huggingface:{model_id}")
                                return r2.content
                            continue
                        else:
                            logger.warning(f"HF {model_id}: HTTP {response.status_code} - {response.text[:100]}")
                            continue

                    except requests.exceptions.Timeout:
                        logger.warning(f"HF {model_id} timeout, trying next...")
                        continue
                    except Exception as e:
                        logger.warning(f"HF {model_id} error: {e}")
                        continue

                logger.warning("All HF router models failed")
                return None
            
            def _try_unsplash_stock(search_query, w, h):
                """
                Fetch relevant stock photos from Unsplash API.
                FREE tier: 50 requests/hour
                Requires UNSPLASH_ACCESS_KEY in .env
                """
                try:
                    if not UNSPLASH_ACCESS_KEY:
                        logger.info("UNSPLASH_ACCESS_KEY not set in .env file, skipping...")
                        return None
                    
                    # Clean search query - extract main keywords
                    stop_words = {'a', 'an', 'the', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for', 'with'}
                    words = [word for word in search_query.lower().split() if word not in stop_words]
                    clean_query = ' '.join(words[:3]) if words else search_query
                    
                    logger.info(f"Searching Unsplash for: '{clean_query}'")
                    
                    # Unsplash Search API
                    headers = {
                        "Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}",
                        "Accept-Version": "v1"
                    }
                    
                    params = {
                        "query": clean_query,
                        "per_page": 1,
                        "orientation": "landscape" if w >= h else "portrait"
                    }
                    
                    resp = requests.get(
                        "https://api.unsplash.com/search/photos",
                        headers=headers,
                        params=params,
                        timeout=10
                    )
                    
                    if resp.status_code != 200:
                        logger.warning(f"Unsplash API returned {resp.status_code}")
                        return None
                    
                    data = resp.json()
                    results = data.get("results", [])
                    
                    if not results:
                        logger.warning(f"No Unsplash results for '{clean_query}'")
                        return None
                    
                    # Get the first (best) result
                    photo = results[0]
                    image_url = photo["urls"]["regular"]  # High quality, ~1080px
                    
                    logger.info(f"Found Unsplash photo by {photo.get('user', {}).get('name', 'Unknown')}")
                    
                    # Download the image
                    img_resp = requests.get(
                        image_url,
                        timeout=15,
                        headers={"User-Agent": "InfoSightAI/3.0"}
                    )
                    
                    if img_resp.status_code == 200 and len(img_resp.content) > 10000:
                        logger.info(f"✓ Unsplash ({len(img_resp.content):,} bytes)")
                        self.cache.set(cache_key + "_source", "unsplash")
                        return img_resp.content
                    
                    return None
                
                except Exception as e:
                    logger.warning(f"Unsplash error: {e}")
                    return None
            
            def _try_pexels_stock(search_query, w, h):
                """
                Fetch relevant stock photos from Pexels API.
                FREE tier: 200 requests/hour, 20,000/month
                Requires PEXELS_API_KEY in .env
                """
                try:
                    if not PEXELS_API_KEY:
                        logger.info("PEXELS_API_KEY not set in .env file, trying public fallback...")
                        return _try_pexels_public(search_query, w, h)
                    
                    # Clean search query
                    stop_words = {'a', 'an', 'the', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for', 'with'}
                    words = [word for word in search_query.lower().split() if word not in stop_words]
                    clean_query = ' '.join(words[:3]) if words else search_query
                    
                    logger.info(f"Searching Pexels for: '{clean_query}'")
                    
                    headers = {
                        "Authorization": PEXELS_API_KEY
                    }
                    
                    params = {
                        "query": clean_query,
                        "per_page": 1,
                        "orientation": "landscape" if w >= h else "portrait",
                        "size": "large"
                    }
                    
                    resp = requests.get(
                        "https://api.pexels.com/v1/search",
                        headers=headers,
                        params=params,
                        timeout=10
                    )
                    
                    if resp.status_code != 200:
                        logger.warning(f"Pexels API returned {resp.status_code}")
                        return None
                    
                    data = resp.json()
                    photos = data.get("photos", [])
                    
                    if not photos:
                        logger.warning(f"No Pexels results for '{clean_query}'")
                        return None
                    
                    # Get best quality image URL
                    photo = photos[0]
                    image_url = photo["src"]["large2x"]  # Highest quality available
                    
                    logger.info(f"Found Pexels photo by {photo.get('photographer', 'Unknown')}")
                    
                    # Download the image
                    img_resp = requests.get(
                        image_url,
                        timeout=15,
                        headers={"User-Agent": "InfoSightAI/3.0"}
                    )
                    
                    if img_resp.status_code == 200 and len(img_resp.content) > 10000:
                        logger.info(f"✓ Pexels ({len(img_resp.content):,} bytes)")
                        self.cache.set(cache_key + "_source", "pexels")
                        return img_resp.content
                    
                    return None
                
                except Exception as e:
                    logger.warning(f"Pexels error: {e}")
                    return None
            
            def _try_pexels_public(search_query, w, h):
                """
                Fallback: Try Pexels public/curated photos endpoint.
                No API key required but less targeted results.
                """
                try:
                    import hashlib as _hl
                    
                    # Map keywords to Pexels popular search terms
                    keyword_map = {
                        'motogp': 'motorcycle racing',
                        'f1': 'formula one racing',
                        'hulk': 'strong man green',
                        'rocket': 'rocket launch space',
                        'spaceship': 'spacecraft',
                        'car': 'sports car',
                        'bike': 'motorcycle',
                    }
                    
                    query = search_query.lower()
                    for key, replacement in keyword_map.items():
                        if key in query:
                            query = replacement
                            break
                    
                    # Use hash to get consistent results for same query
                    query_hash = int(_hl.md5(query.encode()).hexdigest()[:8], 16)
                    page = (query_hash % 10) + 1  # Pages 1-10
                    
                    # Pexels curated photos (no API key needed, but generic)
                    url = f"https://www.pexels.com/search/{_urlparse.quote(query)}/"
                    
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    }
                    
                    resp = requests.get(url, headers=headers, timeout=10)
                    
                    if resp.status_code == 200:
                        # Parse HTML to extract image URLs
                        import re
                        img_pattern = r'https://images\.pexels\.com/photos/\d+/[^"]+\.jpeg\?[^"]*w=1280'
                        matches = re.findall(img_pattern, resp.text)
                        
                        if matches:
                            # Get first good quality image
                            img_url = matches[0]
                            img_resp = requests.get(img_url, headers=headers, timeout=15)
                            
                            if img_resp.status_code == 200 and len(img_resp.content) > 10000:
                                logger.info(f"✓ Pexels public ({len(img_resp.content):,} bytes)")
                                self.cache.set(cache_key + "_source", "pexels_public")
                                return img_resp.content
                    
                    return None
                
                except Exception as e:
                    logger.warning(f"Pexels public fallback error: {e}")
                    return None
            
            def _try_wikimedia_commons(search_query, w, h):
                """
                Search Wikimedia Commons for images - 100% FREE, no API key needed.
                90+ million freely licensed media files.
                """
                try:
                    # Clean query for search
                    stop_words = {'a', 'an', 'the', 'and', 'or', 'of', 'in', 'on', 'at'}
                    words = [word for word in search_query.lower().split() if word not in stop_words]
                    clean_query = ' '.join(words[:4]) if words else search_query

                    logger.info(f"Searching Wikimedia Commons for: '{clean_query}'")

                    # Step 1: Search for image files
                    search_params = {
                        "action": "query",
                        "list": "search",
                        "srsearch": f"{clean_query} filetype:bitmap",
                        "srnamespace": "6",  # File namespace only
                        "srlimit": "5",
                        "format": "json"
                    }

                    headers = {"User-Agent": "InfoSightAI/3.0 (educational project)"}

                    search_resp = requests.get(
                        "https://commons.wikimedia.org/w/api.php",
                        params=search_params,
                        headers=headers,
                        timeout=10
                    )

                    if search_resp.status_code != 200:
                        return None

                    search_data = search_resp.json()
                    results = search_data.get("query", {}).get("search", [])

                    if not results:
                        logger.warning(f"No Wikimedia Commons results for '{clean_query}'")
                        return None

                    # Step 2: Get image URL from first result
                    for result in results:
                        title = result.get("title", "")
                        if not title.startswith("File:"):
                            continue

                        # Get image info
                        info_params = {
                            "action": "query",
                            "titles": title,
                            "prop": "imageinfo",
                            "iiprop": "url|size|mime",
                            "iiurlwidth": min(w, 1280),
                            "format": "json"
                        }

                        info_resp = requests.get(
                            "https://commons.wikimedia.org/w/api.php",
                            params=info_params,
                            headers=headers,
                            timeout=10
                        )

                        if info_resp.status_code != 200:
                            continue

                        info_data = info_resp.json()
                        pages = info_data.get("query", {}).get("pages", {})

                        for page_id, page in pages.items():
                            imageinfo = page.get("imageinfo", [])
                            if not imageinfo:
                                continue

                            info = imageinfo[0]
                            mime = info.get("mime", "")

                            # Only use actual images, not SVG/PDF
                            if "image" not in mime or "svg" in mime:
                                continue

                            thumb_url = info.get("thumburl") or info.get("url")
                            if not thumb_url:
                                continue

                            # Download the image
                            img_resp = requests.get(
                                thumb_url,
                                headers=headers,
                                timeout=15
                            )

                            if img_resp.status_code == 200 and len(img_resp.content) > 10000:
                                logger.info(f"✓ Wikimedia Commons ({len(img_resp.content):,} bytes) - {title}")
                                self.cache.set(cache_key + "_source", "wikimedia_commons")
                                return img_resp.content

                    logger.warning("No usable images found on Wikimedia Commons")
                    return None

                except Exception as e:
                    logger.warning(f"Wikimedia Commons error: {e}")
                    return None

            def _try_lorem_picsum(w, h):
                """
                Lorem Picsum - random high quality photos, no API key needed.
                Use as absolute last stock fallback before failure.
                https://picsum.photos
                """
                try:
                    logger.info(f"Trying Lorem Picsum fallback ({w}x{h})...")

                    # Use seed based on prompt for consistency
                    import hashlib
                    seed = int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16) % 1000

                    url = f"https://picsum.photos/seed/{seed}/{min(w,1024)}/{min(h,1024)}"

                    img_resp = requests.get(
                        url,
                        timeout=15,
                        headers={"User-Agent": "InfoSightAI/3.0"},
                        allow_redirects=True
                    )

                    if img_resp.status_code == 200 and len(img_resp.content) > 10000:
                        logger.info(f"✓ Lorem Picsum fallback ({len(img_resp.content):,} bytes)")
                        self.cache.set(cache_key + "_source", "picsum")
                        self.cache.set(cache_key + "_fallback", True)
                        return img_resp.content

                    return None

                except Exception as e:
                    logger.warning(f"Lorem Picsum error: {e}")
                    return None
            
            def _try_wikipedia_image(entity_name):
                """
                Wikipedia image for famous entities.
                No API key required.
                """
                try:
                    import re
                    
                    api_url = "https://en.wikipedia.org/w/api.php"
                    params = {
                        "action": "query",
                        "titles": entity_name,
                        "prop": "pageimages",
                        "pithumbsize": 1024,
                        "format": "json",
                        "redirects": 1,
                    }
                    
                    resp = requests.get(api_url, params=params, timeout=10)
                    if resp.status_code != 200:
                        return None
                    
                    data = resp.json()
                    pages = data.get("query", {}).get("pages", {})
                    
                    for page_id, page in pages.items():
                        if page_id == "-1":
                            return None
                        
                        thumb = page.get("thumbnail", {})
                        if thumb and thumb.get("source"):
                            thumb_url = thumb["source"]
                            thumb_url = re.sub(r'/(\d+)px-', '/1200px-', thumb_url)
                            
                            img_resp = requests.get(thumb_url, timeout=12, allow_redirects=True)
                            if img_resp.status_code == 200 and len(img_resp.content) > 10000:
                                logger.info(f"✓ Wikipedia ({len(img_resp.content):,} bytes)")
                                self.cache.set(cache_key + "_source", "wikipedia")
                                return img_resp.content
                    
                    return None
                
                except Exception as e:
                    logger.warning(f"Wikipedia error: {e}")
                    return None

            # ═══════════════════════════════════════════════════════════════════════
            # SMART ROUTING: Person → Real Photo | Everything else → AI Generation
            # ═══════════════════════════════════════════════════════════════════════

            # Detect if prompt is a person's name
            # Heuristic: 1-4 words, mostly capitalized, no action/descriptive words
            action_words = {
                'flying', 'running', 'jumping', 'eating', 'swimming', 'fighting',
                'dancing', 'sitting', 'standing', 'burning', 'glowing', 'attacking',
                'dragon', 'robot', 'monster', 'alien', 'fire', 'ice', 'magic',
                'forest', 'ocean', 'space', 'city', 'mountain', 'galaxy', 'universe',
                'diagram', 'chart', 'graph', 'architecture', 'flowchart',
                'car', 'bike', 'rocket', 'spaceship', 'animal', 'dog', 'cat',
                'in', 'on', 'at', 'with', 'and', 'or', 'the', 'a', 'an',
                'beautiful', 'stunning', 'epic', 'dark', 'bright', 'colorful',
                'landscape', 'portrait', 'abstract', 'realistic', 'cartoon'
            }

            words = prompt.strip().split()

            # Normalize to title case for Wikipedia — handles any case combination
            normalized_name = ' '.join(w.capitalize() for w in words)

            # Lowercase version for all comparisons
            prompt_lower_stripped = ' '.join(w.lower() for w in words)

            # Non-person keywords — if any of these exist, it's NOT a person name
            non_person_keywords = {
                'flying', 'running', 'jumping', 'eating', 'swimming', 'fighting',
                'dancing', 'sitting', 'standing', 'burning', 'glowing', 'attacking',
                'riding', 'driving', 'climbing', 'playing', 'singing', 'cooking',
                'dragon', 'robot', 'monster', 'alien', 'zombie', 'vampire', 'wizard',
                'fairy', 'ghost', 'demon', 'angel', 'phoenix', 'unicorn',
                'fire', 'ice', 'water', 'earth', 'wind', 'forest', 'ocean', 'desert',
                'space', 'city', 'mountain', 'galaxy', 'universe', 'island', 'jungle',
                'diagram', 'chart', 'graph', 'architecture', 'flowchart', 'logo',
                'car', 'bike', 'rocket', 'spaceship', 'sword', 'gun', 'shield',
                'house', 'castle', 'tower', 'bridge', 'ship', 'plane', 'train',
                'dog', 'cat', 'lion', 'tiger', 'elephant', 'horse', 'wolf',
                'eagle', 'snake', 'bear', 'shark', 'whale', 'fox', 'deer',
                'abstract', 'realistic', 'cartoon', 'anime', 'dark', 'bright',
                'colorful', 'vintage', 'futuristic', 'magical', 'epic', 'giant',
                'in', 'on', 'at', 'with', 'and', 'or', 'the', 'a', 'an', 'of',
            }

            all_alpha = all(w.isalpha() for w in words)
            has_non_person_word = any(w.lower() in non_person_keywords for w in words)

            # Quick pre-check: could this possibly be a person name?
            could_be_person = (
                1 <= len(words) <= 4 and
                all_alpha and
                not has_non_person_word and
                not is_diagram and
                not is_food and
                not is_vehicle and
                not is_space and
                not is_animal and
                not is_landscape and
                not is_character and
                not is_tech and
                not is_sports
            )

            # ── WIKIPEDIA PERSON VERIFICATION ───────────────────────────────
            # If it looks like it could be a name, ask Wikipedia
            # Wikipedia will confirm if it's a real person
            is_person_name = False

            def _verify_person_via_wikipedia(name):
                """
                Ask Wikipedia if this name is a real person.
                Returns True if Wikipedia finds a person page for this name.
                Fast — only fetches page categories/description, not full content.
                """
                try:
                    # Step 1: Search Wikipedia for the name
                    search_params = {
                        "action": "query",
                        "list": "search",
                        "srsearch": name,
                        "srlimit": 3,
                        "format": "json"
                    }
                    headers = {"User-Agent": "InfoSightAI/3.0 (educational project)"}

                    search_resp = requests.get(
                        "https://en.wikipedia.org/w/api.php",
                        params=search_params,
                        headers=headers,
                        timeout=5
                    )

                    if search_resp.status_code != 200:
                        return False

                    results = search_resp.json().get("query", {}).get("search", [])
                    if not results:
                        return False

                    # Step 2: Check the top result's categories for person indicators
                    top_title = results[0].get("title", "")

                    category_params = {
                        "action": "query",
                        "titles": top_title,
                        "prop": "categories|extracts",
                        "cllimit": 10,
                        "exintro": True,
                        "exsentences": 2,
                        "explaintext": True,
                        "format": "json"
                    }

                    cat_resp = requests.get(
                        "https://en.wikipedia.org/w/api.php",
                        params=category_params,
                        headers=headers,
                        timeout=5
                    )

                    if cat_resp.status_code != 200:
                        return False

                    pages = cat_resp.json().get("query", {}).get("pages", {})

                    for page_id, page in pages.items():
                        if page_id == "-1":
                            continue

                        # Check categories for person indicators
                        categories = [
                            c.get("title", "").lower()
                            for c in page.get("categories", [])
                        ]

                        person_category_keywords = [
                            'births', 'deaths', 'living people', 'people from',
                            'alumni', 'cricketers', 'footballers', 'actors',
                            'actresses', 'politicians', 'businesspeople', 'musicians',
                            'singers', 'athletes', 'players', 'entrepreneurs',
                            'engineers', 'scientists', 'directors', 'writers',
                            'authors', 'journalists', 'models', 'sportspeople',
                        ]

                        for cat in categories:
                            if any(kw in cat for kw in person_category_keywords):
                                logger.info(f"✓ Wikipedia confirms '{name}' is a person (category: {cat})")
                                return True

                        # Also check the extract text for person indicators
                        extract = page.get("extract", "").lower()
                        person_text_indicators = [
                            'is an indian', 'is a indian', 'is an american', 'is a british',
                            'born in', 'born on', 'is a cricketer', 'is an actor',
                            'is a politician', 'is a businessman', 'is a singer',
                            'is a footballer', 'is an entrepreneur', 'is a director',
                            'is a scientist', 'is a musician', 'is a writer',
                            'is the ceo', 'is the founder', 'is the president',
                            'he is', 'she is', 'his career', 'her career',
                        ]

                        if any(indicator in extract for indicator in person_text_indicators):
                            logger.info(f"✓ Wikipedia extract confirms '{name}' is a person")
                            return True

                    return False

                except Exception as e:
                    logger.warning(f"Wikipedia person verify error: {e}")
                    return False

            if could_be_person:
                logger.info(f"🔍 Checking Wikipedia if '{normalized_name}' is a real person...")
                is_person_name = _verify_person_via_wikipedia(normalized_name)
                if not is_person_name:
                    logger.info(f"❌ Wikipedia: '{normalized_name}' is NOT a person → AI generation")
                else:
                    logger.info(f"✅ Wikipedia: '{normalized_name}' IS a person → real photo chain")
            else:
                logger.info(f"⏭️  Skipping person check — keywords indicate non-person content")

            logger.info(f"🧠 Smart routing — is_person_name={is_person_name}, "
                       f"normalized='{normalized_name}', could_be_person={could_be_person}")
            if is_person_name:
                logger.info("👤 PERSON detected → Real photo priority chain")
                logger.info("=" * 70)

                # [1] Wikipedia (most accurate - use normalized Title Case name)
                logger.info(f"📖 [1/4] Trying Wikipedia for '{normalized_name}'...")
                wiki_result = _try_wikipedia_image(normalized_name)
                if wiki_result and len(wiki_result) > 10000:
                    logger.info("✅ SUCCESS: Wikipedia")
                    if use_cache:
                        self.cache.set(cache_key, wiki_result)
                    return wiki_result
                logger.info("❌ Wikipedia failed")

                # [2] Wikimedia Commons (use normalized name)
                logger.info(f"🌐 [2/4] Trying Wikimedia Commons for '{normalized_name}'...")
                wikimedia_result = _try_wikimedia_commons(normalized_name, width, height)
                if wikimedia_result and len(wikimedia_result) > 10000:
                    logger.info("✅ SUCCESS: Wikimedia Commons")
                    if use_cache:
                        self.cache.set(cache_key, wikimedia_result)
                    return wikimedia_result
                logger.info("❌ Wikimedia Commons failed")

                # [3] Unsplash (press/event photos - use normalized name)
                logger.info(f"📷 [3/4] Trying Unsplash for '{normalized_name}'...")
                unsplash_result = _try_unsplash_stock(normalized_name, width, height)
                if unsplash_result and len(unsplash_result) > 10000:
                    logger.info("✅ SUCCESS: Unsplash")
                    if use_cache:
                        self.cache.set(cache_key, unsplash_result)
                    return unsplash_result
                logger.info("❌ Unsplash failed")

                # [4] Pexels (use normalized name)
                logger.info(f"📷 [4/4] Trying Pexels for '{normalized_name}'...")
                pexels_result = _try_pexels_stock(normalized_name, width, height)
                if pexels_result and len(pexels_result) > 10000:
                    logger.info("✅ SUCCESS: Pexels")
                    if use_cache:
                        self.cache.set(cache_key, pexels_result)
                    return pexels_result
                logger.info("❌ Pexels failed")
            else:
                # ── NON-PERSON: AI generation priority chain ───────────────────
                logger.info("🎨 NON-PERSON detected → AI generation priority chain")
                logger.info("=" * 70)

                # [1] HuggingFace AI (best for creative/specific requests)
                logger.info("🎨 [1/6] Trying HuggingFace AI...")
                hf_result = _try_huggingface_2026(ai_prompt, negative_prompt, width, height)
                if hf_result and len(hf_result) > 5000:
                    logger.info("✅ SUCCESS: HuggingFace AI")
                    if use_cache:
                        self.cache.set(cache_key, hf_result)
                    return hf_result
                logger.info("❌ HuggingFace failed")

                # [2] Unsplash
                logger.info("📷 [2/6] Trying Unsplash...")
                unsplash_result = _try_unsplash_stock(prompt, width, height)
                if unsplash_result and len(unsplash_result) > 10000:
                    logger.info("✅ SUCCESS: Unsplash")
                    if use_cache:
                        self.cache.set(cache_key, unsplash_result)
                    return unsplash_result
                logger.info("❌ Unsplash failed")

                # [3] Pexels
                logger.info("📷 [3/6] Trying Pexels...")
                pexels_result = _try_pexels_stock(prompt, width, height)
                if pexels_result and len(pexels_result) > 10000:
                    logger.info("✅ SUCCESS: Pexels")
                    if use_cache:
                        self.cache.set(cache_key, pexels_result)
                    return pexels_result
                logger.info("❌ Pexels failed")

                # [4] Wikimedia Commons
                logger.info("🌐 [4/6] Trying Wikimedia Commons...")
                wikimedia_result = _try_wikimedia_commons(prompt, width, height)
                if wikimedia_result and len(wikimedia_result) > 10000:
                    logger.info("✅ SUCCESS: Wikimedia Commons")
                    if use_cache:
                        self.cache.set(cache_key, wikimedia_result)
                    return wikimedia_result
                logger.info("❌ Wikimedia Commons failed")

                # [5] Wikipedia (for named concepts/entities)
                if is_named_entity:
                    logger.info("📖 [5/6] Trying Wikipedia...")
                    wiki_result = _try_wikipedia_image(prompt.strip())
                    if wiki_result and len(wiki_result) > 10000:
                        logger.info("✅ SUCCESS: Wikipedia")
                        if use_cache:
                            self.cache.set(cache_key, wiki_result)
                        return wiki_result
                    logger.info("❌ Wikipedia failed")

                # [6] Lorem Picsum (last resort)
                logger.info("🎲 [6/6] Trying Lorem Picsum last resort...")
                picsum_result = _try_lorem_picsum(width, height)
                if picsum_result and len(picsum_result) > 10000:
                    logger.info("✅ SUCCESS: Lorem Picsum fallback")
                    if use_cache:
                        self.cache.set(cache_key, picsum_result)
                    return picsum_result

            # All providers failed
            logger.error("🆘 All providers failed - no image could be generated")

            missing_keys = []
            if not HF_API_TOKEN:
                missing_keys.append("HF_API_TOKEN")
            if not UNSPLASH_ACCESS_KEY:
                missing_keys.append("UNSPLASH_ACCESS_KEY")

            error_msg = "Image generation failed across all providers.\n\n"
            if missing_keys:
                error_msg += f"Missing API keys: {', '.join(missing_keys)}\n"
            raise ValueError(error_msg)

        except Exception as e:
            logger.error(f"generate_image error: {str(e)}", exc_info=True)
            if 'rate limit' in str(e).lower():
                raise ValueError(str(e))
            raise ValueError(f"Image generation failed: {str(e)}")

    def generate_both(self, prompt, use_cache=True):
        """Generate both text and image in parallel with full error handling."""
        prompt = self._sanitize_prompt(prompt)
        logger.info(f"Starting parallel generation for: {prompt[:50]}...")

        text_result  = None
        image_result = None
        text_error   = None
        image_error  = None
        model_used   = "llm_router"

        text_future  = self.executor.submit(self._safe_generate_text,  prompt, use_cache)
        image_future = self.executor.submit(self._safe_generate_image, prompt, use_cache)

        # Collect text result
        try:
            text_result_data, text_error = text_future.result(timeout=90)
            if text_result_data and isinstance(text_result_data, tuple):
                text_result, model_used = text_result_data
            else:
                text_result = text_result_data
        except Exception as e:
            text_error  = f"Text generation timeout: {str(e)}"
            text_result = None
            logger.error(text_error)

        # Collect image result
        try:
            image_result, image_error = image_future.result(timeout=180)
        except Exception as e:
            image_error  = f"Image generation timeout: {str(e)}"
            image_result = None
            logger.error(image_error)

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
                            'configured': bool(HF_API_TOKEN),
                            'working': False,
                            'error': None,
                            'token_format_valid': bool(HF_API_TOKEN and HF_API_TOKEN.startswith("hf_"))
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
        clean_text = re.sub('<[^<]+?>', '', text)
        word_count = len(clean_text.split())
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
    width = data.get('width', 1024)
    height = data.get('height', 1024)
    
    logger.info(f"Image generation request: {prompt[:50]}... ({width}x{height})")
    start_time = time.time()
    
    try:
        image_bytes = generator.generate_image(prompt, use_cache=use_cache, width=width, height=height)
        
        if not image_bytes:
            return jsonify({'error': 'Image generation failed - no image data returned'}), 500
        
        elapsed = time.time() - start_time
        logger.info(f"Image generation completed in {elapsed:.2f}s ({len(image_bytes)} bytes)")
        
        # Encode to base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        image_url = f"data:image/png;base64,{image_base64}"
        cached = generator.cache.get(generator._create_cache_key(prompt, 'image')) is not None
        
        is_fallback = generator.cache.get(generator._create_cache_key(prompt, 'image') + "_fallback") or False
        img_source  = generator.cache.get(generator._create_cache_key(prompt, 'image') + "_source") or ""

        if is_fallback or img_source in ('picsum', 'picsum_stock'):
            model_label = 'picsum_stock'
        elif img_source.startswith('huggingface'):
            model_label = 'huggingface'
        elif img_source == 'unsplash':
            model_label = 'unsplash'
        elif img_source in ('pexels', 'pexels_public'):
            model_label = 'pexels'
        elif img_source == 'wikimedia_commons':
            model_label = 'wikimedia_commons'
        elif img_source == 'wikipedia':
            model_label = 'wikipedia'
        else:
            model_label = img_source or 'unknown'

        # Save to database
        user_id = request.remote_addr or 'anonymous'
        db_manager.save_generation(
            user_id=user_id,
            prompt=prompt,
            content_type='image',
            image_url=image_url,
            model_used=model_label,
            generation_time=elapsed,
            cached=cached
        )

        return jsonify({
            'image_url': image_url,
            'cached': cached,
            'generation_time': elapsed,
            'image_size': len(image_bytes),
            'model_used': model_label,
            'is_fallback': bool(is_fallback)
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
    width = data.get('width', 1024)
    height = data.get('height', 1024)
    
    logger.info(f"Combined generation request: {prompt[:50]}... ({width}x{height})")
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

        # Resolve actual image source label
        is_fallback = generator.cache.get(generator._create_cache_key(prompt, 'image') + "_fallback") or False
        img_source  = generator.cache.get(generator._create_cache_key(prompt, 'image') + "_source") or ""

        if is_fallback or img_source in ('picsum', 'picsum_stock'):
            response['model_used'] = 'picsum_stock'
            response['is_fallback'] = True
        elif img_source.startswith('huggingface'):
            response['model_used'] = 'huggingface'
            response['is_fallback'] = False
        elif img_source == 'unsplash':
            response['model_used'] = 'unsplash'
            response['is_fallback'] = True
        elif img_source in ('pexels', 'pexels_public'):
            response['model_used'] = 'pexels'
            response['is_fallback'] = True
        elif img_source == 'wikimedia_commons':
            response['model_used'] = 'wikimedia_commons'
            response['is_fallback'] = True
        elif img_source == 'wikipedia':
            response['model_used'] = 'wikipedia'
            response['is_fallback'] = True
        else:
            response['model_used'] = img_source or 'unknown'
            response['is_fallback'] = False

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

if __name__ == "__main__":
    from flask import Flask
    app = Flask(__name__, template_folder='templates')
    app.config.setdefault('SECRET_KEY', os.getenv('FLASK_SECRET_KEY', 'dev-key-for-standalone-mode'))
    app.register_blueprint(infosight_ai)
    host = os.getenv('APP_HOST','127.0.0.1')
    port = int(os.getenv('APP_PORT', '5006'))
    print(f'Starting infosight_ai standalone mode on {host}:{port}...')
    app.run(debug=True, host=host, port=port, threaded=True)
