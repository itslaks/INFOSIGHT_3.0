from flask import Flask, render_template, jsonify, Blueprint, request, send_file, g
from flask_cors import CORS
import speech_recognition as sr
import pyttsx3
import threading
import queue
import time
import os
import requests
import tempfile
import pygame
from gtts import gTTS
try:
    import whisper
    WHISPER_AVAILABLE = True
except (ImportError, TypeError, OSError) as e:
    whisper = None
    WHISPER_AVAILABLE = False
    # Logger not yet defined, will log later
import librosa
import soundfile as sf
import langdetect
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import concurrent.futures
from collections import defaultdict
import logging
import sqlite3
import hashlib
import re
from typing import Optional, Dict, Any, Tuple, List
from fuzzywuzzy import fuzz
from difflib import SequenceMatcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import centralized security utilities
try:
    from utils.security import rate_limit_api, validate_request, InputValidator
except ImportError:
    logger.warning("Security utils not found, using fallback")
    def rate_limit_api(*args, **kwargs):
        def decorator(f):
            return f
        return decorator
    
    def validate_request(*args, **kwargs):
        def decorator(f):
            return f
        return decorator
    
    class InputValidator:
        @staticmethod
        def validate_string(value, name, max_length=1000, required=False):
            if value is None:
                if required:
                    raise ValueError(f"{name} is required")
                return None
            return str(value)[:max_length]
        
        @staticmethod
        def validate_filename(value, name, required=False):
            if value is None:
                if required:
                    raise ValueError(f"{name} is required")
                return None
            clean = os.path.basename(str(value))
            clean = re.sub(r'[^\w\-\.]', '', clean)
            return clean
        
        @staticmethod
        def validate_integer(value, name, min_value=None, max_value=None, required=False):
            if value is None:
                if required:
                    raise ValueError(f"{name} is required")
                return None
            try:
                val = int(value)
                if min_value is not None and val < min_value:
                    val = min_value
                if max_value is not None and val > max_value:
                    val = max_value
                return val
            except (ValueError, TypeError):
                if required:
                    raise ValueError(f"{name} must be an integer")
                return None

# Groq and Local LLM setup
GROQ_AVAILABLE = False
LOCAL_LLM_AVAILABLE = False

try:
    from groq import Groq
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    if GROQ_API_KEY:
        groq_client = Groq(api_key=GROQ_API_KEY)
        GROQ_AVAILABLE = True
        logger.info("✓ Groq client initialized successfully")
except Exception as e:
    logger.warning(f"⚠ Groq not available: {e}")
    groq_client = None

try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.local_llm_utils import generate_with_ollama, check_ollama_available
    LOCAL_LLM_AVAILABLE = True
    logger.info("✓ Local LLM utilities imported successfully")
except ImportError as e:
    logger.warning(f"⚠ Local LLM not available: {e}")

# Create a blueprint
lana_ai = Blueprint('lana_ai', __name__, template_folder='templates')

# Enable CORS
CORS(lana_ai, resources={
    r"/*": {
        "origins": ["http://localhost:*", "http://127.0.0.1:*"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

# Load API keys
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

NEWS_API_KEY = os.getenv('NEWS_API_KEY')
WEATHER_API_KEY = os.getenv('WEATHER_API_KEY')
SERPAPI_KEY = os.getenv('SERPAPI_KEY')

# Initialize TTS engine
tts_engine = None
tts_lock = threading.Lock()

def init_tts():
    global tts_engine
    if tts_engine is None:
        with tts_lock:
            if tts_engine is None:
                try:
                    tts_engine = pyttsx3.init()
                    voices = tts_engine.getProperty('voices')
                    if voices:
                        for voice in voices:
                            if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                                tts_engine.setProperty('voice', voice.id)
                                break
                    tts_engine.setProperty('rate', 190)
                    tts_engine.setProperty('volume', 1.0)
                    logger.info("✓ TTS engine initialized")
                except Exception as e:
                    logger.error(f"✗ Error initializing TTS: {e}")
                    tts_engine = None
    return tts_engine

# Constants

AUDIO_DIR = "audio"
CACHE_DURATION = 300
MAX_HISTORY = 100
API_TIMEOUT = 8
DB_PATH = "data/lana_ai.db"
RESPONSES_JSON_PATH = "data/responses.json"

TEMP_AUDIO_DIR = "audio/temp"
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

# Language mapping for TTS and detection
LANGUAGE_MAPPING = {
    "en-US": "en",
    "ta-IN": "ta",
    "hi-IN": "hi",
    "ml-IN": "ml",
    "te-IN": "te",
    "kn-IN": "kn",
    "fr-FR": "fr",
    "de-DE": "de",
    "ko-KR": "ko",
    "ja-JP": "ja"
}

REVERSE_LANGUAGE_MAPPING = {
    "en": "en-US",
    "ta": "ta-IN",
    "hi": "hi-IN",
    "ml": "ml-IN",
    "te": "te-IN",
    "kn": "kn-IN",
    "fr": "fr-FR",
    "de": "de-DE",
    "ko": "ko-KR",
    "ja": "ja-JP"
}

FALLBACK_LANGUAGES = {
    "ta-IN": ["en-IN", "en-US"],
    "ml-IN": ["en-IN", "en-US"],
    "te-IN": ["en-IN", "en-US"],
    "kn-IN": ["en-IN", "en-US"],
    "ko-KR": ["en-US"],
    "ja-JP": ["en-US"]
}

# Knowledge base
responses_knowledge_base = []

# Thread pool - Increased workers for better concurrency
executor = concurrent.futures.ThreadPoolExecutor(max_workers=16)

# Conversation state
conversation_history = []
api_cache = {}
stop_event = threading.Event()
user_preferences = {}
long_term_memory = {}
scheduled_reminders = []
db_lock = threading.Lock()

# Queues
audio_queue = queue.Queue()
response_queue = queue.Queue()
transcript_queue = queue.Queue()

is_audio_playing = False
pygame_initialized = False
current_language = "en-US"
audio_data = np.array([])
whisper_enabled = False
whisper_model = None

# Rate limiting
rate_limit_store = defaultdict(lambda: {
    'minute_count': 0,
    'hour_count': 0,
    'minute_reset': datetime.now(),
    'hour_reset': datetime.now()
})

# Ensure directories exist
Path(AUDIO_DIR).mkdir(exist_ok=True)
Path(AUDIO_DIR, "cache").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

# Initialize WhisperASR
whisper_model = None
whisper_enabled = False
if WHISPER_AVAILABLE and whisper is not None:
    try:
        whisper_model = whisper.load_model("base")
        whisper_enabled = True
        logger.info("✓ WhisperASR model loaded successfully")
    except Exception as e:
        whisper_enabled = False
        logger.warning(f"⚠ WhisperASR not available: {e}")
else:
    if not WHISPER_AVAILABLE:
        logger.warning("⚠ WhisperASR not available - whisper module import failed")

def initialize_pygame():
    """Initialize pygame mixer safely"""
    global pygame_initialized
    
    if not pygame_initialized:
        try:
            pygame.mixer.quit()
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)
            pygame_initialized = True
            logger.info("✓ Pygame mixer initialized")
        except Exception as e:
            logger.error(f"✗ Failed to initialize pygame: {e}")
            pygame_initialized = False

initialize_pygame()

# ==================== SQLITE DATABASE ====================
def get_data_path(filename):
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / filename

def init_database():
    try:
        db_path = get_data_path('lana_ai.db')
        
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    user_message TEXT NOT NULL,
                    assistant_message TEXT NOT NULL,
                    model_used TEXT,
                    timestamp REAL NOT NULL,
                    audio_file TEXT,
                    response_time REAL,
                    language TEXT DEFAULT 'en',
                    intent TEXT,
                    sentiment TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_id ON conversations(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON conversations(timestamp DESC)')
            
            conn.commit()
        logger.info("✓ SQLite database initialized")
    except Exception as e:
        logger.error(f"✗ Database initialization error: {e}")

def save_conversation(user_id, user_message, assistant_message, model_used=None, 
                     audio_file=None, response_time=None, language='en', intent=None, sentiment=None):
    try:
        db_path = get_data_path('lana_ai.db')
        
        with db_lock:
            with sqlite3.connect(str(db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO conversations 
                    (user_id, user_message, assistant_message, model_used, timestamp, 
                     audio_file, response_time, language, intent, sentiment)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (user_id, user_message, assistant_message, model_used, 
                      time.time(), audio_file, response_time, language, intent, sentiment))
                conn.commit()
    except Exception as e:
        logger.error(f"✗ Error saving conversation: {e}")

def get_conversation_history(user_id="default", page=1, per_page=20):
    try:
        db_path = get_data_path('lana_ai.db')
        offset = (page - 1) * per_page
        
        with db_lock:
            with sqlite3.connect(str(db_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('SELECT COUNT(*) as total FROM conversations WHERE user_id = ?', (user_id,))
                total = cursor.fetchone()['total']
                
                cursor.execute('''
                    SELECT user_message, assistant_message, model_used, timestamp, 
                           audio_file, response_time, language, intent, sentiment
                    FROM conversations
                    WHERE user_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ? OFFSET ?
                ''', (user_id, per_page, offset))
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        "user": row['user_message'],
                        "assistant": row['assistant_message'],
                        "model_used": row['model_used'],
                        "timestamp": row['timestamp'],
                        "audio_file": row['audio_file'],
                        "response_time": row['response_time'],
                        "language": row['language'],
                        "intent": row['intent'],
                        "sentiment": row['sentiment']
                    })
                
                return {
                    "conversations": results,
                    "total": total,
                    "page": page,
                    "per_page": per_page,
                    "total_pages": (total + per_page - 1) // per_page
                }
    except Exception as e:
        logger.error(f"✗ Error getting conversation history: {e}")
        return {"conversations": [], "total": 0, "page": page, "per_page": per_page, "total_pages": 0}

# Initialize database
init_database()

# ==================== KNOWLEDGE BASE ====================
def load_responses_knowledge_base():
    global responses_knowledge_base
    try:
        responses_path = get_data_path('responses.json')
        
        if responses_path.exists():
            with open(responses_path, 'r', encoding='utf-8') as f:
                responses_knowledge_base = json.load(f)
            logger.info(f"✓ Loaded {len(responses_knowledge_base)} knowledge entries")
        else:
            logger.warning(f"⚠ responses.json not found at {responses_path}")
            responses_knowledge_base = []
    except Exception as e:
        logger.error(f"✗ Error loading responses.json: {e}")
        responses_knowledge_base = []

def search_responses_knowledge_base(query: str) -> Optional[str]:
    if not responses_knowledge_base:
        return None
    
    query_lower = query.lower().strip()
    if not query_lower:
        return None
    
    best_score = 0
    best_answer = None
    
    for entry in responses_knowledge_base:
        if not isinstance(entry, dict):
            continue
        
        question = entry.get('question', '')
        if not isinstance(question, str):
            continue
        
        score = fuzz.token_set_ratio(query_lower, question.lower())
        
        if score > best_score and score >= 60:
            best_score = score
            answer = entry.get('answer', '')
            if isinstance(answer, dict):
                answer_parts = [f"{k}: {v}" for k, v in answer.items()]
                best_answer = '\n'.join(answer_parts)
            else:
                best_answer = str(answer)
    
    if best_answer:
        logger.info(f"📚 Knowledge base HIT (score: {best_score})")
    
    return best_answer

load_responses_knowledge_base()

# ==================== IMPROVED SPEECH RECOGNITION ====================

class ImprovedSpeechRecognizer:
    """Enhanced speech recognition with Whisper, multi-language, and audio visualization"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 2500
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.dynamic_energy_adjustment_damping = 0.15
        self.recognizer.dynamic_energy_ratio = 1.5
        self.recognizer.pause_threshold = 0.7
        self.recognizer.phrase_threshold = 0.3
        self.recognizer.non_speaking_duration = 0.5
    
    def transcribe_with_whisper(self, audio_data) -> Optional[str]:
        """Use Whisper model for transcription"""
        if not whisper_enabled or not whisper_model:
            return None
        
        try:
            wav_data = audio_data.get_wav_data()
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                temp_audio.write(wav_data)
                temp_audio_path = temp_audio.name
            
            try:
                audio, _ = librosa.load(temp_audio_path, sr=16000)
                result = whisper_model.transcribe(audio)
                
                if result and "text" in result:
                    text = result["text"].strip()
                    logger.info(f"✓ Whisper transcription: '{text}'")
                    return text
            finally:
                try:
                    os.unlink(temp_audio_path)
                except:
                    pass
        except Exception as e:
            logger.warning(f"⚠ Whisper error: {e}")
        
        return None
    
    def recognize_with_groq_whisper(self, audio_data) -> Optional[str]:
        """Use Groq's Whisper API for transcription"""
        if not GROQ_AVAILABLE or not groq_client:
            return None
        
        try:
            wav_data = audio_data.get_wav_data()
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                temp_audio.write(wav_data)
                temp_audio_path = temp_audio.name
            
            try:
                with open(temp_audio_path, 'rb') as audio_file:
                    transcription = groq_client.audio.transcriptions.create(
                        file=audio_file,
                        model="whisper-large-v3-turbo",
                        language="en",
                        temperature=0.0,
                        response_format="text"
                    )
                
                text = transcription.strip() if isinstance(transcription, str) else transcription.text.strip()
                
                if text:
                    logger.info(f"✓ Groq Whisper: '{text}'")
                    return text
            finally:
                try:
                    os.unlink(temp_audio_path)
                except:
                    pass
        except Exception as e:
            logger.warning(f"⚠ Groq Whisper error: {e}")
        
        return None
    
    def recognize_with_google(self, audio_data, language='en-IN') -> Optional[str]:
        """Google Speech Recognition with language support"""
        try:
            text = self.recognizer.recognize_google(audio_data, language=language)
            if text:
                logger.info(f"✓ Google ({language}): '{text}'")
                return text.strip()
        except sr.UnknownValueError:
            logger.debug(f"Google couldn't understand ({language})")
        except sr.RequestError as e:
            logger.warning(f"⚠ Google API error: {e}")
        except Exception as e:
            logger.warning(f"⚠ Google error: {e}")
        
        return None
    
    def transcribe(self, timeout=10, phrase_time_limit=15, language="en-US") -> Tuple[str, np.ndarray]:
        """
        Enhanced transcription with audio data return
        Returns: (text, audio_data_array)
        """
        global current_language
        audio_array = np.array([])
        
        # Adjust parameters based on language
        if language in ["ja-JP", "ko-KR", "zh-CN"]:
            self.recognizer.energy_threshold = 280
            self.recognizer.pause_threshold = 1.0
        elif language in ["ta-IN", "ml-IN", "te-IN", "kn-IN", "hi-IN"]:
            self.recognizer.energy_threshold = 270
            self.recognizer.pause_threshold = 0.9
        else:
            self.recognizer.energy_threshold = 2500
            self.recognizer.pause_threshold = 0.7
        
        try:
            with sr.Microphone(sample_rate=16000, chunk_size=1024) as source:
                logger.info(f"🎤 Listening in {language}...")
                
                self.recognizer.adjust_for_ambient_noise(source, duration=0.3)
                
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout, 
                    phrase_time_limit=phrase_time_limit
                )
                
                # Get audio data for visualization
                audio_array = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
                
                logger.info("🔄 Processing audio...")
                
                # Strategy 1: Groq Whisper (fastest, most accurate)
                text = self.recognize_with_groq_whisper(audio)
                if text:
                    return text, audio_array
                
                # Strategy 2: Try primary language with Google
                text = self.recognize_with_google(audio, language=language)
                if text:
                    return text, audio_array
                
                # Strategy 3: Try local Whisper for difficult languages
                if language in ["ta-IN", "ml-IN", "te-IN", "kn-IN", "hi-IN", "ko-KR", "ja-JP"]:
                    text = self.transcribe_with_whisper(audio)
                    if text:
                        return text, audio_array
                
                # Strategy 4: Try fallback languages
                if language in FALLBACK_LANGUAGES:
                    for fallback_lang in FALLBACK_LANGUAGES[language]:
                        text = self.recognize_with_google(audio, language=fallback_lang)
                        if text:
                            return text, audio_array
                
                # Strategy 5: Try English as last resort
                if language != "en-US":
                    text = self.recognize_with_google(audio, language="en-US")
                    if text:
                        return text, audio_array
                
                logger.info("🔇 No speech detected")
                return "", audio_array
                
        except sr.WaitTimeoutError:
            logger.info("⏱️ No speech detected (timeout)")
            return "", audio_array
        except Exception as e:
            logger.error(f"✗ Transcription error: {e}")
            return "", audio_array

# Global recognizer instance
speech_recognizer = ImprovedSpeechRecognizer()

def transcribe_audio_optimized(language="en-US") -> Tuple[str, np.ndarray]:
    """Wrapper for improved speech recognition with audio data"""
    return speech_recognizer.transcribe(timeout=10, phrase_time_limit=15, language=language)

def detect_language(text: str) -> str:
    """Detect language from text"""
    try:
        detected = langdetect.detect(text)
        if detected in REVERSE_LANGUAGE_MAPPING:
            return REVERSE_LANGUAGE_MAPPING[detected]
        return "en-US"
    except:
        return "en-US"

# ==================== AI RESPONSE GENERATION ====================
def generate_ai_response(query: str, context_data=None, user_id="default") -> Tuple[str, str]:
    """
    Generate AI response with Groq primary, Local LLM fallback
    Returns: (response_text, model_used)
    """
    
    # Build context
    context_parts = []
    if context_data and context_data[1]:
        context_parts.append(f"Context: {context_data[1]}")
    
    # Add conversation history (limited for speed)
    try:
        recent = get_conversation_history(user_id=user_id, page=1, per_page=2)
        if recent['conversations']:
            history_text = "\n".join([
                f"User: {c['user']}\nLana: {c['assistant']}"
                for c in reversed(recent['conversations'][:2])
            ])
            context_parts.append(f"Recent:\n{history_text}")
    except:
        pass
    
    context_text = "\n\n".join(context_parts) if context_parts else ""
    
    system_prompt = """You are Lana, an advanced AI assistant. You are helpful, friendly, and conversational.
Respond naturally and concisely in 1-3 sentences unless asked for more detail.
Be empathetic, engaging, and proactive."""
    
    # Try Groq first
    if GROQ_AVAILABLE and groq_client:
        try:
            logger.info("🔄 Using Groq Cloud LLM...")
            start_time = time.time()
            
            completion = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{context_text}\n\n{query}" if context_text else query}
                ],
                temperature=0.7,  # Slightly lower for more consistent responses
                max_tokens=300,  # Reduced for faster responses
                top_p=0.9,
                stream=False  # Disable streaming for faster completion
            )
            
            response = completion.choices[0].message.content.strip()
            latency = (time.time() - start_time) * 1000
            
            if response:
                logger.info(f"✓ Groq response ({len(response)} chars, {latency:.0f}ms)")
                return response, "groq-llama-3.3-70b"
        except Exception as e:
            error_str = str(e).lower()
            # Check for common Groq errors that should trigger fallback
            if any(keyword in error_str for keyword in [
                "resource exhausted", "quota", "rate limit", "429", 
                "503", "500", "timeout", "unavailable", "error", "api key"
            ]):
                logger.warning(f"⚠️ Groq error: {e}, falling back to local LLM...")
            else:
                logger.error(f"✗ Groq error: {e}, falling back to local...")
    
    # Fallback to Local LLM
    if LOCAL_LLM_AVAILABLE:
        try:
            from utils.local_llm_utils import generate_with_ollama, check_ollama_available
            
            if check_ollama_available():
                logger.info("🔄 Using Local LLM (Ollama)...")
                start_time = time.time()
                
                result, success = generate_with_ollama(
                    f"User: {query}\nLana:",
                    system_prompt=system_prompt,
                    temperature=0.7,
                    max_tokens=150
                )
                
                latency = (time.time() - start_time) * 1000
                
                if success and result and result.strip():
                    logger.info(f"✓ Local LLM response ({len(result)} chars, {latency:.0f}ms)")
                    return result.strip(), "ollama-local"
        except Exception as e:
            logger.error(f"✗ Local LLM error: {e}")
    
    # Ultimate fallback
    fallback_responses = [
        "I'm here to help! Could you rephrase that?",
        "I didn't quite catch that. Can you ask in a different way?",
        "Hmm, I'm not sure about that. Can you give me more details?"
    ]
    import random
    return random.choice(fallback_responses), "fallback"

# ==================== INTENT DETECTION ====================
def detect_intent(query: str) -> Tuple[Optional[str], Any]:
    query_lower = query.lower()
    
    # Weather
    if any(word in query_lower for word in ['weather', 'temperature', 'rain', 'forecast']):
        return ('weather', 'Tiruppur')
    
    # News
    if any(word in query_lower for word in ['news', 'latest', 'headline']):
        return ('news', 'india')
    
    # Time
    if any(word in query_lower for word in ['time', 'date', 'today']):
        now = datetime.now()
        return ('time', now.strftime("%I:%M %p on %A, %B %d, %Y"))
    
    return (None, None)

def get_instant_response(query: str) -> Optional[str]:
    query_lower = query.lower().strip()
    
    greetings = {
        'hello': "Hello! How can I help you today?",
        'hi': "Hi there! What can I do for you?",
        'hey': "Hey! I'm here to help.",
        'good morning': "Good morning! How may I assist you?",
        'good evening': "Good evening! What can I help you with?",
    }
    
    for greeting, response in greetings.items():
        if query_lower.startswith(greeting):
            return response
    
    if 'how are you' in query_lower:
        return "I'm doing great! Thanks for asking. How can I assist you today?"
    
    if any(word in query_lower for word in ['thank', 'thanks']):
        return "You're very welcome! Happy to help anytime."
    
    if 'who are you' in query_lower or 'what are you' in query_lower:
        return "I'm Lana, your AI assistant. I can help with weather, news, conversations, and much more!"
    
    return None

# ==================== AUDIO GENERATION ====================
TTS_CACHE_DIR = Path(AUDIO_DIR) / "cache"
TTS_CACHE_DIR.mkdir(exist_ok=True)

def generate_audio_with_gtts(text: str, lang_code: str = "en") -> Optional[str]:
    """Generate audio with gTTS with error handling"""
    try:
        temp_file = os.path.join(TEMP_AUDIO_DIR, f"gtts_{int(time.time() * 1000)}.mp3")
        
        logger.info(f"🔊 Generating audio with gTTS ({lang_code})")
        tts = gTTS(text=text, lang=lang_code, slow=False)
        tts.save(temp_file)
        
        if os.path.exists(temp_file) and os.path.getsize(temp_file) > 1000:
            logger.info(f"✓ gTTS audio saved: {temp_file}")
            return temp_file
        else:
            logger.warning("⚠ gTTS created empty/small file")
            return None
            
    except Exception as e:
        logger.error(f"✗ gTTS error: {e}")
        return None

def play_audio_file(file_path: str) -> bool:
    """Play audio file with pygame"""
    global is_audio_playing, pygame_initialized
    
    if not pygame_initialized:
        initialize_pygame()
        
    if not pygame_initialized:
        logger.error("Cannot play audio - pygame not initialized")
        return False
        
    try:
        pygame.mixer.stop()
        
        is_audio_playing = True
        sound = pygame.mixer.Sound(file_path)
        sound.play()
        
        duration = sound.get_length()
        time.sleep(duration + 0.5)
        
        pygame.mixer.stop()
        is_audio_playing = False
        return True
        
    except Exception as e:
        logger.error(f"✗ Audio playback error: {e}")
        is_audio_playing = False
        initialize_pygame()
        return False

def get_tts_with_fallback(text: str, lang_code: str = "en") -> Optional[str]:
    """Generate TTS with reliable fallback mechanisms"""
    # Try primary language
    audio_file = generate_audio_with_gtts(text, lang_code)
    if audio_file and os.path.exists(audio_file) and os.path.getsize(audio_file) > 1000:
        return audio_file
    
    logger.warning(f"⚠ Primary TTS for {lang_code} failed, trying English")
    
    # For Indian languages, try English
    if lang_code in ["ta", "hi", "ml", "te", "kn"]:
        audio_file = generate_audio_with_gtts(text, "en")
        if audio_file and os.path.exists(audio_file) and os.path.getsize(audio_file) > 1000:
            return audio_file
    
    # Final fallback
    try:
        if lang_code not in ["en", "fr", "de"]:
            fallback_text = "I'm having trouble speaking in this language. Let me try English."
            audio_file = generate_audio_with_gtts(fallback_text, "en")
        else:
            audio_file = generate_audio_with_gtts(text, "en")
            
        if audio_file:
            return audio_file
    except Exception as e:
        logger.error(f"✗ Final fallback TTS error: {e}")
    
    return None

def generate_audio_fast(text: str, language: str = "en-US") -> Optional[str]:
    """
    Enhanced audio generation with language support
    Uses gTTS instead of pyttsx3 for better multi-language support
    """
    try:
        # Detect language if needed
        if not language or language == "en-US":
            detected_lang = detect_language(text)
            lang_code = LANGUAGE_MAPPING.get(detected_lang, "en")
        else:
            lang_code = LANGUAGE_MAPPING.get(language, "en")
        
        # Generate and play audio
        audio_file = get_tts_with_fallback(text, lang_code)
        
        if audio_file:
            # Play the audio
            play_audio_file(audio_file)
            
            # Return path relative to AUDIO_DIR for client access
            # audio_file is in TEMP_AUDIO_DIR (audio/temp/gtts_xxx.mp3)
            # Return as "temp/gtts_xxx.mp3" so client can access via /audio/temp/gtts_xxx.mp3
            return f"temp/{os.path.basename(audio_file)}"
        
        return None
        
    except Exception as e:
        logger.error(f"✗ Audio generation error: {e}")
        return None

def cleanup_temp_files():
    """Clean up temporary audio files"""
    try:
        for file in os.listdir(TEMP_AUDIO_DIR):
            if file.endswith(".wav") or file.endswith(".mp3"):
                try:
                    file_path = os.path.join(TEMP_AUDIO_DIR, file)
                    if time.time() - os.path.getmtime(file_path) > 3600:  # 1 hour old
                        os.remove(file_path)
                except Exception as e:
                    logger.debug(f"Error deleting temp file {file}: {e}")
    except Exception as e:
        logger.error(f"Error cleaning temp files: {e}")

# ==================== MAIN PROCESSING ====================
def process_query(query: str, user_id: str = "default") -> Tuple[str, str, Optional[str]]:
    start_time = time.time()
    logger.info(f"🔍 Processing: '{query[:100]}'")
    
    # Check instant response
    instant = get_instant_response(query)
    if instant:
        logger.info(f"⚡ Instant response")
        return instant, "instant", "greeting"
    
    # Check knowledge base
    knowledge_answer = search_responses_knowledge_base(query)
    if knowledge_answer:
        logger.info(f"📚 Knowledge base response")
        return knowledge_answer, "knowledge-base", "knowledge"
    
    # Detect intent
    intent, data = detect_intent(query)
    
    if intent == 'time':
        return f"The current time is {data}", "time", "time"
    
    # Generate AI response
    response, model_used = generate_ai_response(query, (intent, data), user_id)
    
    elapsed = (time.time() - start_time) * 1000
    logger.info(f"🤖 AI response ({elapsed:.0f}ms) - Model: {model_used}")
    return response, model_used, intent or "general"

# ==================== BACKGROUND WORKER ====================
def audio_worker():
    global conversation_history
    
    while not stop_event.is_set():
        try:
            if not audio_queue.empty():
                request_type = audio_queue.get()
                
                if request_type == "listen":
                    start_time = time.time()
                    
                    # Transcribe with improved recognition (includes audio data)
                    transcript, audio_array = transcribe_audio_optimized(language=current_language)
                    
                    if transcript:
                        # Clear old transcripts
                        while not transcript_queue.empty():
                            try:
                                transcript_queue.get_nowait()
                            except:
                                break
                        
                        # Send transcript immediately
                        transcript_queue.put({
                            "status": "transcript",
                            "transcript": transcript,
                            "is_final": True
                        })
                        
                        logger.info(f"📤 Transcript queued: '{transcript}'")
                        
                        # Process query in parallel with audio generation
                        try:
                            ai_response, model_used, intent = process_query(transcript, user_id="voice_user")
                            
                            if not ai_response:
                                ai_response = "I heard you, but I'm having trouble processing that."
                                model_used = "fallback"
                        except Exception as e:
                            logger.error(f"✗ Error processing: {e}")
                            ai_response = "Sorry, I encountered an error."
                            model_used = "error"
                            intent = "error"
                        
                        # Generate audio with language detection
                        detected_lang = detect_language(ai_response)
                        audio_future = executor.submit(generate_audio_fast, ai_response, detected_lang)
                        
                        audio_filename = None
                        try:
                            audio_filename = audio_future.result(timeout=3)
                        except Exception as e:
                            logger.error(f"Audio generation timeout: {e}")
                        
                        elapsed = time.time() - start_time
                        
                        # Save to DB asynchronously
                        executor.submit(
                            save_conversation,
                            user_id="voice_user",
                            user_message=transcript,
                            assistant_message=ai_response,
                            model_used=model_used,
                            audio_file=audio_filename,
                            response_time=elapsed,
                            intent=intent
                        )
                        
                        logger.info(f"✅ Total time: {elapsed:.2f}s")
                        
                        # Clear old responses first
                        while not response_queue.empty():
                            try:
                                response_queue.get_nowait()
                            except:
                                break
                        
                        # Send response
                        response_data = {
                            "status": "success",
                            "transcript": transcript,
                            "response": ai_response,
                            "audio_file": audio_filename,
                            "response_time": elapsed,
                            "model_used": model_used,
                            "intent": intent,
                            "audio_data": audio_array.tolist() if len(audio_array) > 0 else [],
                            "language": current_language
                        }
                        response_queue.put(response_data)
                        logger.info(f"📤 Response queued: {ai_response[:50]}...")
                        
                    else:
                        response_queue.put({
                            "status": "no_speech",
                            "message": "I didn't catch that. Could you please repeat?"
                        })
            
            time.sleep(0.01)
        except Exception as e:
            logger.error(f"✗ Worker error: {e}")
            response_queue.put({
                "status": "error",
                "message": "Something went wrong. Please try again."
            })

# Initialize worker thread
worker_thread = None

worker_thread = threading.Thread(target=audio_worker, daemon=True)
worker_thread.start()

# ==================== FLASK ROUTES ====================
@lana_ai.route('/')
@lana_ai.route('')
def index():
    """Main route for Lana AI interface"""
    try:
        return render_template('lana.html')
    except Exception as e:
        logger.error(f"Error rendering lana.html: {e}")
        return f"Error loading template: {str(e)}", 500

@lana_ai.route('/listen', methods=['POST'])
@rate_limit_api(requests_per_minute=30, requests_per_hour=300)
def listen():
    """Rate limited voice input endpoint"""
    audio_queue.put("listen")
    return jsonify({"status": "listening"})

@lana_ai.route('/get_response', methods=['GET'])
@rate_limit_api(requests_per_minute=60, requests_per_hour=600)
def get_response():
    """Rate limited response polling endpoint"""
    if not response_queue.empty():
        return jsonify(response_queue.get())
    return jsonify({"status": "processing"})

@lana_ai.route('/get_transcript', methods=['GET'])
@rate_limit_api(requests_per_minute=60, requests_per_hour=600)
def get_transcript():
    """Rate limited transcript polling endpoint"""
    try:
        if not transcript_queue.empty():
            return jsonify(transcript_queue.get())
        return jsonify({"status": "listening"})
    except Exception as e:
        logger.error(f"Transcript endpoint error: {e}")
        return jsonify({"status": "error", "message": str(e)})

@lana_ai.route('/text_input', methods=['POST'])
@rate_limit_api(requests_per_minute=20, requests_per_hour=200)
@validate_request({
    "text": {"type": "string", "required": True, "max_length": 397},
    "user_id": {"type": "string", "required": False, "max_length": 100}
})
def text_input():
    """Validated and rate-limited text input endpoint"""
    try:
        # Use validated data from request context
        data = g.validated_data if hasattr(g, 'validated_data') else request.get_json()
        text = data.get('text')
        user_id = data.get('user_id', 'web_user')
        
        if not text:
            return jsonify({"status": "error", "message": "Text is required"}), 400
        
        start_time = time.time()
        
        # Process query
        ai_response, model_used, intent = process_query(text, user_id=user_id)
        
        # Text input = Text output only (NO audio)
        audio_filename = None
        
        elapsed = time.time() - start_time
        
        # Save to DB asynchronously for faster response
        executor.submit(
            save_conversation,
            user_id=user_id,
            user_message=text,
            assistant_message=ai_response,
            model_used=model_used,
            response_time=elapsed,
            intent=intent
        )
        
        return jsonify({
            "status": "success",
            "transcript": text,
            "response": ai_response,
            "audio_file": None,
            "response_time": elapsed,
            "model_used": model_used,
            "intent": intent
        })
    except Exception as e:
        logger.exception("Text input error")
        return jsonify({"status": "error", "message": str(e)}), 500

@lana_ai.route('/audio/<path:filename>')
@rate_limit_api(requests_per_minute=100, requests_per_hour=1000)
def serve_audio(filename):
    """Rate limited and validated audio file serving"""
    try:
        # Validate and sanitize filename to prevent path traversal
        safe_filename = InputValidator.validate_filename(filename, 'filename', required=True)
        
        # Check if it's a temp file (starts with 'temp/')
        if filename.startswith('temp/'):
            # Extract just the filename part after 'temp/'
            temp_file = filename.replace('temp/', '')
            audio_path = Path(TEMP_AUDIO_DIR) / temp_file
        else:
            audio_path = Path(AUDIO_DIR) / safe_filename
        
        # Fallback to cache directory
        if not audio_path.exists():
            if safe_filename.startswith('cache/'):
                cache_file = safe_filename.replace('cache/', '')
                audio_path = Path(AUDIO_DIR) / 'cache' / cache_file
        
        # Try adding .mp3 extension if not exists
        if not audio_path.exists() and not safe_filename.endswith('.mp3'):
            audio_path = Path(AUDIO_DIR) / 'cache' / f"{safe_filename}.mp3"
        
        if audio_path.exists() and audio_path.is_file():
            response = send_file(str(audio_path), mimetype='audio/mpeg', as_attachment=False)
            response.headers['Accept-Ranges'] = 'bytes'
            response.headers['Cache-Control'] = 'public, max-age=3600'
            return response
        
        return jsonify({"status": "error", "message": "Audio not found"}), 404
    except Exception as e:
        logger.error(f"Error serving audio: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@lana_ai.route('/health', methods=['GET'])
@rate_limit_api(requests_per_minute=60, requests_per_hour=600)
def health():
    """Rate limited health check endpoint"""
    groq_status = "available" if GROQ_AVAILABLE else "unavailable"
    local_status = "unavailable"
    
    if LOCAL_LLM_AVAILABLE:
        try:
            from utils.local_llm_utils import check_ollama_available
            local_status = "available" if check_ollama_available() else "unavailable"
        except:
            pass
    
    active_model = "groq" if GROQ_AVAILABLE else "local" if local_status == "available" else "fallback"
    
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "version": "3.1.0",
        "models": {
            "groq": {
                "available": GROQ_AVAILABLE,
                "status": groq_status,
                "model": "llama-3.3-70b-versatile",
                "priority": 1
            },
            "local": {
                "available": LOCAL_LLM_AVAILABLE,
                "status": local_status,
                "priority": 2
            }
        },
        "active_model": active_model,
        "features": {
            "voice_input": True,
            "text_input": True,
            "audio_output": pygame_initialized,
            "multi_language": True,
            "whisper_local": whisper_enabled,
            "groq_whisper": GROQ_AVAILABLE,
            "google_speech": True,
            "sphinx_offline": True
        },
        "speech_recognition": {
            "primary": "groq-whisper-v3-turbo" if GROQ_AVAILABLE else "google",
            "fallbacks": ["google", "whisper-local", "language-fallbacks"],
            "supported_languages": list(LANGUAGE_MAPPING.keys())
        },
        "current_language": current_language,
        "tts_engine": "gtts",
        "audio_playback": "pygame"
    })

@lana_ai.route('/available_languages', methods=['GET'])
@rate_limit_api(requests_per_minute=60, requests_per_hour=600)
def available_languages():
    """Return list of available languages"""
    languages = [
        {"code": "en-US", "name": "English (US)"},
        {"code": "ta-IN", "name": "Tamil"},
        {"code": "hi-IN", "name": "Hindi"},
        {"code": "ml-IN", "name": "Malayalam"},
        {"code": "te-IN", "name": "Telugu"},
        {"code": "kn-IN", "name": "Kannada"},
        {"code": "fr-FR", "name": "French"},
        {"code": "de-DE", "name": "German"},
        {"code": "ko-KR", "name": "Korean"},
        {"code": "ja-JP", "name": "Japanese"}
    ]
    return jsonify({"status": "success", "languages": languages})

@lana_ai.route('/set_language', methods=['POST'])
@rate_limit_api(requests_per_minute=30, requests_per_hour=300)
@validate_request({
    "language": {"type": "string", "required": True, "max_length": 10}
})
def set_language():
    """Set current language for speech recognition"""
    global current_language
    
    try:
        data = g.validated_data if hasattr(g, 'validated_data') else request.get_json()
        language = data.get('language', 'en-US')
        
        if language in LANGUAGE_MAPPING or language in REVERSE_LANGUAGE_MAPPING:
            current_language = language
            logger.info(f"🌍 Language set to: {current_language}")
            return jsonify({
                "status": "success",
                "language": current_language,
                "message": f"Language changed to {current_language}"
            })
        else:
            return jsonify({
                "status": "error",
                "message": f"Unsupported language: {language}"
            }), 400
            
    except Exception as e:
        logger.exception("Set language error")
        return jsonify({"status": "error", "message": str(e)}), 500

@lana_ai.route('/test_tts', methods=['POST'])
@rate_limit_api(requests_per_minute=10, requests_per_hour=100)
@validate_request({
    "text": {"type": "string", "required": True, "max_length": 500},
    "language": {"type": "string", "required": False, "max_length": 10}
})
def test_tts():
    """Test TTS for a specific language"""
    try:
        data = g.validated_data if hasattr(g, 'validated_data') else request.get_json()
        text = data.get('text', 'Hello, this is a test')
        language = data.get('language', 'en-US')
        
        lang_code = LANGUAGE_MAPPING.get(language, "en")
        
        audio_file = get_tts_with_fallback(text, lang_code)
        
        if audio_file:
            return jsonify({
                "status": "success",
                "message": f"TTS for {language} is working",
                "audio_file": os.path.basename(audio_file)
            })
        else:
            return jsonify({
                "status": "error",
                "message": f"TTS for {language} failed"
            }), 500
            
    except Exception as e:
        logger.exception("Test TTS error")
        return jsonify({"status": "error", "message": str(e)}), 500

@lana_ai.route('/get_history', methods=['GET'])
@rate_limit_api(requests_per_minute=30, requests_per_hour=300)
def get_history():
    """Rate limited and validated history retrieval"""
    try:
        # Validate and sanitize query parameters
        user_id = InputValidator.validate_string(
            request.args.get('user_id', 'default'), 
            'user_id', 
            max_length=100, 
            required=False
        )
        page = InputValidator.validate_integer(
            request.args.get('page', 1, type=int),
            'page',
            min_value=1,
            max_value=1000,
            required=False
        )
        per_page = InputValidator.validate_integer(
            request.args.get('per_page', 20, type=int),
            'per_page',
            min_value=1,
            max_value=100,
            required=False
        )
        
        history_data = get_conversation_history(user_id=user_id, page=page, per_page=per_page)
        
        return jsonify({
            "status": "success",
            "history": history_data['conversations'],
            "pagination": {
                "page": history_data['page'],
                "per_page": history_data['per_page'],
                "total": history_data['total'],
                "total_pages": history_data['total_pages']
            }
        })
    except Exception as e:
        logger.exception("Get history error")
        return jsonify({"status": "error", "message": str(e)}), 500

# Cleanup
def cleanup():
    global is_audio_playing
    
    stop_event.set()
    
    # Stop audio playback
    if is_audio_playing and pygame_initialized:
        try:
            pygame.mixer.stop()
            is_audio_playing = False
        except:
            pass
    
    if worker_thread.is_alive():
        worker_thread.join(timeout=2)
    
    executor.shutdown(wait=False)
    
    # Clean up temp files
    cleanup_temp_files()
    
    # Quit pygame
    if pygame_initialized:
        try:
            pygame.mixer.quit()
        except:
            pass
    
    if tts_engine is not None:
        try:
            tts_engine.stop()
        except:
            pass
    logger.info("Cleanup complete")

import atexit
atexit.register(cleanup)

logger.info("✓ Lana AI Backend loaded with improved speech recognition")