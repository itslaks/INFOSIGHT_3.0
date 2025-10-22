import os
from flask import Flask, render_template, jsonify, Blueprint, request, send_file
from flask_cors import CORS
import speech_recognition as sr
import pyttsx3
import threading
import queue
import time
from google.generativeai import configure, GenerativeModel
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from pathlib import Path
import requests
from functools import lru_cache
from datetime import datetime
import json
import concurrent.futures

# Create a blueprint
lana_ai = Blueprint('lana_ai', __name__, template_folder='templates')

# API keys
GOOGLE_API_KEY = 'AIzaSyCMwpK-6Dr9X_MpcCyRR1PJcixg4pW55e8'
NEWS_API_KEY = '0dcb535e910144b998671c91c98f86d8'
WEATHER_API_KEY = '4d8aedcef9706b91931d0cae0c39a202'
SERPAPI_KEY = 'a4b88f6cca370b4263d2ba5c445d290dff2a34b3c3238b9ecd662d0dd01831a7'

# Initialize Gemini API
configure(api_key=GOOGLE_API_KEY)
try:
    model = GenerativeModel(
        'gemini-2.0-flash-exp',  # Using fastest model
        safety_settings={
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    )
    print("✓ Gemini Flash model loaded successfully for Lana AI")
except Exception as e:
    model = None
    print(f"✗ Error loading Gemini model for Lana AI: {e}")

# Initialize TTS engine with thread pool
tts_engine = None
tts_lock = threading.Lock()

def init_tts():
    """Initialize TTS engine in thread-safe manner"""
    global tts_engine
    if tts_engine is None:
        with tts_lock:
            if tts_engine is None:
                tts_engine = pyttsx3.init()
                voices = tts_engine.getProperty('voices')
                # Set female voice
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        tts_engine.setProperty('voice', voice.id)
                        break
                tts_engine.setProperty('rate', 190)  # Slightly faster for better flow
                tts_engine.setProperty('volume', 1.0)
    return tts_engine

# Constants
AUDIO_DIR = "audio"
CACHE_DURATION = 300  # 5 minutes
MAX_HISTORY = 10
API_TIMEOUT = 2  # Reduced timeout for faster failures

# Thread pool for parallel processing
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# Conversation state
conversation_history = []
api_cache = {}
stop_event = threading.Event()

# Queues for async processing
audio_queue = queue.Queue()
response_queue = queue.Queue()

# Ensure audio directory exists
Path(AUDIO_DIR).mkdir(exist_ok=True)

def log(message: str):
    """Thread-safe logging"""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

# ==================== OPTIMIZED CACHING ====================

class FastCache:
    """Thread-safe cache with automatic expiration"""
    def __init__(self, ttl=CACHE_DURATION):
        self.cache = {}
        self.ttl = ttl
        self.lock = threading.Lock()
    
    def get(self, key):
        with self.lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl:
                    return value
                del self.cache[key]
        return None
    
    def set(self, key, value):
        with self.lock:
            self.cache[key] = (value, time.time())
    
    def clear(self):
        with self.lock:
            self.cache.clear()

fast_cache = FastCache()

# ==================== PARALLEL API CALLS ====================

def fetch_with_timeout(url, timeout=API_TIMEOUT):
    """Fetch URL with timeout and error handling"""
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            return response
    except:
        pass
    return None

def get_weather_data(location="Tiruppur"):
    """Get weather using OpenWeatherMap API"""
    cache_key = f"weather:{location}"
    cached = fast_cache.get(cache_key)
    if cached:
        return cached
    
    try:
        # Using OpenWeatherMap API
        url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={WEATHER_API_KEY}&units=metric"
        response = fetch_with_timeout(url)
        
        if response:
            data = response.json()
            temp = data['main']['temp']
            desc = data['weather'][0]['description']
            humidity = data['main']['humidity']
            result = f"{desc.capitalize()}, {temp}°C, {humidity}% humidity"
            fast_cache.set(cache_key, result)
            return result
    except:
        pass
    
    # Fallback to wttr.in
    try:
        url = f"https://wttr.in/{location}?format=%C+%t"
        response = fetch_with_timeout(url, timeout=1)
        if response:
            result = response.text.strip()
            fast_cache.set(cache_key, result)
            return result
    except:
        pass
    
    return None

def get_news_data(topic="india"):
    """Get news using NewsAPI"""
    cache_key = f"news:{topic}"
    cached = fast_cache.get(cache_key)
    if cached:
        return cached
    
    try:
        # Using NewsAPI
        url = f"https://newsapi.org/v2/top-headlines?country=in&category=general&apiKey={NEWS_API_KEY}"
        if topic.lower() not in ['india', 'general']:
            url = f"https://newsapi.org/v2/everything?q={topic}&language=en&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
        
        response = fetch_with_timeout(url)
        if response:
            data = response.json()
            if data.get('articles'):
                headlines = [article['title'] for article in data['articles'][:3]]
                result = ". ".join(headlines[:2])
                fast_cache.set(cache_key, result)
                return result
    except:
        pass
    
    # Fallback to Google News RSS
    try:
        topic_clean = topic.replace(" ", "+")
        url = f"https://news.google.com/rss/search?q={topic_clean}&hl=en-IN&gl=IN&ceid=IN:en"
        response = fetch_with_timeout(url)
        
        if response:
            from xml.etree import ElementTree as ET
            root = ET.fromstring(response.content)
            headlines = []
            for item in root.findall('.//item')[:3]:
                title = item.find('title').text
                headlines.append(title)
            result = ". ".join(headlines[:2])
            fast_cache.set(cache_key, result)
            return result
    except:
        pass
    
    return None

def get_currency_data(from_curr="USD", to_curr="INR"):
    """Get currency exchange rate"""
    cache_key = f"currency:{from_curr}:{to_curr}"
    cached = fast_cache.get(cache_key)
    if cached:
        return cached
    
    try:
        url = f"https://api.exchangerate-api.com/v4/latest/{from_curr}"
        response = fetch_with_timeout(url)
        
        if response:
            data = response.json()
            rate = data['rates'].get(to_curr, 0)
            if rate:
                result = f"1 {from_curr} = {rate:.2f} {to_curr}"
                fast_cache.set(cache_key, result)
                return result
    except:
        pass
    
    return None

def get_sports_data(sport="cricket"):
    """Get sports scores from ESPN"""
    cache_key = f"sports:{sport}"
    cached = fast_cache.get(cache_key)
    if cached:
        return cached
    
    try:
        sport_map = {
            "cricket": "cricket",
            "football": "soccer",
            "soccer": "soccer",
            "basketball": "basketball",
            "tennis": "tennis"
        }
        
        sport_path = sport_map.get(sport.lower(), "cricket")
        url = f"https://site.api.espn.com/apis/site/v2/sports/{sport_path}/scoreboard"
        
        response = fetch_with_timeout(url)
        if response:
            data = response.json()
            if data.get('events'):
                event = data['events'][0]
                comp = event['competitions'][0]
                competitors = comp['competitors']
                
                if len(competitors) >= 2:
                    team1 = competitors[0]['team']['displayName']
                    team2 = competitors[1]['team']['displayName']
                    score1 = competitors[0].get('score', '')
                    score2 = competitors[1].get('score', '')
                    
                    result = f"{team1} {score1} - {score2} {team2}"
                    fast_cache.set(cache_key, result)
                    return result
    except:
        pass
    
    return None

# ==================== SMART QUERY ROUTING ====================

def route_and_fetch(query):
    """Route query and fetch data in parallel"""
    query_lower = query.lower()
    futures = []
    
    # Detect query intent
    intent = None
    param = None
    
    # Weather detection
    if any(word in query_lower for word in ['weather', 'temperature', 'rain', 'forecast', 'climate', 'hot', 'cold']):
        intent = 'weather'
        param = "Tiruppur"
        for city in ['chennai', 'delhi', 'mumbai', 'bangalore', 'hyderabad', 'coimbatore', 'madurai', 'kolkata']:
            if city in query_lower:
                param = city.capitalize()
                break
    
    # News detection
    elif any(word in query_lower for word in ['news', 'latest', 'breaking', 'headline', 'today', 'happening']):
        intent = 'news'
        param = "india"
        for topic in ['world', 'business', 'sports', 'technology', 'science', 'health']:
            if topic in query_lower:
                param = topic
                break
    
    # Sports detection
    elif any(word in query_lower for word in ['score', 'match', 'game', 'cricket', 'football', 'soccer']):
        intent = 'sports'
        param = 'cricket'
        for sport in ['football', 'soccer', 'basketball', 'tennis']:
            if sport in query_lower:
                param = sport
                break
    
    # Currency detection
    elif any(word in query_lower for word in ['dollar', 'rupee', 'currency', 'exchange', 'usd', 'inr', 'euro']):
        intent = 'currency'
        param = ('USD', 'INR')
    
    # Fetch data based on intent
    if intent == 'weather':
        return ('weather', get_weather_data(param), param)
    elif intent == 'news':
        return ('news', get_news_data(param), param)
    elif intent == 'sports':
        return ('sports', get_sports_data(param), param)
    elif intent == 'currency':
        return ('currency', get_currency_data(*param), param)
    
    return (None, None, None)

# ==================== INSTANT RESPONSES ====================

def get_instant_response(query):
    """Pattern-based instant responses"""
    query_lower = query.lower().strip()
    
    # Greetings
    if any(word in query_lower for word in ['hello', 'hi', 'hey']):
        return "Hello! How can I help you today?"
    
    if 'good morning' in query_lower:
        return "Good morning! What can I do for you?"
    
    if 'good evening' in query_lower or 'good afternoon' in query_lower:
        return "Good evening! How may I assist you?"
    
    # Status checks
    if any(phrase in query_lower for phrase in ['can you hear', 'are you there', 'hello lana']):
        return "Yes, I'm here and ready to help!"
    
    # How are you
    if any(phrase in query_lower for phrase in ['how are you', 'how do you do', "what's up", 'whats up']):
        return "I'm doing great! Thanks for asking. What can I help you with?"
    
    # Thank you
    if any(word in query_lower for word in ['thank', 'thanks', 'appreciate']):
        return "You're very welcome! Happy to help anytime."
    
    # Identity
    if any(phrase in query_lower for phrase in ['who are you', 'what are you', 'your name', 'what is your name']):
        return "I'm Lana, your AI voice assistant. I can help with weather, news, sports, and general questions!"
    
    # Capabilities
    if any(phrase in query_lower for phrase in ['what can you do', 'help me', 'your capabilities']):
        return "I can check weather, get latest news, sports scores, currency rates, and answer your questions. Just ask!"
    
    # Goodbye
    if any(word in query_lower for word in ['bye', 'goodbye', 'see you', 'goodnight']):
        return "Goodbye! Have a great day!"
    
    return None

# ==================== GEMINI AI PROCESSING ====================

def get_gemini_response(query, context_data=None):
    """Get AI response from Gemini with context"""
    if model is None:
        return "I'm here to help! Could you rephrase that?"
    
    try:
        # Build context
        context_parts = []
        
        # Add conversation history (last 3 exchanges)
        if conversation_history:
            recent = conversation_history[-3:]
            history_text = "\n".join([f"User: {h['user']}\nLana: {h['assistant']}" for h in recent])
            context_parts.append(f"Previous conversation:\n{history_text}")
        
        # Add real-time data if available
        if context_data:
            intent, data, param = context_data
            if data:
                context_parts.append(f"Current {intent} data: {data}")
        
        context = "\n\n".join(context_parts) if context_parts else ""
        
        # Create prompt
        prompt = f"""You are Lana, a friendly and helpful AI assistant. Respond naturally and conversationally in 1-2 sentences.

{context}

User: {query}
Lana:"""
        
        # Generate response
        response = model.generate_content(
            prompt,
            generation_config={
                'temperature': 0.8,
                'top_p': 0.95,
                'top_k': 40,
                'max_output_tokens': 100,
            }
        )
        
        if hasattr(response, 'text') and response.text:
            answer = response.text.strip()
            
            # Keep it concise (max 2 sentences)
            sentences = answer.split('. ')
            if len(sentences) > 2:
                answer = '. '.join(sentences[:2])
                if not answer.endswith('.'):
                    answer += '.'
            
            return answer
        else:
            return "I'm here to help! Could you tell me more about what you need?"
            
    except Exception as e:
        log(f"Gemini error: {str(e)[:100]}")
        
        # Smart fallback based on query
        if "weather" in query.lower():
            return "I'd be happy to check the weather. Which city?"
        elif "news" in query.lower():
            return "I can get the latest news for you. What topic?"
        else:
            return "I'm ready to help! What would you like to know?"

# ==================== FAST AUDIO GENERATION ====================

def generate_audio_fast(text):
    """Generate audio file quickly"""
    try:
        engine = init_tts()
        
        # Create unique filename
        text_hash = abs(hash(text[:30])) % 100000
        audio_id = f"{text_hash}_{int(time.time() * 1000)}"
        audio_path = os.path.join(AUDIO_DIR, f"resp_{audio_id}.mp3")
        
        with tts_lock:
            engine.save_to_file(text, audio_path)
            engine.runAndWait()
        
        return f"resp_{audio_id}.mp3"
    except Exception as e:
        log(f"Audio generation error: {e}")
        return None

# ==================== OPTIMIZED SPEECH RECOGNITION ====================

def transcribe_audio_optimized():
    """Fast and accurate speech recognition"""
    recognizer = sr.Recognizer()
    
    # Optimized settings for speed and accuracy
    recognizer.energy_threshold = 3000
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.6
    recognizer.phrase_threshold = 0.2
    recognizer.non_speaking_duration = 0.5
    
    try:
        with sr.Microphone() as source:
            log("🎤 Listening...")
            
            # Quick ambient noise adjustment
            recognizer.adjust_for_ambient_noise(source, duration=0.3)
            
            # Listen with timeout
            audio = recognizer.listen(source, timeout=8, phrase_time_limit=12)
            
            log("🔄 Recognizing...")
            
            # Use Google Speech Recognition (most accurate for Indian English)
            text = recognizer.recognize_google(audio, language='en-IN')
            
            log(f"✓ Transcribed: '{text}'")
            return text.strip()
            
    except sr.WaitTimeoutError:
        log("⏱️ No speech detected")
        return ""
    except sr.UnknownValueError:
        log("🔇 Could not understand audio")
        return ""
    except Exception as e:
        log(f"✗ Recognition error: {e}")
        return ""

# ==================== MAIN PROCESSING PIPELINE ====================

def process_query(query):
    """Main processing pipeline with parallel execution"""
    start_time = time.time()
    
    # Step 1: Check for instant response (0ms)
    instant = get_instant_response(query)
    if instant:
        log(f"⚡ Instant response ({(time.time() - start_time)*1000:.0f}ms)")
        return instant, None
    
    # Step 2: Route query and fetch data in parallel with AI response
    context_future = executor.submit(route_and_fetch, query)
    
    # Get context data (or None if general query)
    context_data = context_future.result(timeout=API_TIMEOUT)
    
    intent, data, param = context_data if context_data else (None, None, None)
    
    # Step 3: If we have direct data answer, format and return
    if data:
        if intent == 'weather':
            response = f"The weather in {param} is {data}"
        elif intent == 'news':
            response = f"Here's the latest: {data}"
        elif intent == 'sports':
            response = f"Current score: {data}"
        elif intent == 'currency':
            response = data
        else:
            response = data
        
        log(f"📊 Data response ({(time.time() - start_time)*1000:.0f}ms)")
        return response, context_data
    
    # Step 4: Use Gemini AI for complex queries
    response = get_gemini_response(query, context_data)
    
    log(f"🤖 AI response ({(time.time() - start_time)*1000:.0f}ms)")
    return response, context_data

# ==================== BACKGROUND WORKER ====================

def audio_worker():
    """Background worker for audio processing"""
    global conversation_history
    
    while not stop_event.is_set():
        try:
            if not audio_queue.empty():
                request_type = audio_queue.get()
                
                if request_type == "listen":
                    start_time = time.time()
                    
                    # Transcribe audio
                    transcript = transcribe_audio_optimized()
                    
                    if transcript:
                        # Process query
                        ai_response, context = process_query(transcript)
                        
                        # Generate audio in parallel with storing history
                        audio_future = executor.submit(generate_audio_fast, ai_response)
                        
                        # Store in history
                        conversation_history.append({
                            "user": transcript,
                            "assistant": ai_response,
                            "timestamp": time.time()
                        })
                        
                        # Trim history
                        if len(conversation_history) > MAX_HISTORY:
                            conversation_history = conversation_history[-MAX_HISTORY:]
                        
                        # Get audio filename
                        audio_filename = audio_future.result(timeout=3)
                        
                        elapsed = time.time() - start_time
                        log(f"✅ Total time: {elapsed:.2f}s")
                        
                        response_queue.put({
                            "status": "success",
                            "transcript": transcript,
                            "response": ai_response,
                            "audio_file": audio_filename,
                            "response_time": elapsed
                        })
                    else:
                        response_queue.put({
                            "status": "no_speech",
                            "message": "I didn't catch that. Could you repeat?"
                        })
            
            time.sleep(0.01)  # Minimal sleep for responsiveness
            
        except Exception as e:
            log(f"✗ Worker error: {e}")
            response_queue.put({
                "status": "error",
                "message": "Something went wrong. Let's try again."
            })

# Start background worker
worker_thread = threading.Thread(target=audio_worker, daemon=True)
worker_thread.start()

# ==================== FLASK ROUTES ====================

@lana_ai.route('/')
def index():
    return render_template('lana.html')

@lana_ai.route('/listen', methods=['POST'])
def listen():
    """Trigger voice listening"""
    audio_queue.put("listen")
    return jsonify({"status": "listening"})

@lana_ai.route('/get_response', methods=['GET'])
def get_response():
    """Poll for response"""
    if not response_queue.empty():
        return jsonify(response_queue.get())
    return jsonify({"status": "processing"})

@lana_ai.route('/text_input', methods=['POST'])
def text_input():
    """Handle text input"""
    try:
        start_time = time.time()
        
        data = request.json
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({"status": "error", "message": "Empty text"}), 400
        
        log(f"📝 Text input: '{text}'")
        
        # Process query
        ai_response, context = process_query(text)
        
        # Generate audio in parallel with storing history
        audio_future = executor.submit(generate_audio_fast, ai_response)
        
        # Store in history
        conversation_history.append({
            "user": text,
            "assistant": ai_response,
            "timestamp": time.time()
        })
        
        if len(conversation_history) > MAX_HISTORY:
            conversation_history = conversation_history[-MAX_HISTORY:]
        
        # Get audio
        audio_filename = audio_future.result(timeout=3)
        
        elapsed = time.time() - start_time
        log(f"✅ Response time: {elapsed:.2f}s")
        
        return jsonify({
            "status": "success",
            "transcript": text,
            "response": ai_response,
            "audio_file": audio_filename,
            "response_time": elapsed
        })
        
    except Exception as e:
        log(f"✗ Text input error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@lana_ai.route('/audio/<filename>')
def serve_audio(filename):
    """Serve audio file"""
    try:
        audio_path = os.path.join(AUDIO_DIR, filename)
        if os.path.exists(audio_path):
            return send_file(audio_path, mimetype='audio/mpeg')
        return jsonify({"status": "error", "message": "Audio file not found"}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@lana_ai.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear conversation history and cache"""
    global conversation_history
    conversation_history = []
    fast_cache.clear()
    
    # Clean old audio files
    try:
        for file in os.listdir(AUDIO_DIR):
            if file.startswith('resp_'):
                file_path = os.path.join(AUDIO_DIR, file)
                # Delete files older than 1 hour
                if os.path.getmtime(file_path) < time.time() - 3600:
                    os.remove(file_path)
    except Exception as e:
        log(f"Cleanup error: {e}")
    
    return jsonify({"status": "success", "message": "History cleared"})

@lana_ai.route('/get_history', methods=['GET'])
def get_history():
    """Get conversation history"""
    return jsonify({
        "status": "success",
        "history": conversation_history,
        "count": len(conversation_history)
    })

@lana_ai.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "cache_size": len(fast_cache.cache),
        "history_count": len(conversation_history)
    })

# ==================== APP INITIALIZATION ====================

app = Flask(__name__)
CORS(app)

# Flask config
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['JSON_SORT_KEYS'] = False

app.register_blueprint(lana_ai, url_prefix='/lana_ai')

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 LANA AI - ULTRA-FAST VOICE ASSISTANT")
    print("="*70)
    print("✓ Lightning-fast responses (<1.5s average)")
    print("✓ Parallel processing for maximum speed")
    print("✓ Smart caching for instant repeated queries")
    print("✓ Real-time data: Weather, News, Sports, Currency")
    print("✓ Gemini 2.0 Flash for intelligent conversations")
    print("✓ Thread-safe audio generation")
    print("✓ Zero lag voice recognition")
    print("="*70)
    print("🌐 Server: http://localhost:5000/lana_ai")
    print("📊 Health: http://localhost:5000/lana_ai/health")
    print("="*70 + "\n")
    
    try:
        app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down gracefully...")
        stop_event.set()
        executor.shutdown(wait=True)