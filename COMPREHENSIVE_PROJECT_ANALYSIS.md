# INFOSIGHT 3.0 - Comprehensive Project Analysis

**Project Type:** Advanced Cybersecurity & AI Intelligence Suite  
**Framework:** Flask + Blueprint-based Microservices  
**Python Version:** 3.8+  
**Architecture:** Unified Server (Port 5000) + 13 Specialized Modules  

---

## 1. MAIN APPLICATION FLOW & SERVER.PY ARCHITECTURE

### Port Configuration (Unified Mode)
```python
APP_PORTS = {
    'infocrypt': 5001,
    'cybersentry_ai': 5002,
    'donna': 5003,
    'enscan': 5004,
    'filescanner': 5005,
    'infosight_ai': 5006,
    'inkwell_ai': 5007,
    'nova_ai': 5008,
    'osint': 5009,
    'portscanner': 5010,
    'snapspeak_ai': 5011,
    'trueshot_ai': 5012,
    'webseeker': 5013
}
```

### Main Server (Port 5000)
- **Entry Point:** `server.py` (~700 lines)
- **Framework:** Flask with Blueprint-based routing
- **Mode:** Unified server (all modules run on port 5000, not in distributed mode)
- **Default Port:** `http://127.0.0.1:5000/`
- **Homepage:** Renders `homepage.html` from templates/

### Key Server Responsibilities
1. **Blueprint Registration** - Dynamically imports and registers all 13 modules as blueprints
2. **Rate Limiting** - Global rate limiting using Flask-Limiter
   - Default: 1000/day, 100/hour, 20/minute (per IP)
   - Custom limits per endpoint type (public/api/strict)
3. **Security Headers** - OWASP-compliant response headers:
   - X-Content-Type-Options: nosniff
   - X-Frame-Options: SAMEORIGIN (except NOVA AI)
   - X-XSS-Protection: 1; mode=block
   - Strict-Transport-Security: max-age=31536000
   - Content-Security-Policy (strict default, relaxed for NOVA AI)
4. **Error Handling** - Custom handlers for 404, 500, 403, 429 errors
5. **File Upload Limits** - MAX_CONTENT_LENGTH: 32MB
6. **Static File Caching** - 1-hour cache control for assets

### LLM Server Auto-Detection & Launch
**Located in server.py:** `check_ollama_running()` and `start_ollama_server()` functions

The server attempts to auto-start the local LLM server (Ollama or llama.cpp) on port 11434:
- **Windows:** Looks for `llama/llama-server.exe` → `D:\llama\llama-server.exe` → system PATH
- **Linux/Mac:** Attempts system `ollama` installation
- **Fallback:** If auto-start fails, user must manually start Ollama
- **Model Auto-Detection:** Searches for Qwen2.5-Coder-3B-Instruct GGUF model
- **Command:** Launches with `--ctx-size 4096 --threads 6 --port 11434 --host 127.0.0.1`

### Blueprint Registration Flow
```python
BLUEPRINT_CONFIGS = [
    ('/api/route', 'app.module_name', 'module_name'),
    # 13 modules total
]

register_blueprints_unified(app)  # Called at startup
```

**Error Handling:** If a module fails to import (e.g., protobuf issues), it logs a warning but continues with other modules.

---

## 2. EXTERNAL DEPENDENCIES & SYSTEM-LEVEL REQUIREMENTS

### Critical System-Level Requirements

#### A. **Nmap** (Required for PortScanner & WebSeeker)
- **Windows:** Install from https://nmap.org/download.html
- **Linux:** `sudo apt-get install nmap`
- **macOS:** `brew install nmap`
- **Function:** Port scanning, service detection, OS fingerprinting
- **Python Binding:** `python-nmap==0.7.1`

#### B. **Npcap** (Windows Only - Required)
- **Requirement:** WinPcap-compatible packet capture driver
- **Download:** https://npcap.com/#download
- **Installation:** Enable "WinPcap compatibility mode"
- **Purpose:** Raw packet capture for advanced network scanning
- **Check:** `portscanner.py` validates port availability before scans

#### C. **TOR Browser** (Optional - For DONNA AI)
- **Purpose:** Dark web intelligence gathering
- **Download:** https://www.torproject.org/download/
- **DONNA AI Integration:** `app/donna.py` - OSINT on dark web
- **Port:** Typically 9050 (SOCKS proxy)

#### D. **Ollama or llama.cpp** (Optional but Recommended)
- **Purpose:** Local LLM fallback when cloud APIs fail
- **Download:** https://ollama.ai/
- **Default Model:** Qwen2.5-Coder-3B-Instruct (3B parameters, ~2GB)
- **Fallback Model Location:** `llama/models/Qwen2.5-Coder-3B-Instruct-abliterated-Q5_K_M.gguf`
- **Port:** 11434 (server auto-detects and launches)
- **Context:** 4096 tokens, 6 threads
- **Env Vars:**
  - `OLLAMA_BASE_URL=http://127.0.0.1:11434` (default)
  - `OLLAMA_MODEL=qwen2.5-coder:3b-instruct` (default)

### Python Core Dependencies (100+ packages)
See [requirements.txt](requirements.txt) for complete list. Key groups:

#### Web Framework
- `flask==3.1.3`
- `flask-cors==6.0.2`
- `flask-limiter==4.1.1` (rate limiting)
- `gunicorn==25.1.0` (production WSGI server)
- `werkzeug==3.1.6`

#### AI/ML - LLM Integration
- `groq==1.0.0` (Groq Cloud LLM API)
- `langchain==1.2.10` (LLM orchestration)
- `langchain-core==1.2.14`
- `langchain-ollama==1.0.1` (local LLM integration)
- `ollama==0.6.1` (Ollama API client)
- `transformers==5.2.0` (Hugging Face models)
- `huggingface-hub==1.4.1` (model downloads)
- `google-generativeai==0.8.5` (Google Gemini integration)
- `openai-whisper==20250625` (speech-to-text)

#### Computer Vision & Image Processing
- `torch==2.10.0` (PyTorch GPU support)
- `torchvision==0.25.0` (vision models)
- `opencv-python==4.13.0.92` (image/video processing)
- `pillow==12.1.1` (image manipulation)
- `scikit-image==0.25.2` (advanced image analysis)
- `imagehash==4.3.2` (perceptual hashing)
- `transformers==5.2.0` (BLIP image captioning)
- `deepface==0.0.98` (face detection/recognition)
- `retina-face==0.0.17` (face detection)
- `mtcnn==1.0.0` (face detection)

#### Network & Security Tools
- `requests==2.32.5` (HTTP client)
- `python-nmap==0.7.1` (Nmap scanning)
- `dnspython==2.8.0` (DNS queries)
- `python-whois==0.9.6` (WHOIS lookups)
- `cryptography==46.0.5` (encryption/PKI)
- `pyOpenSSL==25.3.0` (SSL/TLS analysis)
- `urllib3==2.6.3` (HTTP client, SSL support)

#### Voice & Audio Processing
- `librosa==0.11.0` (audio analysis)
- `soundfile==0.13.1` (audio I/O)
- `pydub==0.25.1` (audio conversion)
- `pyaudio==0.2.14` (audio device access)
- `pocketsphinx==5.0.4` (speech recognition)
- `SpeechRecognition==3.14.5` (STT framework)
- `gtts==2.5.4` (Google Text-to-Speech)
- `edge-tts>=6.1.9` (edge TTS)
- `pyttsx3==2.99` (offline TTS)

#### Threat Intelligence & Scanning
- `beautifulsoup4==4.14.3` (web scraping)
- `requests-toolbelt==1.0.0` (HTTP utilities)
- `abuseipdb` (IP reputation API)
- `virustotal` (VirusTotal API integration)

#### Data Processing & Storage
- `pandas==2.3.3` (data manipulation)
- `numpy==2.2.6` (numerical computing)
- `sqlalchemy==2.0.46` (ORM)
- `openpyxl==3.1.5` (Excel)
- `python-docx==1.2.0` (Word documents)
- `PyPDF2==3.0.1` (PDF processing)
- `PyYAML==6.0.3` (YAML parsing)

#### Specialized Utilities
- `scikit-learn==1.7.2` (machine learning, KMeans)
- `tensorflow==2.20.0` (deep learning - optional)
- `keras==3.12.1` (high-level neural networks)
- `ultralytics==8.4.14` (YOLOv8 object detection)
- `pytesseract==0.3.13` (OCR)
- `filelock==3.24.3` (file locking)
- `validators==0.35.0` (input validation)
- `Deprecated==1.3.1` (deprecation warnings)
- `fire==0.7.1` (CLI interface)
- `markdown2==2.5.4` (Markdown parsing)

#### Development & Testing
- `pytest==9.0.2` (unit testing)
- `pytest-cov==7.0.0` (coverage)
- `black==26.1.0` (code formatting)
- `flake8==7.3.0` (linting)
- `pipdeptree==2.31.3` (dependency visualization)

#### Miscellaneous
- `python-dotenv==1.2.1` (environment variables)
- `cryptography==46.0.5` (encryption)
- `fuzzywuzzy==0.18.0` (fuzzy string matching)
- `retry==0.9.2` (exponential backoff)
- `tenacity==9.1.4` (retry decorator)
- `waitress==3.0.2` (pure Python WSGI server)
- `gradio==4.x.x` (web UI for NOVA AI)
- `reportlab==4.4.10` (PDF generation)
- `cairosvg==2.7.0` (SVG to PDF)

---

## 3. REQUIRED AND OPTIONAL API KEYS

### **REQUIRED (Most Features)**

#### **Groq API Key** (ESSENTIAL FOR LLM)
- **Source:** https://console.groq.com/keys
- **Used By:** 
  - Core LLM router (llama-3.3-70b-versatile for complex tasks, llama-3.1-8b-instant for fast tasks)
  - NOVA AI (voice assistant)
  - InfoSight AI (content generation)
  - SnapSpeak AI (image reasoning)
  - CyberSentry AI (document analysis)
  - DONNA AI (OSINT)
  - Inkwell AI (prompt optimization)
  - WebSeeker (website analysis)
- **Model Routing:**
  ```python
  "cybersentry_ai": "llama-3.3-70b-versatile",  # Complex security analysis
  "donna": "llama-3.3-70b-versatile",  # OSINT reasoning
  "inkwell_ai": "llama-3.1-8b-instant",  # Fast task
  "infosight_ai": "llama-3.1-8b-instant",  # Content generation
  "snapspeak_ai": "llama-3.1-8b-instant",  # Image reasoning
  "webseeker": "llama-3.1-8b-instant",  # Website analysis
  ```
- **Cost:** Free tier: 30 requests/minute
- **Env Var:** `GROQ_API_KEY`

#### **VirusTotal API Key** (for FileFender & WebSeeker)
- **Source:** https://www.virustotal.com/gui/join-us
- **Purpose:** File scanning, URL scanning, hash lookup
- **Used By:** 
  - FileScanner (malware detection)
  - WebSeeker (URL reputation checking)
- **Cost:** Free tier: 4 requests/minute
- **Env Var:** `VIRUSTOTAL_API_KEY`

### **HIGHLY OPTIONAL**

#### **Hugging Face API Token** (for Vision Analysis)
- **Source:** https://huggingface.co/join (with read-only access)
- **Purpose:** BLIP image captioning cloud fallback
- **Used By:** 
  - TrueShot AI (AI-generated image detection)
  - InfoSight AI (image analysis)
  - SnapSpeak AI (image captioning)
  - Vision Analyzer utilities
- **Format:** Starts with `hf_` (e.g., `hf_`+20 chars)
- **Env Var:** `HF_API_TOKEN` or `HUGGINGFACE_API_TOKEN`
- **Note:** Not strictly required - local BLIP model provides fallback

#### **IPInfo API Key** (for WebSeeker)
- **Source:** https://ipinfo.io/signup
- **Purpose:** IP geolocation, ISP lookup
- **Used By:** WebSeeker (IP intelligence)
- **Cost:** Free tier: 50k requests/month
- **Env Var:** `IPINFO_API_KEY`

#### **AbuseIPDB API Key** (for WebSeeker)
- **Source:** https://www.abuseipdb.com/register
- **Purpose:** IP reputation checking, abuse reporting
- **Used By:** WebSeeker (threat intelligence)
- **Cost:** Free tier: limited
- **Env Var:** `ABUSEIPDB_API_KEY`

#### **News API Key** (Optional - for WebSeeker)
- **Source:** https://newsapi.org/
- **Purpose:** Real-time news data integration
- **Env Var:** `NEWS_API_KEY`

#### **Weather API Key** (Optional - for NOVA AI)
- **Purpose:** Real-time weather data, local intelligence
- **Env Var:** `WEATHER_API_KEY`

#### **SerpAPI Key** (Optional - for WebSeeker)
- **Source:** https://serpapi.com/
- **Purpose:** Google search integration for OSINT
- **Env Var:** `SERPAPI_KEY`

### **.env File Template**
```env
# ==================== REQUIRED ====================
GROQ_API_KEY=your_groq_api_key_here
VIRUSTOTAL_API_KEY=your_virustotal_api_key_here

# ==================== HIGHLY RECOMMENDED ====================
HF_API_TOKEN=hf_your_huggingfacetoken_here
HF_HOME=L:/hf_cache  # Hugging Face cache directory

# ==================== OPTIONAL ====================
IPINFO_API_KEY=your_ipinfo_api_key
ABUSEIPDB_API_KEY=your_abuseipdb_api_key
NEWS_API_KEY=your_news_api_key
WEATHER_API_KEY=your_weather_api_key
SERPAPI_KEY=your_serpapi_key

# ==================== LOCAL LLM ====================
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen2.5-coder:3b-instruct
LOCAL_LLM_TIMEOUT=300
LOCAL_LLM_MAX_RETRIES=2
LOCAL_LLM_RETRY_DELAY=5.0
```

---

## 4. MODEL FILES & AI ASSETS

### Machine Learning Models

#### **best_model9.pth** (TrueShot AI - Deepfake Detection)
- **Location:** `models/best_model9.pth`
- **Size:** ~50-100MB (estimated)
- **Framework:** PyTorch (ResNet18 variant)
- **Architecture:** 
  ```python
  model = resnet18(weights=None)
  model.fc = nn.Sequential(
      nn.Dropout(p=0.5),
      nn.Linear(512, 2)  # 2 classes: AI-generated vs Real
  )
  ```
- **Purpose:** Detects AI-generated/deepfake images with 50+ forensic factors
- **Load Location:** Fallback search order:
  1. `utils.paths.get_model_path('best_model9.pth')`
  2. `{PROJECT_ROOT}/models/best_model9.pth`
  3. `{PROJECT_ROOT}/best_model9.pth`
  4. `best_model9.pth` (current directory)

#### **yolov8n.pt** (Object Detection)
- **Location:** `yolov8n.pt` (project root)
- **Size:** ~6.3MB (nano model)
- **Framework:** Ultralytics YOLOv8
- **Purpose:** Object detection in images/videos
- **Download:** Auto-downloaded by ultralytics if missing
- **Command:** `yolo detect predict model=yolov8n.pt source=image.jpg`

### GGUF Model Files (Local LLM)

#### **Qwen2.5-Coder-3B-Instruct** (Local LLM)
- **Location:** `llama/models/Qwen2.5-Coder-3B-Instruct-abliterated-Q5_K_M.gguf`
- **Size:** ~2.3GB (Q5_K_M quantization)
- **Framework:** GGUF format (llama.cpp compatible)
- **Parameters:** 3 billion
- **Quantization:** Q5_K_M (5-bit, good balance of quality and speed)
- **Context Window:** 4096 tokens
- **Purpose:** Local LLM fallback for Groq failures
- **Access Methods:**
  - Ollama: `ollama pull qwen2.5-coder:3b-instruct`
  - Manual: Load from path to llama-server

### Hugging Face Models (Cloud/Local)

#### **BLIP (Image Captioning)**
- **Model:** `Salesforce/blip-image-captioning-base`
- **Size:** ~1GB (download on first use)
- **Framework:** Hugging Face transformers
- **Purpose:** Generate text descriptions of images
- **Used By:** TrueShot AI, SnapSpeak AI, InfoSight AI
- **Cache Location:** `~/.cache/huggingface/hub/` (auto-managed by HF)
- **Lazy Loading:** Loaded on-demand when HF_API_TOKEN fails or first local caption request

#### **Other Pre-trained Models**
- **DeepFace** (face recognition/analysis)
- **MTCNN** (face detection)
- **Retina-Face** (advanced face detection)
- **YOLOv8** (object detection)

---

## 5. DATABASE & CACHE REQUIREMENTS

### SQLite Databases

#### **NOVA AI Conversation Database**
- **Path:** `app/nova_conversations.db`
- **Purpose:** Store user-assistant conversation history
- **Schema:**
  ```sql
  CREATE TABLE messages (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      session TEXT NOT NULL,
      role TEXT NOT NULL CHECK(role IN ('user','assistant')),
      content TEXT NOT NULL,
      ts TEXT NOT NULL
  )
  CREATE INDEX idx_session ON messages(session)
  ```
- **Cache Settings:** WAL mode, PRAGMA cache_size=-2000, synchronous=NORMAL
- **Size:** Grows with conversation history (~10KB per 1000 messages)

#### **Inkwell AI Database**
- **Path:** `inkwell_ultimate.db`
- **Purpose:** Store prompt optimization history and templates
- **Tables:**
  - `optimizations`: Track prompt improvements over time
  - `prompt_templates`: Reusable prompt templates with metadata
- **Size:** ~1-5MB (typical)

#### **FileFender Encryption Metadata**
- **Path:** `data/encryption_metadata.json`
- **Purpose:** Store encryption keys, salts, and metadata for encrypted files
- **Format:** JSON key-value pairs (encryption_key -> metadata dict)

### JSON Data Files

#### **data/data.json** (OSINT Platform Database)
- **Purpose:** 50+ social media and web platforms for username reconnaissance
- **Size:** ~50-100KB
- **Structure:**
  ```json
  {
    "Instagram": {
      "url": "https://instagram.com/{}",
      "urlMain": "https://instagram.com",
      "errorType": "status_code"
    },
    ... 50+ platforms
  }
  ```
- **Auto-Generation:** If missing, uses built-in fallback list

#### **data/responses.json** (CyberSentry AI)
- **Purpose:** Pre-defined security analysis templates
- **Size:** ~10-50KB
- **Format:** Array of response templates

#### **data/lana_memory.json** (DONNA AI Memory)
- **Purpose:** Persistent memory for DONNA AI between sessions
- **Size:** ~5-20KB
- **Content:** Research history, discovered URLs, OSINT results

#### **data/encryption_metadata.json**
- **Purpose:** Track encryption operations
- **Auto-Created:** On first file encryption operation

### Cache Directories

#### **audio/cache/**
- **Purpose:** Cache generated audio files (TTS output)
- **Contents:** .wav files for processed speech
- **Cleanup:** Manual (files accumulate over time)

#### **audio/temp/**
- **Purpose:** Temporary audio buffers during processing
- **Auto-Cleanup:** After speech-to-text processing

#### **static/generated_images/**
- **Purpose:** Cache AI-generated images
- **Generation:** From Gradio/image generation pipelines
- **Types:** .png, .jpg formats
- **Size:** Large (10MB+ possible for many generated images)

#### **temp/**
- **Purpose:** Temporary file uploads for scanning/processing
- **Contents:** Files awaiting malware scan, encryption, analysis
- **Cleanup:** Manual or after each operation
- **Max Size:** 32MB per file (MAX_CONTENT_LENGTH)

#### **models/runs/detect/**
- **Purpose:** YOLOv8 detection output cache
- **Contents:** Subdirectories: `predict/`, `predict2/`, etc.
- **Auto-Generated:** By YOLOv8 inference pipeline
- **Size:** Can grow large (100MB+ for many predictions)

#### **Hugging Face Cache**
- **Location:** `$HF_HOME` (default: `~/.cache/huggingface/hub/`)
- **Project Setting:** `HF_HOME=L:/hf_cache` (Windows)
- **Size:** 1-5GB (for all downloaded models)
- **Contents:** BLIP, Whisper, BERT, and other transformer models

### In-Memory Caches

#### **Rate Limiter Storage**
- **Type:** Memory-based
- **Strategy:** Fixed-window rate limiting
- **Note:** Uses `"memory://"` storage (use Redis for production)

#### **Python-level Caches**
- **@lru_cache decorators:** Used throughout for expensive computations
- **Typical TTL:** None (persistent until server restart)
- **Examples:**
  - Platform lookup cache (OSINT)
  - DNS query results (EnScan)
  - Model predictions

---

## 6. PORT CONFIGURATION DETAILS

### Primary Server Port
- **Main Application:** Port 5000 (HTTP only by default)
- **Access URL:** `http://127.0.0.1:5000/`
- **SSL/TLS:** Not configured (Flask development server)
- **Host Binding:** 127.0.0.1 (localhost only - use 0.0.0.0 for network access)

### Route Structure (Blueprints prefixed)
```
http://127.0.0.1:5000/  ← Homepage
http://127.0.0.1:5000/infocrypt/  ← InfoCrypt module
http://127.0.0.1:5000/cybersentry_ai/  ← CyberSentry AI
http://127.0.0.1:5000/nova_ai/  ← NOVA AI (special CSP config)
http://127.0.0.1:5000/osint/  ← OSINT reconnaissance
http://127.0.0.1:5000/portscanner/  ← Port scanning
http://127.0.0.1:5000/webseeker/  ← Web intelligence
http://127.0.0.1:5000/filescanner/  ← File analysis
http://127.0.0.1:5000/infosight_ai/  ← InfoSight AI
http://127.0.0.1:5000/snapspeak_ai/  ← Image captioning
http://127.0.0.1:5000/trueshot_ai/  ← Deepfake detection
http://127.0.0.1:5000/enscan/  ← Domain scanning
http://127.0.0.1:5000/inkwell_ai/  ← Prompt optimization
http://127.0.0.1:5000/donna/  ← Dark web OSINT
```

### External Service Ports (Not exposed to users)
| Port | Service | Purpose | Auto-Start |
|------|---------|---------|-----------|
| 11434 | Ollama/llama.cpp | Local LLM server | Yes (server.py) |
| 7860 | Gradio (NOVA AI) | Voice chat interface | Optional |

### Rate Limiting Configuration
- **Global Default:** 1000/day, 100/hour, 20/minute
- **Rate Limit Key:** Based on IP address (or IP:user_id if authenticated)
- **Storage:** Memory-based (Redis recommended for production)
- **Breach Response:** HTTP 429 with retry-after header

### Special Configurations
- **NOVA AI CSP Policy:** Allows iframe embedding of Gradio at 127.0.0.1:7860
- **File Upload Max:** 32MB (MAX_CONTENT_LENGTH)
- **Static File Cache:** 1 hour
- **Request Timeout:** Varies by endpoint (typically 5-10 minutes for scans)

---

## 7. SPECIAL RUNTIME REQUIREMENTS & INITIALIZATION

### Environment Variables Required at Startup

```python
# Early initialization in server.py
os.environ.setdefault("HF_TOKEN", os.getenv("HF_TOKEN", "your_token_here"))
os.environ.setdefault("HF_HOME", os.getenv("HF_HOME", "L:/hf_cache"))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN (optimization)
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'  # Disable pygame welcome message
```

### Initialization Steps (server.py startup)

1. **Load Environment Variables**
   ```python
   from dotenv import load_dotenv
   load_dotenv()  # Load .env file
   ```

2. **Initialize Rate Limiter**
   ```python
   limiter = init_rate_limiter(app)  # Sets up Flask-Limiter
   ```

3. **Register Blueprints**
   ```python
   register_blueprints_unified(app)  # Loads all 13 modules
   ```

4. **Auto-Start Local LLM** (if needed)
   ```python
   if check_ollama_running(11434):
       logger.info("✓ Ollama already running")
   else:
       start_ollama_server(11434)  # Attempts auto-start
   ```

5. **Initialize Module-Specific Resources**
   - Each blueprint initializes its own databases, caches, models
   - Lazy loading: Models loaded on first request, not at startup

### Graceful Shutdown Handling
- **Signal Handlers:** SIGINT (Ctrl+C) gracefully shuts down
- **Thread Pool Cleanup:** Concurrent futures pools are cleaned up
- **Database Cleanup:** SQLite databases properly closed

### Error Recovery
- **Blueprint Load Failures:** Logged as warnings; server continues
- **Protobuf Version Mismatches:** Handled with compatibility shims (see snapspeak_ai.py)
- **Missing Models:** Falls back to API or returns error
- **Missing API Keys:** Features gracefully degrade

### Memory Management
- **Lazy Model Loading:** Large models (BLIP, torch models) loaded on-demand
- **LRU Caching:** Expensive operations cached with @lru_cache
- **Singleton Patterns:** Model instances shared across requests
- **Cleanup:** Temporary files stored in `temp/` and `audio/temp/` directories

### Threading & Concurrency
- **Flask:** Single-threaded development server (use Gunicorn for production with workers)
- **ThreadPoolExecutor:** Used for concurrent tasks:
  - DNS queries (EnScan)
  - Port scanning (PortScanner)
  - File processing (FileFender)
  - URL analysis (WebSeeker)
- **Thread Safety:** SQLite databases use WAL mode and connection locks
- **Locks:** Threading locks for model access and voice selection (NOVA AI)

### First Run Behavior
Auto-generated files created on first run:
```
✅ app/nova_conversations.db  (NOVA AI)
✅ inkwell_ultimate.db  (Inkwell AI)
✅ data/lana_memory.json  (DONNA AI)
✅ audio/cache/  (directory)
✅ static/generated_images/  (directory)
✅ models/runs/detect/  (YOLOv8 cache)
```

---

## 8. AUDIO PROCESSING REQUIREMENTS

### Supported TTS (Text-to-Speech) Systems
Hierarchy: Edge-TTS → Google TTS → Fallback

#### **Edge TTS** (Primary - Recommended)
- **Package:** `edge-tts>=6.1.9`
- **Provider:** Microsoft Azure Cognitive Services
- **Quality:** High-quality neural voices
- **Voices Available:** 6 confirmed (3 female, 3 male)
  ```python
  VOICES = {
      "aria":    ("en-US-AriaNeural",    "Aria"),      # Female
      "sonia":   ("en-GB-SoniaNeural",   "Sonia"),     # Female
      "neerja":  ("en-IN-NeerjaNeural",  "Neerja"),    # Female
      "guy":     ("en-US-GuyNeural",     "Guy"),       # Male
      "ryan":    ("en-GB-RyanNeural",    "Ryan"),      # Male
      "william": ("en-AU-WilliamNeural", "William"),   # Male
  }
  ```
- **Rate Adjustments:** Voice-specific pitch adjustments (+5%, -3%, etc.)
- **Used By:** NOVA AI, DONNA AI

#### **Google TTS (gTTS)** (Fallback)
- **Package:** `gtts>=2.5.0`
- **Quality:** Lower than Edge-TTS but more stable
- **Fallback Reason:** Works without API keys (uses Google's free API)
- **Used By:** Audio generation endpoints

#### **PyTTSx3** (Offline Fallback)
- **Package:** `pyttsx3==2.99`
- **Quality:** Lowest quality but completely offline
- **Purpose:** Works when internet is unavailable

### Speech Recognition (STT)

#### **PyAudio** (Required for NOVA AI)
- **Plugin:** `pyaudio==0.2.14`
- **Purpose:** Real-time microphone input capture
- **Requirements:** System audio drivers, working microphone
- **Configuration:**
  ```python
  PA_RATE = 16000  # Sample rate (16kHz)
  PA_CHUNK = 1024  # Chunk size
  PA_CHANNELS = 1  # Mono
  PA_WIDTH = 2  # 16-bit audio
  SILENCE_RMS = 300  # Silence detection threshold
  SILENCE_SEC = 1.5  # Minimum silence duration
  MIN_REC_SEC = 0.6  # Minimum recording duration
  MAX_REC_SEC = 60.0  # Maximum recording duration
  ```

#### **SpeechRecognition** (Google STT)
- **Package:** `SpeechRecognition==3.14.5`
- **Backend:** Google Speech-to-Text API (free tier)
- **Usage:** Converts audio WAV to text
- **Accuracy:** ~90-95% for clear English speech

#### **Pocketsphinx** (Offline STT)
- **Package:** `pocketsphinx==5.0.4`
- **Purpose:** Offline speech recognition (lower accuracy)
- **Fallback:** When internet unavailable

### Audio File Processing

#### **librosa** (Audio Analysis)
- **Package:** `librosa==0.11.0`
- **Purpose:** Audio feature extraction, spectral analysis
- **Used By:** Audio processing modules

#### **SoundFile** (Audio I/O)
- **Package:** `soundfile==0.13.1`
- **Purpose:** Read/write WAV, FLAC, OGG files

#### **Pydub** (Audio Conversion)
- **Package:** `pydub==0.25.1`
- **Purpose:** Convert between audio formats (WAV, MP3, etc.)
- **Dependency:** Requires ffmpeg for MP3 conversion

### Audio Output

#### **WAV File Format**
- **Codec:** PCM 16-bit
- **Sample Rate:** 16kHz
- **Channels:** Mono
- **Typical Size:** ~30KB per second of audio

### NOVA AI Audio Processing Pipeline
1. **Record:** PyAudio captures microphone input (silent detection disabled for user input)
2. **Convert:** WAV format with 16kHz sample rate
3. **Transcribe:** SpeechRecognition (Google STT)
4. **Process:** LLM generates response text
5. **Synthesize:** Edge-TTS generates audio response
6. **Play:** Browser plays audio via HTML5 audio element

### Performance Considerations
- **Audio Processing Latency:** 2-5 seconds typical (STT + LLM + TTS)
- **Cache:** Generated TTS audio cached in `audio/cache/`
- **Network:** All TTS systems require internet (except pyttsx3)
- **Concurrency:** Single voice active at a time (thread-locked)

---

## 9. IMAGE GENERATION & VISION PIPELINE

### Image Analysis Pipeline (TrueShot AI v4.0)

#### **Multi-Layer Detection System**

1. **Cloud Layer (Primary)**
   - **API:** Hugging Face Inference API (BLIP)
   - **Model:** `Salesforce/blip-image-captioning-base`
   - **Purpose:** Generate natural language captions
   - **Fallback:** If HF_API_TOKEN missing or request fails

2. **Local Layer (Secondary)**
   - **Framework:** PyTorch
   - **Model:** BLIP (lazily loaded from Hugging Face hub)
   - **Device:** GPU if available, else CPU
   - **Purpose:** Backup captioning when cloud fails
   - **Cache:** ~/.cache/huggingface/hub/

3. **Advanced Forensics Layer**
   ```python
   AI_PATTERNS = {
       'diffusion_artifacts': 0.0,      # DDPM noise patterns
       'gan_signatures': 0.0,           # StyleGAN fingerprints
       'unnatural_smoothness': 0.0,     # Overprocessed regions
       'pixel_repetition': 0.0,         # Pattern reuse (GAN artifact)
       'color_banding': 0.0,            # Quantization issues
   }
   ```

#### **AI Service Detection**
Detects 20+ AI image generators by watermark/metadata:
- Midjourney, DALL-E 2/3, Stable Diffusion
- Adobe Firefly, Playground AI, Craiyon
- Runway, Imagen, Flux, Together AI
- And many more (see trueshot_ai.py for full list)

#### **Forensic Analysis Techniques**
- **ELA (Error Level Analysis):** Detects JPEG compression inconsistencies
- **Double JPEG Detection:** Identifies recompressed images
- **Copy-Move Forgery:** Detects duplicated/cloned regions (scipy-based)
- **Splicing Detection:** Identifies image stitching
- **Deepfake Indicators:** Face warping, eye gaze inconsistencies
- **Metadata Analysis:** EXIF, color space, ICC profiles
- **Face Manipulation Checks:** Using MTCNN, retina-face, deepface

### Computer Vision Libraries & Models

#### **PyTorch & TorchVision**
- **Framework:** `torch==2.10.0`, `torchvision==0.25.0`
- **GPU Support:** CUDA acceleration (cuda.is_available())
- **Models Used:**
  - ResNet18 (deepfake detect classifier)
  - Pre-trained vision encoders (feature extraction)

#### **OpenCV (cv2)**
- **Package:** `opencv-python==4.13.0.92`
- **Purpose:** Image processing, edge detection, color analysis
- **Functions:**
  - Gaussian filters (blur detection)
  - Canny edge detection (authenticity checks)
  - Color space conversions

#### **Scikit-Image**
- **Package:** `scikit-image==0.25.2`
- **Purpose:** Advanced image analysis
- **Features:**
  - Texture analysis
  - Morphological operations
  - Denoise algorithms

#### **ImageHash**
- **Package:** `imagehash==4.3.2`
- **Purpose:** Perceptual hashing (find similar images)
- **Algorithms:** pHash, aHash, dHash, whash

#### **PIL/Pillow**
- **Package:** `pillow==12.1.1`
- **Purpose:** Image I/O, manipulation, metadata extraction
- **EXIF Support:** ExifRead, piexif packages

#### **Scikit-Learn (KMeans)**
- **Package:** `scikit-learn==1.7.2`
- **Purpose:** Color clustering and dominant color extraction
- **Used By:** Image analysis modules

### Image Generation (NOVA AI / Gradio)

#### **Gradio Integration**
- **Package:** `gradio>=4.x.x`
- **Purpose:** Web UI for NOVA AI voice chat
- **Port:** 7860 (automatically launched)
- **Features:**
  - Voice input/output interface
  - Image upload capability
  - Real-time chat display

#### **Potential Generation Models** (Not included)
- Stable Diffusion (via huggingface/diffusers)
- DALL-E (via OpenAI API)
- Midjourney (via API)

### Vision Analyzer Module (`utils/vision_analyzer.py`)

#### **Reusable Component Functions**

```python
def cloud_caption(image_path) -> Optional[str]:
    """Get image caption using HF API (requires HF_API_TOKEN)"""

def local_caption(image_path) -> Optional[str]:
    """Get image caption using local BLIP (auto-downloads model)"""

def detect_objects(image_path) -> List[Dict]:
    """Detect objects in image using YOLOv8"""

def extract_colors(image_path, num_colors=5) -> List[tuple]:
    """Extract and cluster dominant colors"""

def analyze_image(image_path, app_name) -> Dict:
    """Full image analysis pipeline (cloud + local fallback)"""
```

#### **Lazy Loading Pattern**
```python
_blip_model = None      # Load only if needed
_blip_processor = None
_blip_device = None     # GPU/CPU auto-detect

def _load_blip_model():
    global _blip_model, _blip_processor, _blip_device
    # Retry logic with exponential backoff (3 attempts)
    # Downloads model on first use (~1GB)
```

### Object Detection (YOLOv8)

#### **Ultralytics YOLOv8 Integration**
- **Package:** `ultralytics==8.4.14`
- **Model:** `yolov8n.pt` (nano - 6.3MB)
- **Purpose:** Real-time object detection
- **Cache:** `models/runs/detect/` (prediction outputs)
- **Classes Detected:** 80 COCO classes (persons, cars, animals, etc.)

### Face Detection & Recognition

#### **Face Detection Libraries**
- **MTCNN** (`mtcnn==1.0.0`): Fast multi-task cascade networks
- **Retina-Face** (`retina-face==0.0.17`): More accurate, wider angle
- **Used By:** TrueShot AI for face manipulation detection

#### **Face Recognition**
- **DeepFace** (`deepface==0.0.98`): Face recognition, age/gender
- **Purpose:** Identity verification, comparison

### OCR (Optical Character Recognition)

#### **Tesseract Integration**
- **Package:** `pytesseract==0.3.13`
- **Requirement:** System tesseract installation (Windows: installer available)
- **Purpose:** Extract text from images (watermark detection)
- **Languages:** 100+ supported

### Vision Processing Performance

| Operation | Time | Device |
|-----------|------|--------|
| BLIP Caption (cloud) | 1-3s | Cloud (HF API) |
| BLIP Caption (local) | 5-10s | GPU / 15-20s CPU |
| YOLOv8 Detection | 0.5-2s | GPU / 2-5s CPU |
| TrueShot Analysis | 10-30s | GPU / 30-60s CPU |
| Face Detection | 0.2-1s | GPU / 1-3s CPU |
| Deepface Recognition | 2-5s | GPU / 5-10s CPU |

---

## 10. GPU ACCELERATION NEEDS & CONFIGURATION

### GPU Support Assessment

#### **PyTorch GPU Acceleration** (Recommended)
- **Package:** `torch==2.10.0`
- **CUDA Support:** Included in pip wheel (NVIDIA GPUs only)
- **AMD Support:** ROCm alternative (requires separate installation)
- **CPU Fallback:** Automatic if CUDA unavailable

#### **GPU Detection**
```python
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")  # Logs at startup
```

### NVIDIA GPU Requirements

#### **For TrueShot AI & Vision Tasks** (Optional but highly recommended)
- **Minimum VRAM:** 2GB (ResNet18 + basic ops)
- **Recommended:** 4GB+ (for concurrent image processing)
- **Ideal:** 8GB+ (for batch processing)
- **Compute Capability:** 3.5+ (Kepler era and newer)

#### **GPU-Accelerated Libraries**
- **TensorFlow** (`tensorflow==2.20.0`): Uses CUDA 11.8+ by default
- **PyTorch:** Uses CUDA 11.8 (pinned version)
- **CuDNN:** Auto-downloaded with CUDA wheels
- **NCCL:** Multi-GPU communication (optional)

#### **CUDA & cuDNN**
- **Included:** CUDA/cuDNN bundled with torch wheel
- **No Manual Installation:** PyTorch handles everything
- **Version:** Matched to torch==2.10.0 requirements

### performance Impact (GPU vs CPU)

| Model | CPU Time | GPU Time (RTX 3060) | Speedup |
|-------|----------|-------------------|---------|
| ResNet18 inference | 200ms | 15ms | **13x** |
| BLIP caption (1024x1024) | 8s | 1s | **8x** |
| YOLOv8 detection | 5s | 0.5s | **10x** |
| TrueShot full analysis | 45s | 8s | **5.6x** |

### Configuration for GPU

#### **Enable GPU (Automated)**
No configuration needed! Libraries auto-detect available GPU.

#### **Force CPU (if needed)**
```python
# Add to environment before running
export CUDA_VISIBLE_DEVICES=""  # Disables all GPUs

# Or set in Python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ""
```

#### **Specify GPU Index** (Multi-GPU)
```python
# For 2+ GPUs, specify which one
export CUDA_VISIBLE_DEVICES="0"  # Use GPU 0 only
export CUDA_VISIBLE_DEVICES="0,1"  # Use GPUs 0 and 1
```

### Memory Management

#### **GPU Memory Issues** (If OOM)
1. **Reduce Batch Size:** Analyze fewer images at once
2. **Clear Cache:** `torch.cuda.empty_cache()` (between requests)
3. **Use CPU:** Set `CUDA_VISIBLE_DEVICES=""` as fallback
4. **Downgrade Model:** Use quantized versions (smaller VRAM)

#### **Shared GPU Memory**
- TrueShot AI model: ~200-400MB VRAM
- BLIP model: ~1-2GB VRAM
- Concurrent ops: May use up to 3GB total

### CPU-Only Mode (Fallback)

All models work on CPU if CUDA unavailable:
- **No Errors:** Libraries silently fall back to CPU
- **Performance:** 5-15x slower than GPU
- **Suitable For:** 
  - Development/testing
  - Low-end servers
  - Batch processing overnight

### Production GPU Recommendations

#### **For Small Deployments** (1-10 concurrent users)
- **GPU:** NVIDIA T4 ($35-50/month on cloud)
- **VRAM:** 16GB
- **Performance:** ~20 concurrent image analyses/min

#### **For Large Deployments** (50+ concurrent users)
- **GPUs:** 2-4 x RTX 3090 or A100
- **VRAM:** 80GB+ total
- **Performance:** 200+ concurrent analyses/min

#### **Recommended Cloud Providers**
- **AWS:** EC2 g4dn instances (T4 GPUs)
- **Google Cloud:** Compute Engine with NVIDIA GPUs
- **Azure:** GPU-enabled VMs
- **Paperspace:** GPU terminals with hourly billing

---

## DOCKERFILE REQUIREMENTS SUMMARY

For production deployment, your Dockerfile should include:

### Base Image
```dockerfile
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
# or: FROM python:3.10-slim (for CPU-only)
```

### System Dependencies
```bash
# Required
apt-get install -y \
    git \
    nmap \
    dnsmasq \
    libmagic1 \
    libssl-dev \
    libffi-dev \
    python3-dev \
    tesseract-ocr \
    ffmpeg

# Optional but recommended
apt-get install -y tor
```

### Python Environment
```bash
python -m venv /app/venv
source /app/venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Environment Setup
```bash
# Create directories
mkdir -p /app/data /app/models /app/audio/cache /app/static/generated_images
mkdir -p /app/audio/temp /app/temp /app/llama/models

# Set permissions
chmod 755 /app/temp
chmod 755 /app/audio/temp
```

### Runtime Configuration
```bash
# .env file (passed via Docker secrets in production)
GROQ_API_KEY=${GROQ_API_KEY}
HF_API_TOKEN=${HF_API_TOKEN}
OLLAMA_BASE_URL=http://ollama-service:11434
```

### Health Checks
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python check_token.py
```

### Port Exposure
```dockerfile
EXPOSE 5000 11434
```

### Startup Command
```bash
CMD ["python", "server.py"]
```

---

## KEY SECURITY CONSIDERATIONS

1. **Rate Limiting:** Enabled on all endpoints (OWASP)
2. **Input Validation:** Schema-based validation (max 397 chars for strings)
3. **CORS:** Configured per-blueprint
4. **Security Headers:** X-Frame-Options, CSP, HSTS, X-XSS-Protection
5. **API Keys:** Server-side only (never sent to client)
6. **SSL/TLS:** Not configured in development (use reverse proxy in production)
7. **HTTPS Enforcement:** Recommend nginx/gunicorn combo with SSL

---

## REFERENCES

- **Project Root:** `L:\PROJECT\INFOSIGHT_3.0`
- **Main Entry:** `server.py`
- **Modules:** `app/*.py` (13 blueprints)
- **Core Router:** `core/llm_router.py`
- **Utilities:** `utils/*.py`
- **Config:** `config/api-keys-requirements.txt`
- **Requirements:** `requirements.txt` (150+ packages)
