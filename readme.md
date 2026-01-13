# 🛡️ INFOSIGHT 3.0

<div align="center">

![INFOSIGHT Banner](static/images/logo.png)

**Advanced Cybersecurity & AI Intelligence Suite**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-Latest-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-Private-red.svg)](LICENSE)

*A comprehensive security platform featuring 13 specialized tools for threat detection, data protection, and AI-powered intelligence.*

[Features](#-features) • [Installation](#-installation) • [Tools](#️-tools) • [Usage](#-usage) • [Configuration](#️-configuration)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Installation](#-installation)
- [Prerequisites](#prerequisites)
- [Tools](#️-tools)
- [Usage](#-usage)
- [Configuration](#️-configuration)
- [Project Structure](#-project-structure)
- [Troubleshooting](#-troubleshooting)
- [Support](#-support)

---

## 🌟 Overview

**INFOSIGHT 3.0** is a cutting-edge cybersecurity platform that combines traditional security tools with advanced AI capabilities. Built for security professionals, researchers, and organizations seeking comprehensive threat intelligence and data protection solutions.

The platform integrates 13 specialized modules covering web security, network scanning, file analysis, encryption, OSINT, image forensics, and AI-powered intelligence.

---

## ✨ Features

### 🛡️ **Security Hardening**
- **Rate Limiting**: IP and user-based rate limiting on all endpoints (OWASP compliant)
- **Input Validation**: Schema-based validation with type checking and length limits
- **API Key Security**: All API keys handled server-side only, never exposed to client
- **Security Headers**: Comprehensive security headers (CSP, HSTS, XSS protection)
- **Path Traversal Prevention**: Filename validation prevents directory traversal attacks
- **OWASP Best Practices**: Following OWASP Top 10 security guidelines

### 🔍 **Scanning & Reconnaissance**
- Multi-database threat intelligence (VirusTotal, AbuseIPDB)
- Network vulnerability assessment
- DNS enumeration and analysis
- Real-time port scanning with Nmap integration
- Domain security analysis (SSL, SPF, DMARC, DKIM)

### 🎯 **Threat Detection**
- Multi-engine malware scanning (VirusTotal integration)
- Real-time threat assessment
- Hash-based file verification
- URL/domain risk classification

### 🔐 **Data Protection**
- Military-grade encryption (AES-256, RSA, Fernet)
- Secure file encryption/decryption
- Cryptographic hashing (MD5, SHA-256)
- Secure key management

### 🕵️ **OSINT Capabilities**
- Username reconnaissance across 50+ platforms
- Social media footprint analysis
- Digital presence tracking
- Dark web intelligence gathering (DONNA AI)

### 🖼️ **Image Forensics**
- AI-powered image analysis (BLIP model)
- Automatic image captioning
- Steganography detection
- Comprehensive metadata extraction (EXIF, GPS, camera info)
- Image hashing (perceptual hashing)
- Color analysis and clustering

### 🤖 **AI Intelligence**
- Natural language processing via centralized LLM router (Groq Cloud LLM - Llama 3.3-70B-Versatile for complex tasks, Llama 3.1-8B-Instant for fast tasks)
- Automated threat detection and analysis
- AI content generation (text and images) with multi-model fallback
- Voice-enabled assistance (LANA AI) with sentiment analysis
- Prompt optimization and enhancement (INKWELL AI)
- Real-time data integration (weather, news, sports)
- Centralized LLM router with intelligent model selection and local fallback (Ollama)

### 📊 **Image Authenticity**
- Deepfake detection
- AI-generated image identification
- Digital manipulation analysis
- Multi-factor authenticity verification
- Confidence scoring and detailed analysis

---

## 🚀 Installation

### Prerequisites

#### **Critical Requirements for PortScanner & WebSeeker:**

1. **Nmap Installation** (Required)
   - **Windows:** Download and install from [Nmap Official Site](https://nmap.org/download.html)
   - **Linux:** 
     ```bash
     sudo apt-get update
     sudo apt-get install nmap
     ```
   - **macOS:** 
     ```bash
     brew install nmap
     ```

2. **Npcap Installation** (Windows Only - Required)
   - Download from [Npcap Official Site](https://npcap.com/#download)
   - Install with **WinPcap compatibility mode** enabled
   - Required for packet capture functionality

#### **Optional Requirements:**

3. **Ollama** (For Local LLM Fallback - Recommended)
   - Download from [Ollama Official Site](https://ollama.ai/)
   - **For Qwen2.5-Coder-3B-Instruct model:**
     - If you have the .gguf file, import it: `ollama create qwen2.5-coder:3b-instruct -f Modelfile`
     - Or pull from library: `ollama pull qwen2.5-coder:3b-instruct`
   - **Check your model name:** Run `ollama list` to see available models
   - **Configure model name:** Set `OLLAMA_MODEL` env var to match your model name
   - Required for local AI model fallback when cloud APIs fail
   - Default model: `qwen2.5-coder:3b-instruct` (auto-detects if not found)

4. **TOR** (For DONNA AI - Optional)
   - Download from [TOR Project](https://www.torproject.org/download/)
   - Required for dark web access

#### **General Requirements:**
- Python 3.8 or higher
- pip (Python package manager)
- Git
- 4GB RAM minimum (8GB recommended)
- Internet connection for API-dependent features

### Step-by-Step Installation

#### **1. Clone the Repository**
```bash
git clone https://github.com/itslaks/INFOSIGHT_3.0.git
cd INFOSIGHT_3.0
```

#### **2. Create Virtual Environment** (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

#### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

#### **4. Configure Environment Variables**
Create a `.env` file in the root directory (optional - some features work without API keys):
```env
VIRUSTOTAL_API_KEY=your_virustotal_api_key
GROQ_API_KEY=your_groq_api_key
IPINFO_API_KEY=your_ipinfo_api_key
ABUSEIPDB_API_KEY=your_abuseipdb_api_key
HUGGINGFACE_API_KEY=your_huggingface_api_key
NEWS_API_KEY=your_news_api_key
OPENWEATHER_API_KEY=your_openweather_api_key
SERPAPI_API_KEY=your_serpapi_api_key
```

#### **5. Run the Application**

**Windows:**
```bash
scripts\run_windows.bat
```

**Linux/macOS:**
```bash
chmod +x scripts/run_linux&mac.sh
./scripts/run_linux&mac.sh
```

**Or manually:**
```bash
python server.py
```

#### **6. Access the Application**
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 🛠️ Tools

### 🔍 RECONNAISSANCE

### 1. **WEBSEEKER** - Web Security Scanner
> Comprehensive threat intelligence combining VirusTotal and Nmap scanning

**Features:**
- Domain/URL malware detection via VirusTotal
- Port vulnerability analysis with Nmap
- IP information and geolocation
- AbuseIPDB reputation checking
- Real-time threat assessment
- AI-powered analysis with Groq Cloud LLM (Llama models)
- **Local LLM Fallback**: Automatically uses Ollama (Qwen2.5-Coder) when Groq fails

**⚠️ Requires:** Nmap, Npcap (Windows), VirusTotal API key, IPInfo API key, AbuseIPDB API key, Groq API key  
**💡 Optional:** Ollama with Qwen2.5-Coder model for local fallback

---

### 2. **PORTSCANNER** - Network Port Scanner
> Advanced network reconnaissance and security auditing

**Features:**
- Open port identification
- Service detection and versioning
- Operating system detection
- Vulnerability assessment
- Customizable scan profiles
- Exportable scan results

**⚠️ Requires:** Nmap, Npcap (Windows)

---

### 3. **ENSCAN (Site Index)** - Domain Intelligence
> Multi-layer domain analysis platform

**Features:**
- DNS enumeration (A, AAAA, MX, NS, TXT, CNAME records)
- SSL/TLS certificate analysis
- Security header evaluation
- SPF, DMARC, DKIM email security analysis
- Domain vulnerability scanning
- Comprehensive security scoring

**No API keys required**

---

### 🎯 DETECTION

### 4. **FILESCANNER (File Fender)** - File Scanner
> Multi-engine malware detection system

**Features:**
- Upload file scanning via VirusTotal
- Hash-based threat detection
- Comprehensive virus analysis
- File encryption (AES, RSA, Fernet)
- File decryption
- Hash lookup and verification

**⚠️ Requires:** VirusTotal API key

---

### 🛡️ PROTECTION

### 5. **INFOCRYPT** - Encryption Suite
> Military-grade data protection

**Features:**
- AES-256 encryption/decryption
- RSA encryption/decryption
- Fernet (symmetric) encryption
- Secure key generation
- Text and file encryption support
- Key management

**No API keys required**

---

### 🧠 INTELLIGENCE

### 6. **OSINT (TrackLyst Pro)** - OSINT Tool
> Advanced username reconnaissance platform with modern interface

**Features:**
- Social media profile discovery (50+ platforms)
- Digital footprint analysis
- Cross-platform tracking
- Username availability checking
- Profile link aggregation
- Real-time URL validation
- Exportable results (JSON format)
- Modern futuristic UI with animated gradients
- Enhanced visual analytics and statistics
- Category-based filtering (social, professional, developer, gaming, media)
- Grid and list view modes

**No API keys required**

---

### 7. **DONNA AI** - Dark Web OSINT Intelligence Platform
> Threat intelligence and automated artifact extraction

**Features:**
- Multi-engine dark web & clearnet search aggregation
- Automated extraction of 15+ artifact types
- AI-powered threat assessment with real-time risk scoring
- Comprehensive intelligence reports
- Source attribution
- Exportable investigation reports

**⚠️ Requires:** Ollama (local model), TOR, Groq API key  
**💡 Optional:** Local LLM fallback via centralized router

---

### 8. **SNAPSPEAK AI** - Image Forensics
> AI-powered image analysis and forensics

**Features:**
- Automatic image captioning (BLIP model)
- Steganography detection
- Comprehensive metadata extraction (EXIF, GPS, camera info)
- Image hashing (perceptual hashing)
- Color analysis and clustering
- Face detection
- Deep image analysis

**⚠️ Requires:** Groq API key  
**💡 Optional:** Local LLM fallback via centralized router

---

### 9. **TRUESHOT AI** - Authenticity Verification
> Advanced AI-generated image detection and media validation

**Features:**
- Deepfake detection
- AI-generated image identification (ResNet-18 model)
- Digital manipulation analysis
- Multi-factor authenticity verification
- Noise pattern analysis
- Texture consistency checking
- Frequency domain analysis
- Confidence scoring with detailed reasoning

**No API keys required** (uses local ML model)

---

### 10. **INFOSIGHT AI** - AI Content Generator
> Next-generation AI content studio with advanced generation capabilities

**Features:**
- AI text generation (Groq Cloud LLM - Llama 3.1/3.3 models)
- AI image generation with multi-model fallback chain (FLUX, Stable Diffusion, Realistic Vision, Qwen, Hunyuan)
- Combined text and image generation (hybrid mode)
- Intelligent prompt enhancement
- Generation history tracking (SQLite database)
- Favorites management system
- Response caching with TTL
- Rate limiting (5 requests/minute, 50/hour)
- Multiple model fallback chain with automatic retry
- Professional glassmorphism UI with animated gradients
- Real-time generation progress tracking
- Export capabilities (copy, download, share)
- **Local LLM Fallback**: Automatically uses Ollama (Qwen2.5-Coder) when Groq fails

**⚠️ Requires:** Groq API key, Hugging Face API key  
**💡 Optional:** Ollama with Qwen2.5-Coder model for local fallback

---

### 11. **LANA AI** - AI Voice Assistant
> Ultimate AI voice assistant with sentiment analysis and conversation analytics

**Features:**
- **Voice & Text Interaction**: Seamless voice and text-based communication
- **Natural Language Processing**: Powered by Groq Cloud LLM (Llama 3.1/3.3 models) with intelligent model selection
- **Sentiment Analysis**: Real-time sentiment detection (positive/negative/neutral) with emotion recognition
- **Conversation Analytics**: Advanced insights including sentiment distribution, active hours, top intents, and response time metrics
- **Real-time Data Integration**: Weather, news, sports scores, currency exchange rates
- **Text-to-Speech (TTS)**: High-quality voice responses with female voice support
- **Speech-to-Text (STT)**: Accurate speech recognition with Google Speech Recognition
- **Audio Visualization**: Real-time frequency equalizer with Web Audio API (64-bar visualization)
- **Voice Activity Indicator**: Visual feedback during voice input
- **Conversation History**: Persistent conversation memory with SQLite database
- **Long-term Memory**: User preferences and important facts storage
- **Scheduled Reminders**: Set and manage reminders
- **Multi-language Support**: 12+ languages (English, Spanish, French, German, Italian, Portuguese, Hindi, Chinese, Japanese, Korean, Arabic, Russian)
- **Smart Caching**: Fast response times with intelligent API response caching (5-minute TTL)
- **Rate Limiting**: Built-in protection (20 requests per 60 seconds)
- **Task Automation**: Automated information extraction and memory management
- **Fast Response Times**: Optimized pipeline with parallel processing (<1.5s average)
- **Enhanced Error Handling**: Graceful fallback with detailed error categorization
- **Local LLM Fallback**: Automatically uses Ollama (Qwen2.5-Coder) when Groq fails or rate-limited
- **Model Selection**: Automatically chooses Llama 3.3-70B-Versatile for complex queries, Llama 3.1-8B-Instant for simple ones
- **Response Time Tracking**: Real-time performance metrics display
- **Futuristic UI**: Modern glassmorphism design with animated gradients, sentiment badges, and insights dashboard

**⚠️ Requires:** Groq API key, News API key, OpenWeather API key, SerpAPI key  
**💡 Optional:** Ollama with Qwen2.5-Coder model for local fallback  
**🎨 UI Features:** Modern gradient design, responsive layout, audio visualizer, voice activity bars, typing indicators, sentiment visualization, conversation insights panel

---

### 12. **CYBERSENTRY AI** - Security Monitoring
> Advanced AI-powered cybersecurity assistant and threat detection system

**Features:**
- Real-time threat detection and analysis
- Security question answering with fuzzy matching
- Intelligent alerting system
- Comprehensive cybersecurity guidance
- Threat intelligence queries
- AI-powered security analysis
- Advanced knowledge base with fuzzy matching
- Enhanced modern UI with professional design
- Comprehensive threat assessment
- **Local LLM Fallback**: Automatically uses Ollama (Qwen2.5-Coder) when Groq fails

**⚠️ Requires:** Groq API key  
**💡 Optional:** Ollama with Qwen2.5-Coder model for local fallback

---

### 13. **INKWELL AI** - Prompt Optimizer & Content Generator
> Advanced prompt optimization and content generation platform

**Features:**
- Prompt optimization and enhancement
- Groq-powered prompt refinement
- 9 rule-based prompt transformations (clarity, specificity, structure, etc.)
- Quality metrics calculation (clarity, completeness, quality scores)
- Category detection (creative, technical, marketing, etc.)
- Enhancement levels (light, moderate, aggressive, expert)
- Prompt history tracking (SQLite database)
- Favorites management system
- Batch processing capabilities
- Prompt versioning and comparison
- Advanced analytics and insights
- **Local LLM Fallback**: Automatically uses Ollama (Qwen2.5-Coder) when Groq fails

**⚠️ Requires:** Groq API key  
**💡 Optional:** Ollama with Qwen2.5-Coder model for local fallback

---

## 📖 Usage

### Basic Workflow

1. **Launch Application**
   ```bash
   python server.py
   ```

2. **Select Tool**
   - Navigate to homepage at `http://127.0.0.1:5000`
   - Choose desired security tool
   - Click "LAUNCH" button

3. **Configure Settings**
   - Input target parameters (domain, file, prompt, etc.)
   - Select scan/analysis options
   - Initiate analysis

4. **Review Results**
   - Analyze generated reports
   - Export data (JSON/CSV where available)
   - Take action on findings

### API Integration

Example API call to WEBSEEKER:
```python
import requests

url = "http://127.0.0.1:5000/webseeker/scan"
data = {
    "target": "example.com",
    "scan_type": "comprehensive"
}

response = requests.post(url, json=data)
print(response.json())
```

---

## ⚙️ Configuration

### API Keys Setup

Required API keys for full functionality (see `config/api-keys-requirements.txt` for details):

1. **VirusTotal API** - File/URL scanning (WebSeeker, FileScanner)
   - Get from: https://www.virustotal.com/gui/join-us
   - Add to `.env`: `VIRUSTOTAL_API_KEY=your_key`

2. **Groq API** - AI features (Multiple tools)
   - Get from: https://console.groq.com/keys
   - Add to `.env`: `GROQ_API_KEY=your_key`

3. **IPInfo API** - IP geolocation (WebSeeker)
   - Get from: https://ipinfo.io/signup
   - Add to `.env`: `IPINFO_API_KEY=your_key`

4. **AbuseIPDB API** - IP reputation (WebSeeker)
   - Get from: https://www.abuseipdb.com/register
   - Add to `.env`: `ABUSEIPDB_API_KEY=your_key`

5. **Hugging Face API** - Image generation (InfoSight AI)
   - Get from: https://huggingface.co/join
   - Add to `.env`: `HUGGINGFACE_API_KEY=your_key`

6. **News API** - News data (LANA AI)
   - Get from: https://newsapi.org/register
   - Add to `.env`: `NEWS_API_KEY=your_key`
   - Used for: Real-time news headlines and updates

7. **OpenWeather API** - Weather data (LANA AI)
   - Get from: https://home.openweathermap.org/users/sign_up
   - Add to `.env`: `OPENWEATHER_API_KEY=your_key` or `WEATHER_API_KEY=your_key`
   - Used for: Current weather conditions and forecasts
   - Fallback: wttr.in service if API unavailable

8. **SerpAPI** - Search results (LANA AI)
   - Get from: https://serpapi.com/users/sign_up
   - Add to `.env`: `SERPAPI_API_KEY=your_key`
   - Used for: Web search and information retrieval

**Note:** Many features work without API keys, but with limited functionality. Refer to `config/api-keys-requirements.txt` for detailed requirements per tool.

### Port Configuration

Default port: `5000`

To change port, edit `server.py`:
```python
serve(app, host='127.0.0.1', port=YOUR_PORT)
```

---

## 📁 Project Structure

```
INFOSIGHT_3.0/
├── server.py                 # Main application server
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variables template
├── .env                      # Environment variables (create from .env.example)
├── .gitignore               # Git ignore rules
├── .flake8                  # Linting configuration
├── pyproject.toml           # Tool configurations
├── readme.md                # This documentation
│
├── app/                     # Application modules
│   ├── __init__.py
│   ├── webseeker.py         # Web security scanner (Recon)
│   ├── portscanner.py       # Network port scanner (Recon)
│   ├── enscan.py            # Domain intelligence (Recon)
│   ├── filescanner.py       # File scanner (Detection)
│   ├── infocrypt.py         # Encryption suite (Protection)
│   ├── osint.py             # OSINT tool (Intelligence)
│   ├── donna.py             # Dark web OSINT (Intelligence)
│   ├── snapspeak_ai.py      # Image forensics (Intelligence)
│   ├── trueshot_ai.py       # Authenticity verification (Intelligence)
│   ├── infosight_ai.py      # AI content generator (Intelligence)
│   ├── lana_ai.py           # AI voice assistant (Intelligence)
│   ├── cybersentry_ai.py    # Security monitoring (Intelligence)
│   ├── inkwell_ai.py        # Prompt optimizer (Intelligence)
│   └── validate_api.py      # API validation utility
│
├── config/                  # Configuration
│   ├── __init__.py          # Centralized configuration
│   └── api-keys-requirements.txt # API key documentation
│
├── llama/                   # Local LLM files (excluded from git)
│   └── models/              # Local model files
│
├── core/                    # Core modules
│   ├── __init__.py
│   └── llm_router.py        # Centralized LLM router with intelligent model selection
│
├── utils/                   # Utility modules
│   ├── __init__.py
│   ├── paths.py             # Path management utilities
│   ├── local_llm_utils.py   # Local LLM (Ollama) utilities
│   ├── record.py            # Recording utilities
│   ├── security.py          # Security utilities (rate limiting, validation)
│   ├── llm_logger.py        # LLM request logging
│   └── vision_analyzer.py   # Vision analysis utilities
│
├── tests/                   # Test suite
│   ├── __init__.py
│   ├── conftest.py          # Pytest configuration
│   ├── test_server.py       # Server tests
│   └── test_utils.py        # Utility tests
│
├── models/                  # ML models
│   └── best_model9.pth      # ResNet-18 model
│
├── data/                    # Data files
│   ├── data.json            # OSINT platform data
│   ├── responses.json       # CyberSentry AI responses
│   └── encryption_metadata.json # Encryption metadata
│
├── scripts/                 # Run scripts
│   ├── run_windows.bat      # Windows startup
│   └── run_linux&mac.sh     # Linux/Mac startup
│
├── static/                  # Static assets
│   ├── css/                 # Stylesheets
│   ├── js/                  # JavaScript files
│   ├── images/              # Image assets
│   └── generated_images/    # Generated image storage
│
├── templates/               # HTML templates
│   ├── homepage.html        # Main homepage
│   ├── error.html           # Error page
│   ├── webseeker.html
│   ├── portscanner.html
│   ├── enscan.html
│   ├── filescanner.html
│   ├── infocrypt.html
│   ├── osint.html
│   ├── donna.html
│   ├── snapspeak.html
│   ├── trueshot.html
│   ├── infosight_ai.html
│   ├── lana.html
│   ├── cybersentry_AI.html
│   └── inkwell_ai.html
│
└── docs/                    # Documentation
    ├── README.md            # Documentation index
    ├── SETUP_GUIDE.md       # Setup instructions
    ├── PROJECT_STRUCTURE.md # Project structure details
    ├── architecture/        # Architecture documentation
    ├── technical/           # Technical documentation
    └── interview/           # Interview preparation
```

---

## 🔒 Security Notes

- **Private Repository**: This is a private project - unauthorized access prohibited
- **API Keys**: Never commit API keys to version control
- **Ethical Use**: Tools designed for legitimate security testing only
- **Legal Compliance**: Ensure authorization before scanning external systems
- **Data Privacy**: Handle collected data per applicable regulations
- **Environment Variables**: Store sensitive keys in `.env` file (not tracked by git)

---

## 🐛 Troubleshooting

### Common Issues

**PortScanner/WebSeeker not working:**
- ✅ Verify Nmap installation: `nmap --version`
- ✅ Ensure Npcap is installed (Windows)
- ✅ Run with administrator privileges (Windows)
- ✅ Check firewall settings
- ✅ Verify Nmap is in system PATH

**Module Import Errors:**
```bash
pip install --upgrade -r requirements.txt
```

**API Rate Limiting:**
- Implement request delays in code
- Use premium API keys for higher limits
- Cache results when possible
- Check API key validity

**Image Generation Fails (InfoSight AI):**
- Verify Hugging Face API key is valid
- Check model availability on Hugging Face
- Some models may be loading (503 error) - wait and retry
- Try different models in fallback chain

**Groq API Errors:**
- Verify API key is correct
- Check API quota/limits
- Ensure internet connection
- Some features work with rule-based fallback

**Performance Issues:**
- Increase system resources (RAM)
- Reduce concurrent operations
- Optimize scan parameters
- Use caching where available

**DONNA AI / Local LLM Issues:**
- Ensure Ollama is installed and running: `ollama serve`
- Verify your model is available: `ollama list`
- Pull the model if needed: `ollama pull qwen2.5-coder:3b-instruct`
- Check Ollama is accessible: `curl http://localhost:11434/api/tags`
- Verify TOR is properly configured (for DONNA AI only)
- Set `OLLAMA_MODEL` env var if using a different model name

**LANA AI Issues:**
- **Audio not playing**: Check browser autoplay policy - user interaction required for first audio
- **Voice recognition not working**: Ensure microphone permissions are granted in browser
- **Web Audio API not available**: Browser may not support Web Audio API - fallback visualization will be used
- **Rate limiting errors**: Wait 60 seconds between requests or increase rate limit in code
- **Model fallback**: Check console logs to see if local Ollama model is being used
- **Memory not saving**: Ensure `data/` directory exists and is writable
- **TTS not working**: Verify `pyttsx3` is installed and system has TTS voices available
- **Slow responses**: Check API key validity and internet connection
- **Audio visualization not showing**: Browser may not support Web Audio API - static wave animation will be used instead

---

## 🤝 Contributing

This is a private repository. For authorized contributors:

1. Create feature branch
2. Commit changes with clear messages
3. Submit pull request
4. Await code review

---

## 📄 License

**Private & Confidential**

This software is proprietary and confidential. Unauthorized copying, distribution, or use is strictly prohibited.

---

## 👥 Author

**INFOSIGHT Development Team**

**Built By:** Lakshan  
**GitHub:** [@itslaks](https://github.com/itslaks)

---

## 🙏 Acknowledgments

- VirusTotal API
- Groq Cloud LLM (Llama models)
- Hugging Face
- Nmap Project
- Flask Framework
- Open-source security community
- All API providers and contributors

---

## 📞 Support

For issues or questions:
- 📧 Email: sjlakshan2004@gmail.com
- 🔗 LinkedIn: https://www.linkedin.com/in/lakshan013/
- 🐙 GitHub: https://github.com/itslaks

---

<div align="center">

**⚡ Built with Python & Flask | Powered by AI | Secured by Design ⚡**

**🍂 Built By Lakshan For Tech Community**

*Last Updated: January 2026*

</div>
