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
- Natural language processing (Gemini AI)
- Automated threat detection
- AI content generation (text and images)
- Voice-enabled assistance (LANA AI)
- Prompt optimization and enhancement
- Real-time data integration (weather, news, sports)

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

3. **Ollama** (For DONNA AI - Optional)
   - Download from [Ollama Official Site](https://ollama.ai/)
   - Required for local AI model inference

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
GEMINI_API_KEY=your_gemini_api_key
IPINFO_API_KEY=your_ipinfo_api_key
ABUSEIPDB_API_KEY=your_abuseipdb_api_key
HUGGINGFACE_API_KEY=your_huggingface_api_key
NEWS_API_KEY=your_news_api_key
OPENWEATHER_API_KEY=your_openweather_api_key
SERPAPI_API_KEY=your_serpapi_api_key
FLASK_SECRET_KEY=your_secret_key
```

#### **5. Run the Application**

**Windows:**
```bash
run_windows.bat
```

**Linux/macOS:**
```bash
chmod +x run_linux&mac.sh
./run_linux&mac.sh
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

### 1. **WEBSEEKER** - Web Security Scanner
> Comprehensive threat intelligence combining VirusTotal and Nmap scanning

**Features:**
- Domain/URL malware detection via VirusTotal
- Port vulnerability analysis with Nmap
- IP information and geolocation
- AbuseIPDB reputation checking
- Real-time threat assessment
- AI-powered analysis with Gemini

**⚠️ Requires:** Nmap, Npcap (Windows), VirusTotal API key, IPInfo API key, AbuseIPDB API key, Gemini API key

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

### 6. **OSINT (TrackLyst)** - OSINT Tool
> Username reconnaissance platform

**Features:**
- Social media profile discovery (50+ platforms)
- Digital footprint analysis
- Cross-platform tracking
- Username availability checking
- Profile link aggregation
- Exportable results

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

**⚠️ Requires:** Ollama (local model), TOR, Gemini API key

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

**⚠️ Requires:** Gemini API key

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
> Machine learning-powered content generation platform

**Features:**
- AI text generation (Gemini 2.0 Flash)
- AI image generation (Hugging Face models: FLUX, Stable Diffusion, Realistic Vision)
- Combined text and image generation
- Prompt enhancement
- Response caching
- Rate limiting
- Multiple model fallback chain

**⚠️ Requires:** Gemini API key, Hugging Face API key

---

### 11. **LANA AI** - AI Voice Assistant
> Generative AI assistant with voice capabilities

**Features:**
- Voice and text interaction
- Natural language processing (Gemini 2.0 Flash)
- Real-time data integration (weather, news, sports, currency)
- Text-to-speech (TTS)
- Speech-to-text (STT)
- Conversation history
- Task automation
- Fast response times (<1.5s average)

**⚠️ Requires:** Gemini API key, News API key, OpenWeather API key, SerpAPI key

---

### 12. **CYBERSENTRY AI** - Security Monitoring
> Autonomous cybersecurity AI assistant

**Features:**
- Real-time threat detection
- Security question answering
- Intelligent alerting
- Cybersecurity guidance
- Threat intelligence queries
- AI-powered security analysis

**⚠️ Requires:** Gemini API key

---

### 13. **INKWELL AI** - Prompt Optimizer & Content Generator
> Professional AI prompt optimization and writing engine

**Features:**
- Prompt optimization and enhancement
- Gemini-powered prompt refinement
- Rule-based prompt transformations
- Quality metrics calculation
- Category detection
- Enhancement levels (light, moderate, aggressive, expert)
- History tracking
- Favorites management
- Database storage

**⚠️ Requires:** Gemini API key

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

Required API keys for full functionality (see `API key requirements.txt` for details):

1. **VirusTotal API** - File/URL scanning (WebSeeker, FileScanner)
   - Get from: https://www.virustotal.com/gui/join-us
   - Add to `.env`: `VIRUSTOTAL_API_KEY=your_key`

2. **Google Gemini API** - AI features (Multiple tools)
   - Get from: https://makersuite.google.com/app/apikey
   - Add to `.env`: `GEMINI_API_KEY=your_key`

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

7. **OpenWeather API** - Weather data (LANA AI)
   - Get from: https://home.openweathermap.org/users/sign_up
   - Add to `.env`: `OPENWEATHER_API_KEY=your_key`

8. **SerpAPI** - Search results (LANA AI)
   - Get from: https://serpapi.com/users/sign_up
   - Add to `.env`: `SERPAPI_API_KEY=your_key`

**Note:** Many features work without API keys, but with limited functionality. Refer to `API key requirements.txt` for detailed requirements per tool.

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
├── API key requirements.txt # API key documentation
├── .env                      # Environment variables (create this)
├── .gitignore               # Git ignore rules
├── run_windows.bat          # Windows execution script
├── run_linux&mac.sh        # Linux/macOS execution script
├── readme.md                # This documentation
│
├── static/                  # Static assets
│   ├── css/                 # Stylesheets
│   ├── js/                  # JavaScript files
│   ├── images/              # Image assets
│   └── generated_images/   # Generated image storage
│
├── templates/               # HTML templates
│   ├── homepage.html        # Main homepage
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
├── audio/                   # Audio file storage
├── temp/                    # Temporary files
│
├── Individual tool modules:
├── webseeker.py             # Web security scanner
├── portscanner.py           # Network port scanner
├── enscan.py                # Domain intelligence
├── filescanner.py           # File scanner
├── infocrypt.py             # Encryption suite
├── osint.py                 # OSINT tool
├── donna.py                 # Dark web OSINT
├── snapspeak_ai.py          # Image forensics
├── trueshot_ai.py           # Authenticity verification
├── infosight_ai.py          # AI content generator
├── lana_ai.py               # AI voice assistant
├── cybersentry_ai.py        # Security monitoring
├── inkwell_ai.py            # Prompt optimizer
│
└── Additional files:
    ├── best_model9.pth      # ML model for TrueShot AI
    ├── prompt_optimizer.db  # SQLite database for InkWell AI
    ├── data.json            # Data storage
    ├── responses.json       # Response templates
    └── encryption_metadata.json # Encryption metadata
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

**Gemini API Errors:**
- Verify API key is correct
- Check API quota/limits
- Ensure internet connection
- Some features work with rule-based fallback

**Performance Issues:**
- Increase system resources (RAM)
- Reduce concurrent operations
- Optimize scan parameters
- Use caching where available

**DONNA AI Issues:**
- Ensure Ollama is installed and running
- Verify TOR is properly configured
- Check local model availability
- Review Ollama model list: `ollama list`

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
- Google Gemini AI
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
