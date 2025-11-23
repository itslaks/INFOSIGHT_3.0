# 🛡️ INFOSIGHT 3.0

<div align="center">

![INFOSIGHT Banner](static/images/logo.png)

**Advanced Cybersecurity & AI Intelligence Suite**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-Latest-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-Private-red.svg)](LICENSE)

*A comprehensive security platform featuring 12 specialized tools for threat detection, data protection, and AI-powered intelligence.*

[Features](#features) • [Installation](#installation) • [Tools](#tools) • [Usage](#usage) • [Requirements](#requirements)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Prerequisites](#prerequisites)
- [Tools](#tools)
- [Usage](#usage)
- [Configuration](#configuration)
- [Contributing](#contributing)

---

## 🌟 Overview

**INFOSIGHT 3.0** is a cutting-edge cybersecurity platform that combines traditional security tools with advanced AI capabilities. Built for security professionals, researchers, and organizations seeking comprehensive threat intelligence and data protection solutions.

---

## ✨ Features

### 🔍 **Scanning & Reconnaissance**
- Multi-database threat intelligence
- Network vulnerability assessment
- DNS enumeration and analysis
- Real-time port scanning

### 🎯 **Threat Detection**
- Multi-engine malware scanning
- Real-time threat assessment
- Behavioral analysis

### 🔐 **Data Protection**
- Military-grade encryption (AES, RSA, Fernet)
- Secure file handling
- Cryptographic hashing

### 🕵️ **OSINT Capabilities**
- Username reconnaissance
- Social media footprint analysis
- Digital presence tracking

### 🖼️ **Image Forensics**
- AI-powered image analysis
- Steganography detection
- Metadata extraction
- Deepfake detection

### 🤖 **AI Intelligence**
- Natural language processing
- Automated threat detection
- Content generation
- Voice-enabled assistance

---

## 🚀 Installation

### Prerequisites

#### **Critical Requirements for PortScanner:**

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

#### **General Requirements:**
- Python 3.8 or higher
- pip (Python package manager)
- Git
- 4GB RAM minimum (8GB recommended)
- Internet connection for API-dependent features

### Step-by-Step Installation

#### **1. Clone the Repository**
```bash
git clone https://github.com/yourusername/infosight_3.0.git
cd infosight_3.0
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
Create a `.env` file in the root directory:
```env
VIRUSTOTAL_API_KEY=your_virustotal_api_key
GOOGLE_API_KEY=your_google_api_key
FLASK_SECRET_KEY=your_secret_key
```

#### **5. Run the Application**

**Windows:**
```bash
run.bat
```

**Linux/macOS:**
```bash
chmod +x run.sh
./run.sh
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
 - **⚠️ Requires Nmap and Npcap installation**
- Domain/URL malware detection
- Port vulnerability analysis
- Real-time threat assessment

### 2. **PORTSCANNER** - Network Port Scanner
> Advanced network reconnaissance and security auditing
- **⚠️ Requires Nmap and Npcap installation**
- Open port identification
- Service detection
- Vulnerability assessment

### 3. **SITE INDEX** - Domain Intelligence
> Multi-layer domain analysis platform
- Email domain verification
- URL risk classification
- DNS enumeration
- JSON-formatted security reports

### 4. **FILE FENDER** - File Scanner
> Multi-engine malware detection system
- Upload file scanning
- Hash-based threat detection
- Comprehensive virus analysis
- File Encryption with 3 best Algorithms
- File Decryption
- Hash lookup 

### 5. **INFOCRYPT** - Encryption Suite
> Military-grade data protection
- AES, RSA, Fernet encryption
- Secure key management
  

### 6. **TRACKLYST** - OSINT Tool
> Username reconnaissance platform
- Social media profile discovery
- Digital footprint analysis
- Cross-platform tracking

###  7. **DONNA AI** - Dark Web OSINT Intelligence Platform
> Threat intelligence and automated artifact extraction
- **⚠️ Requires Ollama local model and TOR installation**
- Multi-engine dark web & clearnet search aggregation
- Automated extraction of 15+ artifact types
- AI-powered threat assessment with real-time risk scoring

### 8. **SNAPSPEAK AI** - Image Forensics
> AI-powered image analysis
- Automatic captioning
- Steganography detection
- Metadata extraction
- Image hashing

### 9. **TRUESHOT AI** - Authenticity Verification
> Advanced media validation
- Deepfake detection
- Edit identification
- Digital manipulation analysis

### 10. **INFOSIGHT AI** - Data Intelligence
> Machine learning analytics platform
- Pattern recognition
- Anomaly detection
- Trend analysis

### 11. **LANA AI** - AI Assistant
> Generative AI assistant
- Voice and text interaction
- Natural language processing
- Task automation

### 12. **CYBERSENTRY AI** - Security Monitoring
> Autonomous cybersecurity AI
- Real-time threat detection
- Network monitoring
- Intelligent alerting

### 13. **INKWELL AI** - Content Generator
> Professional AI writing engine
- Article generation
- Technical documentation
- Creative writing

  
---

## 📖 Usage

### Basic Workflow

1. **Launch Application**
```bash
   python server.py
```

2. **Select Tool**
   - Navigate to homepage
   - Choose desired security tool
   - Click "LAUNCH" button

3. **Configure Settings**
   - Input target parameters
   - Select scan options
   - Initiate analysis

4. **Review Results**
   - Analyze generated reports
   - Export data (JSON/CSV)
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

Required API keys for full functionality:

1. **VirusTotal API** - File/URL scanning
   - Get from: https://www.virustotal.com/gui/join-us
   - Add to `.env`: `VIRUSTOTAL_API_KEY=your_key`

2. **Google Generative AI** - AI features
   - Get from: https://makersuite.google.com/app/apikey
   - Add to `.env`: `GOOGLE_API_KEY=your_key`

3. **AND MORE** - Refer API key requirements txt

### Port Configuration

Default port: `5000`

To change port, edit `server.py`:
```python
serve(app, host='127.0.0.1', port=YOUR_PORT)
```

---

## 📁 Project Structure
```
infosight_3.0/
├── server.py                 # Main application server
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables
├── .gitignore               # Git ignore rules
├── run.bat                  # Windows execution script
├── run.sh                   # Linux/macOS execution script
├── README.md                # Documentation
├── static/                  # Static assets
│   ├── css/
│   ├── js/
│   ├── images/
│   
|── templates
|   └── homepage.html
|   |__other html files 
|
|        # Individual tool blueprints
├── webseeker.py
├── portscanner.py
├── infocrypt.py
└── other all python files , 1 ML model and 2 json files

```

---

## 🔒 Security Notes

- **Private Repository**: This is a private project - unauthorized access prohibited
- **API Keys**: Never commit API keys to version control
- **Ethical Use**: Tools designed for legitimate security testing only
- **Legal Compliance**: Ensure authorization before scanning external systems
- **Data Privacy**: Handle collected data per applicable regulations

---

## 🐛 Troubleshooting

### Common Issues

**PortScanner not working:**
- ✅ Verify Nmap installation: `nmap --version`
- ✅ Ensure Npcap is installed (Windows)
- ✅ Run with administrator privileges
- ✅ Check firewall settings

**Module Import Errors:**
```bash
pip install --upgrade -r requirements.txt
```

**API Rate Limiting:**
- Implement request delays
- Use premium API keys
- Cache results when possible

**Performance Issues:**
- Increase system resources
- Reduce concurrent operations
- Optimize scan parameters

---

## 🤝 Contributing

This is a private repository. For authorized contributors:

1. Create feature branch
2. Commit changes
3. Submit pull request
4. Await code review

---

## 📄 License

**Private & Confidential**

This software is proprietary and confidential. Unauthorized copying, distribution, or use is strictly prohibited.

---

## 👥 Authors

**INFOSIGHT Development Team**

---

## 🙏 Acknowledgments

- VirusTotal API
- Google Generative AI
- Nmap Project
- Flask Framework
- Open-source security community

---

## 📞 Support

For issues or questions:
- 📧 Email: sjlakshan2004@gmail.com
- 🔗 Linkedin : https://www.linkedin.com/in/lakshan013/

---

<div align="center">

**⚡ Built with Python & Flask | Powered by AI | Secured by Design ⚡**

**🍂 Built By Lakshan For Tech Community**

*Last Updated: November 2025*

</div>
