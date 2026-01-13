# INFOSIGHT 3.0 - Technical Documentation

## Overview

INFOSIGHT 3.0 is a comprehensive cybersecurity and AI intelligence platform that integrates 13 specialized security tools into a unified web-based application, organized by category: **Recon → Detection → Protection → Intelligence**. The platform combines traditional security scanning capabilities with advanced AI-powered analysis, threat intelligence, and digital forensics tools.

## Architecture

### Core Framework
- **Flask 3.1.2**: Web application framework using blueprint architecture for modularity
- **Waitress**: Production WSGI server for deployment
- **Flask-CORS**: Cross-origin resource sharing support
- **Flask-Limiter**: Rate limiting and API protection

### Centralized LLM Router
- **Location**: `core/llm_router.py`
- **Purpose**: Single entry point for all text-based LLM calls
- **Primary**: Groq Cloud LLM
  - Llama 3.3-70B-Versatile for complex reasoning tasks
  - Llama 3.1-8B-Instant for fast response tasks
- **Fallback**: Ollama local LLM (Qwen2.5-Coder-3B-Instruct)
- **Features**: Intelligent model selection, automatic fallback, error handling

### Application Categories

#### 🔍 Reconnaissance
1. **WebSeeker** - Web security scanner
2. **PortScanner** - Network port scanner
3. **Site Index (EnScan)** - Domain intelligence

#### 🎯 Detection
4. **File Fender (FileScanner)** - File scanner

#### 🛡️ Protection
5. **InfoCrypt** - Encryption suite

#### 🧠 Intelligence
6. **TrackLyst (OSINT)** - Username reconnaissance
7. **DONNA AI** - Dark web OSINT
8. **SnapSpeak AI** - Image forensics
9. **TrueShot AI** - Authenticity verification
10. **InfoSight AI** - AI content generator
11. **LANA AI** - AI voice assistant
12. **CyberSentry AI** - Security monitoring
13. **InkWell AI** - Prompt optimizer

## Technologies

### AI and Machine Learning
- **Groq Cloud LLM**: Primary AI model via centralized router
- **Hugging Face Transformers**: BLIP for image captioning, multiple image generation models
- **PyTorch 2.9.1**: ResNet-18 for AI-generated image detection
- **TensorFlow 2.20.0**: Additional ML capabilities
- **Ollama Integration**: Local LLM fallback

### Security and Network Tools
- **python-nmap**: Network port scanning
- **dnspython**: DNS enumeration
- **OpenSSL/pyOpenSSL**: SSL/TLS certificate analysis
- **VirusTotal API**: Multi-engine malware detection
- **AbuseIPDB API**: IP reputation checking
- **IPInfo API**: IP geolocation

### Image Processing
- **Pillow (PIL)**: Image manipulation
- **OpenCV (cv2)**: Computer vision operations
- **ImageHash**: Perceptual hashing
- **ExifRead/piexif**: EXIF metadata extraction
- **scikit-learn**: K-means clustering

### Data Processing
- **SQLite3**: Local database storage
- **JSON**: Configuration storage
- **ReportLab**: PDF report generation

## Memory Management

### Caching Strategy
- **Response Caching**: 5-minute TTL for API responses
- **Model Caching**: Singleton pattern for ML models
- **Database Caching**: In-memory cache for frequent queries

### Resource Optimization
- **Lazy Loading**: Models loaded on first use
- **Connection Pooling**: Database connection reuse
- **Thread Pooling**: Concurrent request handling
- **Memory Cleanup**: Automatic cleanup of temporary files

### Performance
- **Rate Limiting**: Prevents resource exhaustion
- **Request Timeouts**: Prevents hanging requests
- **Batch Processing**: Efficient bulk operations
- **Async Operations**: Non-blocking I/O where possible

## Security Implementation

### Rate Limiting
- IP-based and user-based rate limiting
- Configurable limits per endpoint type
- Graceful 429 responses

### Input Validation
- Schema-based validation
- Type checking and length limits
- Path traversal prevention
- Injection prevention

### API Key Security
- All keys loaded from environment variables
- Centralized configuration
- Never exposed to client-side

### Security Headers
- X-Content-Type-Options
- X-Frame-Options
- X-XSS-Protection
- Strict-Transport-Security
- Content-Security-Policy

## Problem Solving

### Challenges Addressed
1. **Fragmented Security Tools**: Unified interface for multiple tools
2. **AI Integration Complexity**: Centralized LLM router simplifies integration
3. **Offline Capabilities**: Local LLM fallback for resilience
4. **Resource Management**: Efficient memory and caching strategies
5. **Security Hardening**: OWASP-compliant implementation

### Solutions
- Modular blueprint architecture
- Centralized configuration and utilities
- Intelligent fallback mechanisms
- Comprehensive error handling
- Professional security implementation

## Goals

1. **Unified Platform**: Single interface for all security tools
2. **AI-Enhanced Analysis**: Intelligent threat detection and analysis
3. **Reliability**: Local fallback for continuous operation
4. **Security**: OWASP-compliant security practices
5. **Performance**: Optimized resource usage and caching
6. **Scalability**: Modular architecture for easy extension

## Development Practices

### Code Organization
- Blueprint-based modular architecture
- Shared utilities in `utils/`
- Centralized configuration in `config/`
- Core modules in `core/`

### Error Handling
- Comprehensive try-catch blocks
- Graceful degradation
- Detailed logging
- User-friendly error messages

### Testing
- Unit tests in `tests/`
- Integration testing
- API validation
- Security testing
