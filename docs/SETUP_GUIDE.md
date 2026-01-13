# INFOSIGHT 3.0 - Setup & Installation Guide

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (optional, for version control)

## 🚀 Quick Start

### 1. Clone/Download the Repository

```bash
cd INFOSIGHT_3.0
```

### 2. Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

1. Copy the example environment file:
   ```bash
   # Windows
   copy .env.example .env
   
   # Linux/Mac
   cp .env.example .env
   ```

2. Edit `.env` file and add your API keys:
   ```env
   GROQ_API_KEY=your_actual_key_here
   VIRUSTOTAL_API_KEY=your_actual_key_here
   # ... etc
   ```

### 5. Run the Application

**Option 1: Using Python directly**
```bash
python server.py
```

**Option 2: Using provided scripts**
```bash
# Windows
scripts\run_windows.bat

# Linux/Mac
chmod +x scripts/run_linux&mac.sh
./scripts/run_linux&mac.sh
```

### 6. Access the Application

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

## 🔧 Configuration

### Required API Keys

The following API keys are required for full functionality:

1. **GROQ_API_KEY** - Required for AI features
   - Get from: https://console.groq.com/keys

2. **VIRUSTOTAL_API_KEY** - Required for WebSeeker and FileScanner
   - Get from: https://www.virustotal.com/gui/join-us

3. **HF_API_TOKEN** - Required for InfoSight AI image generation
   - Get from: https://huggingface.co/settings/tokens

### Optional API Keys

These enhance functionality but are not strictly required:

- **IPINFO_API_KEY** - Enhanced IP information (WebSeeker)
  - Get from: https://ipinfo.io/signup
- **ABUSEIPDB_API_KEY** - Abuse detection (WebSeeker)
  - Get from: https://www.abuseipdb.com/register
- **NEWS_API_KEY** - News features (LANA AI)
  - Get from: https://newsapi.org/register
- **WEATHER_API_KEY** or **OPENWEATHER_API_KEY** - Weather features (LANA AI)
  - Get from: https://home.openweathermap.org/users/sign_up
- **SERPAPI_KEY** - Search features (LANA AI)
  - Get from: https://serpapi.com/users/sign_up

### Local LLM Setup (Optional but Recommended)

For local LLM fallback when cloud APIs are unavailable:

1. **Install Ollama**
   - Download from: https://ollama.ai/
   - Install and start the service: `ollama serve`

2. **Pull Required Model**
   ```bash
   ollama pull qwen2.5-coder:3b-instruct
   ```

3. **Verify Installation**
   ```bash
   ollama list
   # Should show qwen2.5-coder:3b-instruct
   ```

4. **Configure (Optional)**
   - Set `OLLAMA_MODEL` environment variable if using a different model name
   - Default: `qwen2.5-coder:3b-instruct` (auto-detected)

The centralized LLM router will automatically use Ollama as fallback when Groq API fails or is rate-limited.

## 🔒 Security Features

INFOSIGHT 3.0 includes comprehensive security hardening:

- **Rate Limiting**: All endpoints are rate-limited to prevent abuse
- **Input Validation**: All user inputs are validated and sanitized
- **API Key Security**: API keys are never exposed to client-side
- **Security Headers**: OWASP-recommended security headers are applied
- **Path Traversal Prevention**: File operations are secured against directory traversal

For detailed security information, see [Security Hardening Guide](./SECURITY_HARDENING.md).

- **SERPAPI_KEY** - Search features (LANA AI)
  - Get from: https://serpapi.com/users/sign_up

### Environment Variables

All configuration is done through environment variables. See `.env.example` for the complete list.

## 🧪 Running Tests

### Install Test Dependencies

```bash
pip install pytest pytest-cov
```

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_server.py

# Run with verbose output
pytest -v
```

## 🏗️ Development Setup

### Code Formatting

```bash
# Install black
pip install black

# Format all Python files
black *.py utils/*.py config/*.py

# Check formatting without making changes
black --check *.py
```

### Linting

```bash
# Install flake8
pip install flake8

# Run linter
flake8 *.py utils/*.py config/*.py

# Run with specific configuration
flake8 --config=.flake8
```

## 📦 Production Deployment

### Using Waitress (Recommended for Windows)

The application already uses Waitress as the WSGI server. For production:

```bash
waitress-serve --host=0.0.0.0 --port=5000 server:app
```

### Using Gunicorn (Linux/Mac)

```bash
gunicorn -w 4 -b 0.0.0.0:5000 server:app
```

### Environment Variables for Production

Set these in your production environment:

```env
FLASK_ENV=production
SERVER_HOST=0.0.0.0
SERVER_PORT=5000
```

## 🐛 Troubleshooting

### Import Errors

If you get import errors:
```bash
# Make sure you're in the project root directory
cd INFOSIGHT_3.0

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Module Not Found

If modules can't be found:
```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Install missing packages
pip install <package-name>
```

### API Key Errors

If you see API key errors:
1. Check that `.env` file exists
2. Verify all required keys are set
3. Restart the application after changing `.env`

### Port Already in Use

If port 5000 is already in use:
```bash
# Change port in .env file
SERVER_PORT=5001

# Or specify when running
python server.py --port 5001
```

## 📚 Additional Resources

- [Flask Documentation](https://flask.palletsprojects.com/)
- [Waitress Documentation](https://docs.pylonsproject.org/projects/waitress/)
- [Pytest Documentation](https://docs.pytest.org/)

## ✅ Verification

After setup, verify everything works:

1. **Check server starts:**
   ```bash
   python server.py
   # Should see: "Starting INFOSIGHT 3.0 server on 127.0.0.1:5000"
   ```

2. **Check homepage loads:**
   - Open http://127.0.0.1:5000
   - Should see the homepage

3. **Run tests:**
   ```bash
   pytest tests/test_server.py -v
   # All tests should pass
   ```

## 🆘 Getting Help

If you encounter issues:

1. Check the logs in the console output
2. Review the `.env` file configuration
3. Verify all dependencies are installed
4. Check the documentation in `docs/` folder

---

**Last Updated**: After cleanup implementation  
**Version**: 3.0
