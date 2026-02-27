# INFOSIGHT 3.0 - Project Structure

## 📁 Directory Organization

```
INFOSIGHT_3.0/
├── 📄 server.py                    # Main Flask application entry point
├── 📄 requirements.txt             # Python dependencies
├── 📄 readme.md                    # Main project README
├── 📄 .gitignore                   # Git ignore rules
│
├── 📂 app/                         # Application modules (main Python files)
│   ├── __init__.py
│   ├── cybersentry_ai.py          # CyberSentry AI (Intelligence)
│   ├── donna.py                   # DONNA AI (Intelligence)
│   ├── enscan.py                  # Site Index (Recon)
│   ├── filescanner.py             # File Fender (Detection)
│   ├── infocrypt.py               # InfoCrypt (Protection)
│   ├── infosight_ai.py            # InfoSight AI (Intelligence)
│   ├── inkwell_ai.py              # Inkwell AI (Intelligence)
│   ├── nova_ai.py                 # NOVA AI (Intelligence)
│   ├── osint.py                   # TrackLyst (Intelligence)
│   ├── portscanner.py             # PortScanner (Recon)
│   ├── snapspeak_ai.py            # SnapSpeak AI (Intelligence)
│   ├── trueshot_ai.py             # TrueShot AI (Intelligence)
│   ├── webseeker.py               # WebSeeker (Recon)
│   └── validate_api.py            # API validation utility
│
├── 📂 core/                        # Core system modules
│   ├── __init__.py
│   └── llm_router.py              # Centralized LLM router with intelligent model selection
│
├── 📂 utils/                       # Utility modules
│   ├── __init__.py
│   ├── local_llm_utils.py          # Local LLM (Ollama) utilities
│   ├── record.py                   # Recording utilities
│   ├── security.py                 # Security utilities (rate limiting, validation)
│   ├── llm_logger.py               # LLM request logging
│   ├── paths.py                    # Path management utilities
│   └── vision_analyzer.py          # Vision analysis utilities
│
├── 📂 models/                      # Machine learning models
│   └── best_model9.pth             # ResNet-18 model for TrueShot AI
│
├── 📂 data/                        # Data files
│   ├── data.json                   # OSINT platform data
│   ├── responses.json              # CyberSentry AI responses
│   └── encryption_metadata.json   # FileScanner encryption metadata
│
├── 📂 config/                      # Configuration files
│   ├── __init__.py                 # Centralized configuration
│   └── api-keys-requirements.txt  # API key requirements documentation
│
├── 📂 llama/                       # Local LLM files (excluded from git)
│   └── models/                     # Local model files (Qwen2.5-Coder, etc.)
│
├── 📂 scripts/                     # Run scripts
│   ├── run_windows.bat             # Windows startup script
│   └── run_linux&mac.sh            # Linux/Mac startup script
│
├── 📂 docs/                        # Documentation
│   ├── README.md                   # Documentation index
│   ├── INDEX.md                    # Quick reference
│   ├── ORGANIZATION_SUMMARY.md     # Organization summary
│   ├── architecture/               # Architecture documentation
│   │   └── memory-management.md   # Memory management guide
│   ├── technical/                  # Technical documentation
│   │   └── technical-documentation.md
│   └── interview/                  # Interview preparation
│       └── memory-management-answer.md
│
├── 📂 templates/                   # HTML templates
│   ├── cybersentry_AI.html
│   ├── donna.html
│   ├── enscan.html
│   ├── filescanner.html
│   ├── homepage.html
│   ├── infocrypt.html
│   ├── infosight_ai.html
│   ├── inkwell_ai.html
│   ├── lana.html
│   ├── osint.html
│   ├── portscanner.html
│   ├── snapspeak.html
│   ├── trueshot.html
│   └── webseeker.html
│
└── 📂 static/                      # Static assets
    ├── css/                        # Stylesheets
    │   ├── filescanner.css
    │   ├── homepage.css
    │   └── inkwell_ai.css
    ├── js/                         # JavaScript files
    │   └── homepage.js
    └── images/                     # Image assets
        ├── logo.png
        ├── cybersentry_AI.jpeg
        ├── donna_ai.png
        ├── enscan.png
        └── [other images...]
```

## 📋 File Categories

### Core Application Files
- **server.py**: Main Flask application with blueprint registration
- **requirements.txt**: Python package dependencies

### Application Modules (app/)
All main application modules organized by category:
- **Recon**: `webseeker.py`, `portscanner.py`, `enscan.py`
- **Detection**: `filescanner.py`
- **Protection**: `infocrypt.py`
- **Intelligence**: `osint.py`, `donna.py`, `snapspeak_ai.py`, `trueshot_ai.py`, `infosight_ai.py`, `nova_ai.py`, `cybersentry_ai.py`, `inkwell_ai.py`

### Organized Directories

#### `core/`
Core system modules:
- `llm_router.py`: Centralized LLM router with intelligent model selection (Groq Cloud LLM with local Ollama fallback)

#### `utils/`
Utility modules used across the application:
- `local_llm_utils.py`: Local LLM integration (Ollama)
- `record.py`: Recording utilities
- `security.py`: Security utilities (rate limiting, input validation, OWASP compliance)
- `llm_logger.py`: LLM request logging and monitoring
- `paths.py`: Path management utilities
- `vision_analyzer.py`: Vision analysis utilities for image processing

#### `models/`
Machine learning model files:
- `best_model9.pth`: Pre-trained ResNet-18 model

#### `data/`
JSON data files:
- `data.json`: OSINT platform configurations
- `responses.json`: AI response templates
- `encryption_metadata.json`: Encryption metadata

#### `config/`
Configuration and setup files:
- `api-keys-requirements.txt`: API key setup guide

#### `scripts/`
Execution scripts:
- `run_windows.bat`: Windows startup
- `run_linux&mac.sh`: Linux/Mac startup

#### `docs/`
All documentation organized by category

#### `templates/`
Flask HTML templates

#### `static/`
Static web assets (CSS, JS, images)

## 🔄 Import Paths

### Updated Import Statements
```python
# Before
from local_llm_utils import generate_with_ollama

# After
from utils.local_llm_utils import generate_with_ollama
```

### Updated File Paths
```python
# Models
models/best_model9.pth

# Data files
data/data.json
data/responses.json
data/encryption_metadata.json
```

## 📝 Notes

- **Application modules** organized in `app/` directory by category (Recon → Detection → Protection → Intelligence)
- **Core modules** in `core/` for centralized system functionality (LLM router)
- **Utility modules** in `utils/` for cross-application utilities
- **Data files** centralized in `data/` directory
- **Models** stored in `models/` directory
- **Local LLM files** in `llama/` directory (excluded from git)
- **Documentation** fully organized in `docs/` with subcategories
- All file references updated in code
- Centralized LLM router provides intelligent model selection and automatic fallback

## ✅ Organization Status

- ✅ Models organized
- ✅ Data files organized
- ✅ Scripts organized
- ✅ Config files organized
- ✅ Utils organized
- ✅ Documentation organized
- ✅ Code references updated
- ✅ Import paths updated
