import os
import sys
import subprocess
import time
import socket
import platform
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

from flask import Flask, render_template, redirect, url_for, jsonify
import logging
import requests

# Security imports
from utils.security import init_rate_limiter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import blueprints from app package
blueprints = {}
blueprint_configs = [
    ('/infocrypt', 'app.infocrypt', 'infocrypt'),
    ('/cybersentry_ai', 'app.cybersentry_ai', 'cybersentry_ai'),
    ('/lana_ai', 'app.lana_ai', 'lana_ai'),
    ('/osint', 'app.osint', 'osint'),
    ('/portscanner', 'app.portscanner', 'portscanner'),
    ('/webseeker', 'app.webseeker', 'webseeker'),
    ('/filescanner', 'app.filescanner', 'filescanner'),
    ('/infosight_ai', 'app.infosight_ai', 'infosight_ai'),
    ('/snapspeak_ai', 'app.snapspeak_ai', 'snapspeak_ai'),
    ('/trueshot_ai', 'app.trueshot_ai', 'trueshot_ai'),
    ('/enscan', 'app.enscan', 'enscan'),
    ('/inkwell_ai', 'app.inkwell_ai', 'inkwell_ai'),
    ('/donna', 'app.donna', 'donna'),
]

for prefix, module_path, blueprint_name in blueprint_configs:
    try:
        module = __import__(module_path, fromlist=[blueprint_name])
        blueprint = getattr(module, blueprint_name)
        blueprints[prefix] = blueprint
        logger.info(f"✓ Registered blueprint: {prefix}")
    except ImportError as e:
        error_msg = str(e)
        if 'protobuf' in error_msg.lower() or 'runtime_version' in error_msg.lower():
            logger.warning(f"⚠️ {module_path}: Protobuf compatibility issue - blueprint disabled")
            logger.warning(f"⚠️ This is a known issue. The module will continue without {blueprint_name} features.")
        else:
            logger.error(f"✗ Failed to import {module_path}: {e}")
    except AttributeError as e:
        logger.error(f"✗ Blueprint {blueprint_name} not found in {module_path}: {e}")
    except Exception as e:
        error_msg = str(e)
        if 'protobuf' in error_msg.lower() or 'runtime_version' in error_msg.lower() or 'cannot import' in error_msg.lower():
            logger.warning(f"⚠️ {module_path}: Protobuf compatibility issue - blueprint disabled")
            logger.warning(f"⚠️ This is a known issue. The module will continue without {blueprint_name} features.")
        else:
            logger.error(f"✗ Unexpected error loading {module_path}: {e}")

app = Flask(__name__, template_folder='templates')

# Performance optimizations
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 3600  # Cache static files for 1 hour
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max upload size

# Initialize rate limiting (OWASP: Rate limiting on all endpoints)
limiter = init_rate_limiter(app)
logger.info("✓ Rate limiting initialized")

# Register blueprints with error handling
for prefix, blueprint in blueprints.items():
    try:
        app.register_blueprint(blueprint, url_prefix=prefix)
        logger.info(f"✓ Registered route: {prefix}")
    except Exception as e:
        logger.error(f"✗ Failed to register blueprint {prefix}: {e}")


# Add response headers for caching and security
@app.after_request
def after_request(response):
    """
    Add security and performance headers
    OWASP: Implement security headers to prevent common attacks
    """
    # Cache static assets
    if response.content_type and 'text/html' not in response.content_type:
        response.cache_control.max_age = 3600
        response.cache_control.public = True
    
    # Security headers (OWASP best practices)
    response.headers['X-Content-Type-Options'] = 'nosniff'  # Prevent MIME sniffing
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'  # Prevent clickjacking
    response.headers['X-XSS-Protection'] = '1; mode=block'  # XSS protection
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'  # HSTS
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdnjs.cloudflare.com https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdnjs.cloudflare.com https://cdn.jsdelivr.net; img-src 'self' data: blob: https:; font-src 'self' data: https://fonts.gstatic.com https://cdnjs.cloudflare.com https://cdn.jsdelivr.net; connect-src 'self' https://api2.amplitude.com;"
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Permissions-Policy'] = 'geolocation=(), microphone=(self), camera=()'
    
    # Remove server header to prevent information disclosure
    response.headers.pop('Server', None)
    
    return response

@app.route('/')
def login():
    return render_template('homepage.html')


# Global error handlers
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    from flask import request
    # Don't log 404s for common browser requests
    if request.path not in ['/favicon.ico', '/robots.txt', '/apple-touch-icon.png']:
        logger.warning(f"404 error: {request.method} {request.path}")
    return render_template('error.html', error_code=404, error_message="Page not found"), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"500 error: {error}")
    return render_template('error.html', error_code=500, error_message="Internal server error"), 500


@app.errorhandler(403)
def forbidden(error):
    """Handle 403 errors"""
    logger.warning(f"403 error: {error}")
    return render_template('error.html', error_code=403, error_message="Forbidden"), 403

@app.errorhandler(429)
def rate_limit_handler(error):
    """Handle 429 rate limit errors with graceful response"""
    logger.warning(f"429 rate limit exceeded: {error}")
    return jsonify({
        "error": "Rate limit exceeded",
        "message": "Too many requests. Please try again later.",
        "success": False
    }), 429


def check_port_available(host: str, port: int, timeout: float = 2.0) -> bool:
    """Check if a port is accessible"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def check_ollama_running(port: int = 11434) -> bool:
    """Check if Ollama/llama.cpp server is running on the specified port"""
    base_url = f"http://127.0.0.1:{port}"
    
    # Try multiple endpoints to detect server type
    endpoints = [
        ("/", "llama.cpp"),  # Root endpoint (most common)
        ("/v1/models", "llama.cpp"),  # OpenAI-compatible
        ("/api/tags", "ollama"),  # Ollama
        ("/health", "llama.cpp"),  # Health check
    ]
    
    for endpoint, server_type in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=2)
            if response.status_code < 500:
                return True
        except Exception:
            continue
    
    return False


def start_ollama_server(port: int = 11434) -> bool:
    """Start Llama/Ollama server if not already running"""
    # First check if it's already running
    if check_ollama_running(port):
        logger.info(f"✓ Llama server is already running on port {port}")
        return True
    
    # Check if port is in use by something else
    if check_port_available("127.0.0.1", port, timeout=1.0):
        logger.warning(f"⚠️ Port {port} is in use but Llama API is not responding")
        logger.warning(f"⚠️ Please check if Llama server is running correctly")
        return False
    
    logger.info(f"🔄 Starting Llama server on port {port}...")
    
    try:
        is_windows = platform.system() == "Windows"
        
        if is_windows:
            # Try llama.cpp server first (prefer project-local llama folder)
            project_root = os.path.dirname(os.path.abspath(__file__))
            llama_paths = [
                os.path.join(project_root, "llama", "llama-server.exe"),  # repo-local llama folder
                r"D:\llama\llama-server.exe",  # legacy path
                os.path.join(os.getenv("USERPROFILE", ""), "llama", "llama-server.exe"),
                "llama-server.exe"
            ]
            
            llama_cmd = None
            llama_dir = None
            
            # Check for llama-server.exe
            for path in llama_paths:
                if os.path.exists(path):
                    llama_cmd = path
                    llama_dir = os.path.dirname(path)
                    logger.info(f"✓ Found llama-server.exe at: {path}")
                    break
            
            if llama_cmd and llama_dir:
                # Use the user's specific setup
                model_path = os.path.join(llama_dir, "models", "Qwen2.5-Coder-3B-Instruct-abliterated-Q5_K_M.gguf")
                
                # If model doesn't exist at expected path, try to find it
                if not os.path.exists(model_path):
                    # Try to find the model file
                    models_dir = os.path.join(llama_dir, "models")
                    if os.path.exists(models_dir):
                        for file in os.listdir(models_dir):
                            if file.endswith(".gguf") and "qwen" in file.lower():
                                model_path = os.path.join(models_dir, file)
                                logger.info(f"✓ Found model: {model_path}")
                                break
                
                # Start llama-server.exe with the user's configuration
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
                
                creation_flags = 0
                if hasattr(subprocess, 'CREATE_NO_WINDOW'):
                    creation_flags = subprocess.CREATE_NO_WINDOW
                
                # Build command
                cmd = [
                    llama_cmd,
                    "-m", model_path,
                    "--ctx-size", "4096",
                    "--threads", "6",
                    "--port", str(port),
                    "--host", "127.0.0.1"
                ]
                
                logger.info(f"🔄 Starting llama-server.exe...")
                process = subprocess.Popen(
                    cmd,
                    cwd=llama_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    startupinfo=startupinfo,
                    creationflags=creation_flags if creation_flags else 0
                )
            else:
                # Fallback to standard Ollama installation
                ollama_paths = [
                    os.path.join(os.getenv("ProgramFiles", ""), "Ollama", "ollama.exe"),
                    os.path.join(os.getenv("LOCALAPPDATA", ""), "Programs", "Ollama", "ollama.exe"),
                    "ollama.exe",
                    "ollama"
                ]
                
                ollama_cmd = None
                for path in ollama_paths:
                    if os.path.exists(path) or path in ["ollama.exe", "ollama"]:
                        try:
                            result = subprocess.run(
                                ["where", path.split()[0]],
                                capture_output=True,
                                text=True,
                                timeout=2
                            )
                            if result.returncode == 0:
                                ollama_cmd = path
                                break
                        except:
                            if os.path.exists(path):
                                ollama_cmd = path
                                break
                
                if not ollama_cmd:
                    logger.warning("⚠️ Llama/Ollama not found. Please ensure llama-server.exe is at D:\\llama\\")
                    logger.warning("⚠️ Or install Ollama from https://ollama.ai/")
                    return False
                
                # Start standard Ollama
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
                
                creation_flags = 0
                if hasattr(subprocess, 'CREATE_NO_WINDOW'):
                    creation_flags = subprocess.CREATE_NO_WINDOW
                
                process = subprocess.Popen(
                    [ollama_cmd, "serve"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    startupinfo=startupinfo,
                    creationflags=creation_flags if creation_flags else 0
                )
        else:
            # On Linux/Mac, try standard Ollama
            process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True
            )
        
        # Wait a bit for the server to start
        logger.info("⏳ Waiting for Llama server to start...")
        for i in range(15):  # Wait up to 15 seconds (llama.cpp may take longer to load model)
            time.sleep(1)
            if check_ollama_running(port):
                logger.info(f"✓ Llama server started successfully on port {port}")
                return True
            if i % 3 == 0 and i > 0:
                logger.info(f"⏳ Still waiting... ({i}/15 seconds)")
        
        logger.warning("⚠️ Llama server process started but API is not responding yet")
        logger.warning("⚠️ It may still be loading the model. The server will continue...")
        return False
        
    except FileNotFoundError:
        logger.warning("⚠️ Llama/Ollama command not found.")
        logger.warning("⚠️ Please ensure llama-server.exe is at D:\\llama\\ or install Ollama")
        return False
    except Exception as e:
        logger.error(f"✗ Failed to start Llama server: {e}")
        logger.warning("⚠️ Please start Llama server manually")
        return False

if __name__ == '__main__':
    from waitress import serve
    try:
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        from config import Config
        host = Config.HOST
        port = Config.PORT
    except (ImportError, AttributeError):
        host = os.getenv('SERVER_HOST', '127.0.0.1')
        port = int(os.getenv('SERVER_PORT', '5000'))
    
    # Start Ollama server if not running
    ollama_port = 11434
    logger.info("="*70)
    logger.info("🔍 Checking Ollama server status...")
    start_ollama_server(ollama_port)
    logger.info("="*70)
    
    logger.info("="*70)
    logger.info("🚀 INFOSIGHT 3.0 - Starting Server")
    logger.info(f"📍 Address: http://{host}:{port}")
    logger.info(f"📊 Blueprints registered: {len(blueprints)}")
    logger.info("="*70)
    serve(app, host=host, port=port)
