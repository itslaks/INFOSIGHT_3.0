import os
import sys
import importlib
# Ensure Hugging Face token is set before any modules that might use HF are imported
# Prefer environment value if available; otherwise placeholder to remind operator to set it
os.environ.setdefault("HF_TOKEN", os.getenv("HF_TOKEN", "your_token_here"))
# Optional: set a dedicated HF cache directory
os.environ.setdefault("HF_HOME", os.getenv("HF_HOME", "L:/hf_cache"))
import subprocess
import time
import socket
import platform
import argparse
import threading
import signal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

from flask import Flask, render_template, redirect, url_for, jsonify
import logging
import requests

from dotenv import load_dotenv
load_dotenv()

# Security imports
from utils.security import init_rate_limiter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import blueprints from app package
# APP Configuration
APP_PORTS = {
    'infocrypt': 5001,
    'cybersentry_ai': 5002,
    'donna': 5003,
    'enscan': 5004,
    'filescanner': 5005,
    'infosight_ai': 5006,
    'inkwell_ai': 5007,
    'lana_ai': 5008,
    'osint': 5009,
    'portscanner': 5010,
    'snapspeak_ai': 5011,
    'trueshot_ai': 5012,
    'webseeker': 5013
}

BLUEPRINT_CONFIGS = [
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

def register_blueprints_unified(app):
    """Register all blueprints for the unified server mode"""
    blueprints = {}
    for prefix, module_path, blueprint_name in BLUEPRINT_CONFIGS:
        try:
            module = importlib.import_module(module_path)
            blueprint = getattr(module, blueprint_name)
            blueprints[prefix] = blueprint
            logger.info(f"✓ Registered blueprint: {prefix}")
        except ImportError as e:
            error_msg = str(e)
            if 'protobuf' in error_msg.lower() or 'runtime_version' in error_msg.lower():
                logger.warning(f"⚠️ {module_path}: Protobuf compatibility issue - blueprint disabled")
            else:
                logger.error(f"✗ Failed to import {module_path}: {e}")
        except AttributeError as e:
            logger.error(f"✗ Blueprint {blueprint_name} not found in {module_path}: {e}")
        except Exception as e:
            logger.error(f"✗ Unexpected error loading {module_path}: {e}")

    # Register with Flask app
    for prefix, blueprint in blueprints.items():
        try:
            app.register_blueprint(blueprint, url_prefix=prefix)
            logger.info(f"✓ Registered route: {prefix}")
        except Exception as e:
            logger.error(f"✗ Failed to register blueprint {prefix}: {e}")

def launch_distributed_mode():
    """Launch all apps as separate subprocesses"""
    logger.info("🚀 Starting INFOSIGHT 3.0 in DISTRIBUTED mode")
    processes = []
    
    # Path to python executable
    python_exe = sys.executable
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    for app_name, port in APP_PORTS.items():
        app_file = os.path.join(project_root, 'app', f"{app_name}.py")
        if not os.path.exists(app_file):
            logger.warning(f"⚠️ App file not found: {app_file}")
            continue
            
        logger.info(f"🚀 Launching {app_name} on port {port}...")
        try:
            # Launch as subprocess
            cmd = [python_exe, app_file]
            # On Windows, use CREATE_NEW_CONSOLE to spawn separate windows or just run in background
            creation_flags = subprocess.CREATE_NEW_CONSOLE if platform.system() == 'Windows' else 0
            
            p = subprocess.Popen(cmd, cwd=project_root, creationflags=creation_flags)
            processes.append((app_name, p))
        except Exception as e:
            logger.error(f"✗ Failed to launch {app_name}: {e}")
            
    logger.info(f"✓ Launched {len(processes)} services")
    logger.info("Press Ctrl+C to stop all services")
    
    return processes

app = Flask(__name__, template_folder='templates')

# Performance optimizations
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 3600  # Cache static files for 1 hour
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max upload size

# Initialize rate limiting (OWASP: Rate limiting on all endpoints)
limiter = init_rate_limiter(app)
logger.info("✓ Rate limiting initialized")

# Register blueprints with error handling
# MOVED TO register_blueprints_unified FUNCTION


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
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='INFOSIGHT 3.0 Server')
    parser.add_argument('--mode', choices=['unified', 'distributed'], default='unified',
                      help='Run mode: unified (single process) or distributed (multi-process)')
    args = parser.parse_args()
    
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
    
    if args.mode == 'distributed':
        # Launch distributed apps
        processes = launch_distributed_mode()
        
        # Start the main server/gateway (lightweight, no blueprints)
        # This acts as the entry point/landing page
        logger.info("="*70)
        logger.info("🚀 INFOSIGHT 3.0 - Starting Gateway Server (Distribution Mode)")
        logger.info(f"📍 Address: http://{host}:{port}")
        logger.info("="*70)
        
        try:
            serve(app, host=host, port=port)
        finally:
            logger.info("🛑 Shutting down distributed services...")
            for name, p in processes:
                p.terminate()
                
    else:
        # Unified mode
        register_blueprints_unified(app)
        
        logger.info("="*70)
        logger.info("🚀 INFOSIGHT 3.0 - Starting Server (Unified Mode)")
        logger.info(f"📍 Address: http://{host}:{port}")
        logger.info("="*70)
        serve(app, host=host, port=port)
