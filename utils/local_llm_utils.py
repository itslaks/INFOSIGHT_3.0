"""
Local LLM Utility Module
Provides fallback to a local LLaMA/LLM server when cloud models fail.

Supports both:
- Ollama API: /api/generate, /api/tags
- llama.cpp server API: /v1/completions (OpenAI-compatible)
"""
import os
import sys
import requests
import json
from typing import Optional, Tuple

# Fix Windows encoding issues
if sys.platform == 'win32':
    import io
    # Set UTF-8 encoding for stdout/stderr on Windows
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    if sys.stderr.encoding != 'utf-8':
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Configuration
# Base URL of the local LLM server (without trailing slash)
LOCAL_LLM_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

# Model name - for Ollama use model name, for llama.cpp use model file name or path
LOCAL_LLM_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:3b-instruct").strip()

# Increased timeout for complex prompts (5 minutes)
LOCAL_LLM_TIMEOUT = int(os.getenv("LOCAL_LLM_TIMEOUT", "300"))  # seconds (default: 300 = 5 minutes)
LOCAL_LLM_MAX_RETRIES = int(os.getenv("LOCAL_LLM_MAX_RETRIES", "2"))  # Number of retries on timeout
LOCAL_LLM_RETRY_DELAY = float(os.getenv("LOCAL_LLM_RETRY_DELAY", "5.0"))  # Initial retry delay in seconds

# Cache for detected server type
_SERVER_TYPE_CACHE = None


def detect_server_type(clear_cache: bool = False) -> str:
    """
    Detect which type of server is running: 'ollama' or 'llamacpp'
    Returns 'ollama', 'llamacpp', or 'unknown'
    
    Args:
        clear_cache: If True, clear the cache and re-detect
    """
    global _SERVER_TYPE_CACHE
    
    # Clear cache if requested
    if clear_cache:
        _SERVER_TYPE_CACHE = None
    
    # Return cached result if available (only cache successful detections)
    if _SERVER_TYPE_CACHE is not None and _SERVER_TYPE_CACHE != "unknown":
        return _SERVER_TYPE_CACHE
    
    base_url = LOCAL_LLM_BASE_URL.rstrip('/')
    
    # Try root endpoint first (most common, fastest)
    try:
        url = f"{base_url}/"
        response = requests.get(url, timeout=3)
        if response.status_code < 500:
            _SERVER_TYPE_CACHE = "llamacpp"
            return "llamacpp"
    except:
        pass
    
    # Try Ollama API
    try:
        url = f"{base_url}/api/tags"
        response = requests.get(url, timeout=3)
        if response.status_code < 500:
            _SERVER_TYPE_CACHE = "ollama"
            return "ollama"
    except:
        pass
    
    # Try llama.cpp/OpenAI-compatible API
    try:
        url = f"{base_url}/v1/models"
        response = requests.get(url, timeout=3)
        if response.status_code < 500:
            _SERVER_TYPE_CACHE = "llamacpp"
            return "llamacpp"
    except:
        pass
    
    # Also try /health endpoint (some llama.cpp servers use this)
    try:
        url = f"{base_url}/health"
        response = requests.get(url, timeout=3)
        if response.status_code < 500:
            _SERVER_TYPE_CACHE = "llamacpp"
            return "llamacpp"
    except:
        pass
    
    # Don't cache "unknown" - allow retries
    return "unknown"


def check_ollama_available(retries: int = 3, delay: float = 2.0) -> bool:
    """
    Check if the local LLM server is running and accessible.
    Supports both Ollama and llama.cpp server APIs.
    Uses retries with exponential backoff to handle server startup delays.
    
    Args:
        retries: Number of retry attempts
        delay: Initial delay between retries in seconds
    """
    global _SERVER_TYPE_CACHE  # Declare global at function level
    base_url = LOCAL_LLM_BASE_URL.rstrip('/')
    
    # Try multiple times with increasing delays (handles server startup)
    for attempt in range(retries):
        try:
            # Clear cache on first attempt to ensure fresh detection
            # First, try to detect server type (this checks multiple endpoints)
            server_type = detect_server_type(clear_cache=(attempt == 0))
            if server_type != "unknown":
                return True
            
            # Fallback: try root endpoint (most servers respond to this)
            try:
                response = requests.get(f"{base_url}/", timeout=3)
                if response.status_code < 500:
                    # Reset cache since we found a server
                    _SERVER_TYPE_CACHE = "llamacpp"
                    return True
            except:
                pass
            
            # Try Ollama endpoint
            try:
                response = requests.get(f"{base_url}/api/tags", timeout=3)
                if response.status_code < 500:
                    _SERVER_TYPE_CACHE = "ollama"
                    return True
            except:
                pass
            
            # Try llama.cpp/OpenAI-compatible endpoint
            try:
                response = requests.get(f"{base_url}/v1/models", timeout=3)
                if response.status_code < 500:
                    _SERVER_TYPE_CACHE = "llamacpp"
                    return True
            except:
                pass
            
            # Try health endpoint
            try:
                response = requests.get(f"{base_url}/health", timeout=3)
                if response.status_code < 500:
                    _SERVER_TYPE_CACHE = "llamacpp"
                    return True
            except:
                pass
            
        except Exception:
            pass
        
        # If not the last attempt, wait before retrying
        if attempt < retries - 1:
            import time
            time.sleep(delay * (attempt + 1))  # Exponential backoff
    
    return False


def generate_with_ollama(
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    retries: int = None,
) -> Tuple[Optional[str], bool]:
    """
    Generate text using a local LLM server (Ollama or llama.cpp).
    Auto-detects server type and uses appropriate API.
    Includes retry logic with exponential backoff for timeout handling.

    Args:
        prompt: User prompt
        system_prompt: Optional system prompt (prepended to the prompt if provided)
        temperature: Temperature for generation (0.0-1.0)
        max_tokens: Maximum tokens to generate
        retries: Number of retries on timeout (defaults to LOCAL_LLM_MAX_RETRIES)

    Returns:
        Tuple of (generated_text, success)
    """
    if retries is None:
        retries = LOCAL_LLM_MAX_RETRIES
    
    if not check_ollama_available():
        return None, False

    server_type = detect_server_type()
    base_url = LOCAL_LLM_BASE_URL.rstrip('/')

    # Merge system prompt + user prompt
    if system_prompt:
        full_prompt = f"{system_prompt.strip()}\n\n{prompt}"
    else:
        full_prompt = prompt

    # Retry loop with exponential backoff
    import time
    last_error = None
    
    for attempt in range(retries + 1):  # +1 for initial attempt
        try:
            model_name = None  # Initialize for retry logic
            if server_type == "ollama":
                # Use Ollama API
                url = f"{base_url}/api/generate"
                payload = {
                    "model": LOCAL_LLM_MODEL,
                    "prompt": full_prompt,
                    "temperature": float(temperature),
                    "num_predict": int(max_tokens),
                    "stream": False,
                }
            else:
                # Use llama.cpp/OpenAI-compatible API (default for user's setup)
                url = f"{base_url}/v1/completions"
                # For llama.cpp, model name can be the filename or empty if only one model loaded
                # Extract just the filename without extension if it's a path
                model_name = LOCAL_LLM_MODEL
                if "/" in model_name or "\\" in model_name:
                    # Extract filename from path
                    import os
                    model_name = os.path.basename(model_name)
                if model_name.endswith(".gguf"):
                    model_name = model_name[:-5]  # Remove .gguf extension
                
                # For llama.cpp with single model, model name might be optional
                # Try with model name first, will retry without if needed
                payload = {
                    "model": model_name if model_name else "",
                    "prompt": full_prompt,
                    "temperature": float(temperature),
                    "max_tokens": int(max_tokens),
                    "stream": False,
                }

            response = requests.post(
                url,
                json=payload,
                timeout=LOCAL_LLM_TIMEOUT,
            )

            if response.status_code != 200:
                # For llama.cpp, if model name fails, try without model name (single model mode)
                if server_type != "ollama" and model_name and response.status_code == 400:
                    try:
                        payload_no_model = payload.copy()
                        payload_no_model["model"] = ""
                        response = requests.post(url, json=payload_no_model, timeout=LOCAL_LLM_TIMEOUT)
                        if response.status_code == 200:
                            data = response.json()
                            if "choices" in data and len(data["choices"]) > 0:
                                generated_text = (data["choices"][0].get("text") or "").strip()
                                if generated_text:
                                    return generated_text, True
                    except:
                        pass
                
                # Safely encode error message for Windows
                error_msg = response.text[:200] if response.text else "No error message"
                try:
                    error_msg = error_msg.encode('utf-8', errors='replace').decode('utf-8')
                except:
                    error_msg = str(error_msg)
                try:
                    print(f"⚠️ Local LLM returned status {response.status_code}: {error_msg}")
                except UnicodeEncodeError:
                    print(f"Local LLM returned status {response.status_code}: {error_msg}")
                return None, False

            data = response.json()

            # Extract response based on API type
            if server_type == "ollama":
                # Ollama format: {"response": "text"}
                generated_text = (data.get("response") or "").strip()
            else:
                # OpenAI/llama.cpp format: {"choices": [{"text": "..."}]}
                if "choices" in data and len(data["choices"]) > 0:
                    generated_text = (data["choices"][0].get("text") or "").strip()
                elif "content" in data:
                    generated_text = (data.get("content") or "").strip()
                else:
                    # Fallback: try to find text in response
                    generated_text = (data.get("text") or data.get("response") or "").strip()

            if not generated_text:
                try:
                    print("⚠️ Local LLM returned empty completion")
                except UnicodeEncodeError:
                    print("Local LLM returned empty completion")
                return None, False

            # Ensure text is properly decoded
            try:
                if isinstance(generated_text, bytes):
                    generated_text = generated_text.decode('utf-8', errors='replace')
                # Normalize to string and ensure UTF-8
                generated_text = str(generated_text).encode('utf-8', errors='replace').decode('utf-8')
            except Exception as e:
                # If encoding fails, try to salvage what we can
                try:
                    generated_text = str(generated_text).encode('ascii', errors='replace').decode('ascii')
                except:
                    generated_text = ""

            return generated_text, True

        except requests.exceptions.Timeout:
            last_error = "timeout"
            if attempt < retries:
                # Exponential backoff: delay increases with each retry
                delay = LOCAL_LLM_RETRY_DELAY * (2 ** attempt)
                try:
                    print(f"⚠️ Local LLM request timed out (attempt {attempt + 1}/{retries + 1}). Retrying in {delay:.1f}s...")
                except UnicodeEncodeError:
                    print(f"Local LLM request timed out (attempt {attempt + 1}/{retries + 1}). Retrying in {delay:.1f}s...")
                time.sleep(delay)
                continue  # Retry
            else:
                # Final attempt failed
                try:
                    print("⚠️ Local LLM request timed out after all retries")
                except UnicodeEncodeError:
                    print("Local LLM request timed out after all retries")
                return None, False
        except requests.exceptions.ConnectionError as e:
            last_error = "connection"
            # Connection errors usually don't benefit from retries, but try once more
            if attempt < retries:
                delay = LOCAL_LLM_RETRY_DELAY * (2 ** attempt)
                error_msg = str(e)
                try:
                    error_msg = error_msg.encode('utf-8', errors='replace').decode('utf-8')
                except:
                    error_msg = repr(e)
                try:
                    print(f"⚠️ Local LLM connection error (attempt {attempt + 1}/{retries + 1}): {error_msg}. Retrying in {delay:.1f}s...")
                except UnicodeEncodeError:
                    print(f"Local LLM connection error (attempt {attempt + 1}/{retries + 1}): {error_msg}. Retrying in {delay:.1f}s...")
                time.sleep(delay)
                continue
            else:
                # Safely encode error message
                error_msg = str(e)
                try:
                    error_msg = error_msg.encode('utf-8', errors='replace').decode('utf-8')
                except:
                    error_msg = repr(e)
                try:
                    print(f"⚠️ Local LLM connection error: {error_msg}")
                except UnicodeEncodeError:
                    print(f"Local LLM connection error: {error_msg}")
                return None, False
        except Exception as e:
            last_error = "unknown"
            # For other errors, only retry if it's a recoverable error
            error_str = str(e).lower()
            is_recoverable = any(keyword in error_str for keyword in [
                "timeout", "connection", "network", "temporary", "503", "502", "504"
            ])
            
            if is_recoverable and attempt < retries:
                delay = LOCAL_LLM_RETRY_DELAY * (2 ** attempt)
                error_msg = str(e)
                try:
                    error_msg = error_msg.encode('utf-8', errors='replace').decode('utf-8')
                except:
                    error_msg = repr(e)
                try:
                    print(f"⚠️ Local LLM error (attempt {attempt + 1}/{retries + 1}): {error_msg}. Retrying in {delay:.1f}s...")
                except UnicodeEncodeError:
                    print(f"Local LLM error (attempt {attempt + 1}/{retries + 1}): {error_msg}. Retrying in {delay:.1f}s...")
                time.sleep(delay)
                continue
            else:
                # Safely encode error message for Windows
                error_msg = str(e)
                try:
                    error_msg = error_msg.encode('utf-8', errors='replace').decode('utf-8')
                except:
                    error_msg = repr(e)
                try:
                    print(f"⚠️ Local LLM error: {error_msg}")
                except UnicodeEncodeError:
                    print(f"Local LLM error: {error_msg}")
                # Reset cache on error to allow retry with different detection
                global _SERVER_TYPE_CACHE
                _SERVER_TYPE_CACHE = None
                return None, False
    
    # Should not reach here, but handle it just in case
    return None, False

def generate_with_fallback(groq_func, prompt: str, *args, **kwargs) -> Tuple[Optional[str], str]:
    """
    Try Groq first, fallback to Ollama on error
    
    Args:
        groq_func: Function that calls Groq API
        prompt: User prompt
        *args, **kwargs: Additional arguments for groq_func
    
    Returns:
        Tuple of (generated_text, model_used)
        model_used: "groq" or "local" or None
    """
    # Try Groq first
    try:
        result = groq_func(prompt, *args, **kwargs)
        if result:
            return result, "groq"
    except Exception as e:
        error_str = str(e).lower()
        # Check for common Groq errors that should trigger fallback
        if any(keyword in error_str for keyword in [
            "resource exhausted", "quota", "rate limit", "429", 
            "503", "500", "timeout", "unavailable", "error"
        ]):
            # Safely encode error message
            error_msg = str(e)
            try:
                error_msg = error_msg.encode('utf-8', errors='replace').decode('utf-8')
            except:
                error_msg = repr(e)
            try:
                print(f"⚠️ Groq error detected: {error_msg}")
                print("🔄 Falling back to local Ollama model...")
            except UnicodeEncodeError:
                print(f"Groq error detected: {error_msg}")
                print("Falling back to local Ollama model...")
            
            # Try local Ollama
            system_prompt = kwargs.get("system_prompt")
            temperature = kwargs.get("temperature", 0.7)
            max_tokens = kwargs.get("max_tokens", 4096)
            
            local_result, success = generate_with_ollama(
                prompt, 
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            if success and local_result:
                return local_result, "local"
    
    # If Groq didn't raise an error but returned None/empty, try local
    try:
        system_prompt = kwargs.get("system_prompt")
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 4096)
        
        local_result, success = generate_with_ollama(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        if success and local_result:
            return local_result, "local"
    except:
        pass
    
    return None, None
