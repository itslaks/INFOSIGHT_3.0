"""
Centralized LLM Router
Single entry point for all LLM operations across the application.
Handles Groq Cloud (primary) and Ollama Local (fallback) routing.
"""

import os
import sys
import logging
from typing import Dict, Optional, Literal
from functools import lru_cache

# Fix Windows encoding issues
if sys.platform == 'win32':
    import io
    # Set UTF-8 encoding for stdout/stderr on Windows
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    if sys.stderr.encoding != 'utf-8':
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

logger = logging.getLogger(__name__)

# Model mapping per app/task type
# Updated to use current Groq models (many old models are decommissioned)
MODEL_MAP = {
    # Deep reasoning tasks (cybersecurity, OSINT, analysis)
    # Using llama-3.3-70b-versatile for complex reasoning tasks
    "cybersentry_ai": "llama-3.3-70b-versatile",  # Best for complex security analysis
    "donna": "llama-3.3-70b-versatile",  # Best for OSINT research
    "enscan": "llama-3.3-70b-versatile",  # Best for security reasoning
    
    # Fast response tasks (chat, assistants, general)
    "lana_ai": "llama-3.1-8b-instant",  # Fast and efficient for chat
    "inkwell_ai": "llama-3.1-8b-instant",  # Fast for prompt optimization
    "webseeker": "llama-3.1-8b-instant",  # Fast for website analysis
    "infosight_ai": "llama-3.1-8b-instant",  # Fast for content generation
    "snapspeak_ai": "llama-3.1-8b-instant",  # Fast for image reasoning
    
    # Default fallback
    "default": "llama-3.1-8b-instant"
}

# Task type overrides
TASK_MODEL_MAP = {
    "deep_reasoning": "llama-3.3-70b-versatile",
    "security_analysis": "llama-3.3-70b-versatile",
    "osint": "llama-3.3-70b-versatile",
    "code_analysis": "llama-3.1-8b-instant",
    "chat": "llama-3.1-8b-instant",
    "image_reasoning": "llama-3.1-8b-instant",
    "prompt_optimization": "llama-3.1-8b-instant"
}


@lru_cache(maxsize=1)
def _get_config():
    """Lazy load config to avoid circular imports"""
    try:
        from config import Config
        return Config
    except ImportError:
        return None


def _get_groq_model(app_name: str, task_type: str = "default") -> str:
    """Determine which Groq model to use based on app and task"""
    # Task type takes precedence
    if task_type in TASK_MODEL_MAP:
        return TASK_MODEL_MAP[task_type]
    
    # App-specific mapping
    return MODEL_MAP.get(app_name, MODEL_MAP["default"])


def _get_groq_client():
    """Get or create Groq client instance"""
    try:
        from groq import Groq
        Config = _get_config()
        
        if Config:
            api_key = Config.GROQ_API_KEY
        else:
            api_key = os.getenv('GROQ_API_KEY')
        
        if not api_key:
            return None
        
        return Groq(api_key=api_key)
    except ImportError:
        logger.warning("groq package not installed")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize Groq client: {e}")
        return None


def _call_groq(prompt: str, system_prompt: Optional[str], model: str, 
               temperature: float = 0.7, max_tokens: int = 2048) -> Optional[str]:
    """Call Groq API"""
    client = _get_groq_client()
    if not client:
        return None
    
    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        if response and response.choices and len(response.choices) > 0:
            return response.choices[0].message.content.strip()
        
        return None
    except Exception as e:
        error_str = str(e).lower()
        if any(keyword in error_str for keyword in ["quota", "rate limit", "429", "api key", "authentication", "401", "403"]):
            logger.warning(f"Groq API error (will fallback): {e}")
        elif "module" in error_str or "import" in error_str:
            logger.warning(f"Groq package not available (will fallback): {e}")
        else:
            logger.error(f"Groq API error: {e}")
        return None


def _call_ollama(prompt: str, system_prompt: Optional[str], 
                 temperature: float = 0.7, max_tokens: int = 4096) -> Optional[str]:
    """Call local Ollama with improved timeout and retry handling"""
    try:
        from utils.local_llm_utils import generate_with_ollama
        
        # generate_with_ollama now has built-in retry logic with exponential backoff
        result, success = generate_with_ollama(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        if success and result:
            return result.strip()
        
        # Log timeout or failure for debugging
        if not success:
            logger.debug("Local LLM call failed or timed out (retries exhausted)")
        
        return None
    except ImportError:
        logger.warning("local_llm_utils not available")
        return None
    except Exception as e:
        # Safely encode error message for Windows
        error_msg = str(e)
        try:
            error_msg = error_msg.encode('utf-8', errors='replace').decode('utf-8')
        except:
            error_msg = repr(e)
        # Don't log timeout errors as errors - they're handled by retry logic
        error_str = str(e).lower()
        if 'timeout' in error_str or 'timed out' in error_str:
            logger.debug(f"Local LLM timeout (handled by retry logic): {error_msg}")
        else:
            logger.error(f"Ollama error: {error_msg}")
        return None


def generate_text(
    prompt: str,
    app_name: str,
    task_type: str = "default",
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    prefer_local: bool = False
) -> Dict[str, any]:
    """
    Centralized LLM text generation with automatic fallback.
    
    Args:
        prompt: User prompt
        app_name: Application name (for model selection)
        task_type: Task type (for model selection override)
        system_prompt: Optional system prompt
        temperature: Temperature (0.0-1.0)
        max_tokens: Maximum tokens
        prefer_local: If True, try local first
    
    Returns:
        {
            "response": str,
            "model": str,
            "source": "groq" | "local"
        }
    """
    # Try local first if preferred
    if prefer_local:
        local_response = _call_ollama(prompt, system_prompt, temperature, max_tokens)
        if local_response:
            logger.debug(f"[{app_name}] Using local LLM (preferred)")
            return {
                "response": local_response,
                "model": _get_config().OLLAMA_MODEL if _get_config() else "ollama",
                "source": "local"
            }
    
    # Try Groq Cloud
    groq_model = _get_groq_model(app_name, task_type)
    groq_response = _call_groq(prompt, system_prompt, groq_model, temperature, max_tokens)
    
    if groq_response:
        logger.debug(f"[{app_name}] Using Groq Cloud ({groq_model})")
        return {
            "response": groq_response,
            "model": groq_model,
            "source": "groq"
        }
    
    # Fallback to local
    logger.debug(f"[{app_name}] Groq failed, falling back to local")
    local_response = _call_ollama(prompt, system_prompt, temperature, max_tokens)
    
    if local_response:
        return {
            "response": local_response,
            "model": _get_config().OLLAMA_MODEL if _get_config() else "ollama",
            "source": "local"
        }
    
    # Both failed
    logger.error(f"[{app_name}] Both Groq and local LLM failed")
    return {
        "response": "",
        "model": "none",
        "source": "none"
    }
