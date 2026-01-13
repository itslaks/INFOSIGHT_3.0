"""
LLM Status Logger Utility
Provides comprehensive logging for LLM availability and operations across all applications
"""
import logging
from typing import Optional, Tuple
from utils.local_llm_utils import check_ollama_available, generate_with_ollama, LOCAL_LLM_BASE_URL

logger = logging.getLogger(__name__)

def log_llm_status(app_name: str) -> Tuple[bool, bool]:
    """
    Log LLM availability status for an application
    
    Args:
        app_name: Name of the application
        
    Returns:
        Tuple of (local_available, cloud_available)
    """
    logger.info("=" * 70)
    logger.info(f"🔍 [{app_name}] Checking LLM availability...")
    
    # Check local LLM with retries (server may still be starting)
    local_available = False
    try:
        # Use more retries and longer delay for initial check (server may be starting)
        local_available = check_ollama_available(retries=5, delay=2.0)
        if local_available:
            logger.info(f"✅ [{app_name}] Local LLM (Ollama/llama.cpp) is AVAILABLE at {LOCAL_LLM_BASE_URL}")
        else:
            logger.warning(f"⚠️  [{app_name}] Local LLM (Ollama/llama.cpp) is NOT AVAILABLE at {LOCAL_LLM_BASE_URL}")
            logger.warning(f"    → Check if llama-server.exe is running on port 11434")
            logger.warning(f"    → Server may still be starting - will retry on first request")
    except Exception as e:
        logger.error(f"❌ [{app_name}] Error checking local LLM: {e}")
        local_available = False
    
    # Check cloud LLM via router
    cloud_available = False
    try:
        # Try to import and check LLM router
        try:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from core.llm_router import generate_text
            
            # Quick test
            test_result = generate_text(
                prompt="test",
                app_name=app_name,
                task_type="chat",
                max_tokens=1
            )
            
            if test_result.get("response") or test_result.get("source") != "none":
                cloud_available = True
                logger.info(f"✅ [{app_name}] Cloud LLM (via router) is AVAILABLE")
            else:
                logger.warning(f"⚠️  [{app_name}] Cloud LLM (via router) not responding")
        except ImportError:
            logger.warning(f"⚠️  [{app_name}] LLM router not available")
        except Exception as e:
            error_str = str(e).lower()
            if 'api key' in error_str or 'quota' in error_str:
                logger.warning(f"⚠️  [{app_name}] Cloud LLM (via router) API key invalid or quota exceeded")
            else:
                logger.warning(f"⚠️  [{app_name}] Cloud LLM (via router) check failed: {e}")
    except Exception as e:
        logger.error(f"❌ [{app_name}] Error checking cloud LLM: {e}")
        cloud_available = False
    
    # Summary
    if local_available and cloud_available:
        logger.info(f"✅ [{app_name}] LLM Status: Both local and cloud available")
    elif local_available:
        logger.info(f"✅ [{app_name}] LLM Status: Local available, cloud unavailable")
    elif cloud_available:
        logger.info(f"✅ [{app_name}] LLM Status: Cloud available, local unavailable")
    else:
        logger.error(f"❌ [{app_name}] LLM Status: BOTH local and cloud are UNAVAILABLE")
        logger.error(f"    → Application may have limited functionality")
    
    logger.info("=" * 70)
    
    return local_available, cloud_available

def log_llm_request(app_name: str, model_type: str, prompt_length: int = 0):
    """
    Log when an LLM request is being made
    
    Args:
        app_name: Name of the application
        model_type: 'local' or 'cloud'
        prompt_length: Length of the prompt
    """
    if model_type == 'local':
        logger.info(f"🔄 [{app_name}] Processing with LOCAL LLM (prompt: {prompt_length} chars)")
    elif model_type == 'cloud':
        logger.info(f"☁️  [{app_name}] Processing with CLOUD LLM (prompt: {prompt_length} chars)")
    else:
        logger.info(f"🔄 [{app_name}] Processing with {model_type} (prompt: {prompt_length} chars)")

def log_llm_success(app_name: str, model_type: str, response_length: int = 0, latency_ms: float = 0):
    """
    Log successful LLM response
    
    Args:
        app_name: Name of the application
        model_type: Model used
        response_length: Length of response
        latency_ms: Response time in milliseconds
    """
    logger.info(f"✅ [{app_name}] LLM response received from {model_type} ({response_length} chars, {latency_ms:.0f}ms)")

def log_llm_error(app_name: str, model_type: str, error: Exception, fallback: bool = False):
    """
    Log LLM error
    
    Args:
        app_name: Name of the application
        model_type: Model that failed
        error: Exception that occurred
        fallback: Whether fallback will be attempted
    """
    error_str = str(error).lower()
    
    if 'timeout' in error_str or 'timed out' in error_str:
        logger.error(f"⏱️  [{app_name}] {model_type} LLM TIMEOUT: {error}")
    elif 'quota' in error_str or 'rate limit' in error_str or '429' in error_str:
        logger.error(f"🚫 [{app_name}] {model_type} LLM QUOTA/RATE LIMIT: {error}")
    elif 'connection' in error_str or 'refused' in error_str:
        logger.error(f"🔌 [{app_name}] {model_type} LLM CONNECTION ERROR: {error}")
        if model_type == 'local':
            logger.error(f"    → Check if llama-server.exe is running: {LOCAL_LLM_BASE_URL}")
    elif 'api key' in error_str or 'authentication' in error_str:
        logger.error(f"🔑 [{app_name}] {model_type} LLM AUTHENTICATION ERROR: {error}")
    else:
        logger.error(f"❌ [{app_name}] {model_type} LLM ERROR: {error}")
    
    if fallback:
        logger.info(f"🔄 [{app_name}] Attempting fallback to alternative LLM...")
    else:
        logger.error(f"❌ [{app_name}] No fallback available - request failed")

def log_llm_fallback(app_name: str, from_model: str, to_model: str):
    """
    Log LLM fallback attempt
    
    Args:
        app_name: Name of the application
        from_model: Model that failed
        to_model: Model being tried as fallback
    """
    logger.warning(f"🔄 [{app_name}] Falling back from {from_model} to {to_model}")

def log_processing_step(app_name: str, step: str, status: str = "processing", details: str = ""):
    """
    Log a processing step
    
    Args:
        app_name: Name of the application
        step: Step name (e.g., "transcription", "query_processing", "response_generation")
        status: Status ("processing", "success", "error")
        details: Additional details
    """
    if status == "processing":
        logger.info(f"🔄 [{app_name}] Step: {step} - {details}")
    elif status == "success":
        logger.info(f"✅ [{app_name}] Step: {step} - SUCCESS - {details}")
    elif status == "error":
        logger.error(f"❌ [{app_name}] Step: {step} - ERROR - {details}")
    else:
        logger.info(f"ℹ️  [{app_name}] Step: {step} - {status} - {details}")
