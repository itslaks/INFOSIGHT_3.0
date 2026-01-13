#!/usr/bin/env python3
"""
Comprehensive API Validation Script for INFOSIGHT 3.0
Tests all external APIs and local LLM to verify configuration and availability
"""
import os
import sys
import requests
import json
from pathlib import Path
import io
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

# Fix Windows encoding issues
if sys.platform == 'win32':
    import io
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    if sys.stderr.encoding != 'utf-8':
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Try to load from config
try:
    from config import Config
    GROQ_API_KEY = Config.GROQ_API_KEY
    HF_API_TOKEN = Config.HF_API_TOKEN
    VIRUSTOTAL_API_KEY = Config.VIRUSTOTAL_API_KEY
    IPINFO_API_KEY = Config.IPINFO_API_KEY
    ABUSEIPDB_API_KEY = Config.ABUSEIPDB_API_KEY
    NEWS_API_KEY = Config.NEWS_API_KEY
    WEATHER_API_KEY = Config.WEATHER_API_KEY
    SERPAPI_KEY = Config.SERPAPI_KEY
    OLLAMA_BASE_URL = Config.OLLAMA_BASE_URL
    OLLAMA_MODEL = Config.OLLAMA_MODEL
except ImportError:
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    HF_API_TOKEN = os.getenv('HF_API_TOKEN')
    VIRUSTOTAL_API_KEY = os.getenv('VIRUSTOTAL_API_KEY')
    IPINFO_API_KEY = os.getenv('IPINFO_API_KEY')
    ABUSEIPDB_API_KEY = os.getenv('ABUSEIPDB_API_KEY')
    NEWS_API_KEY = os.getenv('NEWS_API_KEY')
    WEATHER_API_KEY = os.getenv('WEATHER_API_KEY')
    SERPAPI_KEY = os.getenv('SERPAPI_KEY')
    OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://127.0.0.1:11434')
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'qwen2.5-coder:3b-instruct')

# Results storage
results: Dict[str, Dict[str, Any]] = {}


def mask_token(token: str, show_chars: int = 8) -> str:
    """Mask API token for display"""
    if not token or len(token) < show_chars * 2:
        return "***"
    return token[:show_chars] + "..." + token[-4:]


def check_api_key(api_name: str, key: Optional[str], required: bool = True) -> Tuple[bool, str]:
    """Check if API key exists and format is valid"""
    if not key:
        return False, "NOT CONFIGURED" if required else "OPTIONAL - NOT SET"
    
    key = key.strip()
    if not key or key.startswith('your_') or len(key) < 10:
        return False, "INVALID FORMAT"
    
    return True, mask_token(key)


def test_groq_api() -> Dict[str, Any]:
    """Test Groq API"""
    api_name = "Groq API"
    result = {
        'name': api_name,
        'configured': False,
        'working': False,
        'credits': None,
        'error': None,
        'details': {}
    }
    
    # Check if key exists
    configured, key_preview = check_api_key(api_name, GROQ_API_KEY)
    result['configured'] = configured
    result['details']['key_preview'] = key_preview
    
    if not configured:
        result['error'] = "API key not configured"
        return result
    
    # Test API
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY.strip()}",
            "Content-Type": "application/json"
        }
        payload = {
            "messages": [{"role": "user", "content": "test"}],
            "model": "llama-3.1-8b-instant",
            "max_tokens": 5
        }
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            result['working'] = True
            result['details']['model'] = "llama-3.1-8b-instant"
            # Groq doesn't provide credit info in response, but we can infer from success
            result['credits'] = "Available (no quota info in API)"
        elif response.status_code == 401:
            result['error'] = "Invalid API key (HTTP 401)"
        elif response.status_code == 429:
            result['error'] = "Rate limit exceeded (HTTP 429)"
            result['credits'] = "May be exhausted"
        else:
            result['error'] = f"HTTP {response.status_code}"
            try:
                error_data = response.json()
                result['error'] = error_data.get('error', {}).get('message', result['error'])
            except:
                pass
    except requests.exceptions.Timeout:
        result['error'] = "Connection timeout"
    except requests.exceptions.ConnectionError:
        result['error'] = "Connection error"
    except Exception as e:
        result['error'] = f"{type(e).__name__}: {str(e)}"
    
    return result


def test_huggingface_api() -> Dict[str, Any]:
    """Test Hugging Face API"""
    api_name = "Hugging Face API"
    result = {
        'name': api_name,
        'configured': False,
        'working': False,
        'credits': None,
        'error': None,
        'details': {}
    }
    
    configured, key_preview = check_api_key(api_name, HF_API_TOKEN)
    result['configured'] = configured
    result['details']['key_preview'] = key_preview
    
    if not configured:
        result['error'] = "API token not configured"
        return result
    
    # Check token format
    if not HF_API_TOKEN.strip().startswith('hf_'):
        result['error'] = "Invalid token format (should start with 'hf_')"
        return result
    
    # Test API
    try:
        headers = {"Authorization": f"Bearer {HF_API_TOKEN.strip()}"}
        response = requests.get(
            "https://huggingface.co/api/whoami-v2",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            user_info = response.json()
            result['working'] = True
            result['details']['username'] = user_info.get('name', 'Unknown')
            result['details']['fullname'] = user_info.get('fullname', 'N/A')
            # HF doesn't provide credit info, but we can check if user has access
            result['credits'] = "Available (free tier)"
        elif response.status_code == 401:
            result['error'] = "Invalid or expired token (HTTP 401)"
        else:
            result['error'] = f"HTTP {response.status_code}"
    except requests.exceptions.Timeout:
        result['error'] = "Connection timeout"
    except requests.exceptions.ConnectionError:
        result['error'] = "Connection error"
    except Exception as e:
        result['error'] = f"{type(e).__name__}: {str(e)}"
    
    return result


def test_virustotal_api() -> Dict[str, Any]:
    """Test VirusTotal API"""
    api_name = "VirusTotal API"
    result = {
        'name': api_name,
        'configured': False,
        'working': False,
        'credits': None,
        'error': None,
        'details': {}
    }
    
    configured, key_preview = check_api_key(api_name, VIRUSTOTAL_API_KEY, required=False)
    result['configured'] = configured
    result['details']['key_preview'] = key_preview
    
    if not configured:
        result['error'] = "API key not configured (optional)"
        return result
    
    # Test API
    try:
        headers = {"x-apikey": VIRUSTOTAL_API_KEY.strip()}
        # Use IP address lookup endpoint to validate API key
        response = requests.get(
            "https://www.virustotal.com/api/v3/ip_addresses/8.8.8.8",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json().get("data", {})
            result['working'] = True
            result['details']['test_ip'] = data.get("id", "8.8.8.8")
            result['credits'] = "Available (key valid)"
        elif response.status_code == 401:
            result['error'] = "Invalid API key (HTTP 401)"
        elif response.status_code == 403:
            result['error'] = "Forbidden - check API key permissions"
        else:
            result['error'] = f"HTTP {response.status_code}"
    except requests.exceptions.Timeout:
        result['error'] = "Connection timeout"
    except requests.exceptions.ConnectionError:
        result['error'] = "Connection error"
    except Exception as e:
        result['error'] = f"{type(e).__name__}: {str(e)}"
    
    return result


def test_ipinfo_api() -> Dict[str, Any]:
    """Test IPInfo API"""
    api_name = "IPInfo API"
    result = {
        'name': api_name,
        'configured': False,
        'working': False,
        'credits': None,
        'error': None,
        'details': {}
    }
    
    configured, key_preview = check_api_key(api_name, IPINFO_API_KEY, required=False)
    result['configured'] = configured
    result['details']['key_preview'] = key_preview
    
    if not configured:
        result['error'] = "API key not configured (optional)"
        return result
    
    # Test API with a simple IP lookup
    try:
        response = requests.get(
            f"https://ipinfo.io/8.8.8.8?token={IPINFO_API_KEY.strip()}",
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            result['working'] = True
            result['details']['test_ip'] = data.get('ip', '8.8.8.8')
            result['credits'] = "Available (free tier: 50k requests/month)"
        elif response.status_code == 401:
            result['error'] = "Invalid API key (HTTP 401)"
        elif response.status_code == 429:
            result['error'] = "Rate limit exceeded (HTTP 429)"
            result['credits'] = "Quota exhausted"
        else:
            result['error'] = f"HTTP {response.status_code}"
    except requests.exceptions.Timeout:
        result['error'] = "Connection timeout"
    except requests.exceptions.ConnectionError:
        result['error'] = "Connection error"
    except Exception as e:
        result['error'] = f"{type(e).__name__}: {str(e)}"
    
    return result


def test_abuseipdb_api() -> Dict[str, Any]:
    """Test AbuseIPDB API"""
    api_name = "AbuseIPDB API"
    result = {
        'name': api_name,
        'configured': False,
        'working': False,
        'credits': None,
        'error': None,
        'details': {}
    }
    
    configured, key_preview = check_api_key(api_name, ABUSEIPDB_API_KEY, required=False)
    result['configured'] = configured
    result['details']['key_preview'] = key_preview
    
    if not configured:
        result['error'] = "API key not configured (optional)"
        return result
    
    # Test API
    try:
        headers = {"Key": ABUSEIPDB_API_KEY.strip()}
        params = {"ipAddress": "8.8.8.8", "maxAgeInDays": 90}
        response = requests.get(
            "https://api.abuseipdb.com/api/v2/check",
            headers=headers,
            params=params,
            timeout=10
        )
        
        if response.status_code == 200:
            result['working'] = True
            data = response.json()
            result['details']['test_ip'] = "8.8.8.8"
            # Check remaining quota
            remaining = response.headers.get('X-RateLimit-Remaining')
            if remaining:
                result['credits'] = f"Remaining requests: {remaining}"
            else:
                result['credits'] = "Available"
        elif response.status_code == 401:
            result['error'] = "Invalid API key (HTTP 401)"
        elif response.status_code == 429:
            result['error'] = "Rate limit exceeded (HTTP 429)"
            result['credits'] = "Quota exhausted"
        else:
            result['error'] = f"HTTP {response.status_code}"
    except requests.exceptions.Timeout:
        result['error'] = "Connection timeout"
    except requests.exceptions.ConnectionError:
        result['error'] = "Connection error"
    except Exception as e:
        result['error'] = f"{type(e).__name__}: {str(e)}"
    
    return result


def test_news_api() -> Dict[str, Any]:
    """Test News API"""
    api_name = "News API"
    result = {
        'name': api_name,
        'configured': False,
        'working': False,
        'credits': None,
        'error': None,
        'details': {}
    }
    
    configured, key_preview = check_api_key(api_name, NEWS_API_KEY, required=False)
    result['configured'] = configured
    result['details']['key_preview'] = key_preview
    
    if not configured:
        result['error'] = "API key not configured (optional)"
        return result
    
    # Test API
    try:
        response = requests.get(
            f"https://newsapi.org/v2/top-headlines?country=us&pageSize=1&apiKey={NEWS_API_KEY.strip()}",
            timeout=10
        )
        
        if response.status_code == 200:
            result['working'] = True
            result['credits'] = "Available (free tier: 100 requests/day)"
        elif response.status_code == 401:
            result['error'] = "Invalid API key (HTTP 401)"
        elif response.status_code == 429:
            result['error'] = "Rate limit exceeded (HTTP 429)"
            result['credits'] = "Daily quota exhausted"
        else:
            result['error'] = f"HTTP {response.status_code}"
    except requests.exceptions.Timeout:
        result['error'] = "Connection timeout"
    except requests.exceptions.ConnectionError:
        result['error'] = "Connection error"
    except Exception as e:
        result['error'] = f"{type(e).__name__}: {str(e)}"
    
    return result


def test_weather_api() -> Dict[str, Any]:
    """Test OpenWeather API"""
    api_name = "OpenWeather API"
    result = {
        'name': api_name,
        'configured': False,
        'working': False,
        'credits': None,
        'error': None,
        'details': {}
    }
    
    configured, key_preview = check_api_key(api_name, WEATHER_API_KEY, required=False)
    result['configured'] = configured
    result['details']['key_preview'] = key_preview
    
    if not configured:
        result['error'] = "API key not configured (optional)"
        return result
    
    # Test API
    try:
        response = requests.get(
            f"https://api.openweathermap.org/data/2.5/weather?q=London&appid={WEATHER_API_KEY.strip()}",
            timeout=10
        )
        
        if response.status_code == 200:
            result['working'] = True
            result['credits'] = "Available (free tier: 60 calls/minute)"
        elif response.status_code == 401:
            result['error'] = "Invalid API key (HTTP 401)"
        elif response.status_code == 429:
            result['error'] = "Rate limit exceeded (HTTP 429)"
            result['credits'] = "Quota exhausted"
        else:
            result['error'] = f"HTTP {response.status_code}"
    except requests.exceptions.Timeout:
        result['error'] = "Connection timeout"
    except requests.exceptions.ConnectionError:
        result['error'] = "Connection error"
    except Exception as e:
        result['error'] = f"{type(e).__name__}: {str(e)}"
    
    return result


def test_serpapi() -> Dict[str, Any]:
    """Test SerpAPI"""
    api_name = "SerpAPI"
    result = {
        'name': api_name,
        'configured': False,
        'working': False,
        'credits': None,
        'error': None,
        'details': {}
    }
    
    configured, key_preview = check_api_key(api_name, SERPAPI_KEY, required=False)
    result['configured'] = configured
    result['details']['key_preview'] = key_preview
    
    if not configured:
        result['error'] = "API key not configured (optional)"
        return result
    
    # Test API
    try:
        params = {
            "engine": "google",
            "q": "test",
            "api_key": SERPAPI_KEY.strip()
        }
        response = requests.get(
            "https://serpapi.com/search",
            params=params,
            timeout=15
        )
        
        if response.status_code == 200:
            result['working'] = True
            # Check account info if available
            account_info = response.headers.get('X-SerpAPI-Account')
            if account_info:
                result['credits'] = f"Account: {account_info}"
            else:
                result['credits'] = "Available (check dashboard for quota)"
        elif response.status_code == 401:
            result['error'] = "Invalid API key (HTTP 401)"
        elif response.status_code == 429:
            result['error'] = "Rate limit exceeded (HTTP 429)"
            result['credits'] = "Quota exhausted"
        else:
            result['error'] = f"HTTP {response.status_code}"
    except requests.exceptions.Timeout:
        result['error'] = "Connection timeout"
    except requests.exceptions.ConnectionError:
        result['error'] = "Connection error"
    except Exception as e:
        result['error'] = f"{type(e).__name__}: {str(e)}"
    
    return result


def test_local_llm() -> Dict[str, Any]:
    """Test Local LLM Server"""
    api_name = "Local LLM (Ollama/llama.cpp)"
    result = {
        'name': api_name,
        'configured': False,
        'working': False,
        'credits': "Unlimited (local)",
        'error': None,
        'details': {}
    }
    
    result['configured'] = True  # Always configured if server is running
    result['details']['url'] = OLLAMA_BASE_URL
    result['details']['model'] = OLLAMA_MODEL
    
    # Try multiple endpoints to detect server type
    endpoints_to_try = [
        ("/api/tags", "Ollama API"),
        ("/v1/models", "OpenAI-compatible API"),
        ("/", "Root endpoint"),
    ]
    
    for endpoint, endpoint_name in endpoints_to_try:
        try:
            url = f"{OLLAMA_BASE_URL.rstrip('/')}{endpoint}"
            response = requests.get(url, timeout=3)
            
            if response.status_code < 500:
                result['working'] = True
                result['details']['server_type'] = endpoint_name
                
                if endpoint == "/api/tags":
                    try:
                        models = response.json().get('models', [])
                        if models:
                            model_names = [m.get('name', 'unknown') for m in models[:5]]
                            result['details']['available_models'] = model_names
                    except:
                        pass
                elif endpoint == "/v1/models":
                    try:
                        models_data = response.json()
                        if isinstance(models_data, dict) and 'data' in models_data:
                            models = models_data['data']
                            if models:
                                model_names = [m.get('id', 'unknown') for m in models[:5]]
                                result['details']['available_models'] = model_names
                    except:
                        pass
                
                break
        except requests.exceptions.Timeout:
            continue
        except requests.exceptions.ConnectionError:
            continue
        except Exception as e:
            continue
    
    if not result['working']:
        result['error'] = f"Server not responding at {OLLAMA_BASE_URL}"
        result['details']['suggestion'] = "Check if llama-server.exe is running"
    
    return result


def print_result(result: Dict[str, Any], index: int, total: int):
    """Print formatted result for an API"""
    name = result['name']
    configured = result['configured']
    working = result['working']
    credits = result.get('credits', 'N/A')
    error = result.get('error')
    
    print(f"\n[{index}/{total}] {name}")
    print("-" * 70)
    
    # Status
    if working:
        status_icon = "✓"
        status_text = "WORKING"
    elif configured:
        status_icon = "⚠"
        status_text = "CONFIGURED BUT NOT WORKING"
    else:
        status_icon = "✗"
        status_text = "NOT CONFIGURED"
    
    print(f"{status_icon} Status: {status_text}")
    
    # Configuration
    if configured:
        key_preview = result.get('details', {}).get('key_preview', 'N/A')
        print(f"  Key: {key_preview}")
    else:
        print(f"  Key: Not set")
    
    # Working status
    if working:
        print(f"✓ API Test: SUCCESS")
        if credits:
            print(f"  Credits/Quota: {credits}")
        
        # Additional details
        details = result.get('details', {})
        if 'username' in details:
            print(f"  User: {details['username']}")
        if 'server_type' in details:
            print(f"  Server Type: {details['server_type']}")
        if 'available_models' in details:
            models = details['available_models']
            if len(models) == 1:
                print(f"  Model: {models[0]}")
            else:
                print(f"  Models: {', '.join(models[:3])}{'...' if len(models) > 3 else ''}")
    else:
        if error:
            print(f"✗ Error: {error}")
        else:
            print(f"✗ API Test: FAILED")
    
    # Optional status
    if not configured and not result.get('required', True):
        print("  Note: This API is optional")


def main():
    """Main validation function"""
    print("=" * 70)
    print("INFOSIGHT 3.0 - COMPREHENSIVE API VALIDATION")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # List of all API tests
    api_tests = [
        ("Local LLM", test_local_llm),
        ("Groq API", test_groq_api),
        ("Hugging Face", test_huggingface_api),
        ("VirusTotal", test_virustotal_api),
        ("IPInfo", test_ipinfo_api),
        ("AbuseIPDB", test_abuseipdb_api),
        ("News API", test_news_api),
        ("OpenWeather", test_weather_api),
        ("SerpAPI", test_serpapi),
    ]
    
    # Run all tests
    total = len(api_tests)
    working_count = 0
    configured_count = 0
    
    for index, (name, test_func) in enumerate(api_tests, 1):
        try:
            result = test_func()
            results[name] = result
            print_result(result, index, total)
            
            if result['working']:
                working_count += 1
            if result['configured']:
                configured_count += 1
        except Exception as e:
            print(f"\n[{index}/{total}] {name}")
            print("-" * 70)
            print(f"✗ Test failed: {type(e).__name__}: {str(e)}")
            results[name] = {'error': str(e), 'working': False, 'configured': False}
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Total APIs tested: {total}")
    print(f"Configured: {configured_count}/{total}")
    print(f"Working: {working_count}/{total}")
    print()
    
    # Critical APIs
    critical_apis = ["Local LLM", "Groq API", "Hugging Face"]
    critical_status = []
    for api_name in critical_apis:
        if api_name in results:
            result = results[api_name]
            if result.get('working'):
                critical_status.append(f"✓ {api_name}")
            elif result.get('configured'):
                critical_status.append(f"⚠ {api_name} (configured but not working)")
            else:
                critical_status.append(f"✗ {api_name} (not configured)")
    
    if critical_status:
        print("Critical APIs:")
        for status in critical_status:
            print(f"  {status}")
        print()
    
    # Recommendations
    if working_count < total:
        print("Recommendations:")
        if not results.get("Groq API", {}).get('working'):
            print("  - Configure GROQ_API_KEY for AI features")
        if not results.get("Hugging Face", {}).get('working'):
            print("  - Configure HF_API_TOKEN for image generation")
        if not results.get("Local LLM", {}).get('working'):
            print("  - Start local LLM server (llama-server.exe)")
    
    print("=" * 70)
    
    # Return exit code
    return 0 if working_count > 0 else 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nValidation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFatal error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
