"""
Configuration module for INFOSIGHT 3.0
Loads environment variables and provides configuration settings
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration"""
    
    # Flask settings
    ENV = os.getenv('FLASK_ENV', 'development')
    DEBUG = os.getenv('FLASK_ENV', 'development') == 'development'
    
    # Server settings
    HOST = os.getenv('SERVER_HOST', '127.0.0.1')
    PORT = int(os.getenv('SERVER_PORT', '5000'))
    
    # Groq API
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    
    # Gemini API (deprecated - kept for backward compatibility)
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    
    # Hugging Face
    HF_API_TOKEN = os.getenv('HF_API_TOKEN')
    
    # Security APIs
    VIRUSTOTAL_API_KEY = os.getenv('VIRUSTOTAL_API_KEY')
    IPINFO_API_KEY = os.getenv('IPINFO_API_KEY')
    ABUSEIPDB_API_KEY = os.getenv('ABUSEIPDB_API_KEY')
    
    # News & Weather APIs
    NEWS_API_KEY = os.getenv('NEWS_API_KEY')
    WEATHER_API_KEY = os.getenv('WEATHER_API_KEY')
    
    # Search APIs
    SERPAPI_KEY = os.getenv('SERPAPI_KEY')
    
    # Local LLM Configuration
    OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'qwen2.5-coder:3b-instruct')
    
    @classmethod
    def validate_required(cls, required_keys=None):
        """Validate that required configuration keys are set"""
        if required_keys is None:
            required_keys = []
        
        missing = []
        for key in required_keys:
            value = getattr(cls, key, None)
            if not value:
                missing.append(key)
        
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}. "
                f"Please set them in .env file or environment variables."
            )
        
        return True
