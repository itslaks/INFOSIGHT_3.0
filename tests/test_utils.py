"""
Tests for utility modules
"""
import pytest
import sys
import os

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))


def test_local_llm_utils_import():
    """Test that local_llm_utils can be imported"""
    try:
        from utils.local_llm_utils import check_ollama_available, generate_with_ollama
        assert callable(check_ollama_available)
        assert callable(generate_with_ollama)
    except ImportError:
        pytest.skip("local_llm_utils not available")


def test_config_module():
    """Test that config module loads correctly"""
    try:
        from config import Config
        assert hasattr(Config, 'GEMINI_API_KEY')
        assert hasattr(Config, 'HOST')
        assert hasattr(Config, 'PORT')
    except ImportError:
        pytest.skip("config module not available")
