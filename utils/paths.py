"""
Utility functions for path management
"""
import os
from pathlib import Path


def get_project_root():
    """Get the project root directory"""
    # This file is in utils/, so go up one level
    return Path(__file__).parent.parent


def get_data_path(filename):
    """Get path to a file in the data directory"""
    return get_project_root() / 'data' / filename


def get_model_path(filename):
    """Get path to a file in the models directory"""
    return get_project_root() / 'models' / filename


def get_config_path(filename):
    """Get path to a file in the config directory"""
    return get_project_root() / 'config' / filename
