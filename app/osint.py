from flask import Flask, request, jsonify, Blueprint, render_template, g
from flask_cors import CORS
import json
import requests
import re
from datetime import datetime
from collections import OrderedDict
import concurrent.futures
import time
import os
import logging
from utils.security import rate_limit_api, validate_request, InputValidator

# Configure logging
logger = logging.getLogger(__name__)

# Create Blueprint for OSINT
osint = Blueprint('osint', __name__, template_folder='templates')

# Log application initialization
logger.info("=" * 70)
logger.info("🔍 OSINT - Initializing")
logger.info("=" * 70)

# Load JSON data
def load_data():
    try:
        # Path relative to project root
        try:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from utils.paths import get_data_path
            possible_paths = [
                str(get_data_path('data.json')),
                os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data.json'),  # Fallback
                'data.json',  # Current directory fallback
            ]
        except ImportError:
            project_root = os.path.dirname(os.path.dirname(__file__))
            possible_paths = [
                os.path.join(project_root, 'data', 'data.json'),
                os.path.join(project_root, 'data.json'),  # Fallback
                'data.json',  # Current directory fallback
            ]
        
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"✅ Loading data.json from: {path}")
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Validate data structure
                    if not isinstance(data, dict):
                        logger.warning("⚠️ WARNING: data.json is not a dictionary. Using sample data...")
                        return get_sample_data()
                    
                    # Filter out invalid entries
                    valid_data = {}
                    for key, value in data.items():
                        if isinstance(value, dict) and 'url' in value:
                            valid_data[key] = value
                        else:
                            logger.debug(f"Skipping invalid entry: {key}")
                    
                    if len(valid_data) == 0:
                        logger.warning("⚠️ WARNING: No valid platforms in data.json. Using sample data...")
                        return get_sample_data()
                    
                    logger.info(f"✅ Loaded {len(valid_data)} platforms from data.json")
                    return valid_data
        
        logger.warning("⚠️ WARNING: data.json not found. Using sample data...")
        return get_sample_data()
    except json.JSONDecodeError as e:
        logger.error(f"❌ Error parsing data.json: {e}. Using sample data...")
        return get_sample_data()
    except Exception as e:
        logger.error(f"❌ Error loading data.json: {e}. Using sample data...")
        return get_sample_data()

def get_sample_data():
    """Fallback sample data with more platforms"""
    return {
        "Instagram": {
            "url": "https://instagram.com/{}",
            "urlMain": "https://instagram.com",
            "errorType": "status_code",
            "regexCheck": "^[a-zA-Z0-9._]{1,30}$"
        },
        "GitHub": {
            "url": "https://github.com/{}",
            "urlMain": "https://github.com",
            "errorType": "status_code",
            "regexCheck": "^[a-zA-Z0-9](?:[a-zA-Z0-9]|-(?=[a-zA-Z0-9])){0,38}$"
        },
        "Twitter": {
            "url": "https://x.com/{}",
            "urlMain": "https://x.com",
            "errorType": "status_code",
            "regexCheck": "^[a-zA-Z0-9_]{1,15}$"
        },
        "LinkedIn": {
            "url": "https://linkedin.com/in/{}",
            "urlMain": "https://linkedin.com",
            "errorType": "status_code"
        },
        "Facebook": {
            "url": "https://facebook.com/{}",
            "urlMain": "https://facebook.com",
            "errorType": "status_code"
        },
        "YouTube": {
            "url": "https://youtube.com/@{}",
            "urlMain": "https://youtube.com",
            "errorType": "status_code"
        },
        "Reddit": {
            "url": "https://reddit.com/user/{}",
            "urlMain": "https://reddit.com",
            "errorType": "status_code",
            "regexCheck": "^[a-zA-Z0-9_-]{3,20}$"
        },
        "TikTok": {
            "url": "https://tiktok.com/@{}",
            "urlMain": "https://tiktok.com",
            "errorType": "status_code"
        },
        "Twitch": {
            "url": "https://twitch.tv/{}",
            "urlMain": "https://twitch.tv",
            "errorType": "status_code"
        },
        "Pinterest": {
            "url": "https://pinterest.com/{}",
            "urlMain": "https://pinterest.com",
            "errorType": "status_code"
        }
    }

user_data = load_data()

# Popular platforms for priority display
POPULAR_PLATFORMS = [
    'Instagram', 'Twitter', 'Facebook', 'LinkedIn', 'GitHub', 
    'TikTok', 'YouTube', 'Reddit', 'Twitch', 'Discord',
    'Snapchat', 'Pinterest', 'Telegram'
]

# Platform categories
PLATFORM_CATEGORIES = {
    'social': ['Instagram', 'Twitter', 'Facebook', 'TikTok', 'Snapchat', 'Reddit', 'Pinterest', 'Tumblr', '9GAG', 'Telegram'],
    'professional': ['LinkedIn', 'AngelList', 'Behance', 'Dribbble', 'About.me', 'Medium'],
    'developer': ['GitHub', 'GitLab', 'Stack Overflow', 'LeetCode', 'HackerRank', 'Codepen', 'BitBucket', 'Replit'],
    'gaming': ['Steam Community', 'Xbox Gamertag', 'Twitch', 'Discord', 'Roblox', 'Minecraft'],
    'media': ['YouTube', 'Vimeo', 'SoundCloud', 'Spotify', 'Bandcamp', 'Flickr', 'Dailymotion'],
}

# Search history storage
search_history = []

def get_platform_category(platform_name):
    """Determine platform category"""
    for category, platforms in PLATFORM_CATEGORIES.items():
        if platform_name in platforms:
            return category
    return 'other'

def validate_username(username, regex_check):
    """Validate username against regex pattern"""
    if not regex_check:
        return True
    try:
        pattern = re.compile(regex_check)
        return bool(pattern.match(username))
    except:
        return True

def check_url_exists(url, site_data):
    """Check if URL exists with proper error handling"""
    try:
        headers = site_data.get('headers', {})
        if not headers:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
        
        response = requests.get(url, headers=headers, timeout=8, allow_redirects=True)
        
        # Consider both 200 and 403 as existing (403 means it exists but is protected)
        if response.status_code in [200, 403]:
            return True
        elif response.status_code == 404:
            return False
        else:
            return None
            
    except requests.exceptions.Timeout:
        return None
    except requests.exceptions.RequestException:
        return None
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error checking URL {url}: {e}")
        return None

@osint.route('/')
def index():
    return render_template("osint.html")

@osint.route('/api/search', methods=['POST'])
@rate_limit_api(requests_per_minute=20, requests_per_hour=200)
@validate_request({
    "username": {
        "type": "string",
        "required": True,
        "max_length": 100
    },
    "options": {
        "type": "dict",
        "required": False,
        "nested_schema": {
            "caseInsensitive": {"type": "bool", "required": False},
            "validateUrls": {"type": "bool", "required": False},
            "popularFirst": {"type": "bool", "required": False}
        }
    }
}, strict=True)
def search():
    """
    Enhanced search with validation and categorization
    OWASP: Rate limited, input validated, schema-based validation
    """
    try:
        # Get validated data from request context
        data = g.validated_data
        username = InputValidator.validate_string(
            data.get('username'), 'username', max_length=100, required=True
        )
        options = data.get('options', {})
        
        # Options
        case_insensitive = options.get('caseInsensitive', True)
        validate_urls = options.get('validateUrls', False)
        popular_first = options.get('popularFirst', True)
        
        # Adjust username case
        search_username = username.lower() if case_insensitive else username
        
        results = OrderedDict()
        popular_results = OrderedDict()
        other_results = OrderedDict()
        
        start_time = time.time()
        
        # Process platforms
        for site, site_data in user_data.items():
            if 'url' not in site_data:
                continue
            
            # Validate username format
            regex_check = site_data.get('regexCheck', '')
            if regex_check and not validate_username(search_username, regex_check):
                continue
            
            try:
                url = site_data['url'].format(search_username)
                
                platform_info = {
                    'url': url,
                    'urlMain': site_data.get('urlMain', ''),
                    'category': get_platform_category(site),
                    'errorType': site_data.get('errorType', 'unknown'),
                    'validated': False,
                    'validationStatus': 'pending'
                }
                
                # Categorize results
                if site in POPULAR_PLATFORMS:
                    popular_results[site] = platform_info
                else:
                    other_results[site] = platform_info
                    
            except Exception as e:
                logger.error(f"Error processing {site}: {e}")
                continue
        
        # Combine results based on popular_first option
        if popular_first:
            results = {**popular_results, **other_results}
        else:
            # Alphabetical order
            all_platforms = {**popular_results, **other_results}
            results = OrderedDict(sorted(all_platforms.items()))
        
        search_time = round(time.time() - start_time, 2)
        
        # Save to search history
        search_entry = {
            'username': username,
            'timestamp': datetime.now().isoformat(),
            'resultsCount': len(results),
            'searchTime': search_time
        }
        search_history.insert(0, search_entry)
        if len(search_history) > 50:
            search_history.pop()
        
        return jsonify({
            'success': True,
            'username': username,
            'results': results,
            'stats': {
                'totalPlatforms': len(results),
                'searchTime': search_time,
                'timestamp': datetime.now().isoformat()
            }
        })
    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        return jsonify({"error": str(e), "success": False}), 500

@osint.route('/api/validate-url', methods=['POST'])
@rate_limit_api(requests_per_minute=30, requests_per_hour=300)
@validate_request({
    "url": {
        "type": "string",
        "required": True,
        "max_length": 2048
    },
    "platform": {
        "type": "string",
        "required": False,
        "max_length": 100
    }
}, strict=True)
def validate_url():
    """
    Validate a single URL
    OWASP: Rate limited, input validated
    """
    try:
        # Get validated data from request context
        data = g.validated_data
        url = InputValidator.validate_string(
            data.get('url'), 'url', max_length=2048, required=True
        )
        platform = data.get('platform', '')
        
        site_data = user_data.get(platform, {})
        exists = check_url_exists(url, site_data)
        
        return jsonify({
            'success': True,
            'url': url,
            'exists': exists,
            'status': 'verified' if exists else 'not_found' if exists is False else 'unknown'
        })
    except Exception as e:
        logger.error(f"Validation error: {e}", exc_info=True)
        return jsonify({"error": str(e), "success": False}), 500

@osint.route('/api/batch-validate', methods=['POST'])
@rate_limit_api(requests_per_minute=10, requests_per_hour=100)
@validate_request({
    "urls": {
        "type": "list",
        "required": True,
        "max_items": 50,
        "item_schema": {
            "type": "string",
            "max_length": 2048
        }
    }
}, strict=True)
def batch_validate():
    """
    Validate multiple URLs concurrently
    OWASP: Rate limited, input validated
    """
    try:
        # Get validated data from request context
        data = g.validated_data
        urls = data.get('urls', [])
        
        results = {}
        
        def validate_single(item):
            url = item['url']
            platform = item['platform']
            site_data = user_data.get(platform, {})
            exists = check_url_exists(url, site_data)
            return platform, exists
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(validate_single, item) for item in urls]
            for future in concurrent.futures.as_completed(futures):
                try:
                    platform, exists = future.result()
                    results[platform] = {
                        'exists': exists,
                        'status': 'verified' if exists else 'not_found' if exists is False else 'unknown'
                    }
                except Exception as e:
                    logger.error(f"Error in batch validation: {e}")
                    continue
        
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        logger.error(f"Batch validation error: {e}", exc_info=True)
        return jsonify({"error": str(e), "success": False}), 500

@osint.route('/api/history', methods=['GET'])
def get_history():
    """Get search history"""
    return jsonify({
        'success': True,
        'history': search_history[:20]
    })

@osint.route('/api/platforms', methods=['GET'])
def get_platforms():
    """Get all available platforms with metadata"""
    try:
        platforms = []
        
        for site, site_data in user_data.items():
            # Ensure site_data is a dictionary
            if not isinstance(site_data, dict):
                logger.warning(f"Warning: {site} has invalid data type: {type(site_data)}")
                continue
                
            platforms.append({
                'name': site,
                'category': get_platform_category(site),
                'urlMain': site_data.get('urlMain', ''),
                'hasRegex': bool(site_data.get('regexCheck')),
                'isNSFW': site_data.get('isNSFW', False)
            })
        
        return jsonify({
            'success': True,
            'platforms': sorted(platforms, key=lambda x: x['name']),
            'totalCount': len(platforms),
            'categories': list(PLATFORM_CATEGORIES.keys())
        })
    except Exception as e:
        logger.error(f"Platforms error: {e}", exc_info=True)
        return jsonify({"error": str(e), "success": False}), 500

@osint.route('/api/stats', methods=['GET'])
def get_stats():
    """Get application statistics"""
    try:
        return jsonify({
            'success': True,
            'totalPlatforms': len(user_data),
            'totalSearches': len(search_history),
            'categories': {cat: len(plat) for cat, plat in PLATFORM_CATEGORIES.items()},
            'popularPlatforms': POPULAR_PLATFORMS
        })
    except Exception as e:
        logger.error(f"Stats error: {e}", exc_info=True)
        return jsonify({"error": str(e), "success": False}), 500

@osint.route('/api/compare', methods=['POST'])
@rate_limit_api(requests_per_minute=10, requests_per_hour=100)
@validate_request({
    "usernames": {
        "type": "list",
        "required": True,
        "min_items": 2,
        "max_items": 10,
        "item_schema": {
            "type": "string",
            "max_length": 100
        }
    }
}, strict=True)
def compare_users():
    """
    Compare multiple usernames across platforms
    OWASP: Rate limited, input validated
    """
    try:
        # Get validated data from request context
        data = g.validated_data
        usernames = data.get('usernames', [])
        
        comparison = {}
        
        for username in usernames:
            comparison[username] = {}
            for site, site_data in user_data.items():
                if 'url' in site_data:
                    try:
                        url = site_data['url'].format(username)
                        comparison[username][site] = url
                    except:
                        continue
        
        return jsonify({
            'success': True,
            'comparison': comparison,
            'usernames': usernames
        })
    except Exception as e:
        logger.error(f"Compare error: {e}", exc_info=True)
        return jsonify({"error": str(e), "success": False}), 500

@osint.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'platforms_loaded': len(user_data),
        'timestamp': datetime.now().isoformat()
    })

# Log status when module is imported
logger.info(f"OSINT Blueprint loaded with {len(user_data)} platforms")
