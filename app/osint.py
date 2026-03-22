import sys
import os
import re
import json
import time
import logging
import concurrent.futures
from pathlib import Path
from datetime import datetime
from collections import OrderedDict

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import (
    Flask, request, jsonify, Blueprint,
    render_template, g
)
from flask_cors import CORS
import requests as http_requests

# ── Conditional project imports ──────────────────────────────────────────────
try:
    from utils.security import rate_limit_api, validate_request, InputValidator
    _HAS_SECURITY = True
except ImportError:
    _HAS_SECURITY = False
    def rate_limit_api(*a, **kw):
        def dec(f): return f
        return dec
    def validate_request(*a, **kw):
        def dec(f): return f
        return dec
    class InputValidator:
        @staticmethod
        def validate_string(v, *a, **kw): return v

try:
    from utils.paths import get_data_path
    _HAS_PATHS = True
except ImportError:
    _HAS_PATHS = False

# ── Logging ──────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Blueprint ────────────────────────────────────────────────────────────────
osint = Blueprint('osint', __name__, template_folder='templates')

logger.info("=" * 70)
logger.info("🔍 OSINT - Initializing")
logger.info("=" * 70)

# ═════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

def _candidate_paths():
    base = Path(__file__).resolve().parent.parent
    paths = [
        base / 'data' / 'data.json',
        base / 'data.json',
        Path('data/data.json'),
        Path('data.json'),
    ]
    if _HAS_PATHS:
        try:
            paths.insert(0, Path(get_data_path('data.json')))
        except Exception:
            pass
    return paths


def load_data():
    for p in _candidate_paths():
        if p.exists():
            logger.info(f"✅ Loading data.json from: {p}")
            try:
                raw = json.loads(p.read_text(encoding='utf-8'))
                if not isinstance(raw, dict):
                    logger.warning("⚠️  data.json is not a dict – using fallback")
                    break
                # Skip the $schema meta key, keep only platform entries
                valid = {
                    k: v for k, v in raw.items()
                    if isinstance(v, dict) and 'url' in v and not k.startswith('$')
                }
                if valid:
                    logger.info(f"✅ Loaded {len(valid)} platforms from data.json")
                    return valid
            except (json.JSONDecodeError, OSError) as exc:
                logger.error(f"❌ Failed to read data.json: {exc}")
                break

    logger.warning("⚠️  data.json not found or invalid – using built-in fallback")
    return _builtin_platforms()


def _builtin_platforms():
    """Minimal fallback used only when data.json is unavailable."""
    return {
        "Instagram":    {"url": "https://instagram.com/{}",             "urlMain": "https://instagram.com",     "errorType": "status_code"},
        "Twitter/X":    {"url": "https://x.com/{}",                     "urlMain": "https://x.com",             "errorType": "status_code"},
        "GitHub":       {"url": "https://github.com/{}",                "urlMain": "https://github.com",        "errorType": "status_code"},
        "LinkedIn":     {"url": "https://linkedin.com/in/{}",           "urlMain": "https://linkedin.com",      "errorType": "status_code"},
        "Reddit":       {"url": "https://reddit.com/user/{}",           "urlMain": "https://reddit.com",        "errorType": "status_code"},
        "TikTok":       {"url": "https://tiktok.com/@{}",               "urlMain": "https://tiktok.com",        "errorType": "status_code"},
        "YouTube":      {"url": "https://youtube.com/@{}",              "urlMain": "https://youtube.com",       "errorType": "status_code"},
        "Facebook":     {"url": "https://facebook.com/{}",              "urlMain": "https://facebook.com",      "errorType": "status_code"},
        "Pinterest":    {"url": "https://pinterest.com/{}",             "urlMain": "https://pinterest.com",     "errorType": "status_code"},
        "Twitch":       {"url": "https://twitch.tv/{}",                 "urlMain": "https://twitch.tv",         "errorType": "status_code"},
        "Steam":        {"url": "https://steamcommunity.com/id/{}",     "urlMain": "https://steamcommunity.com","errorType": "status_code"},
        "Telegram":     {"url": "https://t.me/{}",                      "urlMain": "https://telegram.org",      "errorType": "status_code"},
        "Medium":       {"url": "https://medium.com/@{}",               "urlMain": "https://medium.com",        "errorType": "status_code"},
        "Behance":      {"url": "https://behance.net/{}",               "urlMain": "https://behance.net",       "errorType": "status_code"},
        "Dribbble":     {"url": "https://dribbble.com/{}",              "urlMain": "https://dribbble.com",      "errorType": "status_code"},
        "SoundCloud":   {"url": "https://soundcloud.com/{}",            "urlMain": "https://soundcloud.com",    "errorType": "status_code"},
        "Vimeo":        {"url": "https://vimeo.com/{}",                 "urlMain": "https://vimeo.com",         "errorType": "status_code"},
        "Spotify":      {"url": "https://open.spotify.com/user/{}",     "urlMain": "https://spotify.com",       "errorType": "status_code"},
        "GitLab":       {"url": "https://gitlab.com/{}",                "urlMain": "https://gitlab.com",        "errorType": "status_code"},
        "Keybase":      {"url": "https://keybase.io/{}",                "urlMain": "https://keybase.io",        "errorType": "status_code"},
        "Patreon":      {"url": "https://patreon.com/{}",               "urlMain": "https://patreon.com",       "errorType": "status_code"},
        "Last.fm":      {"url": "https://last.fm/user/{}",              "urlMain": "https://last.fm",           "errorType": "status_code"},
        "Goodreads":    {"url": "https://goodreads.com/{}",             "urlMain": "https://goodreads.com",     "errorType": "status_code"},
        "ArtStation":   {"url": "https://artstation.com/{}",            "urlMain": "https://artstation.com",    "errorType": "status_code"},
        "Flickr":       {"url": "https://flickr.com/people/{}",         "urlMain": "https://flickr.com",        "errorType": "status_code"},
        "CodePen":      {"url": "https://codepen.io/{}",                "urlMain": "https://codepen.io",        "errorType": "status_code"},
        "Letterboxd":   {"url": "https://letterboxd.com/{}",            "urlMain": "https://letterboxd.com",    "errorType": "status_code"},
        "LeetCode":     {"url": "https://leetcode.com/{}",              "urlMain": "https://leetcode.com",      "errorType": "status_code"},
        "Kaggle":       {"url": "https://kaggle.com/{}",                "urlMain": "https://kaggle.com",        "errorType": "status_code"},
        "ResearchGate": {"url": "https://researchgate.net/profile/{}",  "urlMain": "https://researchgate.net",  "errorType": "status_code"},
        "Fiverr":       {"url": "https://fiverr.com/{}",                "urlMain": "https://fiverr.com",        "errorType": "status_code"},
        "About.me":     {"url": "https://about.me/{}",                  "urlMain": "https://about.me",          "errorType": "status_code"},
        "Wattpad":      {"url": "https://wattpad.com/user/{}",          "urlMain": "https://wattpad.com",       "errorType": "status_code"},
        "Tumblr":       {"url": "https://{}.tumblr.com",                "urlMain": "https://tumblr.com",        "errorType": "status_code"},
        "Mastodon":     {"url": "https://mastodon.social/@{}",          "urlMain": "https://mastodon.social",   "errorType": "status_code"},
        "Bluesky":      {"url": "https://bsky.app/profile/{}",          "urlMain": "https://bsky.app",          "errorType": "status_code"},
        "Dev.to":       {"url": "https://dev.to/{}",                    "urlMain": "https://dev.to",            "errorType": "status_code"},
        "Hashnode":     {"url": "https://hashnode.com/@{}",             "urlMain": "https://hashnode.com",      "errorType": "status_code"},
    }


# Platform category lookup
PLATFORM_CATEGORIES = {
    'social':       ['Instagram','Twitter/X','Facebook','TikTok','Snapchat','Reddit',
                     'Pinterest','Tumblr','Telegram','Mastodon','Bluesky','Threads',
                     'BeReal','Quora','Clubhouse','Signal','WhatsApp','9GAG',
                     'VKontakte','Ask.fm','Taringa'],
    'developer':    ['GitHub','GitLab','Stack Overflow','LeetCode','HackerRank',
                     'CodePen','Replit','HackerNews','Keybase','npm','PyPI',
                     'Docker Hub','Kaggle','Dev.to','Hashnode','Sourcehut','Gitea',
                     'Codeberg','ORCID','SourceForge'],
    'professional': ['LinkedIn','AngelList','Behance','Dribbble','About.me','Medium',
                     'Substack','ProductHunt','Fiverr','Upwork','ResearchGate',
                     'Academia.edu','ORCID','ArtStation','Freelancer','Toptal'],
    'gaming':       ['Steam','Twitch','Discord','Xbox','PlayStation','Roblox','Itch.io',
                     'Minecraft','Battlenet','Origin','Epic Games','GOG'],
    'media':        ['YouTube','Vimeo','SoundCloud','Spotify','Bandcamp','Flickr',
                     'Dailymotion','Rumble','Patreon','Ko-fi','Wattpad',
                     'Goodreads','MyAnimeList','500px','DeviantArt','Last.fm',
                     'Letterboxd','Gravatar','Mixcloud','ReverbNation'],
}

POPULAR_PLATFORMS = [
    'Instagram','Twitter/X','GitHub','LinkedIn','Facebook','TikTok',
    'YouTube','Reddit','Twitch','Discord','Telegram','Pinterest',
    'Steam','Medium','SoundCloud','Spotify',
]

user_data: dict = load_data()
search_history: list = []


def get_platform_category(name: str) -> str:
    for cat, names in PLATFORM_CATEGORIES.items():
        if name in names:
            return cat
    return 'other'


def validate_username_regex(username: str, regex: str) -> bool:
    if not regex:
        return True
    try:
        return bool(re.compile(regex).match(username))
    except re.error:
        return True


# ═════════════════════════════════════════════════════════════════════════════
#  HTTP CHECK ENGINE
#  NOTE: errorMsg in data.json can be a string OR a list of strings
# ═════════════════════════════════════════════════════════════════════════════

_SESSION = http_requests.Session()
_SESSION.headers.update({
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/124.0.0.0 Safari/537.36'
    ),
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'DNT': '1',
})
_SESSION.max_redirects = 5


def _error_msg_matches(body: str, error_msg) -> bool:
    """
    error_msg from data.json can be:
      - a string  → single pattern
      - a list    → ANY match means "not found"
    """
    if not error_msg:
        return False
    if isinstance(error_msg, list):
        return any(m in body for m in error_msg if isinstance(m, str))
    return str(error_msg) in body


def check_url(url: str, site_data: dict, timeout: int = 10) -> dict:
    """
    Perform a real HTTP check on *url*.
    Returns: { exists, status_code, error, method, response_time_ms }
    """
    error_type = site_data.get('errorType', 'status_code')
    error_msg  = site_data.get('errorMsg', '')
    extra_hdrs = site_data.get('headers', {})

    headers = dict(_SESSION.headers)
    if isinstance(extra_hdrs, dict):
        headers.update(extra_hdrs)

    t0 = time.monotonic()
    try:
        resp = _SESSION.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        ms   = int((time.monotonic() - t0) * 1000)

        if error_type == 'status_code':
            if resp.status_code in (200, 403):
                exists = True
            elif resp.status_code in (404, 410):
                exists = False
            else:
                exists = None
            return {'exists': exists, 'status_code': resp.status_code,
                    'error': None, 'method': 'status_code', 'response_time_ms': ms}

        elif error_type == 'message':
            if resp.status_code == 200 and _error_msg_matches(resp.text, error_msg):
                exists = False
            elif resp.status_code == 200:
                exists = True
            else:
                exists = None
            return {'exists': exists, 'status_code': resp.status_code,
                    'error': None, 'method': 'message', 'response_time_ms': ms}

        elif error_type == 'response_url':
            uname  = url.split('/')[-1].split('?')[0].lstrip('@').lower()
            exists = uname in resp.url.lower() if uname else None
            return {'exists': exists, 'status_code': resp.status_code,
                    'error': None, 'method': 'response_url', 'response_time_ms': ms}

        else:
            exists = resp.status_code in (200, 403)
            return {'exists': exists, 'status_code': resp.status_code,
                    'error': None, 'method': 'fallback', 'response_time_ms': ms}

    except http_requests.exceptions.SSLError as exc:
        return {'exists': None, 'status_code': None, 'error': f'SSL error', 'method': error_type, 'response_time_ms': None}
    except http_requests.exceptions.ConnectionError:
        return {'exists': None, 'status_code': None, 'error': 'Connection error', 'method': error_type, 'response_time_ms': None}
    except http_requests.exceptions.Timeout:
        return {'exists': None, 'status_code': None, 'error': 'Timeout', 'method': error_type, 'response_time_ms': None}
    except Exception as exc:
        logger.error(f"check_url({url}) unexpected: {exc}")
        return {'exists': None, 'status_code': None, 'error': str(exc)[:120], 'method': error_type, 'response_time_ms': None}


# ═════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ═════════════════════════════════════════════════════════════════════════════

@osint.route('/')
def index():
    return render_template("osint.html")


# ── /api/platforms ─────────────────────────────────────────────────────────
@osint.route('/api/platforms', methods=['GET'])
def get_platforms():
    """
    Returns full platform list with metadata.
    Frontend loads this on startup — single source of truth from data.json.
    """
    try:
        platforms = []
        for name, data in user_data.items():
            if not isinstance(data, dict):
                continue
            platforms.append({
                'name':      name,
                'url':       data.get('url', ''),
                'urlMain':   data.get('urlMain', ''),
                'category':  get_platform_category(name),
                'errorType': data.get('errorType', 'status_code'),
                'isNSFW':    data.get('isNSFW', False),
                'hasRegex':  bool(data.get('regexCheck')),
            })
        # Popular first, then alphabetical
        platforms.sort(key=lambda x: (x['name'] not in POPULAR_PLATFORMS, x['name'].lower()))
        return jsonify({
            'success': True,
            'platforms': platforms,
            'totalCount': len(platforms),
            'categories': list(PLATFORM_CATEGORIES.keys()),
            'popularPlatforms': POPULAR_PLATFORMS,
        })
    except Exception as exc:
        logger.error(f"get_platforms: {exc}", exc_info=True)
        return jsonify({'success': False, 'error': str(exc)}), 500


# ── /api/search ─────────────────────────────────────────────────────────────
@osint.route('/api/search', methods=['POST'])
@rate_limit_api(requests_per_minute=20, requests_per_hour=200)
def search():
    """
    Build URL list for *username* across all matching platforms.
    Returns structured platform list — does NOT do HTTP checks.
    The frontend then calls /api/batch-check in chunks to get live results.
    """
    try:
        data      = request.get_json(force=True) or {}
        username  = str(data.get('username', '')).strip()[:100]
        if not username:
            return jsonify({'success': False, 'error': 'username required'}), 400

        options      = data.get('options', {}) if isinstance(data.get('options'), dict) else {}
        popular_first = options.get('popularFirst', True)
        filter_nsfw   = options.get('filterNSFW', False)
        scan_type     = options.get('scanType', 'all')

        results = OrderedDict()
        for name, sdata in user_data.items():
            if not isinstance(sdata, dict) or 'url' not in sdata:
                continue
            if filter_nsfw and sdata.get('isNSFW'):
                continue
            cat = get_platform_category(name)
            if scan_type != 'all' and cat != scan_type:
                continue
            regex = sdata.get('regexCheck', '')
            if regex and not validate_username_regex(username, regex):
                continue
            try:
                url = sdata['url'].format(username)
            except (IndexError, KeyError):
                continue
            results[name] = {
                'url':       url,
                'urlMain':   sdata.get('urlMain', ''),
                'category':  cat,
                'errorType': sdata.get('errorType', 'status_code'),
                'isNSFW':    sdata.get('isNSFW', False),
            }

        if popular_first:
            popular = {k: v for k, v in results.items() if k in POPULAR_PLATFORMS}
            other   = {k: v for k, v in results.items() if k not in POPULAR_PLATFORMS}
            results = OrderedDict(**popular, **other)

        search_history.insert(0, {
            'username': username,
            'timestamp': datetime.now().isoformat(),
            'totalPlatforms': len(results),
        })
        del search_history[50:]

        return jsonify({
            'success': True,
            'username': username,
            'platforms': results,
            'stats': {'totalPlatforms': len(results), 'timestamp': datetime.now().isoformat()},
        })
    except Exception as exc:
        logger.error(f"search: {exc}", exc_info=True)
        return jsonify({'success': False, 'error': str(exc)}), 500


# ── /api/validate-url ───────────────────────────────────────────────────────
@osint.route('/api/validate-url', methods=['POST'])
@rate_limit_api(requests_per_minute=60, requests_per_hour=600)
def validate_url():
    """
    Validate a single URL. Used by the frontend in parallel fetch() calls.
    Body: { "url": "...", "platform": "GitHub" }
    """
    try:
        data     = request.get_json(force=True) or {}
        url      = str(data.get('url', '')).strip()
        platform = str(data.get('platform', '')).strip()[:100]
        timeout  = min(int(data.get('timeout', 10)), 20)

        if not url:
            return jsonify({'success': False, 'error': 'url required'}), 400
        if not url.startswith(('http://', 'https://')):
            return jsonify({'success': False, 'error': 'invalid url scheme'}), 400

        site_data = user_data.get(platform, {})
        result    = check_url(url, site_data, timeout=6)

        return jsonify({'success': True, 'url': url, 'platform': platform, **result})
    except Exception as exc:
        logger.error(f"validate_url: {exc}", exc_info=True)
        return jsonify({'success': False, 'error': str(exc)}), 500


# ── /api/batch-check ─────────────────────────────────────────────────────────
@osint.route('/api/batch-check', methods=['POST'])
@rate_limit_api(requests_per_minute=15, requests_per_hour=150)
def batch_check():
    """
    MAIN SCAN ENDPOINT (Waitress-compatible, no SSE/streaming needed).
    The frontend sends batches of 20 platforms; backend checks them all
    concurrently and returns when done. Frontend fires multiple batches
    in parallel to achieve live-streaming feel.

    Body: {
        "items": [
            {"platform": "GitHub", "url": "https://github.com/alice"},
            ...
        ]
    }
    Returns: {
        "success": true,
        "results": {
            "GitHub": { "exists": true, "status_code": 200, "response_time_ms": 312, ... },
            ...
        }
    }
    """
    try:
        data  = request.get_json(force=True) or {}
        items = data.get('items', [])
        if not isinstance(items, list):
            return jsonify({'success': False, 'error': 'items must be a list'}), 400
        items = [i for i in items if isinstance(i, dict)][:30]  # max 30 per batch

        results = {}

        def check_one(item):
            platform = str(item.get('platform', '')).strip()
            url      = str(item.get('url', '')).strip()
            if not url.startswith(('http://', 'https://')):
                return platform, {'exists': None, 'error': 'invalid url', 'status_code': None, 'response_time_ms': None}
            sdata  = user_data.get(platform, {})
            result = check_url(url, sdata, timeout=6)
            return platform, result

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(items), 30)) as pool:
            for platform, result in pool.map(check_one, items):
                if platform:
                    results[platform] = result

        return jsonify({'success': True, 'results': results})
    except Exception as exc:
        logger.error(f"batch_check: {exc}", exc_info=True)
        return jsonify({'success': False, 'error': str(exc)}), 500


# ── /api/history ─────────────────────────────────────────────────────────────
@osint.route('/api/history', methods=['GET'])
def get_history():
    return jsonify({'success': True, 'history': search_history[:20]})


@osint.route('/api/history', methods=['DELETE'])
def clear_history():
    search_history.clear()
    return jsonify({'success': True})


# ── /api/stats ───────────────────────────────────────────────────────────────
@osint.route('/api/stats', methods=['GET'])
def get_stats():
    try:
        return jsonify({
            'success': True,
            'totalPlatforms': len(user_data),
            'totalSearches': len(search_history),
            'categories': {cat: len(names) for cat, names in PLATFORM_CATEGORIES.items()},
            'popularPlatforms': POPULAR_PLATFORMS,
        })
    except Exception as exc:
        logger.error(f"get_stats: {exc}", exc_info=True)
        return jsonify({'success': False, 'error': str(exc)}), 500


# ── /api/compare ─────────────────────────────────────────────────────────────
@osint.route('/api/compare', methods=['POST'])
@rate_limit_api(requests_per_minute=10, requests_per_hour=100)
def compare_users():
    """Return profile URLs for 2–5 usernames (no HTTP checks)."""
    try:
        data      = request.get_json(force=True) or {}
        usernames = data.get('usernames', [])
        if not isinstance(usernames, list) or len(usernames) < 2:
            return jsonify({'success': False, 'error': 'Need at least 2 usernames'}), 400
        usernames = [str(u).strip()[:100] for u in usernames[:5]]

        comparison = {}
        for uname in usernames:
            comparison[uname] = {}
            for name, sdata in user_data.items():
                if not isinstance(sdata, dict) or 'url' not in sdata:
                    continue
                try:
                    comparison[uname][name] = {
                        'url':      sdata['url'].format(uname),
                        'urlMain':  sdata.get('urlMain', ''),
                        'category': get_platform_category(name),
                    }
                except (IndexError, KeyError):
                    continue

        return jsonify({'success': True, 'comparison': comparison, 'usernames': usernames})
    except Exception as exc:
        logger.error(f"compare_users: {exc}", exc_info=True)
        return jsonify({'success': False, 'error': str(exc)}), 500


# ── /health ───────────────────────────────────────────────────────────────────
@osint.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'platforms_loaded': len(user_data),
        'timestamp': datetime.now().isoformat(),
    })


logger.info(f"OSINT Blueprint loaded — {len(user_data)} platforms ready")


# ── Standalone dev runner ─────────────────────────────────────────────────────
if __name__ == "__main__":
    from flask import Flask
    app = Flask(__name__, template_folder='templates')
    app.config.setdefault('SECRET_KEY', os.getenv('FLASK_SECRET_KEY', 'dev-key-for-standalone-mode'))
    app.register_blueprint(osint)
    host = os.getenv('APP_HOST','127.0.0.1')
    port = int(os.getenv('APP_PORT', '5009'))
    print(f'Starting osint standalone mode on {host}:{port}...')
    app.run(debug=True, host=host, port=port, threaded=True)
