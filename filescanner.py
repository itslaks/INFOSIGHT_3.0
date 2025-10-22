import os
import hashlib
from flask import Flask, Blueprint, request, render_template, jsonify
from werkzeug.utils import secure_filename
import requests
import time
import logging
from datetime import datetime
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

VIRUSTOTAL_API_KEY = 'cff4224acda1132a9e9398ea3499d63087bd9907df2b293f9e753b1bc186d205'
VIRUSTOTAL_API_URL = 'https://www.virustotal.com/api/v3'
UPLOAD_FOLDER = 'temp'
MAX_FILE_SIZE = 32 * 1024 * 1024  # 32MB limit
ALLOWED_EXTENSIONS = {'exe', 'dll', 'pdf', 'doc', 'docx', 'zip', 'rar', 'apk', 'jpg','jpeg', 'png', 'gif', 'txt', 'bin', 'js', 'html'}

app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

filescanner = Blueprint('filescanner', __name__, template_folder='templates')

# In-memory cache for recent scans
scan_cache = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_hash(file_path, hash_type='sha256'):
    """Calculate file hash"""
    hash_func = hashlib.new(hash_type)
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()

def get_file_info(file_path):
    """Get basic file information"""
    stats = os.stat(file_path)
    return {
        'size': stats.st_size,
        'size_readable': format_file_size(stats.st_size),
        'modified': datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
    }

def format_file_size(size):
    """Format file size in human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"

@filescanner.route('/')
def index():
    return render_template('filescanner.html')

@filescanner.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        temp_dir = os.path.join(os.path.dirname(__file__), 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, filename)
        
        try:
            file.save(file_path)
            
            # Get file info
            file_info = get_file_info(file_path)
            
            # Calculate hashes
            sha256_hash = get_file_hash(file_path, 'sha256')
            md5_hash = get_file_hash(file_path, 'md5')
            
            # Check cache first
            if sha256_hash in scan_cache:
                logger.info(f"Cache hit for file: {filename}")
                os.remove(file_path)
                cached_result = scan_cache[sha256_hash]
                cached_result['cached'] = True
                cached_result['file_info'] = file_info
                cached_result['hashes'] = {'sha256': sha256_hash, 'md5': md5_hash}
                return jsonify(cached_result)
            
            # Check if file already exists on VirusTotal
            existing_result = check_existing_file(sha256_hash)
            if existing_result:
                logger.info(f"File already scanned on VirusTotal: {filename}")
                os.remove(file_path)
                result = process_existing_result(existing_result, file_info, sha256_hash, md5_hash)
                scan_cache[sha256_hash] = result
                return jsonify(result)
            
            # Upload new file
            result = scan_file(file_path, file_info, sha256_hash, md5_hash)
            os.remove(file_path)
            
            # Cache the result
            scan_cache[sha256_hash] = result
            
            return jsonify(result)
            
        except TimeoutError as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            logger.warning(f"Scan timeout for file: {filename}")
            return jsonify({'error': 'Scan is taking longer than expected. Please try again later.'}), 202
        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            logger.error(f"Error during file scan: {str(e)}")
            return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

def check_existing_file(file_hash):
    """Check if file already exists on VirusTotal"""
    url = f'{VIRUSTOTAL_API_URL}/files/{file_hash}'
    headers = {'x-apikey': VIRUSTOTAL_API_KEY}
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
    except requests.RequestException as e:
        logger.error(f"Error checking existing file: {str(e)}")
    
    return None

def process_existing_result(result, file_info, sha256_hash, md5_hash):
    """Process existing VirusTotal result"""
    attrs = result['data']['attributes']
    stats = attrs['last_analysis_stats']
    
    total_scans = sum(stats.values())
    malicious = stats.get('malicious', 0)
    suspicious = stats.get('suspicious', 0)
    undetected = stats.get('undetected', 0)
    
    risk_score = (malicious + suspicious) / total_scans * 100 if total_scans > 0 else 0
    
    # Get vendor details
    vendor_results = []
    if 'last_analysis_results' in attrs:
        for engine, details in attrs['last_analysis_results'].items():
            vendor_results.append({
                'engine': engine,
                'category': details['category'],
                'result': details.get('result', 'Clean')
            })
    
    return {
        'risk_score': round(risk_score, 2),
        'total_scans': total_scans,
        'malicious': malicious,
        'suspicious': suspicious,
        'undetected': undetected,
        'file_info': file_info,
        'hashes': {'sha256': sha256_hash, 'md5': md5_hash},
        'vendor_results': vendor_results,
        'scan_date': attrs.get('last_analysis_date', 'Unknown'),
        'cached': False,
        'instant_result': True
    }

def scan_file(file_path, file_info, sha256_hash, md5_hash):
    """Upload and scan file"""
    url = f'{VIRUSTOTAL_API_URL}/files'
    headers = {'x-apikey': VIRUSTOTAL_API_KEY}
    
    with open(file_path, 'rb') as file:
        files = {'file': (os.path.basename(file_path), file)}
        response = requests.post(url, headers=headers, files=files)
    
    response.raise_for_status()
    upload_result = response.json()
    
    if 'data' not in upload_result or 'id' not in upload_result['data']:
        raise ValueError('Failed to upload file to VirusTotal')

    analysis_id = upload_result['data']['id']
    return get_analysis_result(analysis_id, file_info, sha256_hash, md5_hash)

def get_analysis_result(analysis_id, file_info, sha256_hash, md5_hash):
    """Poll for analysis results"""
    url = f'{VIRUSTOTAL_API_URL}/analyses/{analysis_id}'
    headers = {'x-apikey': VIRUSTOTAL_API_KEY}
    
    max_attempts = 30
    wait_time = 5

    for attempt in range(max_attempts):
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            result = response.json()
            
            status = result['data']['attributes']['status']
            
            if status == 'completed':
                return process_analysis_result(result, file_info, sha256_hash, md5_hash)
            
            logger.info(f"Scan in progress. Attempt {attempt + 1}/{max_attempts}")
            time.sleep(wait_time)
        except requests.RequestException as e:
            logger.error(f"Error getting analysis result (attempt {attempt + 1}): {str(e)}")
    
    raise TimeoutError('Analysis timed out')

def process_analysis_result(result, file_info, sha256_hash, md5_hash):
    """Process completed analysis result"""
    attrs = result['data']['attributes']
    stats = attrs['stats']
    
    total_scans = sum(stats.values())
    malicious = stats.get('malicious', 0)
    suspicious = stats.get('suspicious', 0)
    undetected = stats.get('undetected', 0)
    
    risk_score = (malicious + suspicious) / total_scans * 100 if total_scans > 0 else 0
    
    # Get vendor results
    vendor_results = []
    if 'results' in attrs:
        for engine, details in attrs['results'].items():
            vendor_results.append({
                'engine': engine,
                'category': details['category'],
                'result': details.get('result', 'Clean')
            })
    
    return {
        'risk_score': round(risk_score, 2),
        'total_scans': total_scans,
        'malicious': malicious,
        'suspicious': suspicious,
        'undetected': undetected,
        'file_info': file_info,
        'hashes': {'sha256': sha256_hash, 'md5': md5_hash},
        'vendor_results': vendor_results,
        'cached': False,
        'instant_result': False
    }

@filescanner.route('/hash-check', methods=['POST'])
def hash_check():
    """Check a file by hash without uploading"""
    data = request.get_json()
    file_hash = data.get('hash', '').strip()
    
    if not file_hash:
        return jsonify({'error': 'No hash provided'}), 400
    
    result = check_existing_file(file_hash)
    if result:
        processed_result = process_existing_result(result, {}, file_hash, '')
        return jsonify(processed_result)
    else:
        return jsonify({'error': 'File not found in VirusTotal database'}), 404

# Register the blueprint
app.register_blueprint(filescanner, url_prefix='/filescanner')

if __name__ == '__main__':
    # Ensure temp directory exists
    temp_dir = os.path.join(os.path.dirname(__file__), 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    app.run(debug=True, threaded=True)