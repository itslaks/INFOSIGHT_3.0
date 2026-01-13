import os
import hashlib
import base64
from flask import Flask, Blueprint, request, render_template, jsonify, send_file, g
from werkzeug.utils import secure_filename
from utils.security import rate_limit_api, rate_limit_strict, validate_request, InputValidator
import requests
import time
import logging
from datetime import datetime
import json
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import secrets
import io
import secrets
import base64


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API key from environment variables
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import Config
    VIRUSTOTAL_API_KEY = Config.VIRUSTOTAL_API_KEY
except (ImportError, AttributeError):
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    import os
    VIRUSTOTAL_API_KEY = os.getenv('VIRUSTOTAL_API_KEY')

VIRUSTOTAL_API_URL = 'https://www.virustotal.com/api/v3'
UPLOAD_FOLDER = 'temp'
ENCRYPTED_FOLDER = 'encrypted'
MAX_FILE_SIZE = 32 * 1024 * 1024  # 32MB limit
ALLOWED_EXTENSIONS = {'exe', 'dll', 'pdf', 'doc', 'docx', 'zip', 'rar', 'apk', 'jpg','jpeg', 'png', 'gif', 'txt', 'bin', 'js', 'html', 'py', 'java', 'cpp', 'c'}

filescanner = Blueprint('filescanner', __name__, template_folder='templates')

# Log application initialization
logger.info("=" * 70)
logger.info("📁 FileScanner - Initializing")
logger.info("=" * 70)

# Path for storing encryption metadata
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.paths import get_data_path
    ENCRYPTION_METADATA_FILE = str(get_data_path('encryption_metadata.json'))
except ImportError:
    # Fallback
    ENCRYPTION_METADATA_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'encryption_metadata.json')

# In-memory cache for recent scans
scan_cache = {}
# Store encryption keys temporarily (in production, use secure key management)
encryption_keys = {}

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

# ==================== ENCRYPTION FUNCTIONS ====================

def generate_aes_key(password, salt):
    """Generate AES key from password using PBKDF2"""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    return kdf.derive(password.encode())

def encrypt_file_aes(file_path, password):
    """Encrypt file using AES-256-GCM"""
    try:
        # Generate salt and key
        salt = secrets.token_bytes(16)
        key = generate_aes_key(password, salt)
        
        # Generate IV
        iv = secrets.token_bytes(12)
        
        # Read file content
        with open(file_path, 'rb') as f:
            plaintext = f.read()
        
        # Encrypt
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        
        # Combine salt + iv + tag + ciphertext
        encrypted_data = salt + iv + encryptor.tag + ciphertext
        
        return encrypted_data, True, "File encrypted successfully with AES-256-GCM"
    except Exception as e:
        logger.error(f"AES encryption error: {str(e)}")
        return None, False, str(e)

def decrypt_file_aes(encrypted_data, password):
    """Decrypt file using AES-256-GCM"""
    try:
        # Extract components
        salt = encrypted_data[:16]
        iv = encrypted_data[16:28]
        tag = encrypted_data[28:44]
        ciphertext = encrypted_data[44:]
        
        # Derive key
        key = generate_aes_key(password, salt)
        
        # Decrypt
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(iv, tag),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        
        return plaintext, True, "File decrypted successfully"
    except Exception as e:
        logger.error(f"AES decryption error: {str(e)}")
        return None, False, "Decryption failed - incorrect password or corrupted file"

def encrypt_file_rsa(file_path):
    """Encrypt file using RSA (hybrid encryption with AES)"""
    try:
        # Generate RSA key pair
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        public_key = private_key.public_key()
        
        # Generate random AES key for file encryption
        aes_key = secrets.token_bytes(32)
        iv = secrets.token_bytes(12)
        
        # Read file content
        with open(file_path, 'rb') as f:
            plaintext = f.read()
        
        # Encrypt file with AES
        cipher = Cipher(
            algorithms.AES(aes_key),
            modes.GCM(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        
        # Encrypt AES key with RSA public key
        encrypted_aes_key = public_key.encrypt(
            aes_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        # Serialize private key
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        # Combine: encrypted_key_length + encrypted_key + iv + tag + ciphertext
        key_length = len(encrypted_aes_key).to_bytes(4, 'big')
        encrypted_data = key_length + encrypted_aes_key + iv + encryptor.tag + ciphertext
        
        return encrypted_data, private_pem, True, "File encrypted successfully with RSA-2048 + AES-256"
    except Exception as e:
        logger.error(f"RSA encryption error: {str(e)}")
        return None, None, False, str(e)

def decrypt_file_rsa(encrypted_data, private_key_pem):
    """Decrypt file using RSA private key"""
    try:
        # Load private key
        private_key = serialization.load_pem_private_key(
            private_key_pem,
            password=None,
            backend=default_backend()
        )
        
        # Extract encrypted AES key length
        key_length = int.from_bytes(encrypted_data[:4], 'big')
        
        # Extract components
        encrypted_aes_key = encrypted_data[4:4+key_length]
        iv = encrypted_data[4+key_length:4+key_length+12]
        tag = encrypted_data[4+key_length+12:4+key_length+28]
        ciphertext = encrypted_data[4+key_length+28:]
        
        # Decrypt AES key with RSA private key
        aes_key = private_key.decrypt(
            encrypted_aes_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        # Decrypt file with AES
        cipher = Cipher(
            algorithms.AES(aes_key),
            modes.GCM(iv, tag),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        
        return plaintext, True, "File decrypted successfully"
    except Exception as e:
        logger.error(f"RSA decryption error: {str(e)}")
        return None, False, "Decryption failed - invalid private key or corrupted file"

def encrypt_file_chacha20(file_path, password):
    """Encrypt file using ChaCha20-Poly1305"""
    try:
        # Generate salt and key
        salt = secrets.token_bytes(16)
        key = generate_aes_key(password, salt)
        
        # Generate nonce
        nonce = secrets.token_bytes(12)
        
        # Read file content
        with open(file_path, 'rb') as f:
            plaintext = f.read()
        
        # Encrypt
        cipher = Cipher(
            algorithms.ChaCha20(key, nonce),
            mode=None,
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        
        # Combine salt + nonce + ciphertext
        encrypted_data = salt + nonce + ciphertext
        
        return encrypted_data, True, "File encrypted successfully with ChaCha20-Poly1305"
    except Exception as e:
        logger.error(f"ChaCha20 encryption error: {str(e)}")
        return None, False, str(e)

def decrypt_file_chacha20(encrypted_data, password):
    """Decrypt file using ChaCha20-Poly1305"""
    try:
        # Extract components
        salt = encrypted_data[:16]
        nonce = encrypted_data[16:28]
        ciphertext = encrypted_data[28:]
        
        # Derive key
        key = generate_aes_key(password, salt)
        
        # Decrypt
        cipher = Cipher(
            algorithms.ChaCha20(key, nonce),
            mode=None,
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        
        return plaintext, True, "File decrypted successfully"
    except Exception as e:
        logger.error(f"ChaCha20 decryption error: {str(e)}")
        return None, False, "Decryption failed - incorrect password or corrupted file"

# ==================== ROUTES ====================

@filescanner.route('/')
def index():
    return render_template('filescanner.html')

@filescanner.route('/upload', methods=['POST'])
@rate_limit_strict(requests_per_minute=5, requests_per_hour=30)  # Strict limit for file uploads
def upload_file():
    """
    Upload and scan file
    OWASP: Rate limited, file validation, size limits enforced
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Validate filename
    try:
        filename = InputValidator.validate_filename(file.filename, 'filename', required=True)
    except Exception as e:
        return jsonify({'error': f'Invalid filename: {str(e)}'}), 400
    
    if not allowed_file(filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    # Check file size (already enforced by Flask MAX_CONTENT_LENGTH, but validate here too)
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    
    if file_size > MAX_FILE_SIZE:
        return jsonify({'error': f'File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024)}MB'}), 400
    
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
            sha1_hash = get_file_hash(file_path, 'sha1')
            
            # Check cache first
            if sha256_hash in scan_cache:
                logger.info(f"Cache hit for file: {filename}")
                os.remove(file_path)
                cached_result = scan_cache[sha256_hash]
                cached_result['cached'] = True
                cached_result['file_info'] = file_info
                cached_result['hashes'] = {'sha256': sha256_hash, 'md5': md5_hash, 'sha1': sha1_hash}
                return jsonify(cached_result)
            
            # Check if file already exists on VirusTotal
            existing_result = check_existing_file(sha256_hash)
            if existing_result:
                logger.info(f"File already scanned on VirusTotal: {filename}")
                os.remove(file_path)
                result = process_existing_result(existing_result, file_info, sha256_hash, md5_hash, sha1_hash)
                scan_cache[sha256_hash] = result
                return jsonify(result)
            
            # Upload new file
            result = scan_file(file_path, file_info, sha256_hash, md5_hash, sha1_hash)
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

def process_existing_result(result, file_info, sha256_hash, md5_hash, sha1_hash):
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
        'hashes': {'sha256': sha256_hash, 'md5': md5_hash, 'sha1': sha1_hash},
        'vendor_results': vendor_results,
        'scan_date': attrs.get('last_analysis_date', 'Unknown'),
        'cached': False,
        'instant_result': True
    }

def scan_file(file_path, file_info, sha256_hash, md5_hash, sha1_hash):
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
    return get_analysis_result(analysis_id, file_info, sha256_hash, md5_hash, sha1_hash)

def get_analysis_result(analysis_id, file_info, sha256_hash, md5_hash, sha1_hash):
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
                return process_analysis_result(result, file_info, sha256_hash, md5_hash, sha1_hash)
            
            logger.info(f"Scan in progress. Attempt {attempt + 1}/{max_attempts}")
            time.sleep(wait_time)
        except requests.RequestException as e:
            logger.error(f"Error getting analysis result (attempt {attempt + 1}): {str(e)}")
    
    raise TimeoutError('Analysis timed out')

def process_analysis_result(result, file_info, sha256_hash, md5_hash, sha1_hash):
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
        'hashes': {'sha256': sha256_hash, 'md5': md5_hash, 'sha1': sha1_hash},
        'vendor_results': vendor_results,
        'cached': False,
        'instant_result': False
    }

def load_encryption_keys():
    """Load encryption keys from JSON file"""
    if os.path.exists(ENCRYPTION_METADATA_FILE):
        try:
            with open(ENCRYPTION_METADATA_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_encryption_keys(keys):
    """Save encryption keys to JSON file"""
    with open(ENCRYPTION_METADATA_FILE, 'w') as f:
        json.dump(keys, f, indent=2)

# Load existing keys on startup
encryption_keys = load_encryption_keys()



@filescanner.route('/hash-check', methods=['POST'])
def hash_check():
    """Check a file by hash without uploading"""
    data = request.get_json()
    file_hash = data.get('hash', '').strip()
    
    if not file_hash:
        return jsonify({'error': 'No hash provided'}), 400
    
    result = check_existing_file(file_hash)
    if result:
        processed_result = process_existing_result(result, {}, file_hash, '', '')
        return jsonify(processed_result)
    else:
        return jsonify({'error': 'File not found in VirusTotal database'}), 404

@filescanner.route('/encrypt', methods=['POST'])
def encrypt_file_route():
    """Encrypt uploaded file"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    algorithm = request.form.get('algorithm', 'aes')
    password = request.form.get('password', '')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if algorithm in ['aes', 'chacha20'] and not password:
        return jsonify({'error': 'Password required for selected algorithm'}), 400
    
    original_filename = file.filename
    filename = secure_filename(original_filename)
    
    # Use absolute paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(base_dir, 'temp')
    encrypted_dir = os.path.join(base_dir, 'encrypted')
    
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(encrypted_dir, exist_ok=True)
    
    file_path = os.path.join(temp_dir, filename)
    
    try:
        file.save(file_path)
        
        # Keep complete filename with extension
        encrypted_filename = f"{filename}.encrypted"
        encrypted_path = os.path.join(encrypted_dir, encrypted_filename)
        private_key = None
        
        if algorithm == 'aes':
            encrypted_data, success, message = encrypt_file_aes(file_path, password)
        elif algorithm == 'rsa':
            encrypted_data, private_key, success, message = encrypt_file_rsa(file_path)
        elif algorithm == 'chacha20':
            encrypted_data, success, message = encrypt_file_chacha20(file_path, password)
        else:
            return jsonify({'error': 'Invalid algorithm'}), 400
        
        if not success:
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({'error': message}), 500
        
        # Save encrypted file
        with open(encrypted_path, 'wb') as f:
            f.write(encrypted_data)
        
        # Verify file was written
        if not os.path.exists(encrypted_path):
            raise Exception("Failed to save encrypted file")
        
        file_size = os.path.getsize(encrypted_path)
        
        # Calculate hash
        encrypted_hash = get_file_hash(encrypted_path, 'sha256')
        
        # Generate unique ID and store metadata
        encryption_id = secrets.token_urlsafe(16)
        
        encryption_keys[encryption_id] = {
            'filename': encrypted_filename,
            'original_filename': filename,
            'algorithm': algorithm,
            'private_key': base64.b64encode(private_key).decode() if private_key else None,
            'timestamp': datetime.now().isoformat(),
            'file_path': encrypted_path,
            'file_size': file_size
        }
        
        # Save to persistent storage
        save_encryption_keys(encryption_keys)
        
        # Clean up temp file
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Logging
        logger.info(f"=== ENCRYPTION SUCCESS ===")
        logger.info(f"Encryption ID: {encryption_id}")
        logger.info(f"Original file: {filename}")
        logger.info(f"Encrypted file: {encrypted_filename}")
        logger.info(f"Saved at: {encrypted_path}")
        logger.info(f"File exists: {os.path.exists(encrypted_path)}")
        logger.info(f"File size: {file_size} bytes")
        logger.info(f"========================")
        
        return jsonify({
            'success': True,
            'message': message,
            'encryption_id': encryption_id,
            'encrypted_filename': encrypted_filename,
            'encrypted_hash': encrypted_hash,
            'algorithm': algorithm.upper(),
            'private_key': base64.b64encode(private_key).decode() if private_key else None,
            'file_size': format_file_size(file_size)
        })
        
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        logger.error(f"Encryption error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@filescanner.route('/download-encrypted/<encryption_id>')
def download_encrypted(encryption_id):
    """Download encrypted file"""
    # Reload keys from file to get latest data
    global encryption_keys
    encryption_keys = load_encryption_keys()
    
    logger.info(f"=== DOWNLOAD REQUEST ===")
    logger.info(f"Encryption ID: {encryption_id}")
    logger.info(f"Available IDs: {list(encryption_keys.keys())}")
    
    if encryption_id not in encryption_keys:
        logger.error(f"Encryption ID not found!")
        return jsonify({'error': 'Invalid or expired encryption ID'}), 404
    
    encryption_info = encryption_keys[encryption_id]
    file_path = encryption_info['file_path']
    filename = encryption_info['filename']
    
    logger.info(f"File path: {file_path}")
    logger.info(f"Filename: {filename}")
    logger.info(f"File exists: {os.path.exists(file_path)}")
    
    if not os.path.exists(file_path):
        logger.error(f"File not found on disk!")
        logger.error(f"Expected path: {file_path}")
        
        # List files in encrypted directory
        encrypted_dir = os.path.dirname(file_path)
        if os.path.exists(encrypted_dir):
            files = os.listdir(encrypted_dir)
            logger.error(f"Files in encrypted dir: {files}")
        
        return jsonify({'error': 'Encrypted file not found on server'}), 404
    
    try:
        logger.info(f"Sending file: {file_path}")
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename,
            mimetype='application/octet-stream'
        )
    except Exception as e:
        logger.error(f"Error sending file: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Download error: {str(e)}'}), 500


@filescanner.route('/cleanup-encrypted/<encryption_id>', methods=['POST'])
def cleanup_encrypted(encryption_id):
    """Clean up encrypted file after successful download"""
    if encryption_id in encryption_keys:
        encryption_info = encryption_keys[encryption_id]
        file_path = encryption_info['file_path']
        
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                del encryption_keys[encryption_id]
                return jsonify({'success': True, 'message': 'File cleaned up'})
            except Exception as e:
                logger.error(f"Cleanup error: {str(e)}")
                return jsonify({'error': str(e)}), 500
        else:
            del encryption_keys[encryption_id]
            return jsonify({'success': True, 'message': 'Entry removed'})
    
    return jsonify({'error': 'Invalid encryption ID'}), 404

@filescanner.route('/decrypt', methods=['POST'])
def decrypt_file_route():
    """Decrypt uploaded file"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    algorithm = request.form.get('algorithm', 'aes')
    password = request.form.get('password', '')
    private_key_b64 = request.form.get('private_key', '')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if algorithm in ['aes', 'chacha20'] and not password:
        return jsonify({'error': 'Password required for decryption'}), 400
    
    if algorithm == 'rsa' and not private_key_b64:
        return jsonify({'error': 'Private key required for RSA decryption'}), 400
    
    filename = secure_filename(file.filename)
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    file_path = os.path.join(temp_dir, filename)
    
    try:
        file.save(file_path)
        
        # Read encrypted data
        with open(file_path, 'rb') as f:
            encrypted_data = f.read()
        
        # Decrypt based on algorithm
        if algorithm == 'aes':
            decrypted_data, success, message = decrypt_file_aes(encrypted_data, password)
        elif algorithm == 'rsa':
            private_key_pem = base64.b64decode(private_key_b64)
            decrypted_data, success, message = decrypt_file_rsa(encrypted_data, private_key_pem)
        elif algorithm == 'chacha20':
            decrypted_data, success, message = decrypt_file_chacha20(encrypted_data, password)
        else:
            return jsonify({'error': 'Invalid algorithm'}), 400
        
        if not success:
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({'error': message}), 400
        
        # Restore original filename
        if filename.endswith('.encrypted'):
            decrypted_filename = filename[:-10]  # Remove '.encrypted'
        else:
            decrypted_filename = f"decrypted_{filename}"
        
        decrypted_path = os.path.join(temp_dir, decrypted_filename)
        with open(decrypted_path, 'wb') as f:
            f.write(decrypted_data)
        
        # Clean up encrypted file
        if os.path.exists(file_path):
            os.remove(file_path)
        
        logger.info(f"Decryption successful: {decrypted_filename}")
        
        # Return decrypted file
        return send_file(
            decrypted_path,
            as_attachment=True,
            download_name=decrypted_filename,
            mimetype='application/octet-stream'
        )
        
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        logger.error(f"Decryption error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@filescanner.route('/batch-scan', methods=['POST'])
def batch_scan():
    """Batch scan multiple files"""
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files[]')
    results = []
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            temp_dir = os.path.join(os.path.dirname(__file__), 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            file_path = os.path.join(temp_dir, filename)
            
            try:
                file.save(file_path)
                
                # Calculate hashes
                sha256_hash = get_file_hash(file_path, 'sha256')
                file_info = get_file_info(file_path)
                
                # Check existing
                existing_result = check_existing_file(sha256_hash)
                if existing_result:
                    result = process_existing_result(existing_result, file_info, sha256_hash, '', '')
                    result['filename'] = filename
                    results.append(result)
                else:
                    results.append({
                        'filename': filename,
                        'status': 'pending',
                        'message': 'File not in database - requires full scan'
                    })
                
                os.remove(file_path)
                
            except Exception as e:
                results.append({
                    'filename': filename,
                    'status': 'error',
                    'message': str(e)
                })
                if os.path.exists(file_path):
                    os.remove(file_path)
    
    return jsonify({'results': results})

# Blueprint is registered in server.py
# Directories are created on first use