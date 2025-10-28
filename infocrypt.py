from flask import Blueprint, request, jsonify, render_template
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import hashlib
import os
import base64
import secrets
import hmac
from datetime import datetime
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding, hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding as asym_padding
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.fernet import Fernet
from functools import wraps
import logging
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create blueprint
infocrypt = Blueprint('infocrypt', __name__, template_folder='templates')

# Constants
MAX_INPUT_SIZE = 5 * 1024 * 1024  # 5MB
RSA_CHUNK_SIZE = {
    2048: 190,
    4096: 446
}

HASH_ALGORITHMS = [
    'SHA-256', 'SHA-512', 'SHA-384', 'SHA-224',
    'SHA3-256', 'SHA3-512', 'SHA3-384', 'SHA3-224',
    'BLAKE2b', 'BLAKE2s', 'MD5', 'SHA-1'
]

ENCRYPTION_ALGORITHMS = [
    'AES-128-CBC', 'AES-192-CBC', 'AES-256-CBC',
    'AES-128-GCM', 'AES-192-GCM', 'AES-256-GCM',
    'AES-128-CTR', 'AES-256-CTR',
    'ChaCha20-Poly1305',
    'Fernet',
    'RSA-2048', 'RSA-4096',
    'TripleDES'
]

class CryptoError(Exception):
    """Custom exception for cryptographic operations"""
    pass

def validate_input(max_size=MAX_INPUT_SIZE):
    """Decorator to validate input data"""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            data = request.json
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            # Check size for text/ciphertext fields
            text = data.get('text', data.get('ciphertext', ''))
            if text and len(text.encode('utf-8')) > max_size:
                return jsonify({'error': f'Input exceeds {max_size} bytes'}), 400
            
            return f(*args, **kwargs)
        return wrapper
    return decorator

@infocrypt.route('/')
def index():
    return render_template('infocrypt.html')

@infocrypt.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '3.0.0'
    })

# ==================== HASH OPERATIONS ====================

@infocrypt.route('/api/hash', methods=['POST'])
@validate_input()
def hash_endpoint():
    """Generate hash of input text"""
    try:
        data = request.json
        text = data.get('text', '').strip()
        algorithm = data.get('algorithm', 'SHA-256')
        iterations = int(data.get('iterations', 1))
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        if algorithm not in HASH_ALGORITHMS:
            return jsonify({'error': f'Unsupported algorithm: {algorithm}'}), 400
        
        if not 1 <= iterations <= 1000000:
            return jsonify({'error': 'Iterations must be 1-1,000,000'}), 400
        
        # Hash the data
        result = _hash_data(text, algorithm, iterations)
        
        logger.info(f"Hashed data with {algorithm}, {iterations} iterations")
        return jsonify({'success': True, 'data': result})
        
    except Exception as e:
        logger.error(f"Hash error: {str(e)}")
        return jsonify({'error': str(e)}), 500

def _hash_data(text, algorithm, iterations=1):
    """Internal hash function"""
    data_bytes = text.encode('utf-8')
    
    # Map algorithm names to functions
    hash_map = {
        'SHA-256': hashlib.sha256,
        'SHA-512': hashlib.sha512,
        'SHA-384': hashlib.sha384,
        'SHA-224': hashlib.sha224,
        'SHA3-256': hashlib.sha3_256,
        'SHA3-512': hashlib.sha3_512,
        'SHA3-384': hashlib.sha3_384,
        'SHA3-224': hashlib.sha3_224,
        'BLAKE2b': hashlib.blake2b,
        'BLAKE2s': hashlib.blake2s,
        'MD5': hashlib.md5,
        'SHA-1': hashlib.sha1
    }
    
    if algorithm not in hash_map:
        raise CryptoError(f'Unsupported algorithm: {algorithm}')
    
    # Perform hashing with iterations
    result = data_bytes
    for _ in range(iterations):
        h = hash_map[algorithm]()
        h.update(result)
        result = h.digest()
    
    return {
        'hash': result.hex(),
        'algorithm': algorithm,
        'iterations': iterations,
        'length': len(result) * 2,
        'bits': len(result) * 8
    }

# ==================== ENCRYPTION OPERATIONS ====================

@infocrypt.route('/api/generate-key', methods=['POST'])
def generate_key_endpoint():
    """Generate encryption key for specified algorithm"""
    try:
        data = request.json
        algorithm = data.get('algorithm')
        
        if not algorithm or algorithm not in ENCRYPTION_ALGORITHMS:
            return jsonify({'error': 'Invalid algorithm'}), 400
        
        key_data = _generate_key(algorithm)
        logger.info(f"Generated key for {algorithm}")
        
        return jsonify({'success': True, 'data': key_data})
        
    except Exception as e:
        logger.error(f"Key generation error: {str(e)}")
        return jsonify({'error': str(e)}), 500

def _generate_key(algorithm):
    """Generate cryptographic key"""
    key_id = secrets.token_hex(8)
    timestamp = datetime.utcnow().isoformat()
    
    # Determine key size
    if '128' in algorithm:
        key_size = 16
    elif '192' in algorithm:
        key_size = 24
    elif '256' in algorithm or 'ChaCha20' in algorithm:
        key_size = 32
    elif 'TripleDES' in algorithm:
        key_size = 24
    elif algorithm == 'Fernet':
        key = Fernet.generate_key().decode()
        return {
            'key': key,
            'key_id': key_id,
            'algorithm': algorithm,
            'key_size': 256,
            'timestamp': timestamp
        }
    elif 'RSA' in algorithm:
        key_size_bits = int(algorithm.split('-')[1])
        return _generate_rsa_key(key_size_bits, key_id, timestamp)
    else:
        key_size = 32
    
    # Generate symmetric key
    key_bytes = secrets.token_bytes(key_size)
    key = base64.b64encode(key_bytes).decode()
    
    return {
        'key': key,
        'key_id': key_id,
        'algorithm': algorithm,
        'key_size': key_size * 8,
        'timestamp': timestamp
    }

def _generate_rsa_key(key_size, key_id, timestamp):
    """Generate RSA key pair"""
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=key_size,
        backend=default_backend()
    )
    
    public_key = private_key.public_key()
    
    # Serialize keys
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    
    return {
        'private_key': base64.b64encode(private_pem).decode(),
        'public_key': base64.b64encode(public_pem).decode(),
        'key_id': key_id,
        'algorithm': f'RSA-{key_size}',
        'key_size': key_size,
        'timestamp': timestamp
    }

@infocrypt.route('/api/encrypt', methods=['POST'])
@validate_input()
def encrypt_endpoint():
    """Encrypt plaintext"""
    try:
        data = request.json
        text = data.get('text', '').strip()
        algorithm = data.get('algorithm')
        key = data.get('key', '').strip()
        use_password = data.get('use_password', False)
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        if not algorithm or algorithm not in ENCRYPTION_ALGORITHMS:
            return jsonify({'error': 'Invalid algorithm'}), 400
        
        result = _encrypt_data(text, algorithm, key, use_password)
        logger.info(f"Encrypted with {algorithm}")
        
        return jsonify({'success': True, 'data': result})
        
    except Exception as e:
        logger.error(f"Encryption error: {str(e)}")
        return jsonify({'error': str(e)}), 500

def _encrypt_data(text, algorithm, key=None, use_password=False):
    """Encrypt data with specified algorithm"""
    timestamp = datetime.utcnow().isoformat()
    key_id = secrets.token_hex(8)
    
    # Handle key generation or derivation
    if not key:
        # Auto-generate key
        key_info = _generate_key(algorithm)
        
        if 'RSA' in algorithm:
            # RSA encryption
            public_pem = base64.b64decode(key_info['public_key'])
            pub_key = serialization.load_pem_public_key(public_pem, backend=default_backend())
            
            ciphertext = pub_key.encrypt(
                text.encode(),
                asym_padding.OAEP(
                    mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            return {
                'ciphertext': base64.b64encode(ciphertext).decode(),
                'private_key': key_info['private_key'],
                'public_key': key_info['public_key'],
                'key_id': key_info['key_id'],
                'algorithm': algorithm,
                'timestamp': timestamp
            }
        else:
            key = key_info['key']
            result = {'key': key, 'key_id': key_info['key_id']}
    else:
        result = {}
    
    # Decode key
    if use_password and key:
        salt = secrets.token_bytes(16)
        derived_key = _derive_key_from_password(key, salt, algorithm)
        result['salt'] = base64.b64encode(salt).decode()
        result['kdf'] = 'Scrypt'
    else:
        try:
            derived_key = base64.b64decode(key)
        except Exception:
            raise CryptoError("Invalid key format - must be base64 encoded")
    
    # Perform encryption based on algorithm
    if 'CBC' in algorithm:
        ciphertext = _encrypt_aes_cbc(text, derived_key, algorithm)
    elif 'GCM' in algorithm:
        ciphertext = _encrypt_aes_gcm(text, derived_key, algorithm)
        result['authenticated'] = True
    elif 'CTR' in algorithm:
        ciphertext = _encrypt_aes_ctr(text, derived_key, algorithm)
    elif algorithm == 'ChaCha20-Poly1305':
        ciphertext = _encrypt_chacha20_poly1305(text, derived_key)
        result['authenticated'] = True
    elif algorithm == 'TripleDES':
        ciphertext = _encrypt_tripledes(text, derived_key)
    elif algorithm == 'Fernet':
        f = Fernet(key.encode())
        ciphertext = f.encrypt(text.encode()).decode()
    elif 'RSA' in algorithm:
        # Encrypt with provided public key
        public_pem = base64.b64decode(key)
        pub_key = serialization.load_pem_public_key(public_pem, backend=default_backend())
        
        encrypted = pub_key.encrypt(
            text.encode(),
            asym_padding.OAEP(
                mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        ciphertext = base64.b64encode(encrypted).decode()
    else:
        raise CryptoError(f'Unsupported algorithm: {algorithm}')
    
    result.update({
        'ciphertext': ciphertext,
        'algorithm': algorithm,
        'timestamp': timestamp
    })
    
    return result

def _derive_key_from_password(password, salt, algorithm):
    """Derive encryption key from password using Scrypt"""
    if '128' in algorithm:
        key_length = 16
    elif '192' in algorithm:
        key_length = 24
    else:
        key_length = 32
    
    kdf = Scrypt(
        salt=salt,
        length=key_length,
        n=2**14,
        r=8,
        p=1,
        backend=default_backend()
    )
    
    return kdf.derive(password.encode())

def _encrypt_aes_cbc(text, key, algorithm):
    """Encrypt with AES-CBC"""
    key_size = int(algorithm.split('-')[1]) // 8
    key = key[:key_size]
    
    iv = secrets.token_bytes(16)
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    
    padder = padding.PKCS7(128).padder()
    padded = padder.update(text.encode()) + padder.finalize()
    
    encrypted = encryptor.update(padded) + encryptor.finalize()
    return base64.b64encode(iv + encrypted).decode()

def _encrypt_aes_gcm(text, key, algorithm):
    """Encrypt with AES-GCM"""
    key_size = int(algorithm.split('-')[1]) // 8
    key = key[:key_size]
    
    iv = secrets.token_bytes(12)
    cipher = Cipher(algorithms.AES(key), modes.GCM(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    
    encrypted = encryptor.update(text.encode()) + encryptor.finalize()
    return base64.b64encode(iv + encryptor.tag + encrypted).decode()

def _encrypt_aes_ctr(text, key, algorithm):
    """Encrypt with AES-CTR"""
    key_size = int(algorithm.split('-')[1]) // 8
    key = key[:key_size]
    
    nonce = secrets.token_bytes(16)
    cipher = Cipher(algorithms.AES(key), modes.CTR(nonce), backend=default_backend())
    encryptor = cipher.encryptor()
    
    encrypted = encryptor.update(text.encode()) + encryptor.finalize()
    return base64.b64encode(nonce + encrypted).decode()

def _encrypt_chacha20_poly1305(text, key):
    """Encrypt with ChaCha20-Poly1305"""
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
    
    key = key[:32]
    nonce = secrets.token_bytes(12)
    chacha = ChaCha20Poly1305(key)
    encrypted = chacha.encrypt(nonce, text.encode(), None)
    
    return base64.b64encode(nonce + encrypted).decode()

def _encrypt_tripledes(text, key):
    """Encrypt with TripleDES"""
    key = key[:24]
    iv = secrets.token_bytes(8)
    cipher = Cipher(algorithms.TripleDES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    
    padder = padding.PKCS7(64).padder()
    padded = padder.update(text.encode()) + padder.finalize()
    
    encrypted = encryptor.update(padded) + encryptor.finalize()
    return base64.b64encode(iv + encrypted).decode()

@infocrypt.route('/api/decrypt', methods=['POST'])
@validate_input()
def decrypt_endpoint():
    """Decrypt ciphertext"""
    try:
        data = request.json
        ciphertext = data.get('ciphertext', '').strip()
        algorithm = data.get('algorithm')
        key = data.get('key', '')
        salt = data.get('salt', '')
        private_key = data.get('private_key', '')
        
        # Strip strings safely
        if key:
            key = key.strip()
        if salt:
            salt = salt.strip()
        if private_key:
            private_key = private_key.strip()
        
        if not ciphertext:
            return jsonify({'error': 'Ciphertext is required'}), 400
        
        if not algorithm or algorithm not in ENCRYPTION_ALGORITHMS:
            return jsonify({'error': 'Invalid algorithm'}), 400
        
        if not key and not private_key:
            return jsonify({'error': 'Key or private key is required'}), 400
        
        result = _decrypt_data(ciphertext, algorithm, key, salt, private_key)
        logger.info(f"Decrypted with {algorithm}")
        
        return jsonify({'success': True, 'data': result})
        
    except Exception as e:
        logger.error(f"Decryption error: {str(e)}")
        return jsonify({'error': str(e)}), 500

def _decrypt_data(ciphertext, algorithm, key=None, salt=None, private_key=None):
    """Decrypt data"""
    timestamp = datetime.utcnow().isoformat()
    
    # Handle key derivation or decoding
    if salt and key:
        try:
            salt_bytes = base64.b64decode(salt)
            derived_key = _derive_key_from_password(key, salt_bytes, algorithm)
        except Exception as e:
            raise CryptoError(f"Invalid salt format: {str(e)}")
    elif key:
        try:
            derived_key = base64.b64decode(key)
        except Exception as e:
            raise CryptoError(f"Invalid key format - must be base64 encoded: {str(e)}")
    elif private_key:
        derived_key = None
    else:
        raise CryptoError("Key is required")
    
    # Perform decryption
    try:
        if 'CBC' in algorithm:
            plaintext = _decrypt_aes_cbc(ciphertext, derived_key, algorithm)
        elif 'GCM' in algorithm:
            plaintext = _decrypt_aes_gcm(ciphertext, derived_key, algorithm)
        elif 'CTR' in algorithm:
            plaintext = _decrypt_aes_ctr(ciphertext, derived_key, algorithm)
        elif algorithm == 'ChaCha20-Poly1305':
            plaintext = _decrypt_chacha20_poly1305(ciphertext, derived_key)
        elif algorithm == 'TripleDES':
            plaintext = _decrypt_tripledes(ciphertext, derived_key)
        elif algorithm == 'Fernet':
            f = Fernet(key.encode())
            plaintext = f.decrypt(ciphertext.encode()).decode()
        elif 'RSA' in algorithm:
            if not private_key:
                raise CryptoError("Private key is required for RSA decryption")
            
            try:
                private_pem = base64.b64decode(private_key)
                priv_key = serialization.load_pem_private_key(
                    private_pem,
                    password=None,
                    backend=default_backend()
                )
                
                encrypted = base64.b64decode(ciphertext)
                decrypted = priv_key.decrypt(
                    encrypted,
                    asym_padding.OAEP(
                        mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                plaintext = decrypted.decode()
            except Exception as e:
                raise CryptoError(f"RSA decryption failed: {str(e)}")
        else:
            raise CryptoError(f'Unsupported algorithm: {algorithm}')
    except CryptoError:
        raise
    except Exception as e:
        raise CryptoError(f"Decryption failed: {str(e)}")
    
    return {
        'plaintext': plaintext,
        'algorithm': algorithm,
        'timestamp': timestamp
    }

def _decrypt_aes_cbc(ciphertext, key, algorithm):
    """Decrypt AES-CBC"""
    key_size = int(algorithm.split('-')[1]) // 8
    key = key[:key_size]
    
    decoded = base64.b64decode(ciphertext)
    iv = decoded[:16]
    encrypted = decoded[16:]
    
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    
    padded = decryptor.update(encrypted) + decryptor.finalize()
    
    unpadder = padding.PKCS7(128).unpadder()
    plaintext = unpadder.update(padded) + unpadder.finalize()
    
    return plaintext.decode()

def _decrypt_aes_gcm(ciphertext, key, algorithm):
    """Decrypt AES-GCM"""
    key_size = int(algorithm.split('-')[1]) // 8
    key = key[:key_size]
    
    decoded = base64.b64decode(ciphertext)
    iv = decoded[:12]
    tag = decoded[12:28]
    encrypted = decoded[28:]
    
    cipher = Cipher(
        algorithms.AES(key),
        modes.GCM(iv, tag),
        backend=default_backend()
    )
    decryptor = cipher.decryptor()
    
    plaintext = decryptor.update(encrypted) + decryptor.finalize()
    return plaintext.decode()

def _decrypt_aes_ctr(ciphertext, key, algorithm):
    """Decrypt AES-CTR"""
    key_size = int(algorithm.split('-')[1]) // 8
    key = key[:key_size]
    
    decoded = base64.b64decode(ciphertext)
    nonce = decoded[:16]
    encrypted = decoded[16:]
    
    cipher = Cipher(algorithms.AES(key), modes.CTR(nonce), backend=default_backend())
    decryptor = cipher.decryptor()
    
    plaintext = decryptor.update(encrypted) + decryptor.finalize()
    return plaintext.decode()

def _decrypt_chacha20_poly1305(ciphertext, key):
    """Decrypt ChaCha20-Poly1305"""
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
    
    key = key[:32]
    decoded = base64.b64decode(ciphertext)
    nonce = decoded[:12]
    encrypted = decoded[12:]
    
    chacha = ChaCha20Poly1305(key)
    plaintext = chacha.decrypt(nonce, encrypted, None)
    
    return plaintext.decode()

def _decrypt_tripledes(ciphertext, key):
    """Decrypt TripleDES"""
    key = key[:24]
    decoded = base64.b64decode(ciphertext)
    iv = decoded[:8]
    encrypted = decoded[8:]
    
    cipher = Cipher(algorithms.TripleDES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    
    padded = decryptor.update(encrypted) + decryptor.finalize()
    
    unpadder = padding.PKCS7(64).unpadder()
    plaintext = unpadder.update(padded) + unpadder.finalize()
    
    return plaintext.decode()

# ==================== COMPARISON & VERIFICATION ====================

@infocrypt.route('/api/compare', methods=['POST'])
def compare_hashes():
    """Compare two hash values"""
    try:
        data = request.json
        hash1 = data.get('hash1', '').strip()
        hash2 = data.get('hash2', '').strip()
        
        if not hash1 or not hash2:
            return jsonify({'error': 'Both hashes required'}), 400
        
        # Constant-time comparison
        match = hmac.compare_digest(hash1.encode(), hash2.encode())
        
        return jsonify({
            'success': True,
            'data': {
                'match': match,
                'hash1_length': len(hash1),
                'hash2_length': len(hash2)
            }
        })
        
    except Exception as e:
        logger.error(f"Comparison error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@infocrypt.route('/api/verify', methods=['POST'])
@validate_input()
def verify_hash():
    """Verify text against expected hash"""
    try:
        data = request.json
        text = data.get('text', '').strip()
        expected_hash = data.get('hash', '').strip()
        algorithm = data.get('algorithm', 'SHA-256')
        iterations = int(data.get('iterations', 1))
        
        if not text or not expected_hash:
            return jsonify({'error': 'Text and hash required'}), 400
        
        # Compute hash
        result = _hash_data(text, algorithm, iterations)
        computed_hash = result['hash']
        
        # Compare
        verified = hmac.compare_digest(computed_hash.encode(), expected_hash.encode())
        
        return jsonify({
            'success': True,
            'data': {
                'verified': verified,
                'computed_hash': computed_hash,
                'algorithm': algorithm,
                'iterations': iterations,
                'timestamp': datetime.utcnow().isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Verification error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@infocrypt.route('/api/algorithms', methods=['GET'])
def list_algorithms():
    """List supported algorithms"""
    return jsonify({
        'hash_algorithms': HASH_ALGORITHMS,
        'encryption_algorithms': ENCRYPTION_ALGORITHMS
    })

# Error handlers
@infocrypt.errorhandler(413)
def request_too_large(error):
    return jsonify({'error': 'Request too large'}), 413

@infocrypt.errorhandler(429)
def rate_limit_exceeded(error):
    return jsonify({'error': 'Rate limit exceeded'}), 429

@infocrypt.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    infocrypt.run(debug=False)