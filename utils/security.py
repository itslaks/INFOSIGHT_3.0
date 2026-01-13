"""
Security utilities for INFOSIGHT 3.0
Provides centralized rate limiting, input validation, and security helpers
Following OWASP best practices
"""
import re
import time
from functools import wraps
from typing import Dict, Any, Optional, List, Callable
from collections import defaultdict
from flask import request, jsonify, g
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import logging

logger = logging.getLogger(__name__)

# ==================== RATE LIMITING ====================

# Global rate limiter instance (will be initialized in server.py)
limiter: Optional[Limiter] = None

def init_rate_limiter(app):
    """
    Initialize rate limiter with IP and user-based tracking
    OWASP: Implement rate limiting to prevent abuse
    """
    global limiter
    limiter = Limiter(
        app=app,
        key_func=get_rate_limit_key,
        default_limits=["1000 per day", "100 per hour", "20 per minute"],
        storage_uri="memory://",  # Use Redis in production: "redis://localhost:6379"
        strategy="fixed-window",  # or "moving-window" for smoother limiting
        headers_enabled=True,
        on_breach=rate_limit_breach_handler
    )
    return limiter

def get_rate_limit_key():
    """
    Get rate limit key based on IP and user (if authenticated)
    OWASP: Rate limit by IP and user ID when available
    """
    # Get IP address
    ip = get_remote_address()
    
    # If user is authenticated, include user ID
    user_id = getattr(g, 'user_id', None)
    if user_id:
        return f"{ip}:{user_id}"
    
    return ip

def rate_limit_breach_handler(request_limit):
    """
    Handle rate limit breaches with graceful 429 responses
    OWASP: Return proper HTTP status codes and helpful error messages
    """
    logger.warning(f"Rate limit exceeded for {get_rate_limit_key()}: {request_limit}")
    return jsonify({
        "error": "Rate limit exceeded",
        "message": "Too many requests. Please try again later.",
        "retry_after": request_limit.reset_at if hasattr(request_limit, 'reset_at') else None
    }), 429

# Rate limit decorators for different endpoint types
def rate_limit_public(requests_per_minute: int = 20, requests_per_hour: int = 100):
    """
    Rate limit decorator for public endpoints
    Default: 20 requests/minute, 100 requests/hour
    """
    if limiter is None:
        return lambda f: f  # No-op if limiter not initialized
    
    return limiter.limit(f"{requests_per_minute} per minute; {requests_per_hour} per hour")

def rate_limit_api(requests_per_minute: int = 10, requests_per_hour: int = 200):
    """
    Rate limit decorator for API endpoints
    Default: 10 requests/minute, 200 requests/hour
    """
    if limiter is None:
        return lambda f: f
    
    return limiter.limit(f"{requests_per_minute} per minute; {requests_per_hour} per hour")

def rate_limit_strict(requests_per_minute: int = 5, requests_per_hour: int = 50):
    """
    Rate limit decorator for resource-intensive endpoints
    Default: 5 requests/minute, 50 requests/hour
    """
    if limiter is None:
        return lambda f: f
    
    return limiter.limit(f"{requests_per_minute} per minute; {requests_per_hour} per hour")

# ==================== INPUT VALIDATION ====================

class ValidationError(Exception):
    """Custom exception for validation errors"""
    def __init__(self, message: str, field: Optional[str] = None):
        self.message = message
        self.field = field
        super().__init__(self.message)

class InputValidator:
    """
    Schema-based input validator
    OWASP: Validate all inputs, reject unexpected fields, enforce length limits
    """
    
    # Maximum field lengths (OWASP recommendation: reasonable limits)
    # OWASP: Enforce strict length limits to prevent DoS and injection attacks
    MAX_STRING_LENGTH = 397  # As specified by user - strict limit for all user inputs
    MAX_TEXT_LENGTH = 10000  # For longer text fields (templates, batch operations)
    MAX_URL_LENGTH = 2048
    MAX_EMAIL_LENGTH = 254
    MAX_FILENAME_LENGTH = 255
    
    # Allowed characters patterns
    SAFE_STRING_PATTERN = re.compile(r'^[a-zA-Z0-9\s\-_.,!?@#$%^&*()+=\[\]{}|\\:";\'<>?/~`]+$')
    ALPHANUMERIC_PATTERN = re.compile(r'^[a-zA-Z0-9]+$')
    DOMAIN_PATTERN = re.compile(r'^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$')
    IP_PATTERN = re.compile(r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$')
    
    @staticmethod
    def validate_string(value: Any, field_name: str, max_length: int = MAX_STRING_LENGTH, 
                       required: bool = True, allow_empty: bool = False, 
                       pattern: Optional[re.Pattern] = None) -> str:
        """
        Validate string input
        OWASP: Type checking, length limits, pattern validation
        """
        if value is None:
            if required:
                raise ValidationError(f"{field_name} is required", field_name)
            return ""
        
        if not isinstance(value, str):
            raise ValidationError(f"{field_name} must be a string", field_name)
        
        value = value.strip()
        
        if not value and not allow_empty:
            if required:
                raise ValidationError(f"{field_name} cannot be empty", field_name)
            return ""
        
        if len(value) > max_length:
            raise ValidationError(
                f"{field_name} exceeds maximum length of {max_length} characters", 
                field_name
            )
        
        # Check for path traversal attempts
        if '..' in value or value.startswith('/') or '\\' in value:
            if field_name not in ['path', 'filepath']:  # Allow for legitimate path fields
                raise ValidationError(f"{field_name} contains invalid characters", field_name)
        
        # Pattern validation if provided
        if pattern and value and not pattern.match(value):
            raise ValidationError(f"{field_name} contains invalid characters", field_name)
        
        return value
    
    @staticmethod
    def validate_integer(value: Any, field_name: str, min_value: Optional[int] = None,
                        max_value: Optional[int] = None, required: bool = True) -> int:
        """Validate integer input"""
        if value is None:
            if required:
                raise ValidationError(f"{field_name} is required", field_name)
            return 0
        
        try:
            int_value = int(value)
        except (ValueError, TypeError):
            raise ValidationError(f"{field_name} must be an integer", field_name)
        
        if min_value is not None and int_value < min_value:
            raise ValidationError(f"{field_name} must be at least {min_value}", field_name)
        
        if max_value is not None and int_value > max_value:
            raise ValidationError(f"{field_name} must be at most {max_value}", field_name)
        
        return int_value
    
    @staticmethod
    def validate_float(value: Any, field_name: str, min_value: Optional[float] = None,
                      max_value: Optional[float] = None, required: bool = True) -> float:
        """Validate float input"""
        if value is None:
            if required:
                raise ValidationError(f"{field_name} is required", field_name)
            return 0.0
        
        try:
            float_value = float(value)
        except (ValueError, TypeError):
            raise ValidationError(f"{field_name} must be a number", field_name)
        
        if min_value is not None and float_value < min_value:
            raise ValidationError(f"{field_name} must be at least {min_value}", field_name)
        
        if max_value is not None and float_value > max_value:
            raise ValidationError(f"{field_name} must be at most {max_value}", field_name)
        
        return float_value
    
    @staticmethod
    def validate_url(value: Any, field_name: str, required: bool = True) -> str:
        """Validate URL input"""
        value = InputValidator.validate_string(
            value, field_name, max_length=InputValidator.MAX_URL_LENGTH, 
            required=required, allow_empty=not required
        )
        
        if not value and not required:
            return ""
        
        # Basic URL validation
        if not (value.startswith('http://') or value.startswith('https://')):
            raise ValidationError(f"{field_name} must be a valid URL (http:// or https://)", field_name)
        
        # Check for dangerous protocols
        dangerous_protocols = ['javascript:', 'data:', 'file:', 'vbscript:']
        if any(value.lower().startswith(proto) for proto in dangerous_protocols):
            raise ValidationError(f"{field_name} contains invalid protocol", field_name)
        
        return value
    
    @staticmethod
    def validate_domain(value: Any, field_name: str, required: bool = True) -> str:
        """Validate domain name"""
        value = InputValidator.validate_string(
            value, field_name, max_length=253, required=required
        )
        
        if not value and not required:
            return ""
        
        if not InputValidator.DOMAIN_PATTERN.match(value):
            raise ValidationError(f"{field_name} must be a valid domain name", field_name)
        
        return value.lower()
    
    @staticmethod
    def validate_ip(value: Any, field_name: str, required: bool = True) -> str:
        """Validate IP address"""
        value = InputValidator.validate_string(
            value, field_name, max_length=45, required=required  # IPv6 can be up to 45 chars
        )
        
        if not value and not required:
            return ""
        
        # IPv4 validation
        if InputValidator.IP_PATTERN.match(value):
            return value
        
        # Basic IPv6 validation (simplified)
        if ':' in value and value.count(':') <= 7:
            return value
        
        raise ValidationError(f"{field_name} must be a valid IP address", field_name)
    
    @staticmethod
    def validate_email(value: Any, field_name: str, required: bool = True) -> str:
        """Validate email address"""
        value = InputValidator.validate_string(
            value, field_name, max_length=InputValidator.MAX_EMAIL_LENGTH, 
            required=required
        )
        
        if not value and not required:
            return ""
        
        email_pattern = re.compile(
            r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        )
        
        if not email_pattern.match(value):
            raise ValidationError(f"{field_name} must be a valid email address", field_name)
        
        return value.lower()
    
    @staticmethod
    def validate_filename(value: Any, field_name: str, required: bool = True) -> str:
        """Validate filename (prevent path traversal)"""
        value = InputValidator.validate_string(
            value, field_name, max_length=InputValidator.MAX_FILENAME_LENGTH, 
            required=required
        )
        
        if not value and not required:
            return ""
        
        # Prevent path traversal
        if '..' in value or '/' in value or '\\' in value:
            raise ValidationError(f"{field_name} contains invalid characters", field_name)
        
        # Prevent dangerous filenames
        dangerous_names = ['CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 
                          'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 
                          'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 
                          'LPT7', 'LPT8', 'LPT9']
        if value.upper() in dangerous_names:
            raise ValidationError(f"{field_name} is a reserved filename", field_name)
        
        return value
    
    @staticmethod
    def validate_json_schema(data: Dict[str, Any], schema: Dict[str, Dict[str, Any]], 
                            strict: bool = True) -> Dict[str, Any]:
        """
        Validate JSON data against schema
        OWASP: Reject unexpected fields, validate all inputs
        
        Schema format:
        {
            "field_name": {
                "type": "string|int|float|bool|list|dict",
                "required": True/False,
                "max_length": int,
                "min_value": int/float,
                "max_value": int/float,
                "pattern": regex_pattern,
                "allowed_values": [list],
                "nested_schema": {...}  # For dict/list types
            }
        }
        """
        if not isinstance(data, dict):
            raise ValidationError("Request body must be a JSON object")
        
        validated = {}
        
        # Check for unexpected fields if strict mode
        if strict:
            allowed_fields = set(schema.keys())
            provided_fields = set(data.keys())
            unexpected = provided_fields - allowed_fields
            if unexpected:
                raise ValidationError(
                    f"Unexpected fields: {', '.join(unexpected)}. "
                    f"Allowed fields: {', '.join(sorted(allowed_fields))}"
                )
        
        # Validate each field in schema
        for field_name, field_schema in schema.items():
            field_type = field_schema.get('type', 'string')
            required = field_schema.get('required', True)
            value = data.get(field_name)
            
            if value is None:
                if required:
                    raise ValidationError(f"{field_name} is required", field_name)
                continue
            
            # Type validation
            if field_type == 'string':
                max_length = field_schema.get('max_length', InputValidator.MAX_STRING_LENGTH)
                pattern = field_schema.get('pattern')
                validated[field_name] = InputValidator.validate_string(
                    value, field_name, max_length=max_length, 
                    required=required, pattern=pattern
                )
            
            elif field_type == 'int':
                validated[field_name] = InputValidator.validate_integer(
                    value, field_name,
                    min_value=field_schema.get('min_value'),
                    max_value=field_schema.get('max_value'),
                    required=required
                )
            
            elif field_type == 'float':
                validated[field_name] = InputValidator.validate_float(
                    value, field_name,
                    min_value=field_schema.get('min_value'),
                    max_value=field_schema.get('max_value'),
                    required=required
                )
            
            elif field_type == 'bool':
                if not isinstance(value, bool):
                    raise ValidationError(f"{field_name} must be a boolean", field_name)
                validated[field_name] = value
            
            elif field_type == 'list':
                if not isinstance(value, list):
                    raise ValidationError(f"{field_name} must be a list", field_name)
                max_items = field_schema.get('max_items', 100)
                if len(value) > max_items:
                    raise ValidationError(
                        f"{field_name} cannot contain more than {max_items} items", 
                        field_name
                    )
                # Validate list items if schema provided
                item_schema = field_schema.get('item_schema')
                if item_schema:
                    validated[field_name] = [
                        InputValidator.validate_json_schema(
                            item if isinstance(item, dict) else {'value': item},
                            {'value': item_schema}
                        )['value'] for item in value
                    ]
                else:
                    validated[field_name] = value
            
            elif field_type == 'dict':
                if not isinstance(value, dict):
                    raise ValidationError(f"{field_name} must be an object", field_name)
                nested_schema = field_schema.get('nested_schema', {})
                # If nested_schema is empty, allow all fields (flexible schema)
                if nested_schema:
                    validated[field_name] = InputValidator.validate_json_schema(
                        value, nested_schema, strict=strict
                    )
                else:
                    # Empty nested_schema means allow all fields
                    validated[field_name] = value
            
            # Check allowed values
            allowed_values = field_schema.get('allowed_values')
            if allowed_values and validated[field_name] not in allowed_values:
                raise ValidationError(
                    f"{field_name} must be one of: {', '.join(map(str, allowed_values))}", 
                    field_name
                )
        
        return validated

def validate_request(schema: Dict[str, Dict[str, Any]], strict: bool = True):
    """
    Decorator to validate request JSON against schema
    Usage:
        @validate_request({
            "field_name": {"type": "string", "required": True, "max_length": 100}
        })
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                if not request.is_json:
                    return jsonify({
                        "error": "Content-Type must be application/json",
                        "success": False
                    }), 400
                
                data = request.get_json()
                if data is None:
                    return jsonify({
                        "error": "Request body must be valid JSON",
                        "success": False
                    }), 400
                
                # Validate against schema
                validated_data = InputValidator.validate_json_schema(data, schema, strict=strict)
                
                # Store validated data in request context
                g.validated_data = validated_data
                
                return f(*args, **kwargs)
            
            except ValidationError as e:
                logger.warning(f"Validation error: {e.message} (field: {e.field})")
                return jsonify({
                    "error": e.message,
                    "field": e.field,
                    "success": False
                }), 400
            
            except Exception as e:
                logger.error(f"Validation exception: {str(e)}")
                return jsonify({
                    "error": "Invalid request format",
                    "success": False
                }), 400
        
        return decorated_function
    return decorator

# ==================== SANITIZATION ====================

def sanitize_string(value: str) -> str:
    """
    Sanitize string input to prevent XSS and injection attacks
    OWASP: Sanitize all user inputs
    """
    if not isinstance(value, str):
        return ""
    
    # Remove null bytes
    value = value.replace('\x00', '')
    
    # Remove control characters except newlines and tabs
    value = ''.join(char for char in value if ord(char) >= 32 or char in '\n\t')
    
    return value.strip()

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal"""
    # Remove path components
    filename = filename.split('/')[-1].split('\\')[-1]
    
    # Remove dangerous characters
    dangerous_chars = ['<', '>', ':', '"', '|', '?', '*']
    for char in dangerous_chars:
        filename = filename.replace(char, '_')
    
    return filename

# ==================== SECURITY HELPERS ====================

def get_client_ip() -> str:
    """Get client IP address, handling proxies"""
    if request.headers.get('X-Forwarded-For'):
        # Take the first IP in the chain
        return request.headers.get('X-Forwarded-For').split(',')[0].strip()
    elif request.headers.get('X-Real-IP'):
        return request.headers.get('X-Real-IP')
    else:
        return request.remote_addr or '0.0.0.0'

def is_safe_origin(origin: str) -> bool:
    """Check if origin is safe (for CORS)"""
    # In production, implement proper origin whitelist
    safe_patterns = [
        r'^https?://localhost(:\d+)?$',
        r'^https?://127\.0\.0\.1(:\d+)?$',
        r'^https?://.*\.yourdomain\.com$'  # Replace with your domain
    ]
    
    for pattern in safe_patterns:
        if re.match(pattern, origin):
            return True
    
    return False
