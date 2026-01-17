# ==================== PROTOBUF COMPATIBILITY PATCH (MUST BE FIRST) ====================
# Fix for Python 3.13 + protobuf compatibility issue
# Apply this BEFORE any other imports that might use protobuf
try:
    import google.protobuf
    if not hasattr(google.protobuf, 'runtime_version'):
        protobuf_version = getattr(google.protobuf, '__version__', '4.25.8')
        version_parts = protobuf_version.split('.')
        
        # Create complete runtime_version compatibility shim
        class RuntimeVersion:
            def __init__(self):
                self.major = int(version_parts[0]) if len(version_parts) > 0 and version_parts[0].isdigit() else 4
                self.minor = int(version_parts[1]) if len(version_parts) > 1 and version_parts[1].isdigit() else 25
                self.patch = int(version_parts[2]) if len(version_parts) > 2 and version_parts[2].isdigit() else 8
                
                # Domain enum for tensorflow compatibility
                class Domain:
                    PUBLIC = 1
                    INTERNAL = 2
                
                self.Domain = Domain()
            
            def __str__(self):
                return f"{self.major}.{self.minor}.{self.patch}"
            
            def ValidateProtobufRuntimeVersion(self, *args, **kwargs):
                """Validate protobuf runtime version (required by tensorflow and transformers)"""
                # Accept any arguments for compatibility
                return True
        
        google.protobuf.runtime_version = RuntimeVersion()
except (ImportError, Exception):
    # If protobuf isn't available or patch fails, continue anyway
    pass
# ==================== END PROTOBUF PATCH ====================

from flask import Blueprint, request, jsonify, render_template, current_app
from flask_cors import CORS
import torch
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import io
import time
import imagehash
import traceback
import warnings
from collections import Counter
import cv2
import numpy as np
import binascii
from sklearn.cluster import KMeans
import hashlib
import struct
import json
import os
import logging
from datetime import datetime
import exifread
import piexif

# Configure logging first (before any logger usage)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Lazy import transformers to handle protobuf version compatibility issues
# This must be done carefully to prevent module import failures
BlipForConditionalGeneration = None
BlipProcessor = None

def _safe_import_transformers():
    """Safely import transformers, handling protobuf compatibility issues"""
    global BlipForConditionalGeneration, BlipProcessor
    try:
        # Suppress all warnings during import
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Try importing transformers
            from transformers import BlipForConditionalGeneration, BlipProcessor
            return True
    except (ImportError, AttributeError, ModuleNotFoundError, TypeError) as e:
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ['protobuf', 'runtime_version', 'cannot import']):
            logger.warning("⚠️ Protobuf version compatibility issue. BLIP model will be unavailable.")
            logger.warning("⚠️ This is a known issue with protobuf 5.x and transformers. BLIP features will be disabled.")
        else:
            logger.warning(f"⚠️ Transformers not available: {e}")
        return False
    except Exception as e:
        # Catch any other errors during import (including protobuf runtime errors)
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ['protobuf', 'runtime_version', 'cannot import', 'import name']):
            logger.warning("⚠️ Protobuf version compatibility issue. BLIP model will be unavailable.")
            logger.warning("⚠️ This is a known issue with protobuf 5.x and transformers. BLIP features will be disabled.")
        else:
            logger.warning(f"⚠️ Error loading transformers: {e}")
        return False

# Attempt to import transformers safely
try:
    _safe_import_transformers()
except Exception as e:
    # Final fallback - ensure variables are set to None
    logger.warning(f"⚠️ Failed to import transformers: {e}")
    BlipForConditionalGeneration = None
    BlipProcessor = None

# Suppress transformers logging (if transformers is available)
try:
    if BlipForConditionalGeneration is not None:
        from transformers import logging as tf_logging
        tf_logging.set_verbosity_error()
except:
    pass

# Local LLM fallback
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.local_llm_utils import generate_with_ollama, check_ollama_available
    from utils.llm_logger import log_llm_status, log_llm_request, log_llm_success, log_llm_error, log_llm_fallback, log_processing_step
    LOCAL_LLM_AVAILABLE = True
except ImportError as e:
    LOCAL_LLM_AVAILABLE = False
    # Create dummy functions
    def log_llm_status(*args, **kwargs): return (False, False)
    def log_llm_request(*args, **kwargs): pass
    def log_llm_success(*args, **kwargs): pass
    def log_llm_error(*args, **kwargs): pass
    def log_llm_fallback(*args, **kwargs): pass
    def log_processing_step(*args, **kwargs): pass

# Vision Analyzer Integration (Hugging Face + Local Fallback)
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.vision_analyzer import analyze_image as vision_analyze_image, cloud_caption, local_caption, detect_objects, extract_colors
    VISION_ANALYZER_AVAILABLE = True
    logger.info("✓ Vision Analyzer: AVAILABLE (Hugging Face + Local BLIP/YOLOv8)")
except ImportError as e:
    VISION_ANALYZER_AVAILABLE = False
    logger.warning(f"⚠️ Vision Analyzer: NOT AVAILABLE ({e})")
    # Create dummy functions
    def vision_analyze_image(*args, **kwargs):
        return {"caption": "", "dominant_colors": [], "objects_detected": [], "analysis_source": "none"}
    def cloud_caption(*args, **kwargs): return None
    def local_caption(*args, **kwargs): return None
    def detect_objects(*args, **kwargs): return []
    def extract_colors(*args, **kwargs): return []

snapspeak_ai = Blueprint('snapspeak_ai', __name__, template_folder='templates')
CORS(snapspeak_ai)

# Use centralized LLM router (for GROQ_CONFIGURED) and local LLM fallback
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.llm_router import generate_text
    LLM_ROUTER_AVAILABLE = True
except ImportError as e:
    LLM_ROUTER_AVAILABLE = False
    logger.warning(f"⚠️ LLM router not available: {e}")
    def generate_text(*args, **kwargs):
        return {"response": "", "model": "none", "source": "none"}

# Groq initialization removed - using centralized router
GROQ_CONFIGURED = LLM_ROUTER_AVAILABLE

# Import security utilities
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.security import rate_limit_strict, rate_limit_api, validate_request, InputValidator
except ImportError:
    # Fallback if security utils not available
    def rate_limit_strict(*args, **kwargs):
        def decorator(f):
            return f
        return decorator
    def rate_limit_api(*args, **kwargs):
        def decorator(f):
            return f
        return decorator
    def validate_request(*args, **kwargs):
        def decorator(f):
            return f
        return decorator
    InputValidator = None

# Global configurations
FACE_DETECTION_CONFIDENCE_THRESHOLD = 0.85
COLOR_CLUSTER_COUNT = 8
IMAGE_RESIZE_DIMENSION = 150

# Initialize models with protobuf compatibility handling
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"✓ Using device: {device}")

model = None
processor = None
BLIP_AVAILABLE = False

if BlipForConditionalGeneration is not None and BlipProcessor is not None:
    try:
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        BLIP_AVAILABLE = True
        logger.info("✓ BLIP Model: LOADED")
    except Exception as e:
        model = None
        processor = None
        BLIP_AVAILABLE = False
        logger.error(f"✗ BLIP Model: FAILED - {str(e)}")
else:
    logger.warning("⚠️ BLIP Model: UNAVAILABLE (Transformers import failed)")

# Initialize face detection
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    logger.info("✓ DeepFace: AVAILABLE")
except ImportError:
    DEEPFACE_AVAILABLE = False
    logger.info("✓ DeepFace: NOT AVAILABLE (Using OpenCV fallback)")


def get_color_name(r, g, b):
    """Get approximate color name from RGB values"""
    colors = {
        'Black': [0, 0, 0],
        'White': [255, 255, 255],
        'Red': [255, 0, 0],
        'Green': [0, 255, 0],
        'Blue': [0, 0, 255],
        'Yellow': [255, 255, 0],
        'Cyan': [0, 255, 255],
        'Magenta': [255, 0, 255],
        'Orange': [255, 165, 0],
        'Purple': [128, 0, 128],
        'Pink': [255, 192, 203],
        'Brown': [165, 42, 42],
        'Gray': [128, 128, 128],
        'Olive': [128, 128, 0],
        'Navy': [0, 0, 128],
        'Teal': [0, 128, 128],
        'Maroon': [128, 0, 0]
    }
    
    closest_color = 'Unknown'
    min_distance = float('inf')
    
    for name, (cr, cg, cb) in colors.items():
        distance = ((r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2) ** 0.5
        if distance < min_distance:
            min_distance = distance
            closest_color = name
    
    brightness = (r + g + b) / 3
    saturation = max(r, g, b) - min(r, g, b)
    
    descriptor = ''
    if brightness < 50:
        descriptor = 'Dark '
    elif brightness > 200:
        descriptor = 'Light '
    elif saturation < 30:
        descriptor = 'Pale '
    
    return descriptor + closest_color


def advanced_vision_analysis(image):
    """Advanced image analysis using vision_analyzer module (Hugging Face + Local Fallback)"""
    if not VISION_ANALYZER_AVAILABLE:
        return None
    
    try:
        # Use the reusable vision analyzer to get base caption, colors, and objects
        vision_result = vision_analyze_image(image, n_colors=8)
        
        result = {
            'caption': vision_result.get('caption', ''),
            'dominant_colors': vision_result.get('dominant_colors', []),
            'objects_detected': vision_result.get('objects_detected', []),
            'analysis_source': vision_result.get('analysis_source', 'local'),
            'source': vision_result.get('analysis_source', 'local')
        }
        
        # If LLM router is available, generate a richer human-readable description
        if LLM_ROUTER_AVAILABLE:
            try:
                base_caption = result.get('caption', '')
                objects = result.get('objects_detected') or []
                colors = result.get('dominant_colors') or []
                
                context = {
                    "base_caption": base_caption,
                    "objects_detected": objects,
                    "dominant_colors": colors,
                }
                
                prompt = (
                    "You are an expert visual forensic analyst. "
                    "You will receive a brief machine-generated caption of an image, "
                    "a list of detected objects, and the dominant colors. "
                    "Using this information, write a detailed, accurate, and objective description "
                    "of what is visible in the image in 3-5 sentences. "
                    "Focus on visual details (people, objects, environment, actions, mood). "
                    "Do NOT guess identities or personal details; only describe what can be seen.\n\n"
                    f"Machine caption: \"{base_caption}\"\n"
                    f"Detected objects: {json.dumps(objects, ensure_ascii=False)}\n"
                    f"Dominant colors: {json.dumps(colors, ensure_ascii=False)}\n\n"
                    "Detailed description:"
                )
                
                log_llm_request("snapspeak_ai", "image_reasoning", prompt)
                llm_result = generate_text(
                    prompt=prompt,
                    app_name="snapspeak_ai",
                    task_type="image_reasoning",
                    system_prompt="You are a professional image forensics expert. Provide clear, factual, and detailed visual descriptions without making unverifiable assumptions about identity or intent.",
                    temperature=0.6,
                    max_tokens=220
                )
                
                if llm_result and llm_result.get('response'):
                    result['detailed_caption'] = llm_result['response'].strip()
                    result['analysis_source'] = llm_result.get('source', result['analysis_source'])
                    result['model_used'] = llm_result.get('model')
                    log_llm_success("snapspeak_ai", llm_result.get('model', 'unknown'))
                else:
                    logger.warning("LLM router did not return a detailed caption, using base caption only")
            except Exception as llm_err:
                logger.error(f"Error in detailed caption generation: {llm_err}")
                log_llm_error("snapspeak_ai", str(llm_err))
        
        return result
    except Exception as e:
        logger.error(f"✗ Vision analysis error: {e}")
        return None


def deep_metadata_extraction(image, image_bytes):
    """Extract comprehensive metadata with use case explanations"""
    try:
        metadata = {
            'basic_info': {},
            'technical_specs': {},
            'camera_settings': {},
            'gps_location': {},
            'software_info': {},
            'color_profile': {},
            'compression_info': {},
            'timestamps': {},
            'hidden_data': {}
        }
        
        use_cases = {}
        
        # Basic Information
        metadata['basic_info'] = {
            'Format': image.format or 'Unknown',
            'Mode': image.mode,
            'Size': f"{image.width}x{image.height}",
            'Aspect_Ratio': f"{round(image.width/image.height, 2)}:1",
            'Total_Pixels': f"{image.width * image.height:,}",
            'File_Size_Bytes': len(image_bytes),
            'File_Size_KB': f"{len(image_bytes) / 1024:.2f} KB",
            'File_Size_MB': f"{len(image_bytes) / (1024*1024):.2f} MB"
        }
        use_cases['basic_info'] = "Essential for determining image compatibility, storage requirements, and display optimization"
        
        # Use exifread for deeper extraction
        img_file = io.BytesIO(image_bytes)
        tags = exifread.process_file(img_file, details=True)
        
        # Camera Settings
        camera_tags = ['Image Make', 'Image Model', 'EXIF LensModel', 'EXIF FocalLength', 
                      'EXIF FNumber', 'EXIF ExposureTime', 'EXIF ISOSpeedRatings',
                      'EXIF ExposureProgram', 'EXIF MeteringMode', 'EXIF Flash',
                      'EXIF WhiteBalance', 'EXIF FocalLengthIn35mmFilm']
        
        for tag in camera_tags:
            if tag in tags:
                clean_tag = tag.split()[-1]
                metadata['camera_settings'][clean_tag] = str(tags[tag])
        
        if metadata['camera_settings']:
            use_cases['camera_settings'] = "Recreate similar photos, verify camera authenticity, analyze photography technique"
        
        # GPS Data
        gps_tags = {k: v for k, v in tags.items() if 'GPS' in k}
        if gps_tags:
            for tag, value in gps_tags.items():
                clean_tag = tag.replace('GPS ', '').replace('GPS', '')
                metadata['gps_location'][clean_tag] = str(value)
            use_cases['gps_location'] = "⚠️ PRIVACY RISK: Reveals exact location where photo was taken"
        
        # Timestamps
        time_tags = ['EXIF DateTimeOriginal', 'EXIF DateTimeDigitized', 'Image DateTime']
        for tag in time_tags:
            if tag in tags:
                clean_tag = tag.split()[-1]
                metadata['timestamps'][clean_tag] = str(tags[tag])
        
        if metadata['timestamps']:
            use_cases['timestamps'] = "Verify photo timeline, detect backdated images, establish chronology"
        
        # Software & Processing
        software_tags = ['Image Software', 'Image ProcessingSoftware', 'EXIF Software']
        for tag in software_tags:
            if tag in tags:
                clean_tag = tag.split()[-1]
                metadata['software_info'][clean_tag] = str(tags[tag])
        
        if metadata['software_info']:
            use_cases['software_info'] = "Detect edited images, identify editing tools, trace manipulation history"
        
        # Technical Specifications
        tech_tags = ['EXIF ColorSpace', 'EXIF ExifImageWidth', 'EXIF ExifImageHeight',
                    'EXIF Compression', 'EXIF YCbCrPositioning', 'Image Orientation',
                    'EXIF ExifVersion', 'EXIF ComponentsConfiguration']
        for tag in tech_tags:
            if tag in tags:
                clean_tag = tag.split()[-1]
                metadata['technical_specs'][clean_tag] = str(tags[tag])
        
        # Color Profile Analysis
        if 'icc_profile' in image.info:
            metadata['color_profile']['ICC_Profile'] = "Present"
            metadata['color_profile']['Profile_Size'] = f"{len(image.info['icc_profile'])} bytes"
            use_cases['color_profile'] = "Ensure accurate color reproduction, detect professional vs amateur photos"
        
        # Compression Analysis
        if image.format == 'JPEG':
            try:
                qtables = image.quantization
                if qtables:
                    metadata['compression_info']['Quality_Tables'] = f"{len(qtables)} present"
                    avg_quality = sum(sum(q) for q in qtables.values()) / (len(qtables) * 64)
                    metadata['compression_info']['Estimated_Quality'] = f"{min(100, int(avg_quality))}%"
                    use_cases['compression_info'] = "Detect re-compressed images, estimate original quality"
            except:
                pass
        
        # Entropy Analysis
        if image.mode == 'RGB':
            img_array = np.array(image)
            hist = cv2.calcHist([img_array], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
            hist = hist.flatten()
            hist = hist / (hist.sum() + 1e-10)
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            metadata['hidden_data']['Image_Entropy'] = f"{entropy:.4f}"
            metadata['hidden_data']['Entropy_Level'] = "High" if entropy > 7.5 else "Normal"
            use_cases['hidden_data'] = "High entropy may indicate hidden data or steganography"
        
        # Thumbnail analysis
        if hasattr(image, 'thumbnail_size'):
            metadata['hidden_data']['Embedded_Thumbnail'] = "Yes"
        
        # Clean empty sections
        metadata = {k: v for k, v in metadata.items() if v}
        
        return {
            'metadata': metadata,
            'use_cases': use_cases,
            'extraction_depth': 'DEEP',
            'sections_found': len(metadata)
        }
        
    except Exception as e:
        logger.error(f"Error in deep metadata extraction: {str(e)}")
        traceback.print_exc()
        return {
            'metadata': {},
            'use_cases': {},
            'error': str(e)
        }


def convert_to_json_serializable(obj):
    """Convert numpy and other non-serializable types to JSON-compatible types"""
    if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64,
                        np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, bool):
        return obj
    return obj


def detect_lsb_steganography(img_array):
    """Detect actual LSB steganography patterns"""
    try:
        lsb_plane = img_array & 1
        flat_lsb = lsb_plane.ravel()
        
        # Check bit transitions
        transitions = np.abs(np.diff(flat_lsb.astype(int)))
        transition_rate = np.mean(transitions)
        
        # Real LSB stego should have ~0.5 transition rate
        is_suspicious = 0.48 < transition_rate < 0.52
        
        # Check for payload indicators in first 32 bits
        if len(flat_lsb) >= 32:
            header_bits = flat_lsb[:32]
            try:
                potential_length = int(''.join(map(str, header_bits[:24])), 2)
                if 0 < potential_length < len(flat_lsb) // 8:
                    is_suspicious = True
            except:
                pass
        
        return is_suspicious, transition_rate
    except:
        return False, 0.5


def advanced_steganography_detection(image):
    """Improved multi-method steganography detection with better accuracy"""
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_array = np.array(image)
        results = {
            'basic': {},
            'advanced': {},
            'overall_risk': 'LOW',
            'confidence': 0.0,
            'methods_detected': []
        }
        
        suspicious_flags = []
        
        # 1. Enhanced LSB Analysis
        lsb = img_array & 1
        lsb_flat = lsb.ravel()
        lsb_counts = np.bincount(lsb_flat, minlength=2)
        lsb_entropy = 0.0
        
        if lsb_counts.sum() > 0:
            lsb_probs = lsb_counts / lsb_counts.sum()
            lsb_probs = lsb_probs[lsb_probs > 0]
            lsb_entropy = -np.sum(lsb_probs * np.log2(lsb_probs))
        
        # Detect actual LSB steganography
        lsb_detected, lsb_transition = detect_lsb_steganography(img_array)
        if lsb_detected:
            suspicious_flags.append(('LSB Pattern Match', 0.5))
            results['methods_detected'].append('LSB Steganography Pattern')
        
        results['basic']['LSB_Entropy'] = float(lsb_entropy)
        results['basic']['LSB_Transition_Rate'] = float(lsb_transition)
        results['basic']['LSB_Pattern_Detected'] = bool(lsb_detected)
        results['basic']['LSB_Suspicious'] = bool(lsb_entropy > 0.999 or lsb_detected)
        
        # 2. Chi-Square Attack (stricter)
        observed = np.bincount(img_array.ravel(), minlength=256).astype(float)
        total_pixels = img_array.size
        expected = np.full(256, total_pixels / 256.0)
        
        chi_square = np.sum((observed - expected) ** 2 / (expected + 1e-10))
        chi_threshold = 2000 if total_pixels > 1000000 else 1500
        chi_suspicious = chi_square > chi_threshold
        
        results['advanced']['Chi_Square_Value'] = float(chi_square)
        results['advanced']['Chi_Square_Suspicious'] = bool(chi_suspicious)
        
        if chi_suspicious:
            suspicious_flags.append(('Chi-Square', 0.25))
            results['methods_detected'].append('Chi-Square Anomaly')
        
        # 3. Improved RS Steganalysis
        def calculate_smoothness(pixels):
            diff = np.abs(np.diff(pixels.ravel().astype(float)))
            return np.mean(diff) if len(diff) > 0 else 0
        
        rs_ratios = []
        for _ in range(5):
            mask = np.random.rand(*img_array.shape[:2]) > 0.5
            if mask.sum() > 0 and (~mask).sum() > 0:
                regular_smooth = calculate_smoothness(img_array[mask])
                singular_smooth = calculate_smoothness(img_array[~mask])
                if singular_smooth > 0:
                    rs_ratios.append(regular_smooth / singular_smooth)
        
        rs_ratio = np.mean(rs_ratios) if rs_ratios else 1.0
        rs_std = np.std(rs_ratios) if len(rs_ratios) > 1 else 0
        
        rs_suspicious = (abs(rs_ratio - 1.0) > 0.25) or (rs_std > 0.15)
        
        results['advanced']['RS_Ratio'] = float(rs_ratio)
        results['advanced']['RS_Std_Dev'] = float(rs_std)
        results['advanced']['RS_Suspicious'] = bool(rs_suspicious)
        
        if rs_suspicious:
            suspicious_flags.append(('RS Analysis', 0.25))
            results['methods_detected'].append('RS Analysis Anomaly')
        
        # 4. Histogram Analysis
        channels_variance = []
        for channel in range(3):
            hist = cv2.calcHist([img_array], [channel], None, [256], [0, 256])
            channels_variance.append(float(np.var(hist)))
        
        hist_variance = np.var(channels_variance)
        hist_threshold = 1e11 if total_pixels > 1000000 else 5e10
        hist_suspicious = hist_variance > hist_threshold
        
        results['advanced']['Histogram_Variance'] = float(hist_variance)
        results['advanced']['Histogram_Suspicious'] = bool(hist_suspicious)
        
        if hist_suspicious:
            suspicious_flags.append(('Histogram', 0.2))
            results['methods_detected'].append('Histogram Anomaly')
        
        # 5. Pixel Pair Analysis
        h_pairs = img_array[:, :-1] ^ img_array[:, 1:]
        v_pairs = img_array[:-1, :] ^ img_array[1:, :]
        
        h_entropy = 0.0
        v_entropy = 0.0
        
        if h_pairs.size > 0:
            h_flat = h_pairs.ravel()
            h_counts = np.bincount(h_flat, minlength=256)
            if h_counts.sum() > 0:
                h_probs = h_counts / h_counts.sum()
                h_probs = h_probs[h_probs > 0]
                h_entropy = -np.sum(h_probs * np.log2(h_probs))
        
        if v_pairs.size > 0:
            v_flat = v_pairs.ravel()
            v_counts = np.bincount(v_flat, minlength=256)
            if v_counts.sum() > 0:
                v_probs = v_counts / v_counts.sum()
                v_probs = v_probs[v_probs > 0]
                v_entropy = -np.sum(v_probs * np.log2(v_probs))
        
        avg_pair_entropy = (h_entropy + v_entropy) / 2
        pair_suspicious = avg_pair_entropy > 6.5
        
        results['advanced']['Pair_Entropy'] = float(avg_pair_entropy)
        results['advanced']['Pair_Suspicious'] = bool(pair_suspicious)
        
        if pair_suspicious:
            suspicious_flags.append(('Pixel Pairs', 0.2))
            results['methods_detected'].append('Pixel Pair Anomaly')
        
        # Calculate weighted confidence with stricter criteria
        total_weight = sum(weight for _, weight in suspicious_flags)
        confidence = min(100.0, total_weight * 100)
        
        results['confidence'] = float(confidence)
        
        # Much stricter risk levels
        if confidence >= 70 and len(suspicious_flags) >= 3:
            results['overall_risk'] = 'HIGH'
        elif confidence >= 40 and len(suspicious_flags) >= 2:
            results['overall_risk'] = 'MEDIUM'
        else:
            results['overall_risk'] = 'LOW'
        
        return results
        
    except Exception as e:
        logger.error(f"Error in steganography detection: {str(e)}")
        traceback.print_exc()
        return {
            'basic': {},
            'advanced': {},
            'overall_risk': 'UNKNOWN',
            'confidence': 0.0,
            'methods_detected': [],
            'error': str(e)
        }


def enhanced_color_analysis(image):
    """Advanced color analysis with color theory insights"""
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize for processing
        thumb = image.copy()
        thumb.thumbnail((IMAGE_RESIZE_DIMENSION, IMAGE_RESIZE_DIMENSION))
        pixels = np.array(thumb).reshape(-1, 3)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=COLOR_CLUSTER_COUNT, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        colors = kmeans.cluster_centers_
        labels = kmeans.labels_
        counts = np.bincount(labels)
        total_pixels = sum(counts)
        
        # Sort by frequency
        sorted_indices = np.argsort(counts)[::-1]
        
        color_info = []
        for idx in sorted_indices:
            r, g, b = map(int, colors[idx])
            hex_color = f'#{r:02x}{g:02x}{b:02x}'
            percentage = (counts[idx] / total_pixels) * 100
            
            # Calculate HSV for color theory insights
            hsv = cv2.cvtColor(np.uint8([[colors[idx]]]), cv2.COLOR_RGB2HSV)[0][0]
            h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
            
            # Color temperature
            temp = 'Warm' if (h < 30 or h > 150) else 'Cool'
            
            # Color name approximation
            color_name = get_color_name(r, g, b)
            
            color_info.append({
                'hex': hex_color,
                'rgb': f'rgb({r},{g},{b})',
                'hsv': f'hsv({h},{s},{v})',
                'hsv_values': {'h': int(h), 's': int(s), 'v': int(v)},
                'percentage': float(round(percentage, 2)),
                'name': color_name,
                'temperature': temp,
                'saturation': 'High' if s > 150 else 'Medium' if s > 80 else 'Low',
                'brightness': 'High' if v > 180 else 'Medium' if v > 100 else 'Low'
            })
        
        # Overall color scheme analysis
        scheme = determine_color_scheme(color_info[:5])
        
        return {
            'colors': color_info,
            'color_scheme': scheme,
            'total_colors_analyzed': COLOR_CLUSTER_COUNT,
            'accuracy': 'HIGH'
        }
        
    except Exception as e:
        logger.error(f"Error in color analysis: {str(e)}")
        traceback.print_exc()
        return {'colors': [], 'error': str(e)}


def determine_color_scheme(colors):
    """Determine overall color scheme"""
    if not colors:
        return 'Unknown'
    
    saturations = [c.get('hsv_values', {}).get('s', 0) for c in colors[:3]]
    
    if all(s < 80 for s in saturations):
        return 'Monochromatic/Desaturated'
    elif any(s > 150 for s in saturations):
        return 'Vibrant/Bold'
    else:
        return 'Balanced/Natural'


def multiple_hash_generation(image):
    """Generate multiple types of image hashes"""
    try:
        hashes = {
            'average_hash': str(imagehash.average_hash(image)),
            'perceptual_hash': str(imagehash.phash(image)),
            'difference_hash': str(imagehash.dhash(image)),
            'wavelet_hash': str(imagehash.whash(image)),
            'color_hash': str(imagehash.colorhash(image))
        }
        
        # MD5 and SHA256 of image data
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        
        hashes['md5'] = hashlib.md5(img_bytes).hexdigest()
        hashes['sha256'] = hashlib.sha256(img_bytes).hexdigest()
        
        return hashes
    except Exception as e:
        logger.error(f"Error generating hashes: {str(e)}")
        return {}


def object_detection_analysis(image):
    """Basic object detection and scene understanding"""
    try:
        # Convert to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Edge detection
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Brightness and contrast
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        # Blur detection
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = 'Sharp' if laplacian_var > 100 else 'Slightly Blurred' if laplacian_var > 50 else 'Blurred'
        
        return {
            'edge_density': float(edge_density),
            'avg_brightness': float(brightness),
            'contrast': float(contrast),
            'blur_assessment': blur_score,
            'laplacian_variance': float(laplacian_var)
        }
    except Exception as e:
        logger.error(f"Error in object detection: {str(e)}")
        return {}


def enhanced_face_detection(image):
    """Enhanced face detection with analysis"""
    try:
        np_image = np.array(image)
        
        if len(np_image.shape) == 2:
            np_image = cv2.cvtColor(np_image, cv2.COLOR_GRAY2RGB)
        elif np_image.shape[2] == 4:
            np_image = cv2.cvtColor(np_image, cv2.COLOR_RGBA2RGB)

        face_locations = []

        # Try DeepFace first if available
        if DEEPFACE_AVAILABLE:
            try:
                # DeepFace works better with file paths to avoid KerasTensor issues
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                    # Convert numpy array to PIL Image and save
                    pil_image = Image.fromarray(np_image)
                    pil_image.save(tmp_file.name, 'JPEG', quality=95)
                    tmp_path = tmp_file.name
                
                try:
                    # Use file path instead of numpy array to avoid KerasTensor issues
                    faces = DeepFace.extract_faces(
                        tmp_path,
                        detector_backend='retinaface',
                        enforce_detection=False,
                        align=True
                    )

                    for face in faces:
                        # Handle both dict and array return types
                        if isinstance(face, dict):
                            confidence = face.get('confidence', 0.9)
                            facial_area = face.get('facial_area', {})
                        else:
                            # If face is just an array, use default values
                            confidence = 0.9
                            facial_area = {'x': 0, 'y': 0, 'w': 0, 'h': 0}
                        
                        if confidence > FACE_DETECTION_CONFIDENCE_THRESHOLD and facial_area:
                            face_locations.append({
                                'x': int(facial_area.get('x', 0)),
                                'y': int(facial_area.get('y', 0)),
                                'width': int(facial_area.get('w', 0)),
                                'height': int(facial_area.get('h', 0)),
                                'confidence': float(confidence),
                                'detector': 'retinaface'
                            })
                    logger.info(f"  DeepFace detected {len(face_locations)} faces")
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"  DeepFace detection failed: {str(e)}")

        # Fallback to OpenCV if DeepFace failed or unavailable
        if not face_locations:
            try:
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                for (x, y, w, h) in faces:
                    face_locations.append({
                        'x': int(x), 
                        'y': int(y), 
                        'width': int(w), 
                        'height': int(h),
                        'confidence': 0.8, 
                        'detector': 'opencv'
                    })
                logger.info(f"  OpenCV detected {len(face_locations)} faces")
            except Exception as e:
                logger.warning(f"  OpenCV detection failed: {str(e)}")

        # Perform advanced face analysis if faces found
        face_analysis = None
        if face_locations and GROQ_CONFIGURED:
            try:
                logger.info(f"  Running advanced face analysis...")
                face_analysis = analyze_face_with_gemini(image)
            except Exception as e:
                logger.error(f"  LLM face analysis failed: {str(e)}")

        return convert_to_json_serializable({
            'count': len(face_locations), 
            'locations': face_locations,
            'face_analysis': face_analysis,
            'detector_used': face_locations[0]['detector'] if face_locations else 'none'
        })
        
    except Exception as e:
        logger.error(f"Error in face detection: {str(e)}")
        traceback.print_exc()
        return {'count': 0, 'locations': [], 'error': str(e)}


def analyze_face_with_gemini(image):
    """Advanced face analysis using LLM router (Groq/Ollama) - function name kept for backward compatibility"""
    if not GROQ_CONFIGURED:
        return None
    
    try:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        prompt = """Analyze any faces in this image based on the description I will give you. Provide:

1. PERSON DESCRIPTION: Age range, gender (if identifiable), ethnicity, distinctive features
2. FACIAL EXPRESSIONS: Emotions, mood, what they might be feeling
3. CONTEXT: Setting, what the person might be doing, social context
4. APPEARANCE: Clothing style, accessories, grooming, overall presentation
5. IDENTIFICATION CLUES: Any visible text, logos, badges, or identifying markers (DO NOT attempt to identify specific individuals)
6. AUTHENTICITY: Does this appear to be a real photograph or potentially AI-generated/manipulated?

IMPORTANT: Do NOT attempt to identify specific individuals by name. Focus on observable characteristics only."""

        # Use LLM router for better timeout handling and retry logic
        if not LLM_ROUTER_AVAILABLE:
            logger.warning("LLM router not available for face analysis.")
            return None

        # For now we only pass a generic description; bounding-box level details could be threaded in future.
        description = "The previous step detected one or more faces in the image. Provide only high-level, non-identifying observations."

        try:
            # Use router with prefer_local=True to ensure local LLM is used
            # Router has built-in retry logic and better timeout handling
            llm_result = generate_text(
                prompt=f"{prompt}\n\nImage context: {description}",
                app_name="snapspeak_ai",
                task_type="image_reasoning",
                system_prompt="You are an expert facial analysis AI. Provide detailed, structured observations about faces without identifying specific people.",
                temperature=0.7,
                max_tokens=1024,
                prefer_local=True  # Prefer local LLM for face analysis
            )
            
            if llm_result and llm_result.get('response'):
                analysis_text = llm_result['response'].strip()
                source = llm_result.get('source', 'unknown')
                model_used = llm_result.get('model', 'unknown')
                
                logger.info(f"✓ Face analysis generated via {source} ({model_used})")
                return {
                    'analysis': analysis_text,
                    'source': source,
                    'model_used': model_used
                }

            logger.warning("LLM router did not return a successful face analysis response.")
            return None
        except Exception as llm_error:
            logger.error(f"LLM face analysis error: {str(llm_error)}")
            return None
    except Exception as e:
        logger.error(f"Error in analyze_face_with_gemini: {e}")
        traceback.print_exc()
        return None


def detect_ai_generation(image):
    """Detect if image is AI-generated"""
    try:
        results = {
            'assessment': 'UNKNOWN',
            'confidence': 0.0,
            'indicators': []
        }
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # 1. High-frequency noise analysis
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        high_freq_ratio = np.sum(magnitude_spectrum > np.percentile(magnitude_spectrum, 90)) / magnitude_spectrum.size
        
        if high_freq_ratio < 0.05:
            results['indicators'].append('Suspiciously low high-frequency content')
            results['confidence'] += 15
        
        # 2. Color distribution uniformity
        if len(img_array.shape) == 3:
            color_std = np.std([np.std(img_array[:,:,i]) for i in range(3)])
            if color_std < 10:
                results['indicators'].append('Uniform color distribution')
                results['confidence'] += 10
        
        # 3. Texture repetition
        patches = []
        h, w = img_array.shape[:2]
        patch_size = 32
        for i in range(0, h - patch_size, patch_size):
            for j in range(0, w - patch_size, patch_size):
                patch = img_array[i:i+patch_size, j:j+patch_size]
                patches.append(patch.flatten())
        
        if len(patches) > 1:
            similarities = []
            for i in range(min(10, len(patches))):
                for j in range(i+1, min(10, len(patches))):
                    corr = np.corrcoef(patches[i], patches[j])[0,1]
                    similarities.append(corr)
            
            if similarities and np.mean(similarities) > 0.7:
                results['indicators'].append('High texture repetition')
                results['confidence'] += 20
        
        # 4. Edge consistency
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        if edge_density < 0.05 or edge_density > 0.3:
            results['indicators'].append('Unusual edge characteristics')
            results['confidence'] += 10
        
        # Determine assessment
        if results['confidence'] >= 40:
            results['assessment'] = 'LIKELY_AI_GENERATED'
        elif results['confidence'] >= 20:
            results['assessment'] = 'POSSIBLY_AI_GENERATED'
        else:
            results['assessment'] = 'LIKELY_REAL_PHOTO'
        
        return results
        
    except Exception as e:
        logger.error(f"Error in AI detection: {str(e)}")
        return {
            'assessment': 'UNKNOWN',
            'confidence': 0.0,
            'indicators': [],
            'error': str(e)
        }


@torch.no_grad()
def generate_caption(image):
    """Generate image caption using BLIP model"""
    try:
        if not BLIP_AVAILABLE or model is None or processor is None:
            return "BLIP model not available (protobuf compatibility issue or model failed to load)"
        
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        output_ids = model.generate(pixel_values, max_length=50, num_beams=4)
        return processor.decode(output_ids[0], skip_special_tokens=True)
    except Exception as e:
        logger.error(f"Error in caption generation: {str(e)}")
        return "Error generating caption"


@snapspeak_ai.route('/')
def index():
    return render_template('snapspeak.html')


@snapspeak_ai.route('/api/analyze/', methods=['POST'])
@rate_limit_strict(requests_per_minute=5, requests_per_hour=50)  # Strict limit for image processing
def analyze_image():
    """
    Main image analysis endpoint
    OWASP: Rate limited, file validation
    """
    start_time = time.time()
    
    try:
        # Validate file upload
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if not file.filename:
            return jsonify({'error': 'No selected file'}), 400
        
        # Validate filename
        try:
            filename = InputValidator.validate_filename(file.filename, 'filename', required=True)
        except Exception as e:
            return jsonify({'error': f'Invalid filename: {str(e)}'}), 400
        
        # Load image
        try:
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Perform comprehensive analyses
        logger.info(f"\n{'='*50}")
        logger.info(f"Analyzing image: {file.filename}")
        logger.info(f"{'='*50}")
        
        # Vision analysis (Hugging Face + Local Fallback)
        vision_analysis = advanced_vision_analysis(image) if VISION_ANALYZER_AVAILABLE else None
        if vision_analysis:
            caption = vision_analysis.get('detailed_caption') or vision_analysis.get('caption', '')
            logger.info(f"✓ Vision analysis complete (source: {vision_analysis.get('analysis_source', 'unknown')})")
            logger.info(f"  - Caption: {caption[:160]}..." if caption else "  - No caption generated")
            logger.info(f"  - Objects detected: {len(vision_analysis.get('objects_detected', []))}")
            logger.info(f"  - Dominant colors: {len(vision_analysis.get('dominant_colors', []))}")
        else:
            # Fallback to existing BLIP caption if vision analyzer not available
            caption = generate_caption(image)
            logger.info(f"✓ Caption generated (fallback BLIP)")
        
        # Deep metadata
        metadata_result = deep_metadata_extraction(image, image_bytes)
        logger.info(f"✓ Metadata extracted: {metadata_result.get('sections_found', 0)} sections")
        
        # Enhanced color analysis (use vision_analyzer if available, else fallback)
        if vision_analysis and vision_analysis.get('dominant_colors'):
            color_data = {
                'colors': vision_analysis['dominant_colors'],
                'source': 'vision_analyzer'
            }
            logger.info(f"✓ Color analysis: {len(color_data.get('colors', []))} colors (from vision_analyzer)")
        else:
            color_data = enhanced_color_analysis(image)
            logger.info(f"✓ Color analysis: {len(color_data.get('colors', []))} colors (fallback)")
        
        # Advanced steganography
        steg_result = advanced_steganography_detection(image)
        logger.info(f"✓ Steganography check: {steg_result['overall_risk']} risk ({steg_result['confidence']:.1f}% confidence)")
        
        # Multiple hashes
        hashes = multiple_hash_generation(image)
        logger.info(f"✓ Hashes generated: {len(hashes)} types")
        
        # Object analysis (use vision_analyzer if available, else fallback)
        if vision_analysis and vision_analysis.get('objects_detected'):
            object_analysis = {
                'objects': vision_analysis['objects_detected'],
                'count': len(vision_analysis['objects_detected']),
                'source': 'vision_analyzer'
            }
            logger.info(f"✓ Object detection: {object_analysis['count']} objects (from vision_analyzer)")
        else:
            object_analysis = object_detection_analysis(image)
            logger.info(f"✓ Object detection complete (fallback)")
        
        # Face detection with advanced analysis
        face_data = enhanced_face_detection(image)
        logger.info(f"✓ Face detection: {face_data.get('count', 0)} faces found")
        if face_data.get('face_analysis'):
            logger.info(f"  ✓ Advanced face analysis completed")
        
        # AI Generation Detection
        ai_detection = detect_ai_generation(image)
        logger.info(f"✓ AI Detection: {ai_detection['assessment']} ({ai_detection['confidence']:.1f}% confidence)")
        
        processing_time = time.time() - start_time
        logger.info(f"\n⏱  Total processing time: {processing_time:.2f}s")
        logger.info(f"{'='*50}\n")
        
        analysis_results = {
            'basic_caption': caption,
            'vision_analysis': vision_analysis,  # Vision analyzer + optional detailed_caption
            'gemini_analysis': vision_analysis,  # Backward-compatible field used by frontend
            'ai_detection': ai_detection,
            'metadata': metadata_result,
            'color_analysis': color_data,
            'steganography': steg_result,
            'image_hashes': hashes,
            'technical_analysis': object_analysis,
            'faces': face_data,
            'processing_time': float(processing_time),
            'gemini_enabled': GROQ_CONFIGURED,  # Keep field name for backward compatibility
            'deepface_available': DEEPFACE_AVAILABLE
        }
        
        # Convert all data to JSON serializable format
        analysis_results = convert_to_json_serializable(analysis_results)
        
        return jsonify(analysis_results)
        
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Error in analyze_image: {str(e)}")
        logger.error(f"Traceback: {error_trace}")
        return jsonify({
            'error': str(e),
            'traceback': error_trace if current_app.debug else 'Enable debug mode for traceback'
        }), 500


# ============================================================
# ADVANCED FORENSICS & UTILITY HELPERS
# ============================================================

def _load_image_from_request(req, field_name='file'):
    """Shared helper to safely load an image from a Flask request"""
    if field_name not in req.files:
        return None, None, jsonify({'error': f'No file field "{field_name}" provided'}), 400

    file = req.files[field_name]
    if not file.filename:
        return None, None, jsonify({'error': 'No selected file'}), 400

    # Validate filename when InputValidator is available
    if InputValidator is not None:
        try:
            InputValidator.validate_filename(file.filename, 'filename', required=True)
        except Exception as e:
            return None, None, jsonify({'error': f'Invalid filename: {str(e)}'}), 400

    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
    except Exception:
        return None, None, jsonify({'error': 'Invalid image file'}), 400

    return image, image_bytes, None, None


def _safe_get_json():
    """Safe JSON body extraction"""
    try:
        return request.get_json(force=True, silent=True) or {}
    except Exception:
        return {}


# ----------------- OCR & TEXT ANALYSIS ----------------------
try:
    import pytesseract
    OCR_AVAILABLE = True
    logger.info("✓ OCR (pytesseract): AVAILABLE")
except Exception as e:
    OCR_AVAILABLE = False
    logger.warning(f"⚠️ OCR (pytesseract): NOT AVAILABLE ({e})")


def perform_ocr(image, languages="eng"):
    """Perform OCR with optional multi-language support"""
    if not OCR_AVAILABLE:
        return {
            'text': '',
            'languages': [],
            'engine': 'none',
            'error': 'pytesseract not installed'
        }
    try:
        if image.mode not in ['L', 'RGB']:
            image = image.convert('RGB')
        config = ''
        text = pytesseract.image_to_string(image, lang=languages, config=config)
        return {
            'text': text.strip(),
            'languages': languages.split('+') if languages else ['eng'],
            'engine': 'pytesseract',
            'confidence': None
        }
    except Exception as e:
        logger.error(f"OCR error: {e}")
        return {
            'text': '',
            'languages': languages.split('+') if languages else ['eng'],
            'engine': 'pytesseract',
            'error': str(e)
        }


# ----------------- QUALITY METRICS -------------------------
def compute_technical_quality(image):
    """Compute basic no-reference quality metrics (BRISQUE/NIQE stubs, sharpness, noise)."""
    try:
        if image.mode != 'L':
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        else:
            gray = np.array(image)

        # Sharpness via Laplacian variance
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Noise estimate via local variance
        noise = cv2.GaussianBlur(gray, (3, 3), 0)
        noise = gray.astype('float32') - noise.astype('float32')
        noise_level = float(np.std(noise))

        # Dynamic range
        min_val, max_val = int(np.min(gray)), int(np.max(gray))
        dynamic_range = max_val - min_val

        # Simple BRISQUE/NIQE/PIQE placeholders
        return {
            'sharpness_laplacian_var': float(lap_var),
            'noise_std': noise_level,
            'dynamic_range': int(dynamic_range),
            'min_intensity': min_val,
            'max_intensity': max_val,
            'brisque_score': None,
            'niqe_score': None,
            'piqe_score': None,
            'implementation_note': 'Classical metrics implemented; BRISQUE/NIQE/PIQE fields reserved for future ML models.'
        }
    except Exception as e:
        logger.error(f"Quality metric error: {e}")
        return {'error': str(e)}


def compute_aesthetic_score(image, color_analysis=None, object_analysis=None, faces=None):
    """
    Lightweight heuristic aesthetic assessment.
    This is intentionally interpretable and can be replaced by ML models later.
    """
    try:
        if image.mode != 'L':
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        else:
            gray = np.array(image)

        h, w = gray.shape[:2]

        # Rule of thirds: check if brightest region lies near thirds intersections
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray)
        thirds_x = [w / 3, 2 * w / 3]
        thirds_y = [h / 3, 2 * h / 3]
        bx, by = max_loc
        dist_to_thirds = min(
            np.hypot(bx - tx, by - ty)
            for tx in thirds_x for ty in thirds_y
        )
        thirds_score = max(0.0, 1.0 - dist_to_thirds / max(w, h))

        # Subject prominence: use edge density near center vs edges
        edges = cv2.Canny(gray, 50, 150)
        cy, cx = h // 2, w // 2
        center_r = min(h, w) // 6
        Y, X = np.ogrid[:h, :w]
        center_mask = (X - cx) ** 2 + (Y - cy) ** 2 <= center_r ** 2
        edge_center = np.mean(edges[center_mask] > 0) if center_mask.any() else 0
        edge_global = np.mean(edges > 0)
        subject_prominence = edge_center / (edge_global + 1e-6) if edge_global > 0 else 1.0

        # Color harmony from color_analysis if provided
        harmony = 'Unknown'
        if isinstance(color_analysis, dict):
            harmony = color_analysis.get('color_scheme', 'Unknown')

        # Simple aggregate aesthetic score
        score = float(
            min(100.0, max(0.0, 60 * thirds_score + 40 * min(subject_prominence, 2.0) / 2.0))
        )

        return {
            'aesthetic_score': score,
            'rule_of_thirds_score': float(thirds_score),
            'subject_prominence_ratio': float(subject_prominence),
            'color_harmony': harmony,
            'notes': 'Heuristic aesthetic estimate; hook for future learned models.'
        }
    except Exception as e:
        logger.error(f"Aesthetic score error: {e}")
        return {'error': str(e)}


# ----------------- CAMERA FINGERPRINTING (PRNU SCAFFOLD) ----
def estimate_prnu_signature(image):
    """
    Very lightweight PRNU-style sensor noise estimate.
    Full PRNU matching against large databases would be implemented as an external service.
    """
    try:
        if image.mode != 'L':
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        else:
            gray = np.array(image)

        # Denoise then subtract to approximate sensor pattern
        denoised = cv2.GaussianBlur(gray, (3, 3), 0)
        residual = gray.astype('float32') - denoised.astype('float32')
        norm_residual = residual / (np.std(residual) + 1e-6)

        # Downsample to fixed size for compact fingerprint
        target_size = (64, 64)
        prnu_map = cv2.resize(norm_residual, target_size, interpolation=cv2.INTER_AREA)
        prnu_flat = prnu_map.flatten().astype('float32')

        # Hash-like compact representation for identification
        prnu_hash = hashlib.sha256(prnu_flat.tobytes()).hexdigest()

        return {
            'prnu_hash': prnu_hash,
            'fingerprint_shape': list(prnu_map.shape),
            'approximate': True,
            'note': 'Approximate PRNU-like fingerprint; robust matching requires dedicated service and database.'
        }
    except Exception as e:
        logger.error(f"PRNU estimation error: {e}")
        return {'error': str(e)}


def analyze_edit_software_from_metadata(metadata_sections):
    """Infer editing software and possible edit history from metadata."""
    software_info = metadata_sections.get('software_info', {})
    software_str = ' '.join(str(v) for v in software_info.values()).lower()
    hints = []
    timeline = []

    known_tools = ['photoshop', 'lightroom', 'gimp', 'snapseed', 'vsco', 'instagram', 'pixelmator']
    for tool in known_tools:
        if tool in software_str:
            hints.append(f'Evidence of editing with {tool.title()} or related tools.')

    timestamps = metadata_sections.get('timestamps', {})
    if timestamps:
        for k, v in timestamps.items():
            timeline.append({'tag': k, 'value': v})

    return {
        'software_string': software_str,
        'software_hints': hints,
        'edit_timeline': timeline
    }


# ----------------- ADVANCED STEGO HELPERS -------------------
def dct_stego_analysis(image):
    """JPEG DCT-based stego heuristics for F5/OutGuess-style methods."""
    try:
        if image.format != 'JPEG':
            return {'supported': False, 'reason': 'DCT analysis only for JPEG images.'}

        # Re-encode to JPEG in-memory to access DCT coefficients
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='JPEG', quality=95)

        # Very lightweight heuristic using JPEG quantization if available
        try:
            qtables = image.quantization
            q_stats = {k: {'min': int(np.min(v)), 'max': int(np.max(v)), 'mean': float(np.mean(v))}
                       for k, v in qtables.items()}
        except Exception:
            qtables = None
            q_stats = {}

        return {
            'supported': True,
            'qtables_present': bool(qtables),
            'qtables_stats': q_stats,
            'note': 'Placeholder DCT analysis; full F5/OutGuess detection requires specialized models.'
        }
    except Exception as e:
        logger.error(f"DCT stego analysis error: {e}")
        return {'error': str(e)}


def metadata_stego_analysis(metadata_sections):
    """Look for suspicious patterns in EXIF/IPTC/XMP-like fields."""
    try:
        suspicious_fields = []
        total_bytes = 0

        for section_name, section in metadata_sections.items():
            if not isinstance(section, dict):
                continue
            for key, value in section.items():
                s_val = str(value)
                total_bytes += len(s_val.encode('utf-8', errors='ignore'))
                if len(s_val) > 512:
                    suspicious_fields.append({
                        'section': section_name,
                        'field': key,
                        'reason': 'Unusually long text field (possible payload).',
                        'length': len(s_val)
                    })

        return {
            'total_metadata_bytes': total_bytes,
            'suspicious_fields': suspicious_fields
        }
    except Exception as e:
        logger.error(f"Metadata stego analysis error: {e}")
        return {'error': str(e)}


def extract_lsb_payload(image, max_bytes=4096):
    """Attempt naive LSB payload extraction from RGB images."""
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        arr = np.array(image)
        bits = (arr & 1).flatten().astype(np.uint8)
        num_bits = min(len(bits), max_bytes * 8)
        bits = bits[:num_bits]
        bytes_arr = np.packbits(bits)
        hex_preview = binascii.hexlify(bytes_arr[:64]).decode('ascii')
        return {
            'payload_preview_hex': hex_preview,
            'payload_bytes_extracted': int(len(bytes_arr)),
            'note': 'Naive LSB extraction; decoding/format recognition performed separately.'
        }
    except Exception as e:
        logger.error(f"LSB extraction error: {e}")
        return {'error': str(e)}


# ============================================================
# API ENDPOINTS: FORENSICS
# ============================================================


@snapspeak_ai.route('/api/forensics/camera-fingerprint', methods=['POST'])
@rate_limit_strict(requests_per_minute=5, requests_per_hour=50)
def api_camera_fingerprint():
    """Estimate camera fingerprint (PRNU-style) and related metadata."""
    image, image_bytes, err_resp, status = _load_image_from_request(request)
    if err_resp is not None:
        return err_resp, status

    try:
        base_metadata = deep_metadata_extraction(image, image_bytes)
        prnu = estimate_prnu_signature(image)
        hashes = multiple_hash_generation(image)

        return jsonify(convert_to_json_serializable({
            'prnu': prnu,
            'image_hashes': hashes,
            'metadata': base_metadata,
            'camera_fingerprinting_note': 'For high-confidence matching against large camera databases, integrate external PRNU service.'
        }))
    except Exception as e:
        logger.error(f"camera-fingerprint error: {e}")
        return jsonify({'error': str(e)}), 500


@snapspeak_ai.route('/api/forensics/location-intelligence', methods=['POST'])
@rate_limit_strict(requests_per_minute=10, requests_per_hour=100)
def api_location_intelligence():
    """
    Extract and analyze GPS metadata.
    Reverse geocoding / elevation are exposed as placeholders to integrate with external APIs.
    """
    image, image_bytes, err_resp, status = _load_image_from_request(request)
    if err_resp is not None:
        return err_resp, status

    try:
        meta = deep_metadata_extraction(image, image_bytes)
        gps_info = meta.get('metadata', {}).get('gps_location', {})

        # Placeholder hooks for external reverse-geocoding services
        location_details = None
        if gps_info:
            location_details = {
                'reverse_geocode_supported': False,
                'message': 'Integrate with external geocoding API (e.g., OpenStreetMap, Google Maps) for full address/elevation.'
            }

        return jsonify(convert_to_json_serializable({
            'gps_raw': gps_info,
            'location_details': location_details,
            'use_case': 'Location forensics & geo-privacy risk assessment.'
        }))
    except Exception as e:
        logger.error(f"location-intelligence error: {e}")
        return jsonify({'error': str(e)}), 500


@snapspeak_ai.route('/api/forensics/edit-history', methods=['POST'])
@rate_limit_strict(requests_per_minute=10, requests_per_hour=100)
def api_edit_history():
    """Reconstruct probable edit history using metadata and technical cues."""
    image, image_bytes, err_resp, status = _load_image_from_request(request)
    if err_resp is not None:
        return err_resp, status

    try:
        meta = deep_metadata_extraction(image, image_bytes)
        meta_core = meta.get('metadata', {})
        software_analysis = analyze_edit_software_from_metadata(meta_core)

        # Use object & stego analysis to add hints
        steg = advanced_steganography_detection(image)
        tech = object_detection_analysis(image)

        return jsonify(convert_to_json_serializable({
            'metadata_sections': meta_core,
            'software_analysis': software_analysis,
            'stego_summary': {
                'overall_risk': steg.get('overall_risk'),
                'confidence': steg.get('confidence'),
                'methods_detected': steg.get('methods_detected', [])
            },
            'technical_artifacts': tech,
            'note': 'Edit history reconstruction is heuristic; for legal-grade timelines integrate with dedicated forensic tools.'
        }))
    except Exception as e:
        logger.error(f"edit-history error: {e}")
        return jsonify({'error': str(e)}), 500


@snapspeak_ai.route('/api/forensics/validate-timestamp', methods=['POST'])
@rate_limit_strict(requests_per_minute=10, requests_per_hour=100)
def api_validate_timestamp():
    """
    Validate EXIF timestamps for consistency.
    Advanced weather/shadow/sun-position checks are left as external integration hooks.
    """
    image, image_bytes, err_resp, status = _load_image_from_request(request)
    if err_resp is not None:
        return err_resp, status

    try:
        meta = deep_metadata_extraction(image, image_bytes)
        timestamps = meta.get('metadata', {}).get('timestamps', {})

        # Basic logical checks (e.g. original <= digitized)
        consistency_flags = []
        if 'DateTimeOriginal' in timestamps and 'DateTimeDigitized' in timestamps:
            if timestamps['DateTimeOriginal'] != timestamps['DateTimeDigitized']:
                consistency_flags.append('Original and digitized timestamps differ.')

        return jsonify(convert_to_json_serializable({
            'timestamps': timestamps,
            'consistency_flags': consistency_flags,
            'advanced_checks': {
                'weather_validation': 'Not implemented – integrate with external weather/time APIs.',
                'sun_position_analysis': 'Not implemented – require geo + solar position modeling.',
                'timezone_consistency': 'Not implemented – requires explicit timezone data.'
            }
        }))
    except Exception as e:
        logger.error(f"validate-timestamp error: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================
# API ENDPOINTS: NEXT-GEN STEGANOGRAPHY
# ============================================================


@snapspeak_ai.route('/api/stego/deep-scan', methods=['POST'])
@rate_limit_strict(requests_per_minute=5, requests_per_hour=50)
def api_stego_deep_scan():
    """Comprehensive steganography scan combining multiple methods."""
    image, image_bytes, err_resp, status = _load_image_from_request(request)
    if err_resp is not None:
        return err_resp, status

    try:
        base_steg = advanced_steganography_detection(image)
        meta = deep_metadata_extraction(image, image_bytes)
        meta_steg = metadata_stego_analysis(meta.get('metadata', {}))
        dct = dct_stego_analysis(image)

        # Ensemble-style summary
        indicators = base_steg.get('methods_detected', [])
        if meta_steg.get('suspicious_fields'):
            indicators.append('Metadata Stego Suspicion')
        if dct.get('supported') and dct.get('qtables_present'):
            indicators.append('DCT/Quantization Anomaly (heuristic)')

        return jsonify(convert_to_json_serializable({
            'lsb_rs_histogram_analysis': base_steg,
            'metadata_stego': meta_steg,
            'dct_analysis': dct,
            'ensemble_summary': {
                'overall_risk': base_steg.get('overall_risk', 'UNKNOWN'),
                'confidence': base_steg.get('confidence', 0.0),
                'indicators': indicators
            },
            'ml_classifier': {
                'available': False,
                'note': 'Deep learning stego classifier is not bundled; integrate external model/service here.'
            }
        }))
    except Exception as e:
        logger.error(f"stego-deep-scan error: {e}")
        return jsonify({'error': str(e)}), 500


@snapspeak_ai.route('/api/stego/extract-payload', methods=['POST'])
@rate_limit_strict(requests_per_minute=3, requests_per_hour=20)
def api_stego_extract_payload():
    """Attempt naive payload extraction from LSB and report basic format hints."""
    image, image_bytes, err_resp, status = _load_image_from_request(request)
    if err_resp is not None:
        return err_resp, status

    body = _safe_get_json()
    max_bytes = int(body.get('max_bytes', 4096)) if isinstance(body, dict) else 4096

    try:
        lsb_payload = extract_lsb_payload(image, max_bytes=max_bytes)
        preview_hex = lsb_payload.get('payload_preview_hex', '')
        format_hint = 'unknown'
        if preview_hex.startswith('89504e47'):
            format_hint = 'possible PNG image'
        elif preview_hex.startswith('ffd8ffe0') or preview_hex.startswith('ffd8ffe1'):
            format_hint = 'possible JPEG image'
        elif preview_hex and all(
            32 <= int(preview_hex[i:i + 2], 16) <= 126
            for i in range(0, min(40, len(preview_hex)), 2)
        ):
            format_hint = 'likely ASCII text'

        return jsonify(convert_to_json_serializable({
            'lsb_payload': lsb_payload,
            'payload_format_hint': format_hint,
            'automatic_decoding': {
                'attempted': False,
                'note': 'Automatic decoding (base64/compressed payloads) can be added on top of raw bytes.'
            }
        }))
    except Exception as e:
        logger.error(f"stego-extract-payload error: {e}")
        return jsonify({'error': str(e)}), 500


@snapspeak_ai.route('/api/stego/tool-identification', methods=['POST'])
@rate_limit_strict(requests_per_minute=5, requests_per_hour=50)
def api_stego_tool_identification():
    """
    Heuristic identification of possible stego tools based on metadata and container anomalies.
    """
    image, image_bytes, err_resp, status = _load_image_from_request(request)
    if err_resp is not None:
        return err_resp, status

    try:
        meta = deep_metadata_extraction(image, image_bytes)
        software_info = meta.get('metadata', {}).get('software_info', {})
        software_str = ' '.join(str(v).lower() for v in software_info.values())

        candidates = []
        tool_signatures = {
            'outguess': ['outguess'],
            'stegHide': ['steghide'],
            'openStego': ['openstego'],
            'jsteg': ['jsteg'],
        }
        for tool, sigs in tool_signatures.items():
            if any(s in software_str for s in sigs):
                candidates.append(tool)

        return jsonify(convert_to_json_serializable({
            'software_metadata': software_info,
            'probable_tools': candidates,
            'note': 'Signature database is minimal; extend with known stego tool fingerprints for higher accuracy.'
        }))
    except Exception as e:
        logger.error(f"stego-tool-identification error: {e}")
        return jsonify({'error': str(e)}), 500


@snapspeak_ai.route('/api/stego/statistical-analysis', methods=['POST'])
@rate_limit_strict(requests_per_minute=5, requests_per_hour=50)
def api_stego_statistical_analysis():
    """Expose raw statistical metrics used in stego detection for expert analysis."""
    image, image_bytes, err_resp, status = _load_image_from_request(request)
    if err_resp is not None:
        return err_resp, status

    try:
        base_steg = advanced_steganography_detection(image)
        return jsonify(convert_to_json_serializable(base_steg))
    except Exception as e:
        logger.error(f"stego-statistical-analysis error: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================
# API ENDPOINTS: REVERSE IMAGE SEARCH & SIMILARITY
# ============================================================


@snapspeak_ai.route('/api/reverse-search/multi-engine', methods=['POST'])
@rate_limit_strict(requests_per_minute=3, requests_per_hour=20)
def api_reverse_search_multi_engine():
    """
    Stub for multi-engine reverse image search.
    This endpoint prepares payloads and clearly documents which engines can be wired up.
    """
    image, image_bytes, err_resp, status = _load_image_from_request(request)
    if err_resp is not None:
        return err_resp, status

    hashes = multiple_hash_generation(Image.open(io.BytesIO(image_bytes)))

    return jsonify(convert_to_json_serializable({
        'supported_engines': {
            'google_images': {'configured': False},
            'tineye': {'configured': False},
            'yandex': {'configured': False},
            'bing_visual_search': {'configured': False},
        },
        'image_fingerprints': hashes,
        'integration_note': 'Wire this endpoint to external reverse image APIs / CLIP-based search service. '
                            'Current implementation only provides robust fingerprints.'
    }))


@snapspeak_ai.route('/api/reverse-search/provenance', methods=['POST'])
@rate_limit_strict(requests_per_minute=3, requests_per_hour=20)
def api_reverse_search_provenance():
    """High-level provenance stub built on top of multi-engine search."""
    image, image_bytes, err_resp, status = _load_image_from_request(request)
    if err_resp is not None:
        return err_resp, status

    hashes = multiple_hash_generation(Image.open(io.BytesIO(image_bytes)))

    return jsonify(convert_to_json_serializable({
        'image_fingerprints': hashes,
        'provenance_timeline': [],
        'note': 'Connect to multi-engine reverse search results to populate first-appearance dates and usage timelines.'
    }))


@snapspeak_ai.route('/api/reverse-search/find-duplicates', methods=['POST'])
@rate_limit_strict(requests_per_minute=10, requests_per_hour=100)
def api_reverse_search_find_duplicates():
    """
    Perceptual hash-based duplicate / near-duplicate detection for multiple images in one request.
    Accepts multiple files in the "files" field.
    """
    if 'files' not in request.files:
        return jsonify({'error': 'No files field "files" provided'}), 400

    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No files provided'}), 400

    results = []
    for f in files:
        try:
            img_bytes = f.read()
            img = Image.open(io.BytesIO(img_bytes))
            h = multiple_hash_generation(img)
            results.append({'filename': f.filename, 'hashes': h})
        except Exception as e:
            results.append({'filename': f.filename, 'error': str(e)})

    # Simple pairwise comparison using average_hash / phash Hamming distance
    def hamming(a, b):
        if not a or not b or len(a) != len(b):
            return None
        return sum(ch1 != ch2 for ch1, ch2 in zip(a, b))

    pairs = []
    for i in range(len(results)):
        if 'hashes' not in results[i]:
            continue
        for j in range(i + 1, len(results)):
            if 'hashes' not in results[j]:
                continue
            ah1 = results[i]['hashes'].get('average_hash')
            ah2 = results[j]['hashes'].get('average_hash')
            d = hamming(ah1, ah2)
            if d is not None:
                pairs.append({
                    'file_a': results[i]['filename'],
                    'file_b': results[j]['filename'],
                    'average_hash_distance': d
                })

    return jsonify(convert_to_json_serializable({
        'files': results,
        'pairwise_distances': pairs
    }))


@snapspeak_ai.route('/api/reverse-search/copyright-check', methods=['POST'])
@rate_limit_strict(requests_per_minute=3, requests_per_hour=20)
def api_reverse_search_copyright():
    """Stub for copyright database checks."""
    image, image_bytes, err_resp, status = _load_image_from_request(request)
    if err_resp is not None:
        return err_resp, status

    hashes = multiple_hash_generation(Image.open(io.BytesIO(image_bytes)))
    return jsonify(convert_to_json_serializable({
        'image_fingerprints': hashes,
        'copyright_matches': [],
        'note': 'Integrate with stock-photo / copyright registries (Getty, Shutterstock, etc.) to populate matches.'
    }))


# ============================================================
# API ENDPOINTS: ENHANCED VISION & SCENE UNDERSTANDING
# ============================================================


@snapspeak_ai.route('/api/vision/advanced-objects', methods=['POST'])
@rate_limit_strict(requests_per_minute=5, requests_per_hour=50)
def api_vision_advanced_objects():
    """
    Multi-model object detection stub.
    Currently reuses vision_analyzer when available and falls back to edge/texture metrics.
    """
    image, image_bytes, err_resp, status = _load_image_from_request(request)
    if err_resp is not None:
        return err_resp, status

    try:
        vision = advanced_vision_analysis(image) if VISION_ANALYZER_AVAILABLE else None
        basic = object_detection_analysis(image)
        return jsonify(convert_to_json_serializable({
            'vision_analyzer_objects': vision.get('objects_detected', []) if vision else [],
            'vision_source': vision.get('analysis_source', None) if vision else None,
            'technical_analysis': basic,
            'yolo_sam_3dpose': {
                'available': False,
                'note': 'Connect to YOLOv8/SAM/pose-estimation service for full advanced object detection.'
            }
        }))
    except Exception as e:
        logger.error(f"advanced-objects error: {e}")
        return jsonify({'error': str(e)}), 500


@snapspeak_ai.route('/api/vision/scene-understanding', methods=['POST'])
@rate_limit_strict(requests_per_minute=5, requests_per_hour=50)
def api_vision_scene_understanding():
    """Heuristic scene understanding built on top of existing analyses."""
    image, image_bytes, err_resp, status = _load_image_from_request(request)
    if err_resp is not None:
        return err_resp, status

    try:
        vision = advanced_vision_analysis(image) if VISION_ANALYZER_AVAILABLE else None
        tech = object_detection_analysis(image)

        brightness = tech.get('avg_brightness', 0)
        time_of_day = 'night' if brightness < 60 else 'day' if brightness > 140 else 'dusk/dawn'

        return jsonify(convert_to_json_serializable({
            'caption': vision.get('caption') if vision else None,
            'objects': vision.get('objects_detected', []) if vision else [],
            'dominant_colors': vision.get('dominant_colors', []) if vision else [],
            'time_of_day_estimate': time_of_day,
            'weather_estimate': None,
            'scene_category': None,
            'note': 'Scene category and weather classifiers can be plugged in via dedicated ML models.'
        }))
    except Exception as e:
        logger.error(f"scene-understanding error: {e}")
        return jsonify({'error': str(e)}), 500


@snapspeak_ai.route('/api/vision/ocr-advanced', methods=['POST'])
@rate_limit_strict(requests_per_minute=10, requests_per_hour=100)
def api_vision_ocr_advanced():
    """Advanced OCR endpoint with multi-language support."""
    image, image_bytes, err_resp, status = _load_image_from_request(request)
    if err_resp is not None:
        return err_resp, status

    body = _safe_get_json()
    languages = body.get('languages', 'eng') if isinstance(body, dict) else 'eng'

    ocr_result = perform_ocr(image, languages=languages)
    return jsonify(convert_to_json_serializable(ocr_result))


@snapspeak_ai.route('/api/vision/face-attributes', methods=['POST'])
@rate_limit_strict(requests_per_minute=5, requests_per_hour=50)
def api_vision_face_attributes():
    """Expose enhanced face detection plus LLM-based attribute analysis when available."""
    image, image_bytes, err_resp, status = _load_image_from_request(request)
    if err_resp is not None:
        return err_resp, status

    face_data = enhanced_face_detection(image)
    return jsonify(convert_to_json_serializable(face_data))


@snapspeak_ai.route('/api/vision/document-parse', methods=['POST'])
@rate_limit_strict(requests_per_minute=10, requests_per_hour=100)
def api_vision_document_parse():
    """
    Document-type parsing built on top of OCR.
    Attempts simple structural hints; full table/ID parsing requires dedicated models.
    """
    image, image_bytes, err_resp, status = _load_image_from_request(request)
    if err_resp is not None:
        return err_resp, status

    ocr_result = perform_ocr(image)
    text = ocr_result.get('text', '')

    doc_type = 'unknown'
    lowered = text.lower()
    if any(k in lowered for k in ['invoice', 'total', 'amount due']):
        doc_type = 'invoice'
    elif any(k in lowered for k in ['passport', 'nationality']):
        doc_type = 'passport'
    elif any(k in lowered for k in ['receipt', 'cash', 'change']):
        doc_type = 'receipt'

    return jsonify(convert_to_json_serializable({
        'ocr': ocr_result,
        'document_type': doc_type,
        'note': 'For robust key-value extraction and table parsing, integrate with specialized OCR/vision models.'
    }))


# ============================================================
# API ENDPOINTS: BLOCKCHAIN & DIGITAL PROVENANCE
# ============================================================


@snapspeak_ai.route('/api/blockchain/c2pa-verify', methods=['POST'])
@rate_limit_strict(requests_per_minute=3, requests_per_hour=20)
def api_blockchain_c2pa_verify():
    """
    Stub endpoint for C2PA/CAI content credentials verification.
    """
    image, image_bytes, err_resp, status = _load_image_from_request(request)
    if err_resp is not None:
        return err_resp, status

    # At this time no C2PA parser is bundled; this endpoint is designed to be wired into such a library.
    return jsonify({
        'c2pa_supported': False,
        'credentials': None,
        'note': 'Integrate with a C2PA/CAI verification library or service to extract and verify content credentials.'
    })


@snapspeak_ai.route('/api/blockchain/nft-check', methods=['POST'])
@rate_limit_strict(requests_per_minute=3, requests_per_hour=20)
def api_blockchain_nft_check():
    """
    Stub endpoint for NFT/blockchain verification.
    """
    image, image_bytes, err_resp, status = _load_image_from_request(request)
    if err_resp is not None:
        return err_resp, status

    hashes = multiple_hash_generation(Image.open(io.BytesIO(image_bytes)))
    return jsonify(convert_to_json_serializable({
        'image_fingerprints': hashes,
        'nft_matches': [],
        'note': 'Connect to blockchain indexers (e.g., Ethereum/Polygon NFT APIs) to look up ownership and provenance.'
    }))


@snapspeak_ai.route('/api/blockchain/digital-signature', methods=['POST'])
@rate_limit_strict(requests_per_minute=3, requests_per_hour=20)
def api_blockchain_digital_signature():
    """
    Stub endpoint for embedded digital signature validation.
    """
    image, image_bytes, err_resp, status = _load_image_from_request(request)
    if err_resp is not None:
        return err_resp, status

    return jsonify({
        'signature_found': False,
        'validation_result': None,
        'note': 'Wire this endpoint to a signature-extraction and cryptographic verification module.'
    })


# ============================================================
# API ENDPOINTS: SMART IMAGE COMPARISON & DIFF
# ============================================================


@snapspeak_ai.route('/api/compare/visual-diff', methods=['POST'])
@rate_limit_strict(requests_per_minute=5, requests_per_hour=50)
def api_compare_visual_diff():
    """
    Visual diff heatmap between two images.
    Expects file fields 'file_a' and 'file_b'.
    """
    img_a, bytes_a, err_a, status_a = _load_image_from_request(request, field_name='file_a')
    if err_a is not None:
        return err_a, status_a
    img_b, bytes_b, err_b, status_b = _load_image_from_request(request, field_name='file_b')
    if err_b is not None:
        return err_b, status_b

    try:
        # Resize to smallest common size
        w = min(img_a.width, img_b.width)
        h = min(img_a.height, img_b.height)
        a_resized = img_a.resize((w, h)).convert('RGB')
        b_resized = img_b.resize((w, h)).convert('RGB')

        arr_a = np.array(a_resized).astype('float32')
        arr_b = np.array(b_resized).astype('float32')
        diff = np.abs(arr_a - arr_b)
        diff_gray = np.mean(diff, axis=2)

        # Normalize to 0-255 for visualization and compute summary stats
        norm = (255 * (diff_gray / (diff_gray.max() + 1e-6))).astype('uint8')
        avg_diff = float(np.mean(diff_gray))

        # Encode diff heatmap as PNG bytes for size reference
        heatmap_img = Image.fromarray(norm)
        buf = io.BytesIO()
        heatmap_img.save(buf, format='PNG')
        heatmap_size = buf.tell()

        return jsonify({
            'width': int(w),
            'height': int(h),
            'average_pixel_difference': avg_diff,
            'heatmap_bytes': heatmap_size,
            'note': 'Raw heatmap PNG bytes are generated server-side; expose via separate download endpoint if needed.'
        })
    except Exception as e:
        logger.error(f"visual-diff error: {e}")
        return jsonify({'error': str(e)}), 500


@snapspeak_ai.route('/api/compare/batch-similarity', methods=['POST'])
@rate_limit_strict(requests_per_minute=5, requests_per_hour=50)
def api_compare_batch_similarity():
    """Batch similarity matrix for multiple images based on perceptual hashes."""
    if 'files' not in request.files:
        return jsonify({'error': 'No files field "files" provided'}), 400

    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No files provided'}), 400

    imgs = []
    for f in files:
        try:
            img_bytes = f.read()
            img = Image.open(io.BytesIO(img_bytes))
            h = multiple_hash_generation(img)
            imgs.append({'filename': f.filename, 'hashes': h})
        except Exception as e:
            imgs.append({'filename': f.filename, 'error': str(e)})

    def hamming(a, b):
        if not a or not b or len(a) != len(b):
            return None
        return sum(ch1 != ch2 for ch1, ch2 in zip(a, b))

    matrix = []
    for i in range(len(imgs)):
        row = []
        for j in range(len(imgs)):
            if i == j or 'hashes' not in imgs[i] or 'hashes' not in imgs[j]:
                row.append(0)
            else:
                d = hamming(imgs[i]['hashes'].get('average_hash'), imgs[j]['hashes'].get('average_hash'))
                row.append(d if d is not None else -1)
        matrix.append(row)

    return jsonify(convert_to_json_serializable({
        'files': imgs,
        'average_hash_distance_matrix': matrix
    }))


@snapspeak_ai.route('/api/compare/detect-edits', methods=['POST'])
@rate_limit_strict(requests_per_minute=5, requests_per_hour=50)
def api_compare_detect_edits():
    """Edit detection using stego, technical, and metadata heuristics."""
    image, image_bytes, err_resp, status = _load_image_from_request(request)
    if err_resp is not None:
        return err_resp, status

    try:
        steg = advanced_steganography_detection(image)
        meta = deep_metadata_extraction(image, image_bytes)
        tech = object_detection_analysis(image)

        clues = []
        if meta.get('use_cases', {}).get('software_info'):
            clues.append('Editing software detected in metadata.')
        if steg.get('overall_risk') in ['MEDIUM', 'HIGH']:
            clues.append('Suspicious stego patterns that may indicate synthetic edits.')

        return jsonify(convert_to_json_serializable({
            'steganography': steg,
            'metadata': meta,
            'technical': tech,
            'edit_clues': clues
        }))
    except Exception as e:
        logger.error(f"detect-edits error: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================
# API ENDPOINTS: PRIVACY & SECURITY ANALYSIS
# ============================================================


@snapspeak_ai.route('/api/privacy/pii-detect', methods=['POST'])
@rate_limit_strict(requests_per_minute=5, requests_per_hour=50)
def api_privacy_pii_detect():
    """
    Basic PII detection based on faces and OCR text.
    """
    image, image_bytes, err_resp, status = _load_image_from_request(request)
    if err_resp is not None:
        return err_resp, status

    ocr_result = perform_ocr(image)
    faces = enhanced_face_detection(image)

    text = ocr_result.get('text', '')
    pii_patterns = {
        'credit_card_like': bool(len(text) > 0 and any(c.isdigit() for c in text)),
    }

    return jsonify(convert_to_json_serializable({
        'faces': faces,
        'ocr': ocr_result,
        'pii_patterns': pii_patterns,
        'note': 'For production-grade PII detection, integrate regex/rule-based + ML entity recognizers.'
    }))


@snapspeak_ai.route('/api/privacy/risk-assessment', methods=['POST'])
@rate_limit_strict(requests_per_minute=5, requests_per_hour=50)
def api_privacy_risk_assessment():
    """Compute a privacy risk score based on metadata and visual cues."""
    image, image_bytes, err_resp, status = _load_image_from_request(request)
    if err_resp is not None:
        return err_resp, status

    meta = deep_metadata_extraction(image, image_bytes)
    faces = enhanced_face_detection(image)
    ocr_result = perform_ocr(image)

    score = 0
    reasons = []

    if meta.get('metadata', {}).get('gps_location'):
        score += 30
        reasons.append('GPS location present in metadata.')
    if faces.get('count', 0) > 0:
        score += 30
        reasons.append('Faces detected in image.')
    if len(ocr_result.get('text', '').strip()) > 0:
        score += 20
        reasons.append('Readable text present (may contain identifiers).')

    score = min(100, score)

    return jsonify(convert_to_json_serializable({
        'privacy_risk_score': score,
        'reasons': reasons,
        'exif_privacy_score': score,
        'metadata': meta.get('metadata', {})
    }))


@snapspeak_ai.route('/api/privacy/auto-redact', methods=['POST'])
@rate_limit_strict(requests_per_minute=3, requests_per_hour=20)
def api_privacy_auto_redact():
    """
    Stub for auto-redaction. Returns regions that should be redacted
    (faces & optionally text regions when OCR bounding boxes are available).
    """
    image, image_bytes, err_resp, status = _load_image_from_request(request)
    if err_resp is not None:
        return err_resp, status

    faces = enhanced_face_detection(image)

    return jsonify(convert_to_json_serializable({
        'regions_to_redact': faces.get('locations', []),
        'note': 'Apply blurring/redaction on client or downstream processor based on these regions.'
    }))


# ============================================================
# API ENDPOINTS: PROFESSIONAL IMAGE QUALITY METRICS
# ============================================================


@snapspeak_ai.route('/api/quality/technical-assessment', methods=['POST'])
@rate_limit_strict(requests_per_minute=10, requests_per_hour=100)
def api_quality_technical_assessment():
    """Return technical quality metrics (sharpness, noise, dynamic range)."""
    image, image_bytes, err_resp, status = _load_image_from_request(request)
    if err_resp is not None:
        return err_resp, status

    quality = compute_technical_quality(image)
    return jsonify(convert_to_json_serializable(quality))


@snapspeak_ai.route('/api/quality/aesthetic-score', methods=['POST'])
@rate_limit_strict(requests_per_minute=10, requests_per_hour=100)
def api_quality_aesthetic_score():
    """Return heuristic aesthetic score with composition insights."""
    image, image_bytes, err_resp, status = _load_image_from_request(request)
    if err_resp is not None:
        return err_resp, status

    color_data = enhanced_color_analysis(image)
    tech = object_detection_analysis(image)
    faces = enhanced_face_detection(image)

    score = compute_aesthetic_score(image, color_analysis=color_data, object_analysis=tech, faces=faces)
    return jsonify(convert_to_json_serializable({
        'aesthetic': score,
        'color_analysis': color_data
    }))


@snapspeak_ai.route('/api/quality/professional-report', methods=['POST'])
@rate_limit_strict(requests_per_minute=5, requests_per_hour=50)
def api_quality_professional_report():
    """Aggregate key technical & aesthetic metrics for professional review."""
    image, image_bytes, err_resp, status = _load_image_from_request(request)
    if err_resp is not None:
        return err_resp, status

    quality = compute_technical_quality(image)
    color_data = enhanced_color_analysis(image)
    tech = object_detection_analysis(image)
    faces = enhanced_face_detection(image)
    aesthetic = compute_aesthetic_score(image, color_analysis=color_data, object_analysis=tech, faces=faces)

    return jsonify(convert_to_json_serializable({
        'technical_quality': quality,
        'aesthetic': aesthetic,
        'color_analysis': color_data,
        'faces': faces
    }))


# ============================================================
# API ENDPOINTS: BATCH PROCESSING & WORKFLOWS
# ============================================================


@snapspeak_ai.route('/api/batch/upload-folder', methods=['POST'])
@rate_limit_strict(requests_per_minute=2, requests_per_hour=10)
def api_batch_upload_folder():
    """
    Simplified batch endpoint: accept multiple files and return immediate per-image summaries.
    Full async batch IDs and progress tracking are exposed in separate endpoints.
    """
    if 'files' not in request.files:
        return jsonify({'error': 'No files field "files" provided'}), 400

    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No files provided'}), 400

    results = []
    for f in files:
        try:
            img_bytes = f.read()
            img = Image.open(io.BytesIO(img_bytes))
            hashes = multiple_hash_generation(img)
            quality = compute_technical_quality(img)
            results.append({
                'filename': f.filename,
                'hashes': hashes,
                'quality': quality
            })
        except Exception as e:
            results.append({'filename': f.filename, 'error': str(e)})

    # In a full implementation this would create a batch_id and store state in DB
    return jsonify(convert_to_json_serializable({
        'batch_id': None,
        'images': results,
        'note': 'Synchronous batch summary; wire this to a task queue and DB for large asynchronous runs.'
    }))


@snapspeak_ai.route('/api/batch/check-progress', methods=['POST'])
@rate_limit_strict(requests_per_minute=10, requests_per_hour=100)
def api_batch_check_progress():
    """Progress stub for asynchronous batches."""
    body = _safe_get_json()
    batch_id = body.get('batch_id')
    return jsonify({
        'batch_id': batch_id,
        'status': 'not_implemented',
        'progress': 0.0,
        'note': 'Implement persistent batch tracking (e.g., Redis + worker queue) to support long-running analyses.'
    })


@snapspeak_ai.route('/api/batch/results/<batch_id>', methods=['GET'])
@rate_limit_strict(requests_per_minute=10, requests_per_hour=100)
def api_batch_results(batch_id):
    """Results stub for asynchronous batches."""
    return jsonify({
        'batch_id': batch_id,
        'status': 'not_implemented',
        'results': [],
        'note': 'Batch results storage not yet wired; connect to DB where batch summaries are persisted.'
    })


@snapspeak_ai.route('/api/workflows/create-pipeline', methods=['POST'])
@rate_limit_strict(requests_per_minute=5, requests_per_hour=50)
def api_workflows_create_pipeline():
    """Define a custom analysis pipeline configuration (stub only)."""
    body = _safe_get_json()
    return jsonify({
        'pipeline_id': None,
        'pipeline_definition': body,
        'note': 'Persist this configuration and attach it to batch jobs for reusable pipelines.'
    })


# ============================================================
# API ENDPOINTS: EXPORT & REPORTING
# ============================================================


@snapspeak_ai.route('/api/export/pdf-report', methods=['POST'])
@rate_limit_strict(requests_per_minute=2, requests_per_hour=10)
def api_export_pdf_report():
    """
    Stub for generating professional PDF forensic reports.
    Currently returns structured JSON that a reporting service can turn into a PDF.
    """
    image, image_bytes, err_resp, status = _load_image_from_request(request)
    if err_resp is not None:
        return err_resp, status

    caption = generate_caption(image)
    meta = deep_metadata_extraction(image, image_bytes)
    quality = compute_technical_quality(image)
    steg = advanced_steganography_detection(image)
    ai_det = detect_ai_generation(image)

    report = {
        'summary': {
            'caption': caption,
            'ai_generation': ai_det,
            'overall_stego_risk': steg.get('overall_risk', 'UNKNOWN')
        },
        'metadata': meta,
        'technical_quality': quality,
        'steganography': steg
    }
    return jsonify(convert_to_json_serializable({
        'report': report,
        'note': 'Feed this JSON into a PDF generation service to produce a full forensic report.'
    }))


@snapspeak_ai.route('/api/export/forensic-package', methods=['POST'])
@rate_limit_strict(requests_per_minute=2, requests_per_hour=10)
def api_export_forensic_package():
    """
    Stub for packaging all analysis artifacts (JSON, hashes, thumbnails) into a downloadable bundle.
    """
    image, image_bytes, err_resp, status = _load_image_from_request(request)
    if err_resp is not None:
        return err_resp, status

    img = Image.open(io.BytesIO(image_bytes))
    hashes = multiple_hash_generation(img)
    meta = deep_metadata_extraction(img, image_bytes)

    return jsonify(convert_to_json_serializable({
        'hashes': hashes,
        'metadata': meta,
        'artifacts_packaged': False,
        'note': 'Integrate with archive/zip creation to produce downloadable forensic packages.'
    }))


@snapspeak_ai.route('/api/visualize/heatmap', methods=['POST'])
@rate_limit_strict(requests_per_minute=5, requests_per_hour=50)
def api_visualize_heatmap():
    """
    Attention/edge heatmap visualization stub.
    Currently returns simple edge-density heatmap metadata.
    """
    image, image_bytes, err_resp, status = _load_image_from_request(request)
    if err_resp is not None:
        return err_resp, status

    try:
        if image.mode != 'L':
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        else:
            gray = np.array(image)

        edges = cv2.Canny(gray, 50, 150)
        density = float(np.mean(edges > 0))

        return jsonify({
            'edge_density': density,
            'note': 'Hook this endpoint to model attention maps for full “what the AI looked at” visualizations.'
        })
    except Exception as e:
        logger.error(f"visualize-heatmap error: {e}")
        return jsonify({'error': str(e)}), 500


@snapspeak_ai.route('/api/visualize/timeline', methods=['POST'])
@rate_limit_strict(requests_per_minute=5, requests_per_hour=50)
def api_visualize_timeline():
    """
    Metadata-based timeline visualization stub.
    """
    image, image_bytes, err_resp, status = _load_image_from_request(request)
    if err_resp is not None:
        return err_resp, status

    meta = deep_metadata_extraction(image, image_bytes)
    timestamps = meta.get('metadata', {}).get('timestamps', {})
    timeline = [{'tag': k, 'value': v} for k, v in timestamps.items()]

    return jsonify(convert_to_json_serializable({
        'timeline': timeline,
        'note': 'Render this timeline on the front-end as an interactive visualization.'
    }))
