from flask import Blueprint, request, jsonify, render_template, current_app
from flask_cors import CORS
from transformers import BlipForConditionalGeneration, BlipProcessor, logging
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
from datetime import datetime
import exifread
import piexif

# Gemini API Integration
try:
    import google.generativeai as genai
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'AIzaSyCMwpK-6Dr9X_MpcCyRR1PJcixg4pW55e8')
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
        GEMINI_CONFIGURED = True
        print("✓ Gemini API: CONFIGURED for Snapspeak_AI")
    else:
        GEMINI_CONFIGURED = False
        print("✗ Gemini API: NOT CONFIGURED (Set GEMINI_API_KEY environment variable)")
except ImportError:
    GEMINI_CONFIGURED = False
    print("✗ Gemini API: NOT CONFIGURED (google-generativeai not installed)")

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
logging.set_verbosity_error()

snapspeak_ai = Blueprint('snapspeak_ai', __name__, template_folder='templates')
CORS(snapspeak_ai)

# Global configurations
FACE_DETECTION_CONFIDENCE_THRESHOLD = 0.85
COLOR_CLUSTER_COUNT = 8
IMAGE_RESIZE_DIMENSION = 150

# Initialize models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✓ Using device: {device}")

try:
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    print("✓ BLIP Model: LOADED")
except Exception as e:
    model = None
    processor = None
    print(f"✗ BLIP Model: FAILED - {str(e)}")

# Initialize face detection
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    print("✓ DeepFace: AVAILABLE")
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("✓ DeepFace: NOT AVAILABLE (Using OpenCV fallback)")


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


def advanced_gemini_analysis(image):
    """Advanced image analysis using Gemini Vision API"""
    if not GEMINI_CONFIGURED:
        return None
    
    try:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        prompt = """Analyze this image comprehensively and provide insights in this EXACT format:

DETAILED DESCRIPTION:
Describe what you see - subject, setting, colors, composition, lighting, and atmosphere. Be specific and detailed.

CONTEXT & PURPOSE:
What might this image be used for? What story does it tell? What is the likely intent behind this image?

TECHNICAL ANALYSIS:
Analyze the photography technique, composition rules applied, artistic elements, image quality, and technical execution.

AUTHENTICITY CHECK:
- Is this a real photograph, AI-generated, or AI-altered?
- Provide specific indicators that led to your conclusion
- Signs of manipulation, editing, or artificial generation
- Confidence level: High/Medium/Low
- List observable evidence

HIDDEN DETAILS:
Point out subtle elements, background details, or nuances that might be missed at first glance.

EMOTIONAL TONE:
What feelings, mood, or atmosphere does this image convey? How does it make you feel?

TIME & LOCATION ESTIMATE:
Based on visible clues, estimate when and where this might have been taken. Explain your reasoning.

Be direct, clear, and engaging. Avoid excessive formatting. Focus on actionable insights."""

        response = gemini_model.generate_content([prompt, {"mime_type": "image/png", "data": img_byte_arr}])
        
        return {
            'detailed_analysis': response.text,
            'source': 'gemini-2.0-flash-exp'
        }
    except Exception as e:
        print(f"Gemini analysis error: {str(e)}")
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
        print(f"Error in deep metadata extraction: {str(e)}")
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
        print(f"Error in steganography detection: {str(e)}")
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
        print(f"Error in color analysis: {str(e)}")
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
        print(f"Error generating hashes: {str(e)}")
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
        print(f"Error in object detection: {str(e)}")
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
                faces = DeepFace.extract_faces(
                    np_image,
                    detector_backend='retinaface',
                    enforce_detection=False,
                    align=True
                )

                for face in faces:
                    confidence = face.get('confidence', 0)
                    if confidence > FACE_DETECTION_CONFIDENCE_THRESHOLD:
                        facial_area = face['facial_area']
                        face_locations.append({
                            'x': int(facial_area['x']),
                            'y': int(facial_area['y']),
                            'width': int(facial_area['w']),
                            'height': int(facial_area['h']),
                            'confidence': float(confidence),
                            'detector': 'retinaface'
                        })
                print(f"  DeepFace detected {len(face_locations)} faces")
            except Exception as e:
                print(f"  DeepFace detection failed: {str(e)}")

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
                print(f"  OpenCV detected {len(face_locations)} faces")
            except Exception as e:
                print(f"  OpenCV detection failed: {str(e)}")

        # Perform advanced face analysis if faces found
        face_analysis = None
        if face_locations and GEMINI_CONFIGURED:
            try:
                print(f"  Running advanced face analysis...")
                face_analysis = analyze_face_with_gemini(image)
            except Exception as e:
                print(f"  Gemini face analysis failed: {str(e)}")

        return convert_to_json_serializable({
            'count': len(face_locations), 
            'locations': face_locations,
            'face_analysis': face_analysis,
            'detector_used': face_locations[0]['detector'] if face_locations else 'none'
        })
        
    except Exception as e:
        print(f"Error in face detection: {str(e)}")
        traceback.print_exc()
        return {'count': 0, 'locations': [], 'error': str(e)}


def analyze_face_with_gemini(image):
    """Advanced face analysis using Gemini AI"""
    if not GEMINI_CONFIGURED:
        return None
    
    try:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        prompt = """Analyze any faces in this image. Provide:

1. PERSON DESCRIPTION: Age range, gender (if identifiable), ethnicity, distinctive features
2. FACIAL EXPRESSIONS: Emotions, mood, what they might be feeling
3. CONTEXT: Setting, what the person might be doing, social context
4. APPEARANCE: Clothing style, accessories, grooming, overall presentation
5. IDENTIFICATION CLUES: Any visible text, logos, badges, or identifying markers (DO NOT attempt to identify specific individuals)
6. AUTHENTICITY: Does this appear to be a real photograph or potentially AI-generated/manipulated?

IMPORTANT: Do NOT attempt to identify specific individuals by name. Focus on observable characteristics only."""

        response = gemini_model.generate_content([prompt, {"mime_type": "image/png", "data": img_byte_arr}])
        
        return {
            'analysis': response.text,
            'source': 'gemini-face-analysis'
        }
    except Exception as e:
        print(f"Gemini face analysis error: {str(e)}")
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
        print(f"Error in AI detection: {str(e)}")
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
        if model is None or processor is None:
            return "BLIP model not available"
        
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        output_ids = model.generate(pixel_values, max_length=50, num_beams=4)
        return processor.decode(output_ids[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Error in caption generation: {str(e)}")
        return "Error generating caption"


@snapspeak_ai.route('/')
def index():
    return render_template('snapspeak.html')


@snapspeak_ai.route('/api/analyze/', methods=['POST'])
def analyze_image():
    """Main image analysis endpoint"""
    start_time = time.time()
    
    try:
        # Validate file upload
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if not file.filename:
            return jsonify({'error': 'No selected file'}), 400
        
        # Load image
        try:
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Perform comprehensive analyses
        print(f"\n{'='*50}")
        print(f"Analyzing image: {file.filename}")
        print(f"{'='*50}")
        
        # Basic caption
        caption = generate_caption(image)
        print(f"✓ Caption generated")
        
        # Gemini advanced analysis
        gemini_analysis = advanced_gemini_analysis(image) if GEMINI_CONFIGURED else None
        print(f"✓ Gemini analysis: {'Complete' if gemini_analysis else 'Skipped'}")
        
        # Deep metadata
        metadata_result = deep_metadata_extraction(image, image_bytes)
        print(f"✓ Metadata extracted: {metadata_result.get('sections_found', 0)} sections")
        
        # Enhanced color analysis
        color_data = enhanced_color_analysis(image)
        print(f"✓ Color analysis: {len(color_data.get('colors', []))} colors")
        
        # Advanced steganography
        steg_result = advanced_steganography_detection(image)
        print(f"✓ Steganography check: {steg_result['overall_risk']} risk ({steg_result['confidence']:.1f}% confidence)")
        
        # Multiple hashes
        hashes = multiple_hash_generation(image)
        print(f"✓ Hashes generated: {len(hashes)} types")
        
        # Object analysis
        object_analysis = object_detection_analysis(image)
        print(f"✓ Technical analysis complete")
        
        # Face detection with advanced analysis
        face_data = enhanced_face_detection(image)
        print(f"✓ Face detection: {face_data.get('count', 0)} faces found")
        if face_data.get('face_analysis'):
            print(f"  ✓ Advanced face analysis completed")
        
        # AI Generation Detection
        ai_detection = detect_ai_generation(image)
        print(f"✓ AI Detection: {ai_detection['assessment']} ({ai_detection['confidence']:.1f}% confidence)")
        
        processing_time = time.time() - start_time
        print(f"\n⏱  Total processing time: {processing_time:.2f}s")
        print(f"{'='*50}\n")
        
        analysis_results = {
            'basic_caption': caption,
            'gemini_analysis': gemini_analysis,
            'ai_detection': ai_detection,
            'metadata': metadata_result,
            'color_analysis': color_data,
            'steganography': steg_result,
            'image_hashes': hashes,
            'technical_analysis': object_analysis,
            'faces': face_data,
            'processing_time': float(processing_time),
            'gemini_enabled': GEMINI_CONFIGURED,
            'deepface_available': DEEPFACE_AVAILABLE
        }
        
        # Convert all data to JSON serializable format
        analysis_results = convert_to_json_serializable(analysis_results)
        
        return jsonify(analysis_results)
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in analyze_image: {str(e)}")
        print(f"Traceback: {error_trace}")
        return jsonify({
            'error': str(e),
            'traceback': error_trace if current_app.debug else 'Enable debug mode for traceback'
        }), 500 