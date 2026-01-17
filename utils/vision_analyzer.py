"""
Vision Analysis Module
Provides cloud (Hugging Face) and local (BLIP, YOLOv8) fallback for image analysis.
Designed as a reusable component for cybersecurity/AI intelligence projects.
"""

import os
import io
import logging
from typing import Dict, List, Optional, Tuple
from PIL import Image
import numpy as np
import requests
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

# ==================== Configuration ====================
# Use Inference API endpoint (more reliable than router)
HF_API_BASE = "https://api-inference.huggingface.co/models"
HF_MODEL = "Salesforce/blip-image-captioning-base"
HF_TIMEOUT = 15  # seconds (increased for reliability)

# ==================== Cloud Layer (Primary) ====================

def _get_hf_api_token() -> Optional[str]:
    """Get Hugging Face API token from config or environment"""
    try:
        from config import Config
        return Config.HF_API_TOKEN
    except (ImportError, AttributeError):
        return os.getenv('HF_API_TOKEN')


def cloud_caption(image_path: str) -> Optional[str]:
    """
    Get image caption using Hugging Face Inference API (free tier)
    
    Args:
        image_path: Path to image file or PIL Image object
        
    Returns:
        Caption string or None if failed
    """
    api_token = _get_hf_api_token()
    if not api_token:
        logger.debug("HF_API_TOKEN not configured, skipping cloud caption")
        return None
    
    try:
        # Handle both file path and PIL Image
        if isinstance(image_path, str):
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
        else:
            # PIL Image object
            img_byte_arr = io.BytesIO()
            image_path.save(img_byte_arr, format='PNG')
            image_bytes = img_byte_arr.getvalue()
        
        headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/octet-stream"
        }
        
        response = requests.post(
            f"{HF_API_BASE}/{HF_MODEL}",
            headers=headers,
            data=image_bytes,
            timeout=HF_TIMEOUT
        )
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                caption = result[0].get('generated_text', '')
                logger.info(f"✓ Cloud caption generated: {caption[:50]}...")
                return caption
            elif isinstance(result, dict) and 'generated_text' in result:
                caption = result['generated_text']
                logger.info(f"✓ Cloud caption generated: {caption[:50]}...")
                return caption
        elif response.status_code == 503:
            logger.warning("⚠️ Hugging Face model is loading, will use local fallback")
        else:
            logger.warning(f"⚠️ Hugging Face API error {response.status_code}: {response.text[:100]}")
            
    except requests.exceptions.Timeout:
        logger.warning("⚠️ Hugging Face API timeout, using local fallback")
    except requests.exceptions.RequestException as e:
        logger.warning(f"⚠️ Hugging Face API request failed: {e}")
    except Exception as e:
        logger.error(f"✗ Cloud caption error: {e}")
    
    return None


# ==================== Local Fallback Layer ====================

# BLIP Model (lazy loading)
_blip_model = None
_blip_processor = None
_blip_device = None

def _load_blip_model():
    """Lazy load BLIP model for local captioning with retry logic"""
    global _blip_model, _blip_processor, _blip_device
    
    if _blip_model is not None:
        return _blip_model, _blip_processor, _blip_device
    
    try:
        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor
        import time
        
        _blip_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Retry logic for Hugging Face model loading with exponential backoff
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                # Increase timeout for model loading
                import os
                os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '60'
                
                _blip_processor = BlipProcessor.from_pretrained(
                    "Salesforce/blip-image-captioning-base",
                    local_files_only=False,
                    timeout=30
                )
                _blip_model = BlipForConditionalGeneration.from_pretrained(
                    "Salesforce/blip-image-captioning-base",
                    local_files_only=False,
                    timeout=30
                ).to(_blip_device)
                _blip_model.eval()
                
                logger.info(f"✓ BLIP model loaded on {_blip_device}")
                return _blip_model, _blip_processor, _blip_device
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"⚠️ Hugging Face connection timeout (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"✗ Failed to load BLIP model after {max_retries} attempts: {e}")
                    return None, None, None
            except Exception as e:
                # For non-network errors, don't retry
                logger.error(f"✗ Failed to load BLIP model: {e}")
                return None, None, None
                
    except ImportError:
        logger.warning("⚠️ transformers not available, BLIP fallback disabled")
        return None, None, None
    except Exception as e:
        logger.error(f"✗ Failed to load BLIP model: {e}")
        return None, None, None


def local_caption(image_path: str) -> Optional[str]:
    """
    Get image caption using local BLIP model
    
    Args:
        image_path: Path to image file or PIL Image object
        
    Returns:
        Caption string or None if failed
    """
    model, processor, device = _load_blip_model()
    if model is None:
        return None
    
    try:
        # Handle both file path and PIL Image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            # PIL Image object
            image = image_path.convert('RGB')
        
        import torch
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=50)
        
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"✓ Local BLIP caption generated: {caption[:50]}...")
        return caption
        
    except Exception as e:
        logger.error(f"✗ Local caption error: {e}")
        return None


# YOLOv8 Model (lazy loading)
_yolo_model = None

def _load_yolo_model():
    """Lazy load YOLOv8 model for object detection"""
    global _yolo_model
    
    if _yolo_model is not None:
        return _yolo_model
    
    try:
        from ultralytics import YOLO
        
        # Use nano model for lightweight operation
        _yolo_model = YOLO('yolov8n.pt')  # nano model - smallest and fastest
        logger.info("✓ YOLOv8 model loaded")
        return _yolo_model
    except ImportError:
        logger.warning("⚠️ ultralytics not available, YOLOv8 fallback disabled")
        return None
    except Exception as e:
        logger.error(f"✗ Failed to load YOLOv8 model: {e}")
        return None


def detect_objects(image_path: str) -> List[str]:
    """
    Detect objects in image using local YOLOv8
    
    Args:
        image_path: Path to image file or PIL Image object
        
    Returns:
        List of detected object class names
    """
    model = _load_yolo_model()
    if model is None:
        return []
    
    try:
        # Handle both file path and PIL Image
        if isinstance(image_path, str):
            results = model(image_path, verbose=False)
        else:
            # PIL Image object - save temporarily with proper file handling
            import tempfile
            import time
            tmp_path = None
            try:
                # Create temporary file and ensure it's closed before YOLOv8 uses it
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    image_path.save(tmp.name, 'JPEG', quality=95)
                    tmp_path = tmp.name
                
                # Ensure file is fully written and closed before YOLOv8 accesses it
                time.sleep(0.1)  # Small delay to ensure file is released
                
                # Run YOLOv8 detection
                results = model(tmp_path, verbose=False)
                
                # Ensure results are processed before deleting file
                if results:
                    _ = list(results)  # Force evaluation
                
            finally:
                # Clean up temporary file with retry logic for Windows file locking
                if tmp_path and os.path.exists(tmp_path):
                    max_cleanup_retries = 5
                    for retry in range(max_cleanup_retries):
                        try:
                            os.unlink(tmp_path)
                            break
                        except (OSError, PermissionError) as e:
                            if retry < max_cleanup_retries - 1:
                                time.sleep(0.2 * (retry + 1))  # Increasing delay
                            else:
                                logger.warning(f"⚠️ Could not delete temp file {tmp_path}: {e}")
        
        if results and len(results) > 0:
            detected_classes = set()
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    for box in result.boxes:
                        if hasattr(box, 'cls') and box.cls is not None:
                            class_id = int(box.cls.item())
                            class_name = model.names[class_id]
                            detected_classes.add(class_name)
            
            objects = sorted(list(detected_classes))
            logger.info(f"✓ Detected {len(objects)} object classes: {', '.join(objects[:5])}...")
            return objects
        
    except Exception as e:
        logger.error(f"✗ Object detection error: {e}")
    
    return []


def extract_colors(image_path: str, n_colors: int = 5) -> List[Dict[str, any]]:
    """
    Extract dominant colors using KMeans clustering
    
    Args:
        image_path: Path to image file or PIL Image object
        n_colors: Number of dominant colors to extract
        
    Returns:
        List of dicts with 'rgb' and 'hex' keys
    """
    try:
        # Handle both file path and PIL Image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            # PIL Image object
            image = image_path.convert('RGB')
        
        # Resize for faster processing (max 300px on longest side)
        image.thumbnail((300, 300), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(image)
        img_array = img_array.reshape(-1, 3)
        
        # Remove pure black/white pixels (often background)
        img_array = img_array[(img_array.sum(axis=1) > 30) & (img_array.sum(axis=1) < 750)]
        
        if len(img_array) < n_colors:
            # Fallback: use all pixels
            if isinstance(image_path, str):
                image = Image.open(image_path).convert('RGB')
            else:
                image = image_path.convert('RGB')
            image.thumbnail((300, 300), Image.Resampling.LANCZOS)
            img_array = np.array(image).reshape(-1, 3)
        
        # Apply KMeans
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(img_array)
        
        # Get cluster centers (dominant colors)
        colors = kmeans.cluster_centers_.astype(int)
        
        # Sort by frequency (cluster size)
        labels = kmeans.labels_
        color_counts = [(np.sum(labels == i), colors[i]) for i in range(n_colors)]
        color_counts.sort(reverse=True, key=lambda x: x[0])
        
        # Format results
        result = []
        for count, color in color_counts:
            r, g, b = color
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            result.append({
                "rgb": [int(r), int(g), int(b)],
                "hex": hex_color
            })
        
        logger.info(f"✓ Extracted {len(result)} dominant colors")
        return result
        
    except Exception as e:
        logger.error(f"✗ Color extraction error: {e}")
        return []


# ==================== Main Entry Function ====================

def analyze_image(image_path: str, n_colors: int = 5) -> Dict:
    """
    Main entry function for image analysis
    
    Args:
        image_path: Path to image file or PIL Image object
        n_colors: Number of dominant colors to extract
        
    Returns:
        Dictionary with:
        - caption: str
        - dominant_colors: List[Dict with 'rgb' and 'hex']
        - objects_detected: List[str]
        - analysis_source: 'cloud' | 'local'
    """
    result = {
        "caption": "",
        "dominant_colors": [],
        "objects_detected": [],
        "analysis_source": "local"
    }
    
    # Try cloud caption first
    caption = cloud_caption(image_path)
    if caption:
        result["caption"] = caption
        result["analysis_source"] = "cloud"
    else:
        # Fallback to local BLIP
        caption = local_caption(image_path)
        if caption:
            result["caption"] = caption
            result["analysis_source"] = "local"
        else:
            result["caption"] = "Unable to generate caption"
    
    # Extract colors (always local, no cloud option)
    result["dominant_colors"] = extract_colors(image_path, n_colors)
    
    # Detect objects (always local, no cloud option)
    result["objects_detected"] = detect_objects(image_path)
    
    return result
