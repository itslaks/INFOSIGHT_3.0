# SnapSpeak AI - Image Forensics Platform

## Overview

SnapSpeak AI is an intelligent image forensics platform powered by AI. It generates detailed captions, detects hidden steganography, extracts comprehensive metadata, and creates unique image hashes for security professionals and investigators. The system combines computer vision, deep learning, and forensic analysis techniques for comprehensive image investigation.

## Core Architecture

### Multi-Layer Analysis System
- **Vision Layer**: `utils.vision_analyzer` (BLIP/YOLO stack with lazy loading)
- **Forensics Layer**: Advanced steganography, metadata, PRNU, and timestamp validation
- **Analysis Layer**: Color analysis, face detection, OCR, quality and aesthetic scoring
- **Intelligence Layer**: Reverse image “fingerprinting”, provenance hooks, and blockchain stubs

### Key Features

**1. AI-Powered Image Captioning**
- Generates captions using BLIP (via `vision_analyzer`) with local + cloud fallback
- Optional detailed, forensic-style descriptions via Groq LLM router
- Context-aware, security-focused wording

**2. Steganography Detection**
- Multi-method detection (LSB patterns, chi-square, RS analysis, histogram, pixel-pair analysis)
- Confidence scoring and overall risk level (LOW / MEDIUM / HIGH)
- Metadata-based stego heuristics and DCT quantisation inspection (JPEG)

**3. Comprehensive Metadata Extraction**
- Deep EXIF / GPS / timestamp / software and technical fields
- Geolocation intelligence and GPS-derived privacy assessment
- Camera fingerprinting using approximate PRNU-like signatures

**4. Image Hashing**
- Multiple perceptual and cryptographic hashes (average, pHash, dHash, wHash, color hash, MD5, SHA-256)
- Duplicate and similarity analysis across batches

**5. Color Analysis**
- K-means based dominant colors with HSV/temperature/brightness labelling
- High-level color harmony classification (e.g. vibrant vs monochromatic)

**6. Face & Privacy Analysis**
- Face detection via DeepFace (if available) with OpenCV fallback
- Bounding boxes, detectors used, and optional LLM-based facial attribute summaries
- PII-aware OCR and privacy risk scoring / auto-redaction hints

**7. Advanced Forensics**
- Camera fingerprinting
- Location intelligence and sun-position / timezone / weather cross-checks
- Edit history and software fingerprints
- Tampering and synthetic edit indicators

**8. Reverse Image Fingerprinting**
- Generation of strong hashes for use with external reverse image search engines
- Duplicate detection and provenance hooks (no direct external API calls bundled)

**9. Blockchain & Provenance Hooks**
- C2PA, NFT, and digital signature endpoints wired as extensible stubs

**10. Batch & Reporting**
- Batch similarity and quality assessments
- JSON forensic packages and report payloads for PDF/archiving services

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Backend** | Flask (Python 3.8+) | Web framework & API routing |
| **Image Processing** | Pillow (PIL) | Image manipulation |
| **Computer Vision** | OpenCV (cv2) | Core vision operations |
| **Hashing** | ImageHash, hashlib | Perceptual + cryptographic hashing |
| **Metadata** | exifread, piexif | Deep EXIF/GPS/technical extraction |
| **Vision Models** | `utils.vision_analyzer` (BLIP/YOLOv8) | Captioning, objects, colors |
| **Clustering** | scikit-learn | K-means color clustering |
| **AI Analysis** | Groq Cloud LLM via router | Forensic descriptions and reasoning |
| **OCR** | pytesseract (optional) | Text extraction from images |
| **Frontend** | Vanilla JS + CSS3 | Interactive UI |

## System Components

### 1. Image Analysis Pipeline

```
Image Upload → Format Validation → Multiple Analysis Modules
           ↓
         Captioning → Metadata Extraction → Steganography Detection
           ↓
         Color Analysis → Face Detection → Forensics
           ↓
         Result Aggregation → Response Delivery
```

**Analysis Modules**:
- **Captioning**: BLIP model for AI-generated descriptions
- **Metadata**: EXIF, GPS, camera information extraction
- **Steganography**: Hidden data detection
- **Hashing**: Perceptual hash generation
- **Color**: Dominant color extraction
- **Face**: Face detection and counting
- **Forensics**: Advanced forensic analysis
- **Reverse Search**: Multi-engine image search
- **Blockchain**: C2PA, NFT, signature verification
- **Privacy**: PII detection and risk assessment

### 2. Steganography Detection System

**Detection Methods**:
- LSB (Least Significant Bit) analysis
- Statistical analysis for steganographic patterns
- Pattern recognition
- Confidence scoring

**Detection Process**:
- Image analysis for hidden data patterns
- Statistical anomaly detection
- Pattern matching
- Confidence calculation

### 3. Metadata Extraction System

**Extracted Information**:
- **EXIF**: Camera settings, timestamps, GPS coordinates
- **GPS**: Location data with coordinates
- **Camera Info**: Make, model, serial number
- **Editing History**: Software used, edit timestamps
- **Technical**: Resolution, color space, compression

### 4. Storage & State

- **No persistent DB**: SnapSpeak does not write its own database tables
- **Stateless API**: Analyses are computed on demand and returned in the response
- **Ephemeral files**: Temporary files (e.g., DeepFace JPEGs) are immediately cleaned up
- **Fingerprints & reports**: Returned to the caller for external storage, search, or reporting

## Memory Management

SnapSpeak itself does not maintain an in-process results cache; heavy vision models are
lazy-loaded inside `utils.vision_analyzer` and reused there. Images are downsampled
aggressively before heavy NumPy work (stego, PRNU, FFT, KMeans) to keep RAM usage
bounded even for large uploads.

## API Reference

### Core Endpoints

**POST /snapspeak_ai/api/analyze/**
```json
Request: multipart/form-data
  file: <binary image>

Response: {
  "basic_caption": "short_or_detailed_caption",
  "vision_analysis": {...},         // BLIP/YOLO + LLM enriched vision data
  "gemini_analysis": {...},        // alias of vision_analysis for backward compatibility
  "ai_detection": {...},           // heuristic AI-generated vs real assessment
  "metadata": {...},               // deep metadata + use case hints
  "color_analysis": {...},         // dominant colors + scheme
  "steganography": {...},          // multi-method stego analysis + risk/confidence
  "image_hashes": {...},           // perceptual + cryptographic hashes
  "technical_analysis": {...},     // edge/blur/contrast metrics
  "faces": {...},                  // counts, locations, optional LLM analysis
  "processing_time": 2.31,
  "gemini_enabled": true,
  "deepface_available": true
}
```

**POST /snapspeak_ai/api/forensics/*`
- `/forensics/camera-fingerprint` – PRNU-like sensor fingerprint + hashes
- `/forensics/location-intelligence` – GPS extraction and geo-privacy context
- `/forensics/edit-history` – software/timeline + stego/technical clues
- `/forensics/validate-timestamp` – timestamp vs sun, timezone, historical weather

**POST /snapspeak_ai/api/stego/*`
- `/stego/deep-scan` – ensemble stego analysis (LSB/RS/histogram/metadata/DCT)
- `/stego/extract-payload` – naive LSB payload extraction + format hints
- `/stego/tool-identification` – heuristic stego tool fingerprinting
- `/stego/statistical-analysis` – direct access to core stego metrics

**POST /snapspeak_ai/api/reverse-search/*`
- Fingerprint-only endpoints ready to plug into external reverse image search services

**POST /snapspeak_ai/api/vision/*`
- Advanced objects, scene understanding, OCR, face attributes, document parsing

**POST /snapspeak_ai/api/blockchain/*`
- C2PA, NFT, and digital signature stubs for wiring to real provenance services

**POST /snapspeak_ai/api/compare/*`
- Visual-diff, batch similarity matrices, and edit-clue comparisons

**POST /snapspeak_ai/api/privacy/*`
- PII-focused OCR and privacy risk scoring / auto-redaction hints

**POST /snapspeak_ai/api/quality/*`
- Technical quality metrics and heuristic aesthetic scoring

**POST /snapspeak_ai/api/batch/*`
- Batch uploads and similarity/quality summaries

**POST /snapspeak_ai/api/export/*`
- JSON forensic reports and package payloads suitable for PDFs or archives

### Rate Limits

Rate limiting is enforced per-endpoint using centralized decorators. Typical limits:

| Endpoint group          | Per Minute | Per Hour |
|-------------------------|-----------|----------|
| `/api/analyze/`         | 5         | 50       |
| `/api/forensics/*`      | 5–10      | 50–100   |
| `/api/stego/*`          | 3–5       | 20–50    |
| `/api/reverse-search/*` | 3–10      | 20–100   |
| `/api/batch/*`          | 2         | 10       |

## Problem Statement & Solution

### Challenges Addressed

1. **Manual Image Analysis**
   - Problem: Investigators manually analyze images for forensics
   - Solution: Automated comprehensive image forensics with multiple analysis modules

2. **Steganography Detection**
   - Problem: Hidden data in images is difficult to detect manually
   - Solution: Automated steganography detection with multiple detection methods

3. **Metadata Extraction**
   - Problem: Extracting all available image metadata is time-consuming
   - Solution: Comprehensive metadata extraction with complete analysis

4. **Image Authentication**
   - Problem: Verifying image authenticity and provenance is complex
   - Solution: Blockchain verification and forensic analysis for authentication

5. **Privacy Concerns**
   - Problem: PII and privacy risks in images are hard to identify
   - Solution: Automated PII detection and privacy risk assessment

6. **Reverse Search**
   - Problem: Finding image sources and duplicates manually is inefficient
   - Solution: Multi-engine reverse image search for source identification

### Business Value

- **Time Savings**: Automated analysis vs. manual investigation
- **Comprehensive Coverage**: Multiple analysis modules vs. single-purpose tools
- **Accuracy**: AI-powered captioning and detection vs. manual analysis
- **Security**: Steganography detection and privacy analysis for security
- **Efficiency**: Batch processing accelerates workflows

## Deployment & Configuration

### Prerequisites
```bash
# Python dependencies
pip install flask pillow opencv-python imagehash exifread piexif transformers scikit-learn

# Optional: Local BLIP model
# Download BLIP model from Hugging Face

# Environment variables
export GROQ_API_KEY="your-groq-key"  # Optional for AI analysis
export HUGGINGFACE_API_KEY="your-hf-key"  # Optional for cloud BLIP
```

### Directory Structure
```
snapspeak_ai/
├── snapspeak_ai.py        # Main Flask blueprint
├── templates/
│   └── snapspeak_ai.html  # Frontend interface
├── models/                # BLIP model storage (optional)
└── utils/
    ├── vision_analyzer.py # Vision analysis utilities
    ├── llm_router.py      # Centralized LLM (in parent)
    └── security.py        # Rate limiting & validation (in parent)
```

### Configuration Options

**Analysis Settings**:
- Max image size: 10MB (configurable)
- Supported formats: JPEG, PNG, GIF, BMP, TIFF
- BLIP model: Cloud (Hugging Face) or local fallback

**Processing Settings**:
- Model caching: Singleton pattern (loaded once)
- Result caching: 5-minute TTL
- Batch size: Up to 10 images

## Performance Characteristics

### Response Time (P50/P95/P99)
- **Captioning (Cached)**: 100ms / 200ms / 500ms
- **Captioning (Cloud)**: 2s / 5s / 10s
- **Captioning (Local)**: 5s / 15s / 30s
- **Metadata Extraction**: 200ms / 500ms / 1s
- **Steganography Detection**: 1s / 3s / 5s
- **Full Analysis**: 5s / 15s / 30s

### Throughput
- Concurrent users: 15+ (tested)
- Analyses per hour: ~300 (5/min rate limit)
- Cache hit rate: ~40% for similar images
- Model loading: Once at startup (singleton)

### Scalability Considerations
- Horizontal: Deploy multiple Flask instances behind load balancer
- Vertical: GPU acceleration for BLIP model inference
- Caching: Add Redis layer for distributed caching
- Storage: Use cloud storage for model files

## Monitoring & Observability

### Built-in Logging
```python
logger.info(f"[ANALYZE] Image: {filename}, Types: {analysis_types}")
logger.info(f"[STEGO] Detected: {detected}, Confidence: {confidence}")
logger.error(f"[ERROR] Analysis failed: {exception}")
```

### Key Metrics
- Analysis counts by type
- Steganography detection rates
- Metadata extraction success rates
- Cache hit/miss rates
- Average analysis latency
- Model usage (cloud vs. local)

## Security Features

**OWASP-Compliant Protection**:
- Rate limiting: 5 requests/min, 50/hour (analysis endpoints)
- Input validation: File size limits (10MB), format validation
- HTML sanitization for XSS prevention
- Secure file handling
- Temporary file cleanup

## Future Enhancements

1. **Advanced Features**
   - Deepfake detection
   - Advanced tampering detection
   - Real-time analysis streaming
   - Custom model fine-tuning

2. **Integration Capabilities**
   - SIEM connectors for security teams
   - Cloud storage integration
   - API webhooks for automation
   - Database storage for analysis history

3. **Performance Optimization**
   - GPU acceleration for all models
   - Distributed processing with Celery
   - Advanced caching strategies
   - Model quantization for faster inference

## License & Compliance

- **Framework**: MIT License (Flask)
- **Computer Vision**: Apache 2.0 License (OpenCV)
- **AI Models**: Hugging Face Terms of Service, Groq Cloud Terms
- **Data Privacy**: Images processed temporarily, no persistent storage
- **Security Standards**: Follows OWASP Top 10 guidelines

---