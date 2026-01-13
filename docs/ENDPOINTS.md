# INFOSIGHT 3.0 - API Endpoints Documentation

## Base URL
All endpoints are prefixed with their respective blueprint path:
- `/webseeker` - WebSeeker endpoints
- `/portscanner` - PortScanner endpoints
- `/enscan` - Site Index endpoints
- `/filescanner` - File Fender endpoints
- `/infocrypt` - InfoCrypt endpoints
- `/osint` - TrackLyst endpoints
- `/donna` - DONNA AI endpoints
- `/snapspeak_ai` - SnapSpeak AI endpoints
- `/trueshot_ai` - TrueShot AI endpoints
- `/infosight_ai` - InfoSight AI endpoints
- `/lana_ai` - LANA AI endpoints
- `/cybersentry_ai` - CyberSentry AI endpoints
- `/inkwell_ai` - InkWell AI endpoints

## Rate Limiting
All endpoints are rate-limited:
- **Public endpoints**: 20 req/min, 100 req/hour
- **API endpoints**: 10 req/min, 200 req/hour
- **Resource-intensive**: 5 req/min, 50 req/hour

## Common Response Format
```json
{
  "success": true/false,
  "data": {...},
  "error": "error message if failed"
}
```

---

## 🔍 RECONNAISSANCE

### WebSeeker

#### `GET /webseeker/`
Main interface page

#### `POST /webseeker/api/analyze`
Analyze website security
- **Body**: `{"domain": "example.com", "scanType": "quick|comprehensive|stealth"}`
- **Rate Limit**: 5/min, 50/hour

#### `GET /webseeker/api/quick-check?domain=example.com`
Quick domain check
- **Rate Limit**: 20/min, 200/hour

---

### PortScanner

#### `GET /portscanner/`
Main interface page

#### `GET /portscanner/api/scan-types`
Get available scan types

#### `POST /portscanner/api/scan`
Perform port scan
- **Body**: `{"target": "192.168.1.1", "scanType": "intense_scan", "ports": "1-1024"}`
- **Rate Limit**: 10/min, 200/hour

---

### Site Index (EnScan)

#### `GET /enscan/`
Main interface page

#### `GET /enscan/docs`
Documentation page

#### `POST /enscan/api/scan`
Scan domain
- **Body**: `{"domain": "example.com"}`
- **Rate Limit**: 5/min, 50/hour

#### `GET /enscan/api/health`
Health check

#### `POST /enscan/api/groq-reasoning`
AI-powered domain reasoning
- **Body**: `{"domain": "example.com", "data": {...}}`

---

## 🎯 DETECTION

### File Fender (FileScanner)

#### `GET /filescanner/`
Main interface page

#### `POST /filescanner/upload`
Upload and scan file
- **Body**: Multipart form data with file
- **Rate Limit**: 5/min, 50/hour

#### `POST /filescanner/hash-check`
Check file hash
- **Body**: `{"hash": "sha256_hash"}`

#### `POST /filescanner/encrypt`
Encrypt file
- **Body**: Multipart form data with file and encryption parameters

#### `GET /filescanner/download-encrypted/<encryption_id>`
Download encrypted file

#### `POST /filescanner/decrypt`
Decrypt file
- **Body**: Multipart form data with encrypted file

#### `POST /filescanner/batch-scan`
Batch file scanning
- **Body**: Multipart form data with multiple files

---

## 🛡️ PROTECTION

### InfoCrypt

#### `GET /infocrypt/`
Main interface page

#### `GET /infocrypt/health`
Health check

#### `POST /infocrypt/api/hash`
Generate hash
- **Body**: `{"text": "input", "algorithm": "SHA-256"}`

#### `POST /infocrypt/api/generate-key`
Generate encryption key

#### `POST /infocrypt/api/encrypt`
Encrypt text/file
- **Body**: `{"text": "data", "algorithm": "AES-256-CBC", "key": "..."}`

#### `POST /infocrypt/api/decrypt`
Decrypt text/file
- **Body**: `{"ciphertext": "...", "algorithm": "AES-256-CBC", "key": "..."}`

#### `POST /infocrypt/api/compare`
Compare hashes
- **Body**: `{"hash1": "...", "hash2": "..."}`

#### `POST /infocrypt/api/verify`
Verify hash
- **Body**: `{"text": "...", "hash": "...", "algorithm": "SHA-256"}`

#### `GET /infocrypt/api/algorithms`
Get available algorithms

---

## 🧠 INTELLIGENCE

### TrackLyst (OSINT)

#### `GET /osint/`
Main interface page

#### `POST /osint/api/search`
Search username across platforms
- **Body**: `{"username": "username"}`
- **Rate Limit**: 10/min, 200/hour

#### `POST /osint/api/validate-url`
Validate URL
- **Body**: `{"url": "https://..."}`

#### `POST /osint/api/batch-validate`
Batch validate URLs
- **Body**: `{"urls": ["url1", "url2"]}`

#### `GET /osint/api/history`
Get search history

#### `GET /osint/api/platforms`
Get available platforms

#### `GET /osint/api/stats`
Get statistics

#### `POST /osint/api/compare`
Compare usernames
- **Body**: `{"usernames": ["user1", "user2"]}`

#### `GET /osint/health`
Health check

---

### DONNA AI

#### `GET /donna/`
Main interface page

#### `GET /donna/health`
Health check

#### `POST /donna/search`
Dark web search
- **Body**: `{"query": "search term", "sources": ["ahmia", "tor.link"]}`
- **Rate Limit**: 5/min, 50/hour

#### `POST /donna/export-pdf`
Export investigation report as PDF
- **Body**: `{"investigation_id": "..."}`

#### `GET /donna/history`
Get investigation history

#### `GET /donna/investigation/<inv_id>`
Get specific investigation

#### `POST /donna/clear-cache`
Clear cache

#### `GET /donna/stats`
Get statistics

#### `GET /donna/system-info`
Get system information

---

### SnapSpeak AI

#### `GET /snapspeak_ai/`
Main interface page

#### `POST /snapspeak_ai/api/analyze/`
Main image analysis
- **Body**: Multipart form data with image
- **Rate Limit**: 5/min, 50/hour

#### `POST /snapspeak_ai/api/forensics/camera-fingerprint`
Camera fingerprint analysis

#### `POST /snapspeak_ai/api/forensics/location-intelligence`
Location intelligence extraction

#### `POST /snapspeak_ai/api/forensics/edit-history`
Edit history detection

#### `POST /snapspeak_ai/api/forensics/validate-timestamp`
Timestamp validation

#### `POST /snapspeak_ai/api/stego/deep-scan`
Deep steganography scan

#### `POST /snapspeak_ai/api/stego/extract-payload`
Extract steganographic payload

#### `POST /snapspeak_ai/api/stego/tool-identification`
Identify steganography tool

#### `POST /snapspeak_ai/api/stego/statistical-analysis`
Statistical steganography analysis

#### `POST /snapspeak_ai/api/reverse-search/multi-engine`
Multi-engine reverse image search

#### `POST /snapspeak_ai/api/reverse-search/provenance`
Image provenance analysis

#### `POST /snapspeak_ai/api/reverse-search/find-duplicates`
Find duplicate images

#### `POST /snapspeak_ai/api/reverse-search/copyright-check`
Copyright check

#### `POST /snapspeak_ai/api/vision/advanced-objects`
Advanced object detection

#### `POST /snapspeak_ai/api/vision/scene-understanding`
Scene understanding

#### `POST /snapspeak_ai/api/vision/ocr-advanced`
Advanced OCR

#### `POST /snapspeak_ai/api/vision/face-attributes`
Face attribute analysis

#### `POST /snapspeak_ai/api/vision/document-parse`
Document parsing

#### `POST /snapspeak_ai/api/blockchain/c2pa-verify`
C2PA verification

#### `POST /snapspeak_ai/api/blockchain/nft-check`
NFT check

#### `POST /snapspeak_ai/api/blockchain/digital-signature`
Digital signature verification

#### `POST /snapspeak_ai/api/compare/visual-diff`
Visual difference comparison

#### `POST /snapspeak_ai/api/compare/batch-similarity`
Batch similarity check

#### `POST /snapspeak_ai/api/compare/detect-edits`
Detect edits

#### `POST /snapspeak_ai/api/privacy/pii-detect`
PII detection

#### `POST /snapspeak_ai/api/privacy/risk-assessment`
Privacy risk assessment

#### `POST /snapspeak_ai/api/privacy/auto-redact`
Auto redaction

#### `POST /snapspeak_ai/api/quality/technical-assessment`
Technical quality assessment

#### `POST /snapspeak_ai/api/quality/aesthetic-score`
Aesthetic scoring

#### `POST /snapspeak_ai/api/quality/professional-report`
Generate professional report

#### `POST /snapspeak_ai/api/batch/upload-folder`
Batch upload folder

#### `POST /snapspeak_ai/api/batch/check-progress`
Check batch progress

#### `GET /snapspeak_ai/api/batch/results/<batch_id>`
Get batch results

#### `POST /snapspeak_ai/api/workflows/create-pipeline`
Create analysis pipeline

#### `POST /snapspeak_ai/api/export/pdf-report`
Export PDF report

#### `POST /snapspeak_ai/api/export/forensic-package`
Export forensic package

#### `POST /snapspeak_ai/api/visualize/heatmap`
Generate heatmap visualization

#### `POST /snapspeak_ai/api/visualize/timeline`
Generate timeline visualization

---

### TrueShot AI

#### `GET /trueshot_ai/`
Main interface page

#### `POST /trueshot_ai/analyze`
Analyze image authenticity
- **Body**: Multipart form data with image
- **Rate Limit**: 5/min, 50/hour

#### `GET /trueshot_ai/health`
Health check

#### `POST /trueshot_ai/batch-analyze`
Batch image analysis
- **Body**: Multipart form data with multiple images

---

### InfoSight AI

#### `GET /infosight_ai/`
Main interface page

#### `GET /infosight_ai/health`
Health check

#### `GET /infosight_ai/api-status`
Check API status

#### `POST /infosight_ai/generate-text`
Generate text
- **Body**: `{"prompt": "...", "style": "..."}`
- **Rate Limit**: 5/min, 50/hour

#### `POST /infosight_ai/generate-image`
Generate image
- **Body**: `{"prompt": "...", "model": "flux", "style": "..."}`
- **Rate Limit**: 5/min, 50/hour

#### `POST /infosight_ai/generate-both`
Generate both text and image
- **Body**: `{"prompt": "...", "style": "..."}`

#### `POST /infosight_ai/enhance-prompt`
Enhance prompt
- **Body**: `{"prompt": "..."}`

#### `GET /infosight_ai/stats`
Get statistics

#### `GET /infosight_ai/styles`
Get available styles

#### `POST /infosight_ai/batch-generate`
Batch generation
- **Body**: `{"prompts": ["...", "..."], "type": "text|image|both"}`

#### `POST /infosight_ai/clear-cache`
Clear cache

#### `POST /infosight_ai/reset-rate-limit`
Reset rate limit

#### `GET /infosight_ai/history`
Get generation history

#### `POST /infosight_ai/favorites`
Add to favorites
- **Body**: `{"item_id": "..."}`

#### `GET /infosight_ai/favorites`
Get favorites

---

### LANA AI

#### `GET /lana_ai/`
Main interface page

#### `POST /lana_ai/listen`
Start voice listening
- **Rate Limit**: 20/min, 200/hour

#### `GET /lana_ai/get_response`
Get AI response

#### `GET /lana_ai/get_transcript`
Get transcript

#### `POST /lana_ai/text_input`
Text input
- **Body**: `{"query": "..."}`
- **Rate Limit**: 20/min, 200/hour

#### `GET /lana_ai/audio/<filename>`
Get audio file

#### `GET /lana_ai/health`
Health check

#### `GET /lana_ai/get_history`
Get conversation history

---

### CyberSentry AI

#### `GET /cybersentry_ai/`
Main interface page

#### `POST /cybersentry_ai/ask`
Ask security question
- **Body**: `{"question": "..."}`
- **Rate Limit**: 10/min, 200/hour

#### `POST /cybersentry_ai/reload-responses`
Reload responses database

#### `GET /cybersentry_ai/stats`
Get statistics

#### `POST /cybersentry_ai/threat-analysis`
Threat analysis
- **Body**: `{"threat": "..."}`

#### `GET /cybersentry_ai/analytics`
Get analytics

#### `POST /cybersentry_ai/execute-code`
Execute code (sandboxed)
- **Body**: `{"code": "..."}`

#### `GET /cybersentry_ai/history`
Get query history

#### `POST /cybersentry_ai/ask-all`
Ask multiple questions
- **Body**: `{"questions": ["...", "..."]}`

---

### InkWell AI

#### `GET /inkwell_ai/`
Main interface page

#### `GET /inkwell_ai/favicon.ico`
Favicon

#### `GET /inkwell_ai/api/health`
Health check

#### `POST /inkwell_ai/api/optimize`
Optimize prompt
- **Body**: `{"prompt": "...", "strategies": [...], "level": "moderate", "use_ai": true}`
- **Rate Limit**: 10/min, 100/hour

#### `POST /inkwell_ai/api/analyze`
Analyze prompt
- **Body**: `{"prompt": "..."}`
- **Rate Limit**: 20/min, 200/hour

#### `POST /inkwell_ai/api/variations`
Generate prompt variations
- **Body**: `{"prompt": "...", "count": 3}`

#### `GET /inkwell_ai/api/insights/<user_id>`
Get user insights

#### `GET /inkwell_ai/api/templates`
Get prompt templates

#### `POST /inkwell_ai/api/templates`
Create prompt template
- **Body**: `{"name": "...", "template": "..."}`

#### `POST /inkwell_ai/api/batch`
Batch prompt optimization
- **Body**: `{"prompts": ["...", "..."]}`

#### `GET /inkwell_ai/api/batch/<job_id>`
Get batch job status

---

## Error Codes

- **400**: Bad Request - Invalid input
- **403**: Forbidden - Access denied
- **404**: Not Found - Resource not found
- **429**: Too Many Requests - Rate limit exceeded
- **500**: Internal Server Error - Server error

## Authentication

Currently, endpoints do not require authentication. All security is handled via:
- Rate limiting
- Input validation
- API key management (server-side only)
