# WebSeeker - Web Security Scanner

## Overview

WebSeeker is a comprehensive web security scanner that combines VirusTotal's threat intelligence with Nmap vulnerability scanning. It provides real-time threat assessment by detecting malware, analyzing open ports, and identifying security weaknesses across domains and URLs, delivering unified security analysis in a single interface.

## Core Architecture

### Hybrid Security System
- **Threat Intelligence Layer**: VirusTotal API integration for malware detection
- **Network Scanning Layer**: Nmap integration for port and vulnerability scanning
- **Intelligence Layer**: IP information and geolocation data
- **Analysis Layer**: AI-powered executive summaries via centralized LLM router

### Key Features

**1. Domain/URL Malware Detection**
- Scans domains and URLs using VirusTotal's multi-engine threat detection
- 70+ antivirus engine aggregation
- Real-time threat assessment
- Risk scoring and classification

**2. Port Vulnerability Analysis**
- Performs Nmap-based port scanning to identify open ports and services
- Service detection and version identification
- Vulnerability assessment
- Risk classification

**3. IP Information & Geolocation**
- Retrieves detailed IP information including geolocation data
- IPInfo API integration
- Location intelligence
- Network information

**4. AbuseIPDB Reputation Checking**
- Checks IP reputation against AbuseIPDB database
- Reputation scoring
- Threat classification
- Historical abuse data

**5. AI-Powered Analysis**
- Generates executive summaries using centralized LLM router
- Context-aware security analysis
- Actionable recommendations
- Comprehensive reporting

**6. Real-time Threat Assessment**
- Provides immediate security scoring and risk classification
- Multi-source threat intelligence
- Unified security assessment
- Prioritized recommendations

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Backend** | Flask (Python 3.8+) | Web framework & API routing |
| **Network Scanning** | python-nmap | Network port scanning |
| **Threat Intelligence** | VirusTotal API | Malware detection |
| **IP Intelligence** | IPInfo API | IP geolocation and information |
| **Reputation** | AbuseIPDB API | IP reputation checking |
| **AI Analysis** | Groq Cloud LLM | AI-powered analysis via router |
| **Local AI** | Ollama | Offline fallback processing |
| **Concurrency** | ThreadPoolExecutor | Concurrent scanning |
| **Frontend** | Vanilla JS + CSS3 | Interactive UI |

## System Components

### 1. Security Analysis Pipeline

```
Domain/URL Input → Validation → Parallel Analysis
           ↓
         VirusTotal Scan → Nmap Port Scan → IP Intelligence
           ↓
         AbuseIPDB Check → AI Analysis → Result Aggregation
           ↓
         Response Delivery
```

**Analysis Modules**:
- **VirusTotal**: Domain/URL malware detection
- **Nmap**: Port scanning and service detection
- **IPInfo**: IP geolocation and information
- **AbuseIPDB**: IP reputation checking
- **AI Analysis**: Executive summary generation

### 2. Threat Intelligence System

**VirusTotal Integration**:
- Domain/URL scanning via API
- 70+ antivirus engine aggregation
- Threat scoring and classification
- Historical threat data

**AbuseIPDB Integration**:
- IP reputation checking
- Abuse history analysis
- Threat classification
- Confidence scoring

### 3. Network Scanning System

**Nmap Integration**:
- Port scanning for open ports
- Service detection and version identification
- Vulnerability assessment
- Risk classification

### 4. Caching Architecture

**Response Caching**:
- TTL-based caching (1 hour default)
- Hash-based cache keys (domain/URL)
- Automatic invalidation on cache expiry
- Memory-efficient storage

## Memory Management

### Caching Strategy
```python
# Response Cache (TTL-based)
_cache = {}  # target_hash → (results, timestamp)
TTL = 3600  # 1 hour

# No persistent storage
```

**Memory Footprint**:
- Base: ~60MB (Flask + dependencies)
- Response cache: ~15MB (100 entries × 150KB avg)
- ThreadPoolExecutor: ~10MB per worker thread
- Nmap process: ~20MB per scan
- VirusTotal API responses: ~5MB (cached)

**Cleanup Mechanisms**:
- Automatic TTL-based cache eviction
- Thread pool automatic cleanup
- Nmap process cleanup after scan completion
- Temporary data cleanup after analysis
- Session cleanup on browser close

## API Reference

### Core Endpoints

**POST /webseeker/api/analyze**
```json
Request: {
  "target": "string (domain or URL)",
  "scan_ports": boolean (optional, default: true),
  "check_reputation": boolean (optional, default: true)
}

Response: {
  "success": true,
  "target": "string",
  "virustotal": {
    "detected": boolean,
    "engines": int,
    "detections": int,
    "risk_score": float,
    "threat_level": "LOW|MEDIUM|HIGH|CRITICAL"
  },
  "port_scan": {
    "open_ports": [
      {
        "port": int,
        "protocol": "tcp|udp",
        "service": "string",
        "version": "string",
        "risk": "LOW|MEDIUM|HIGH"
      }
    ],
    "total_ports_scanned": int,
    "vulnerabilities": [...]
  },
  "ip_info": {
    "ip": "string",
    "country": "string",
    "city": "string",
    "org": "string",
    "asn": "string"
  },
  "reputation": {
    "abuse_confidence": int,
    "is_public": boolean,
    "is_whitelisted": boolean,
    "usage_type": "string",
    "threat_level": "LOW|MEDIUM|HIGH"
  },
  "ai_summary": {
    "summary": "executive_summary_text",
    "recommendations": ["array of recommendations"],
    "risk_assessment": "string"
  },
  "overall_risk": "LOW|MEDIUM|HIGH|CRITICAL",
  "timestamp": "2025-01-13T10:30:00"
}
```

**GET /webseeker/api/quick-check**
```json
Request: {
  "target": "string (domain or URL)"
}

Response: {
  "success": true,
  "target": "string",
  "quick_assessment": {
    "threat_detected": boolean,
    "risk_level": "LOW|MEDIUM|HIGH|CRITICAL",
    "summary": "brief_summary"
  },
  "timestamp": "2025-01-13T10:30:00"
}
```

### Rate Limits

| Endpoint | Per Minute | Per Hour |
|----------|-----------|----------|
| /analyze | 5 | 50 |
| /quick-check | 10 | 100 |

## Problem Statement & Solution

### Challenges Addressed

1. **Fragmented Security Tools**
   - Problem: Users need multiple tools for different security checks
   - Solution: Unified platform combining threat intelligence, port scanning, and IP analysis

2. **Manual Threat Analysis**
   - Problem: Security professionals manually analyze threats from multiple sources
   - Solution: Automated threat intelligence gathering and AI-powered analysis

3. **Time-Consuming Security Audits**
   - Problem: Comprehensive security assessments take significant time
   - Solution: Quick comprehensive security assessments in seconds

4. **Limited Visibility**
   - Problem: Single-source threat intelligence provides limited visibility
   - Solution: Multi-source threat intelligence aggregation for better visibility

5. **Complex Reporting**
   - Problem: Compiling security reports from multiple sources is complex
   - Solution: AI-powered executive summaries automatically generated

### Business Value

- **Time Savings**: Automated analysis vs. manual multi-tool investigation
- **Comprehensive Coverage**: Multi-source intelligence vs. single-source checks
- **Efficiency**: Unified interface vs. switching between multiple tools
- **Intelligence**: AI-powered analysis vs. manual report compilation
- **Visibility**: Complete security picture vs. fragmented information

## Deployment & Configuration

### Prerequisites
```bash
# Python dependencies
pip install flask python-nmap requests

# System dependencies
# Nmap (system installation required)
# Npcap (Windows, for packet capture)

# Environment variables
export VIRUSTOTAL_API_KEY="your-vt-key"  # Required
export IPINFO_API_KEY="your-ipinfo-key"  # Optional
export ABUSEIPDB_API_KEY="your-abuseipdb-key"  # Optional
export GROQ_API_KEY="your-groq-key"  # Optional for AI analysis
```

### Directory Structure
```
webseeker/
├── webseeker.py            # Main Flask blueprint
├── templates/
│   └── webseeker.html      # Frontend interface
└── utils/
    ├── llm_router.py       # Centralized LLM (in parent)
    ├── local_llm_utils.py  # Local LLM fallback (in parent)
    ├── llm_logger.py       # LLM logging (in parent)
    └── security.py         # Rate limiting & validation (in parent)
```

### Configuration Options

**Scan Settings**:
- Default timeout: 300 seconds (comprehensive scans)
- Max concurrent scans: 3
- Cache TTL: 1 hour (3600 seconds)

**API Settings**:
- VirusTotal: Required for malware detection
- IPInfo: Optional for IP geolocation
- AbuseIPDB: Optional for IP reputation
- Groq: Optional for AI analysis

## Performance Characteristics

### Response Time (P50/P95/P99)
- **Quick Check**: 2s / 5s / 10s
- **Full Analysis (Cached)**: 100ms / 200ms / 500ms
- **Full Analysis (New)**: 30s / 60s / 120s
- **VirusTotal Only**: 5s / 15s / 30s
- **Port Scan Only**: 20s / 40s / 60s

### Throughput
- Concurrent users: 15+ (tested)
- Analyses per hour: ~300 (5/min rate limit)
- Cache hit rate: ~40% for repeated targets
- Max concurrent scans: 3 (configurable)

### Scalability Considerations
- Horizontal: Deploy multiple Flask instances behind load balancer
- Vertical: Increase thread pool size for more concurrent scans
- Caching: Add Redis layer for distributed caching
- Database: Add PostgreSQL for persistent scan history

## Monitoring & Observability

### Built-in Logging
```python
logger.info(f"[ANALYZE] Target: {target}, Threat: {threat_detected}")
logger.info(f"[SCAN] Ports found: {ports_count}, Vulnerabilities: {vuln_count}")
logger.error(f"[ERROR] Analysis failed: {exception}")
```

### Key Metrics
- Analysis counts by target type (domain/URL)
- Threat detection rates
- Port scan results distribution
- Cache hit/miss rates
- Average analysis latency
- API usage (VirusTotal, IPInfo, AbuseIPDB)
- AI analysis usage

## Security Features

**OWASP-Compliant Protection**:
- Rate limiting: 5 requests/min, 50/hour (analysis endpoints)
- Input validation: Domain/URL format, length limits
- Schema-based request validation
- HTML sanitization for XSS prevention
- API key security (server-side only)

**Network Security**:
- Respects network policies
- Configurable scan intensity
- Timeout protection prevents resource exhaustion
- Rate limiting prevents API abuse

## Future Enhancements

1. **Advanced Features**
   - Scheduled scans
   - Scan comparison (before/after)
   - Automated monitoring
   - Advanced vulnerability detection

2. **Integration Capabilities**
   - SIEM connectors for security teams
   - Ticketing system integration
   - API webhooks for automation
   - Database storage for scan history

3. **Performance Optimization**
   - Distributed scanning with multiple workers
   - Background job processing
   - Advanced caching strategies
   - Real-time streaming results

## License & Compliance

- **Framework**: MIT License (Flask)
- **Nmap**: Nmap License (GPL v2)
- **VirusTotal**: VirusTotal API Terms of Service
- **Data Privacy**: Scan data processed temporarily, cached for 1 hour
- **Security Standards**: Follows OWASP Top 10 guidelines
- **Usage**: Authorized security assessment only

---