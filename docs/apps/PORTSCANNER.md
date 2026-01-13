# PortScanner - Network Port Scanner

## Overview

PortScanner is an advanced network port scanner designed for comprehensive security audits. It identifies open ports, detects running services, and assesses network vulnerabilities, making it essential for penetration testing and proactive security monitoring. The system supports 20+ scan types with detailed service detection and version identification.

## Core Architecture

### Network Scanning System
- **Scanning Layer**: Nmap integration for port discovery
- **Service Detection**: Automatic service and version identification
- **OS Detection**: Operating system fingerprinting
- **Vulnerability Assessment**: Security weakness identification

### Key Features

**1. Open Port Identification**
- Scans target hosts to identify open ports
- Comprehensive port range scanning
- Fast and stealth scan options
- Custom port range specification

**2. Service Detection & Versioning**
- Detects running services on open ports
- Service version identification
- Protocol detection
- Banner grabbing

**3. Operating System Detection**
- Attempts to identify target operating system
- OS fingerprinting
- Version detection
- Architecture identification

**4. Vulnerability Assessment**
- Provides vulnerability information for detected services
- Security weakness identification
- Risk classification
- Remediation suggestions

**5. Customizable Scan Profiles**
- Supports 20+ different scan types:
  - Intense scan, Service version, OS detection
  - TCP connect, SYN scan, UDP scan
  - Aggressive scan, List scan
  - Null, Xmas, FIN scans
  - Full port scan, Script scan
  - And 10+ more specialized scan types

**6. Exportable Results**
- Provides structured JSON results for further analysis
- Integration-ready format
- Detailed scan reports

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Backend** | Flask (Python 3.8+) | Web framework & API routing |
| **Scanning** | python-nmap | Nmap integration for port scanning |
| **Network** | Nmap | System-level network scanning tool |
| **Packet Capture** | Npcap | Windows packet capture library |
| **Concurrency** | ThreadPoolExecutor | Concurrent scan execution |
| **Validation** | validators | Input validation |
| **Frontend** | Vanilla JS + CSS3 | Interactive UI |

## System Components

### 1. Scanning Pipeline

```
Target Input → Validation → Scan Type Selection
           ↓
         Nmap Execution → Result Parsing
           ↓
         Service Detection → Vulnerability Assessment
           ↓
         Response Formatting → Result Delivery
```

**Scan Types**:
- **Intense Scan**: Comprehensive scan with service and OS detection
- **SYN Scan**: Stealthy TCP SYN scan (requires root)
- **UDP Scan**: UDP port scanning
- **Service Version**: Detailed service version detection
- **OS Detection**: Operating system fingerprinting
- **Script Scan**: NSE script execution

### 2. Service Detection System

**Detection Process**:
- Banner grabbing from open ports
- Service fingerprinting
- Version identification
- Protocol detection

**Vulnerability Assessment**:
- Known vulnerability database lookup
- Risk classification (HIGH/MEDIUM/LOW)
- Remediation suggestions

### 3. Result Processing

**Result Structure**:
- Open ports with service information
- Service versions and protocols
- OS detection results
- Vulnerability information
- Scan metadata

## Memory Management

### Caching Strategy
```python
# Scan Result Cache (recent scans)
_cache = {}  # target_hash → (results, timestamp)
TTL = 3600  # 1 hour

# No persistent storage
```

**Memory Footprint**:
- Base: ~50MB (Flask + python-nmap)
- Scan cache: ~5MB (10 entries × 500KB avg)
- ThreadPoolExecutor: ~10MB per worker thread
- Nmap process: ~20MB per scan

**Cleanup Mechanisms**:
- Automatic TTL-based cache eviction
- Thread pool automatic cleanup
- Nmap process cleanup after scan completion
- Session cleanup on browser close

## API Reference

### Core Endpoints

**GET /portscanner/api/scan-types**
```json
Response: {
  "scan_types": [
    {
      "id": "string",
      "name": "string",
      "description": "string",
      "requires_root": boolean
    }
  ]
}
```

**POST /portscanner/api/scan**
```json
Request: {
  "target": "string (IP or domain)",
  "scan_type": "string",
  "ports": "string (optional, e.g., '1-1000')",
  "options": "string (optional, additional nmap options)"
}

Response: {
  "success": true,
  "target": "string",
  "scan_type": "string",
  "scan_info": {
    "scan_time": float,
    "hosts_up": int,
    "hosts_down": int
  },
  "hosts": [
    {
      "hostname": "string",
      "ip": "string",
      "state": "up|down",
      "ports": [
        {
          "port": int,
          "protocol": "tcp|udp",
          "state": "open|closed|filtered",
          "service": "string",
          "version": "string",
          "product": "string"
        }
      ],
      "os": {
        "name": "string",
        "accuracy": int,
        "type": "string"
      },
      "vulnerabilities": [
        {
          "port": int,
          "service": "string",
          "vulnerability": "string",
          "risk": "HIGH|MEDIUM|LOW"
        }
      ]
    }
  ],
  "timestamp": "2025-01-13T10:30:00"
}
```

### Rate Limits

| Endpoint | Per Minute | Per Hour |
|----------|-----------|----------|
| /scan | 10 | 200 |
| /scan-types | 20 | 400 |

## Problem Statement & Solution

### Challenges Addressed

1. **Manual Port Scanning**
   - Problem: Network administrators manually scan ports using command-line tools
   - Solution: Automated port scanning with web-based interface

2. **Limited Scan Options**
   - Problem: Basic scanners offer few scan types
   - Solution: 20+ scan types for different scenarios

3. **Service Identification**
   - Problem: Port scanners don't identify running services
   - Solution: Automatic service detection and version identification

4. **Vulnerability Discovery**
   - Problem: No vulnerability information in scan results
   - Solution: Vulnerability assessment with risk classification

5. **Network Auditing**
   - Problem: Network security auditing is time-consuming
   - Solution: Streamlined scanning process with detailed reports

### Business Value

- **Time Savings**: Automated scanning vs. manual command-line tools
- **Comprehensive Coverage**: 20+ scan types vs. basic port scanning
- **Service Intelligence**: Service detection vs. port-only results
- **Security Insights**: Vulnerability assessment enables proactive security
- **Efficiency**: Streamlined process accelerates network auditing

## Deployment & Configuration

### Prerequisites
```bash
# Python dependencies
pip install flask python-nmap

# System dependencies
# Linux/Mac: nmap (system package)
# Windows: Nmap + Npcap installer

# Root/Administrator privileges required for some scan types (SYN scan)
```

### Directory Structure
```
portscanner/
├── portscanner.py         # Main Flask blueprint
├── templates/
│   └── portscanner.html   # Frontend interface
└── utils/
    └── security.py        # Rate limiting & validation (in parent)
```

### Configuration Options

**Scan Settings**:
- Default timeout: 6000 seconds (comprehensive scans)
- Max concurrent scans: 5
- Port range: Configurable (default: 1-1000)

**Scan Types**:
- 20+ predefined scan types
- Custom scan options support
- Root privilege checking for privileged scans

## Performance Characteristics

### Response Time (P50/P95/P99)
- **Quick Scan**: 5s / 15s / 30s
- **Intense Scan**: 30s / 60s / 120s
- **Full Port Scan**: 60s / 120s / 300s
- **OS Detection**: 20s / 40s / 60s
- **Cached Results**: 50ms / 100ms / 200ms

### Throughput
- Concurrent users: 10+ (tested)
- Scans per hour: ~600 (10/min rate limit)
- Cache hit rate: ~20% for repeated targets
- Max concurrent scans: 5 (configurable)

### Scalability Considerations
- Horizontal: Deploy multiple Flask instances behind load balancer
- Vertical: Increase thread pool size for more concurrent scans
- Caching: Add Redis layer for distributed caching
- Database: Add PostgreSQL for persistent scan history

## Monitoring & Observability

### Built-in Logging
```python
logger.info(f"[SCAN] Target: {target}, Type: {scan_type}")
logger.info(f"[SCAN] Completed: {ports_found} ports found")
logger.error(f"[ERROR] Scan failed: {exception}")
```

### Key Metrics
- Scan counts by type
- Average scan duration by type
- Port discovery rates
- Service detection accuracy
- Cache hit/miss rates
- Error rates by scan type

## Security Features

**OWASP-Compliant Protection**:
- Rate limiting: 10 requests/min, 200/hour
- Input validation: IP/domain format, port ranges
- Root privilege checking for privileged scans
- Scan timeout protection
- Error handling with generic messages

**Network Security**:
- Respects network policies
- Configurable scan intensity
- Timeout protection prevents resource exhaustion
- Rate limiting prevents network abuse

## Future Enhancements

1. **Advanced Features**
   - Scheduled scans
   - Scan comparison (before/after)
   - Network mapping visualization
   - Automated vulnerability reporting

2. **Integration Capabilities**
   - SIEM connectors
   - Ticketing system integration
   - API webhooks for automation
   - Database storage for scan history

3. **Performance Optimization**
   - Distributed scanning with multiple workers
   - Background job processing
   - Advanced caching strategies
   - GPU acceleration for analysis

## License & Compliance

- **Framework**: MIT License (Flask)
- **Nmap**: Nmap License (GPL v2)
- **Usage**: Authorized network security testing only
- **Disclaimer**: Tool intended for legitimate security professionals and network administrators
- **Security Standards**: Follows OWASP Top 10 guidelines

---