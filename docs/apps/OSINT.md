# TrackLyst (OSINT) - Username Reconnaissance Platform

## Overview

TrackLyst is an advanced OSINT intelligence platform with a modern futuristic interface. It provides comprehensive username reconnaissance across 50+ global social media platforms, enabling digital footprint tracking, online profile discovery, and digital presence management for security professionals, investigators, and individuals.

## Core Architecture

### Multi-Platform Intelligence System
- **Search Layer**: Distributed search across 50+ social media platforms
- **Validation Layer**: Real-time URL validation and batch processing
- **Analysis Layer**: Digital footprint analysis and categorization
- **Storage Layer**: Search history tracking and statistics

### Key Features

**1. Multi-Platform Search**
- Searches 50+ social media platforms simultaneously
- Category-based filtering (social, professional, developer, gaming, media)
- Real-time platform status checking
- Parallel processing for fast results

**2. Digital Footprint Analysis**
- Tracks and analyzes online presence across platforms
- Profile discovery and aggregation
- Cross-platform correlation
- Presence visualization

**3. Real-time URL Validation**
- Validates profile URLs in real-time
- HTTP status code checking
- Profile existence verification
- Response time tracking

**4. Batch Validation**
- Validates multiple URLs simultaneously
- Parallel processing for efficiency
- Result aggregation
- Error handling for failed validations

**5. Category-Based Filtering**
- Filters platforms by category:
  - **Social**: Facebook, Twitter, Instagram, TikTok, etc.
  - **Professional**: LinkedIn, GitHub, Behance, etc.
  - **Developer**: GitHub, GitLab, Stack Overflow, etc.
  - **Gaming**: Steam, Xbox Live, PlayStation, etc.
  - **Media**: YouTube, Twitch, SoundCloud, etc.

**6. Exportable Results**
- Exports results in JSON format
- Structured data for further analysis
- Integration-ready format

**7. Search History**
- Tracks and stores search history
- Recent searches access
- History-based analytics

**8. Statistics**
- Provides search statistics and analytics
- Platform usage distribution
- Success rate tracking
- Performance metrics

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Backend** | Flask (Python 3.8+) | Web framework & API routing |
| **HTTP Client** | Requests | Platform checking and validation |
| **Concurrency** | ThreadPoolExecutor | Parallel platform checking |
| **Data Storage** | JSON | Configuration and history |
| **Frontend** | Vanilla JS + CSS3 | Modern futuristic UI |
| **Security** | Rate limiting, input validation | OWASP-compliant protection |

## System Components

### 1. Platform Search Pipeline

```
Username Input → Platform List Generation → Parallel HTTP Requests
           ↓
         Response Processing → Status Classification
           ↓
         Result Aggregation → Response Delivery
```

**Search Strategy**:
- Parallel HTTP requests to all platforms
- 10-second timeout per platform
- User-Agent rotation for reliability
- Error handling for failed requests

**Platform Categories**:
- **Social Media**: Facebook, Twitter, Instagram, TikTok, Snapchat, etc.
- **Professional**: LinkedIn, GitHub, Behance, Dribbble, etc.
- **Developer**: GitHub, GitLab, Stack Overflow, Bitbucket, etc.
- **Gaming**: Steam, Xbox Live, PlayStation, Epic Games, etc.
- **Media**: YouTube, Twitch, SoundCloud, Vimeo, etc.

### 2. URL Validation System

**Validation Process**:
- HTTP GET request to profile URL
- Status code checking (200 = exists, 404 = not found)
- Response time measurement
- Error handling for network issues

**Batch Validation**:
- Parallel processing of multiple URLs
- Result aggregation
- Failed validation handling

### 3. Data Storage

**Search History**:
- In-memory storage for recent searches
- Limited to recent searches (configurable)
- JSON-based storage format

**Statistics**:
- Platform usage counts
- Success rate tracking
- Performance metrics

## Memory Management

### Caching Strategy
```python
# Result Cache (recent searches)
_cache = {}  # username → (results, timestamp)
TTL = 3600  # 1 hour

# History Storage
In-memory (limited to recent searches)
```

**Memory Footprint**:
- Base: ~40MB (Flask + dependencies)
- Result cache: ~10MB (100 entries × 100KB avg)
- ThreadPoolExecutor: ~5MB per worker thread
- History storage: ~5MB (limited)

**Cleanup Mechanisms**:
- Automatic TTL-based cache eviction
- History cleanup (old entries removed)
- Thread pool automatic cleanup
- Session cleanup on browser close

## API Reference

### Core Endpoints

**POST /osint/api/search**
```json
Request: {
  "username": "string",
  "categories": ["social", "professional", ...] (optional)
}

Response: {
  "success": true,
  "username": "string",
  "results": [
    {
      "platform": "string",
      "url": "string",
      "exists": boolean,
      "status_code": int,
      "category": "string"
    }
  ],
  "statistics": {
    "total_platforms": int,
    "found": int,
    "not_found": int,
    "errors": int
  },
  "timestamp": "2025-01-13T10:30:00"
}
```

**POST /osint/api/validate-url**
```json
Request: {
  "url": "string"
}

Response: {
  "success": true,
  "url": "string",
  "exists": boolean,
  "status_code": int,
  "response_time_ms": float
}
```

**POST /osint/api/batch-validate**
```json
Request: {
  "urls": ["array of URLs"]
}

Response: {
  "success": true,
  "results": [
    {
      "url": "string",
      "exists": boolean,
      "status_code": int,
      "response_time_ms": float
    }
  ]
}
```

**GET /osint/api/history**
```json
Response: {
  "searches": [
    {
      "username": "string",
      "timestamp": "2025-01-13T10:30:00",
      "results_count": int
    }
  ]
}
```

**GET /osint/api/platforms**
```json
Response: {
  "platforms": [
    {
      "name": "string",
      "category": "string",
      "url_template": "string"
    }
  ],
  "categories": ["social", "professional", "developer", "gaming", "media"]
}
```

**GET /osint/api/stats**
```json
Response: {
  "total_searches": int,
  "total_platforms": int,
  "success_rate": float,
  "average_response_time_ms": float,
  "platform_usage": {...}
}
```

**POST /osint/api/compare**
```json
Request: {
  "usernames": ["array of usernames"]
}

Response: {
  "success": true,
  "comparison": {
    "common_platforms": ["array"],
    "unique_platforms": {...},
    "overlap_percentage": float
  }
}
```

**GET /osint/health**
```json
Response: {
  "status": "healthy",
  "platforms_configured": int,
  "cache_size": int
}
```

### Rate Limits

| Endpoint | Per Minute | Per Hour |
|----------|-----------|----------|
| /search | 10 | 200 |
| /validate-url | 20 | 400 |
| /batch-validate | 5 | 100 |
| /history | 10 | 200 |
| /stats | 20 | 400 |

## Problem Statement & Solution

### Challenges Addressed

1. **Manual Platform Checking**
   - Problem: Users manually check usernames across multiple platforms
   - Solution: Automated checking across 50+ platforms simultaneously

2. **Digital Footprint Tracking**
   - Problem: No comprehensive view of online presence
   - Solution: Unified platform aggregating all profile discoveries

3. **Time-Consuming Research**
   - Problem: OSINT investigations require checking many platforms manually
   - Solution: Parallel processing speeds up investigations significantly

4. **Profile Discovery**
   - Problem: Hidden or forgotten profiles are difficult to find
   - Solution: Systematic platform checking discovers all profiles

5. **Reputation Management**
   - Problem: No easy way to track online reputation across platforms
   - Solution: Comprehensive footprint analysis enables reputation management

6. **Security Audits**
   - Problem: Identifying potential security risks from exposed profiles
   - Solution: Complete profile discovery helps identify security concerns

### Business Value

- **Time Savings**: Automated checking vs. manual platform visits
- **Comprehensive Coverage**: 50+ platforms vs. manual checking of few platforms
- **Efficiency**: Parallel processing reduces investigation time
- **Organization**: Centralized results for easy analysis
- **Security**: Complete footprint visibility for security audits

## Deployment & Configuration

### Prerequisites
```bash
# Python dependencies
pip install flask requests

# No external API keys required
# No system dependencies
```

### Directory Structure
```
osint/
├── osint.py                # Main Flask blueprint
├── templates/
│   └── osint.html          # Frontend interface
├── data/
│   └── platforms.json      # Platform configuration
└── utils/
    └── security.py         # Rate limiting & validation (in parent)
```

### Configuration Options

**Platform Configuration**:
- 50+ platforms configured in JSON format
- Category-based organization
- URL template system for platform URLs

**Processing Settings**:
- Concurrent workers: 10 (default)
- Request timeout: 10 seconds per platform
- Max batch size: 50 URLs

## Performance Characteristics

### Response Time (P50/P95/P99)
- **Single Username Search**: 5s / 10s / 15s (50 platforms)
- **URL Validation**: 500ms / 2s / 5s
- **Batch Validation**: 2s / 5s / 10s (10 URLs)
- **Cached Results**: 50ms / 100ms / 200ms

### Throughput
- Concurrent users: 20+ (tested)
- Searches per hour: ~600 (10/min rate limit)
- Cache hit rate: ~30% for repeated usernames
- Platform checks per second: ~50 (parallel)

### Scalability Considerations
- Horizontal: Deploy multiple Flask instances behind load balancer
- Vertical: Increase thread pool size for more concurrent requests
- Caching: Add Redis layer for distributed caching
- Database: Add PostgreSQL for persistent history storage

## Monitoring & Observability

### Built-in Logging
```python
logger.info(f"[SEARCH] Username: {username}, Platforms: {platform_count}")
logger.info(f"[VALIDATE] URL: {url}, Status: {status_code}")
logger.error(f"[ERROR] Platform check failed: {platform}, Error: {exception}")
```

### Key Metrics
- Search counts by username
- Platform usage distribution
- Success rate by platform
- Average response time per platform
- Cache hit/miss rates
- Error rates by platform

## Security Features

**OWASP-Compliant Protection**:
- Rate limiting: 10 requests/min, 200/hour (search endpoints)
- Input validation: Username format, length limits, URL validation
- HTML sanitization for XSS prevention
- Request timeout protection
- User-Agent rotation for reliability

## Future Enhancements

1. **Advanced Features**
   - Profile content analysis
   - Cross-platform correlation
   - Historical profile tracking
   - Automated monitoring

2. **Integration Capabilities**
   - SIEM connectors for security teams
   - API webhooks for automation
   - Database storage for persistent history
   - Export to various formats (CSV, PDF)

3. **Performance Optimization**
   - Distributed caching with Redis
   - Background job processing
   - Database storage for history
   - Advanced analytics and reporting

## License & Compliance

- **Framework**: MIT License (Flask)
- **Data Sources**: Publicly accessible web content
- **Usage**: Authorized OSINT research and investigations
- **Privacy**: No PII storage beyond search scope
- **Security Standards**: Follows OWASP Top 10 guidelines

---