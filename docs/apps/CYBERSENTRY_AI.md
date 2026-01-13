# CyberSentry AI - Advanced Cybersecurity Intelligence Platform

## Overview

CyberSentry AI is an enterprise-grade cybersecurity assistant that provides real-time threat intelligence, security guidance, and vulnerability analysis through a hybrid AI architecture. The system combines cloud-based LLM reasoning with local fallback capabilities and a curated knowledge base to deliver 24/7 security operations support.

## Core Architecture

### Hybrid Intelligence System
- **Primary Layer**: Cloud LLM via centralized router (Groq API)
- **Fallback Layer**: Local Ollama instance for offline operation
- **Knowledge Layer**: Fuzzy-matched JSON database with 300+ security patterns
- **Smart Routing**: Automatic failover with <100ms detection time

### Key Features

**1. Multi-Source Query Resolution**
- Fuzzy matching against curated security knowledge base (70%+ accuracy threshold)
- Cloud LLM for complex threat analysis and novel scenarios
- Local LLM fallback ensuring zero downtime
- Parallel source comparison mode for critical queries

**2. Threat Intelligence Engine**
- Real-time threat level classification (CRITICAL/HIGH/MEDIUM/LOW)
- Risk scoring algorithm with indicator extraction
- Actionable recommendations based on threat severity
- Threat intelligence caching with MD5 query hashing

**3. Conversation Management**
- SQLite-based persistent storage for audit trails
- Conversation history with search and replay
- Export capabilities (JSON, TXT, Markdown, PDF)
- User session tracking with IP-based identification

**4. Advanced UI/UX**
- Real-time status monitoring dashboard
- Reading mode with adjustable content density
- Section navigation with smooth scrolling
- Code block syntax highlighting with one-click copy
- Voice input support (Web Speech API)
- Responsive design with dark/light themes

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Backend** | Flask (Python 3.8+) | Web framework & API routing |
| **AI Core** | Groq Cloud LLM | Primary intelligence via router |
| **Local AI** | Ollama | Offline fallback processing |
| **Database** | SQLite3 | Conversation history & analytics |
| **Matching** | FuzzyWuzzy + difflib | Knowledge base search |
| **Rendering** | Markdown2 | Response formatting |
| **Frontend** | Vanilla JS + CSS3 | Interactive UI with Web APIs |

## System Components

### 1. Question Processing Pipeline

```
User Query → Input Validation → Fuzzy Matching (JSON) 
           ↓ (if no match)
         LLM Router → Cloud LLM (Groq)
           ↓ (if failure)
         Local LLM (Ollama) → Response Formatting
```

**Fuzzy Matching Algorithm**:
- Token set ratio (40% weight)
- Partial ratio (20% weight)  
- Token sort ratio (30% weight)
- Sequence matching (10% weight)
- Keyword extraction with stop-word filtering

### 2. LLM Integration

**Centralized Router** (`core.llm_router.generate_text`):
- Automatic provider selection (cloud-first, local fallback)
- Request logging with latency tracking
- Token usage monitoring
- Error handling with graceful degradation

**Specialized Prompts**:
- System prompt optimized for cybersecurity domain
- Markdown formatting instructions
- Code block and bullet point guidance
- Ethical hacking compliance guardrails

### 3. Database Schema

**Conversations Table**:
- `id`, `user_id`, `question`, `answer`
- `source`, `model_used`, `confidence`
- `timestamp`, `response_time`

**Threat Intel Cache**:
- `query_hash`, `threat_level`, `risk_score`
- `indicators` (JSON), `recommendations` (JSON)
- TTL-based invalidation

**Analytics Events**:
- `event_type`, `event_data` (JSON)
- Performance metrics and usage patterns

### 4. Security Features

**OWASP-Compliant Protection**:
- Rate limiting: 20 requests/min, 200/hour
- Input validation with max length enforcement (2000 chars)
- HTML sanitization for XSS prevention
- SQL injection protection via parameterized queries
- CSRF token validation on state-changing operations

**Sandboxed Code Execution**:
- Temporary file isolation
- 5-second timeout enforcement
- Subprocess confinement
- Output capture and sanitization

## Memory Management

### Caching Strategy
```python
# L1: LRU Cache (in-memory)
@lru_cache(maxsize=100)  # Frequent queries

# L2: Response Cache (5-min TTL)
_responses_cache = None  # JSON knowledge base
_cache_time = None

# L3: Database Cache
threat_intel table  # Persistent threat data
```

**Memory Footprint**:
- Base: ~50MB (Flask + dependencies)
- Response cache: ~5MB (JSON data)
- LRU cache: ~10MB (100 entries × 100KB avg)
- SQLite: Growing (conversation history)

**Cleanup Mechanisms**:
- Automatic LRU eviction on cache overflow
- 5-minute TTL for JSON response cache
- Database vacuuming on history cleanup
- Session cleanup on browser close

## API Reference

### Core Endpoints

**POST /cybersentry_ai/ask**
```json
Request: {
  "question": "string (max 2000 chars)",
  "force_source": "json|ai|null"
}

Response: {
  "answer": "string (formatted HTML)",
  "source": "JSON|AI|Fallback",
  "model_used": "groq|local|json",
  "confidence": "high|medium|low",
  "can_regenerate": boolean,
  "threat_analysis": {
    "threat_level": "CRITICAL|HIGH|MEDIUM|LOW",
    "risk_score": 0-100,
    "indicators": ["string"],
    "recommendations": ["string"]
  }
}
```

**POST /cybersentry_ai/ask-all**
- Parallel processing across all sources
- Returns comparison view with JSON, Cloud LLM, and Local LLM results
- Useful for critical queries requiring consensus

**POST /cybersentry_ai/threat-analysis**
- Dedicated endpoint for threat assessment
- Query hash caching for repeated analysis
- Returns structured threat intelligence

**GET /cybersentry_ai/analytics**
- Conversation statistics
- Source distribution metrics
- Threat level distribution
- Recent activity monitoring

### Rate Limits

| Endpoint | Per Minute | Per Hour |
|----------|-----------|----------|
| /ask | 20 | 200 |
| /ask-all | 10 | 50 |
| /threat-analysis | 10 | 100 |
| /reload-responses | 5 | 20 |

## Problem Statement & Solution

### Challenges Addressed

1. **Security Knowledge Fragmentation**
   - Problem: Security teams juggle multiple documentation sources
   - Solution: Unified AI interface with curated knowledge base

2. **24/7 Availability Requirements**
   - Problem: Cloud API outages disrupt security operations
   - Solution: Local LLM fallback ensures continuous operation

3. **Context-Aware Guidance**
   - Problem: Generic security advice lacks operational relevance
   - Solution: Threat-aware responses with severity-based recommendations

4. **Query Response Latency**
   - Problem: Real-time incident response requires instant answers
   - Solution: Multi-tier caching (LRU → JSON → Database) with <500ms P95

5. **Audit Trail Compliance**
   - Problem: Security decisions must be documented
   - Solution: Complete conversation history with export capabilities

### Business Value

- **Reduced MTTD** (Mean Time To Detect): Instant threat queries vs. manual research
- **Improved MTTR** (Mean Time To Respond): Actionable guidance at query time
- **Team Productivity**: 24/7 "virtual security analyst" reduces on-call burden
- **Knowledge Retention**: Curated database prevents expertise loss during turnover

## Deployment & Configuration

### Prerequisites
```bash
# Python dependencies
pip install flask fuzzywuzzy markdown2 sqlite3

# Optional: Local LLM
ollama pull llama2  # or preferred model

# Environment variables
export GROQ_API_KEY="your-groq-key"  # Optional for cloud LLM
```

### Directory Structure
```
cybersentry_ai/
├── cybersentry_ai.py          # Main Flask blueprint
├── templates/
│   └── cybersentry_AI.html    # Frontend interface
├── data/
│   └── responses.json         # Knowledge base
├── utils/
│   ├── llm_router.py         # Centralized LLM integration
│   ├── local_llm_utils.py    # Ollama wrapper
│   ├── llm_logger.py         # Request logging
│   └── security.py           # Rate limiting & validation
└── cybersentry_ai.db         # SQLite database (auto-created)
```

### Knowledge Base Format
```json
[
  {
    "question": "What is SQL injection?",
    "answer": "Detailed explanation with examples...",
    "category": "web_security",
    "tags": ["injection", "database", "OWASP"]
  }
]
```

## Performance Characteristics

### Response Time (P50/P95/P99)
- **JSON Match**: 50ms / 120ms / 200ms
- **Cloud LLM**: 800ms / 2.5s / 5s
- **Local LLM**: 2s / 8s / 15s
- **Fallback Chain**: 3s / 10s / 20s (worst case)

### Throughput
- Concurrent users: 50+ (tested)
- Queries per second: 10-15 (with caching)
- Database writes: ~100 TPS (conversation history)

### Scalability Considerations
- Horizontal: Deploy multiple Flask instances behind load balancer
- Vertical: SQLite → PostgreSQL for high-volume deployments
- Caching: Add Redis layer for distributed caching

## Monitoring & Observability

### Built-in Logging
```python
# LLM request tracking
log_llm_request(app_name, source, query_length)
log_llm_success(app_name, source, response_length, latency_ms)
log_llm_error(app_name, source, exception, fallback=bool)
log_llm_fallback(from_source, to_source)
```

### Key Metrics
- LLM request counts by source (Groq/Ollama/JSON)
- Average response latency per source
- Fallback trigger frequency
- Threat level distribution
- User session duration

## Future Enhancements

1. **Integration Plugins**
   - SIEM connectors (Splunk, ELK)
   - Ticketing system webhooks (Jira, ServiceNow)
   - Vulnerability scanner APIs

2. **Advanced Features**
   - Multi-turn conversational context
   - Proactive threat hunting suggestions
   - Custom knowledge base training
   - Role-based access control (RBAC)

3. **Performance Optimization**
   - Vector database for semantic search (Pinecone/Weaviate)
   - Response streaming for long-form analysis
   - GraphQL API for flexible queries

## License & Compliance

- **Framework**: MIT License (Flask)
- **AI Models**: Groq Cloud Terms of Service
- **Data Privacy**: User conversations stored locally (GDPR-compliant with proper configuration)
- **Security Standards**: Follows OWASP Top 10 guidelines

---