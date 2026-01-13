# SITE INDEX - Domain Security Intelligence Platform

## Description

EnScan is a comprehensive domain security assessment platform that provides real-time analysis of DNS infrastructure, SSL/TLS certificates, security headers, email authentication protocols, port scanning, and vulnerability detection. It combines automated parallel scanning with AI-powered security reasoning to deliver actionable intelligence for security professionals and network administrators.

## Goal

To automate the tedious process of manual domain security audits by providing comprehensive security assessments across multiple attack vectors in under 20 seconds. The platform enables security professionals to quickly identify misconfigurations, vulnerabilities, and compliance gaps through a unified intelligence gathering system with AI-enhanced analysis and professional reporting.

## Core Functionality

1. **DNS Reconnaissance**: Multi-record query system (A, AAAA, MX, NS, TXT, CNAME, SOA, CAA, SRV, PTR) with 6-server fallback mechanism (Google DNS, Cloudflare, OpenDNS, Quad9) and 5-second timeout protection

2. **SSL/TLS Certificate Analysis**: Comprehensive X.509 certificate validation with expiration monitoring, cipher suite evaluation, protocol version checking, vulnerability detection (SSLv2/v3, weak ciphers), and multi-hash fingerprinting (SHA-256, SHA-1, MD5)

3. **Security Headers Assessment**: Analyzes 10 critical security headers (HSTS, CSP, X-Frame-Options, X-Content-Type-Options, Referrer-Policy, Permissions-Policy, COEP, COOP, CORP) with weighted scoring system and letter grading (A+ to F)

4. **Email Security Validation**: Automated SPF record parsing, DMARC policy analysis, DKIM selector discovery (22 common selectors), and MX record enumeration with risk-based scoring

5. **Port & Service Detection**: Scans 27 common ports with service identification, risk classification (HIGH/MEDIUM/LOW), and dangerous HTTP method detection (PUT, DELETE, TRACE, CONNECT)

6. **Vulnerability Scanning**: Subdomain takeover detection, technology stack fingerprinting (CMS, frameworks, analytics), WHOIS intelligence gathering, and geolocation mapping

7. **AI-Powered Security Reasoning**: Server-side LLM integration for contextual security analysis, generating detailed explanations for scores, vulnerabilities, and actionable recommendations

8. **Risk Assessment Engine**: Calculates overall security score (0-10) using weighted algorithm and generates comprehensive risk reports with prioritized issue lists

## Modules Used

- **flask.Blueprint**: Web framework for modular routing and application organization
- **dns.resolver (dnspython)**: Multi-resolver DNS query system with automatic fallback
- **ssl + OpenSSL.crypto**: SSL/TLS certificate parsing, validation, and fingerprint generation
- **socket**: Low-level network operations for port scanning and SSL connections
- **whois**: Domain registration information retrieval and WHOIS record parsing
- **requests**: HTTP client for security header analysis and web scraping
- **concurrent.futures.ThreadPoolExecutor**: Parallel execution of I/O-bound operations (DNS, ports, HTTP)
- **hashlib**: Cryptographic hash generation for certificate fingerprints
- **datetime + pytz**: Timezone-aware timestamp handling and expiration calculations
- **re**: Regular expression pattern matching for domain validation and parsing
- **json + dataclasses**: Structured data serialization and type-safe results
- **logging**: Application-wide structured logging with severity levels
- **core.llm_router**: Centralized LLM integration with Groq Cloud + Ollama fallback chain
- **utils.security**: OWASP-compliant rate limiting, input validation, and schema enforcement

## Technology Stack

- **Backend**: Python 3.8+ with Flask Blueprint architecture
- **Concurrency**: ThreadPoolExecutor for parallel I/O operations (10 workers default)
- **DNS**: dnspython with multi-server fallback (Google, Cloudflare, OpenDNS, Quad9)
- **SSL/TLS**: pyOpenSSL for X.509 certificate parsing and validation
- **Domain Intelligence**: python-whois for registration data
- **HTTP Client**: requests library with custom user-agent rotation
- **AI Integration**: Groq Cloud LLM (llama3-70b) + Ollama local fallback
- **Security**: Custom rate limiter (5 req/min, 50 req/hour), schema-based validation
- **Frontend**: Vanilla JavaScript with dynamic tab rendering and real-time progress indicators
- **Styling**: Modern CSS3 with glassmorphism effects and responsive design

## Problem It Solves

1. **Manual Domain Audits**: Automates 8+ security checks that would take 30+ minutes manually into a single 15-20 second scan

2. **Certificate Management**: Identifies expired or misconfigured SSL certificates before they cause service outages, with 30-day expiration warnings

3. **Email Spoofing Prevention**: Validates SPF, DMARC, and DKIM configurations to prevent domain impersonation and phishing attacks

4. **Security Header Compliance**: Detects missing critical security headers (HSTS, CSP) that expose applications to XSS, clickjacking, and MITM attacks

5. **Attack Surface Mapping**: Enumerates open ports and dangerous services that increase the attack surface area

6. **Configuration Drift**: Identifies security regressions and misconfigurations across multiple security domains in a single unified report

7. **Vulnerability Assessment**: Detects subdomain takeover risks, outdated TLS protocols, weak ciphers, and dangerous HTTP methods

8. **Threat Prioritization**: Provides risk-based scoring to help security teams prioritize remediation efforts based on actual threat levels

## Memory Management

**Caching Strategy**:
- DNS resolver internal cache with 1-hour TTL to reduce redundant lookups
- LLM response caching per session (category + domain hash) to prevent duplicate API calls
- No persistent storage - all scan data stored in response object only
- Request-scoped memory model ensures automatic cleanup after response completion

**Concurrency Model**:
- ThreadPoolExecutor with 10 worker threads for I/O-bound operations (DNS queries, port scans, HTTP requests)
- Parallel DNS queries for all record types simultaneously (2-3 second total vs. 10+ seconds sequential)
- Concurrent port scanning with 2-second timeout per port (15 parallel connections)
- Automatic thread pool shutdown and resource cleanup after scan completion

**Resource Management**:
- Per-operation timeouts prevent resource exhaustion (5s DNS, 10s SSL, 2s port)
- Global scan timeout of 15 seconds with graceful degradation
- Socket automatic closure after use
- Memory-efficient streaming for large responses
- Rate limiting prevents DoS attacks and memory exhaustion (5/min, 50/hour)

**Performance Characteristics**:
- Base application memory: ~50MB (Flask + dependencies)
- Peak memory during scan: ~120MB (concurrent operations)
- ThreadPoolExecutor overhead: ~15MB per worker thread
- Typical scan duration: 8-15 seconds for full assessment
- Supports 10-15 concurrent users with 8-worker configuration

## License & Compliance

- **Framework**: MIT License (Flask)
- **DNS Library**: BSD License (dnspython)
- **SSL Library**: Apache 2.0 License (pyOpenSSL)
- **Data Privacy**: Scan data processed temporarily, no persistent storage
- **Security Standards**: Follows OWASP Top 10 guidelines
- **Usage**: Authorized domain security assessment only

---