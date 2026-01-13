from flask import Flask, request, jsonify, render_template, Blueprint, g
import re
import dns.resolver
import requests
from urllib.parse import urlparse
import concurrent.futures
import whois
from datetime import datetime
import socket
import ssl
import OpenSSL
from typing import Dict, List, Union, Optional
import hashlib
import time
import json
from dataclasses import dataclass, asdict
import logging
from functools import wraps
import traceback
import pytz
from dateutil import parser as date_parser
from utils.security import rate_limit_api, validate_request, InputValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create Blueprint
enscan = Blueprint("enscan", __name__, template_folder="templates")

# Log application initialization
logger.info("=" * 70)
logger.info("🛡️  EnScan - Initializing")
logger.info("=" * 70)


@dataclass
class ScanResult:
    """Structured scan result"""

    success: bool
    data: Optional[Dict] = None
    error: Optional[str] = None
    timing: float = 0.0


class DNSResolver:
    """Dedicated DNS resolution handler with multiple fallback servers"""

    def __init__(self):
        self.resolver = dns.resolver.Resolver(configure=False)
        # Multiple DNS servers for reliability
        self.resolver.nameservers = [
            "8.8.8.8",  # Google Primary
            "8.8.4.4",  # Google Secondary
            "1.1.1.1",  # Cloudflare Primary
            "1.0.0.1",  # Cloudflare Secondary
            "208.67.222.222",  # OpenDNS
            "9.9.9.9",  # Quad9
        ]
        self.resolver.timeout = 5
        self.resolver.lifetime = 10

    def query(self, domain: str, record_type: str) -> List[str]:
        """Query DNS with error handling and retries"""
        try:
            answers = self.resolver.resolve(domain, record_type)
            results = []
            for rdata in answers:
                value = str(rdata).strip('"').rstrip(".")
                results.append(value)
            return results
        except dns.resolver.NoAnswer:
            return []
        except dns.resolver.NXDOMAIN:
            raise ValueError(f"Domain {domain} does not exist")
        except dns.resolver.Timeout:
            raise TimeoutError(f"DNS query timeout for {record_type}")
        except Exception as e:
            logger.error(f"DNS query failed for {domain} ({record_type}): {str(e)}")
            raise Exception(f"DNS query failed: {str(e)}")


class SSLAnalyzer:
    """Advanced SSL/TLS certificate analyzer with comprehensive checks"""

    @staticmethod
    def analyze(domain: str) -> Dict:
        """Comprehensive SSL analysis with vulnerability checks"""
        try:
            context = ssl.create_default_context()

            with socket.create_connection((domain, 443), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=domain) as secure_sock:
                    # Get binary certificate
                    cert_bin = secure_sock.getpeercert(binary_form=True)
                    cert_dict = secure_sock.getpeercert()

                    # Parse with OpenSSL
                    x509 = OpenSSL.crypto.load_certificate(
                        OpenSSL.crypto.FILETYPE_ASN1, cert_bin
                    )

                    # Extract dates
                    not_before = datetime.strptime(
                        cert_dict["notBefore"], "%b %d %H:%M:%S %Y %Z"
                    )
                    not_after = datetime.strptime(
                        cert_dict["notAfter"], "%b %d %H:%M:%S %Y %Z"
                    )
                    days_remaining = (not_after - datetime.now()).days

                    # Get subject and issuer
                    subject = dict(x[0] for x in cert_dict.get("subject", []))
                    issuer = dict(x[0] for x in cert_dict.get("issuer", []))

                    # Get cipher and protocol
                    cipher = secure_sock.cipher()
                    protocol = secure_sock.version()

                    # Get extensions
                    san_list = cert_dict.get("subjectAltName", [])

                    # Calculate fingerprints
                    sha256_fp = hashlib.sha256(cert_bin).hexdigest().upper()
                    sha1_fp = hashlib.sha1(cert_bin).hexdigest().upper()
                    md5_fp = hashlib.md5(cert_bin).hexdigest().upper()

                    # Format fingerprints
                    sha256_formatted = ":".join(
                        [sha256_fp[i : i + 2] for i in range(0, len(sha256_fp), 2)]
                    )
                    sha1_formatted = ":".join(
                        [sha1_fp[i : i + 2] for i in range(0, len(sha1_fp), 2)]
                    )
                    md5_formatted = ":".join(
                        [md5_fp[i : i + 2] for i in range(0, len(md5_fp), 2)]
                    )

                    # Check for vulnerabilities
                    vulnerabilities = SSLAnalyzer._check_vulnerabilities(
                        protocol, cipher, days_remaining
                    )

                    # Calculate SSL score
                    ssl_score = SSLAnalyzer._calculate_ssl_score(
                        protocol, cipher, days_remaining, vulnerabilities
                    )

                    return {
                        "valid": days_remaining > 0,
                        "days_remaining": days_remaining,
                        "subject": subject,
                        "issuer": issuer,
                        "common_name": subject.get("commonName", "N/A"),
                        "organization": issuer.get("organizationName", "N/A"),
                        "valid_from": not_before.strftime("%Y-%m-%d %H:%M:%S UTC"),
                        "valid_until": not_after.strftime("%Y-%m-%d %H:%M:%S UTC"),
                        "serial_number": format(x509.get_serial_number(), "X"),
                        "signature_algorithm": x509.get_signature_algorithm().decode(),
                        "version": cert_dict["version"],
                        "san": san_list,
                        "san_count": len(san_list),
                        "protocol_version": protocol,
                        "cipher_suite": {
                            "name": cipher[0] if cipher else "Unknown",
                            "protocol": cipher[1] if cipher else "Unknown",
                            "bits": cipher[2] if cipher else 0,
                        },
                        "public_key_bits": x509.get_pubkey().bits(),
                        "fingerprint_sha256": sha256_formatted,
                        "fingerprint_sha1": sha1_formatted,
                        "fingerprint_md5": md5_formatted,
                        "self_signed": subject == issuer,
                        "vulnerabilities": vulnerabilities,
                        "ssl_score": ssl_score,
                        "expires_soon": days_remaining < 30,
                    }
        except socket.timeout:
            raise TimeoutError("SSL connection timeout")
        except ssl.SSLError as e:
            raise Exception(f"SSL Error: {str(e)}")
        except Exception as e:
            logger.error(f"SSL analysis failed: {str(e)}")
            raise Exception(f"SSL analysis failed: {str(e)}")

    @staticmethod
    def _check_vulnerabilities(
        protocol: str, cipher: tuple, days_remaining: int
    ) -> List[str]:
        """Check for SSL/TLS vulnerabilities"""
        vulnerabilities = []

        # Check protocol version
        if protocol in ["SSLv2", "SSLv3"]:
            vulnerabilities.append(f"Insecure protocol {protocol} (deprecated)")
        elif protocol in ["TLSv1.0", "TLSv1.1"]:
            vulnerabilities.append(
                f"Outdated protocol {protocol} (should upgrade to TLS 1.2+)"
            )

        # Check cipher strength
        if cipher and cipher[2] < 128:
            vulnerabilities.append(f"Weak cipher strength ({cipher[2]} bits)")

        # Check certificate expiration
        if days_remaining < 0:
            vulnerabilities.append("Certificate expired")
        elif days_remaining < 7:
            vulnerabilities.append("Certificate expires in less than 7 days")
        elif days_remaining < 30:
            vulnerabilities.append("Certificate expires in less than 30 days")

        return vulnerabilities

    @staticmethod
    def _calculate_ssl_score(
        protocol: str, cipher: tuple, days_remaining: int, vulnerabilities: List[str]
    ) -> float:
        """Calculate SSL security score"""
        score = 10.0

        # Deduct for protocol issues
        if protocol in ["SSLv2", "SSLv3"]:
            score -= 5
        elif protocol in ["TLSv1.0", "TLSv1.1"]:
            score -= 2

        # Deduct for cipher issues
        if cipher and cipher[2] < 128:
            score -= 3
        elif cipher and cipher[2] < 256:
            score -= 1

        # Deduct for expiration issues
        if days_remaining < 0:
            score -= 5
        elif days_remaining < 7:
            score -= 3
        elif days_remaining < 30:
            score -= 1

        return max(0, round(score, 1))


class SecurityHeadersAnalyzer:
    """Advanced security headers analyzer with detailed grading"""

    HEADERS_CONFIG = {
        "Strict-Transport-Security": {
            "weight": 2.5,
            "category": "critical",
            "description": "Enforces HTTPS connections and prevents protocol downgrade attacks",
            "recommendation": "Add: Strict-Transport-Security: max-age=31536000; includeSubDomains; preload",
        },
        "Content-Security-Policy": {
            "weight": 2.5,
            "category": "critical",
            "description": "Prevents XSS, clickjacking, and code injection attacks",
            "recommendation": "Implement a strict CSP policy with specific directives",
        },
        "X-Frame-Options": {
            "weight": 1.5,
            "category": "important",
            "description": "Protects against clickjacking attacks",
            "recommendation": "Add: X-Frame-Options: DENY or SAMEORIGIN",
        },
        "X-Content-Type-Options": {
            "weight": 1.0,
            "category": "important",
            "description": "Prevents MIME-type sniffing attacks",
            "recommendation": "Add: X-Content-Type-Options: nosniff",
        },
        "Referrer-Policy": {
            "weight": 0.8,
            "category": "recommended",
            "description": "Controls referrer information leakage",
            "recommendation": "Add: Referrer-Policy: strict-origin-when-cross-origin",
        },
        "Permissions-Policy": {
            "weight": 1.0,
            "category": "important",
            "description": "Controls browser feature access",
            "recommendation": "Add: Permissions-Policy with restricted features",
        },
        "X-XSS-Protection": {
            "weight": 0.5,
            "category": "legacy",
            "description": "Legacy XSS protection for older browsers",
            "recommendation": "Add: X-XSS-Protection: 1; mode=block",
        },
        "Cross-Origin-Embedder-Policy": {
            "weight": 0.7,
            "category": "advanced",
            "description": "Enables cross-origin isolation",
            "recommendation": "Add: Cross-Origin-Embedder-Policy: require-corp",
        },
        "Cross-Origin-Opener-Policy": {
            "weight": 0.7,
            "category": "advanced",
            "description": "Isolates browsing context group",
            "recommendation": "Add: Cross-Origin-Opener-Policy: same-origin",
        },
        "Cross-Origin-Resource-Policy": {
            "weight": 0.8,
            "category": "advanced",
            "description": "Prevents cross-origin resource loading",
            "recommendation": "Add: Cross-Origin-Resource-Policy: same-origin",
        },
    }

    @classmethod
    def analyze(cls, domain: str) -> Dict:
        """Analyze security headers with detailed assessment"""
        try:
            response = None
            final_url = None

            # Try HTTPS first, then HTTP
            for protocol in ["https", "http"]:
                try:
                    url = f"{protocol}://{domain}"
                    response = requests.get(
                        url,
                        timeout=10,
                        allow_redirects=True,
                        headers={
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                        },
                        verify=True,
                    )
                    final_url = response.url
                    break
                except requests.exceptions.SSLError:
                    if protocol == "https":
                        try:
                            response = requests.get(
                                url,
                                timeout=10,
                                allow_redirects=True,
                                headers={
                                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                                },
                                verify=False,
                            )
                            final_url = response.url
                            break
                        except:
                            continue
                except:
                    continue

            if not response:
                raise Exception("Could not connect to domain")

            headers = response.headers
            analyzed_headers = {}
            score = 0
            max_score = sum(h["weight"] for h in cls.HEADERS_CONFIG.values())

            for header_name, config in cls.HEADERS_CONFIG.items():
                value = headers.get(header_name, None)
                present = value is not None

                analyzed_headers[header_name] = {
                    "present": present,
                    "value": value if present else "Not Set",
                    "category": config["category"],
                    "description": config["description"],
                    "recommendation": config["recommendation"],
                    "weight": config["weight"],
                }

                if present:
                    score += config["weight"]

            # Calculate normalized score (0-10)
            normalized_score = round((score / max_score) * 10, 1)

            # Determine grade
            grade, grade_reasoning = cls._calculate_grade(
                normalized_score, analyzed_headers
            )

            # Analyze cookies
            cookies_analysis = cls._analyze_cookies(response.cookies)

            # Check for information disclosure
            info_disclosure = cls._check_info_disclosure(headers)

            return {
                "score": normalized_score,
                "max_score": 10,
                "grade": grade,
                "grade_reasoning": grade_reasoning,
                "headers": analyzed_headers,
                "missing_critical": [
                    name
                    for name, data in analyzed_headers.items()
                    if not data["present"] and data["category"] == "critical"
                ],
                "missing_important": [
                    name
                    for name, data in analyzed_headers.items()
                    if not data["present"] and data["category"] == "important"
                ],
                "server": headers.get("Server", "Not Disclosed"),
                "powered_by": headers.get("X-Powered-By", "Not Disclosed"),
                "final_url": final_url,
                "status_code": response.status_code,
                "redirect_count": len(response.history),
                "cookies": cookies_analysis,
                "https_enabled": (
                    final_url.startswith("https://") if final_url else False
                ),
                "info_disclosure": info_disclosure,
                "response_time": response.elapsed.total_seconds(),
            }
        except Exception as e:
            logger.error(f"Security headers analysis failed: {str(e)}")
            raise Exception(f"Security headers analysis failed: {str(e)}")

    @staticmethod
    def _calculate_grade(score: float, headers: Dict) -> tuple:
        """Calculate letter grade and reasoning"""
        if score >= 9.5:
            grade = "A+"
            reasoning = "Exceptional security configuration with comprehensive protection mechanisms"
        elif score >= 9.0:
            grade = "A"
            reasoning = (
                "Excellent security with all critical protections properly implemented"
            )
        elif score >= 8.0:
            grade = "A-"
            reasoning = "Very good security posture with minor improvements possible"
        elif score >= 7.0:
            grade = "B"
            reasoning = (
                "Good security foundation but missing some important protections"
            )
        elif score >= 6.0:
            grade = "C"
            reasoning = "Adequate security with several areas requiring improvement"
        elif score >= 4.0:
            grade = "D"
            reasoning = "Poor security configuration with significant vulnerabilities"
        else:
            grade = "F"
            reasoning = "Critical security failure requiring immediate remediation"

        return grade, reasoning

    @staticmethod
    def _analyze_cookies(cookies) -> Dict:
        """Comprehensive cookie security analysis"""
        total = len(cookies)
        secure = 0
        httponly = 0
        samesite = 0
        insecure = []

        for cookie in cookies:
            if cookie.secure:
                secure += 1
            if cookie.has_nonstandard_attr("HttpOnly"):
                httponly += 1
            if cookie.has_nonstandard_attr("SameSite"):
                samesite += 1

            # Track insecure cookies
            if not cookie.secure or not cookie.has_nonstandard_attr("HttpOnly"):
                insecure.append(cookie.name)

        return {
            "total": total,
            "secure": secure,
            "httponly": httponly,
            "samesite": samesite,
            "insecure_cookies": insecure,
            "security_score": (
                round((secure + httponly + samesite) / (total * 3) * 10, 1)
                if total > 0
                else 10
            ),
        }

    @staticmethod
    def _check_info_disclosure(headers: Dict) -> List[str]:
        """Check for information disclosure in headers"""
        disclosures = []

        if "Server" in headers:
            disclosures.append(f"Server version disclosed: {headers['Server']}")
        if "X-Powered-By" in headers:
            disclosures.append(f"Technology stack disclosed: {headers['X-Powered-By']}")
        if "X-AspNet-Version" in headers:
            disclosures.append(
                f"ASP.NET version disclosed: {headers['X-AspNet-Version']}"
            )
        if "X-AspNetMvc-Version" in headers:
            disclosures.append(
                f"ASP.NET MVC version disclosed: {headers['X-AspNetMvc-Version']}"
            )

        return disclosures


class EmailSecurityAnalyzer:
    """Comprehensive email security analyzer"""

    def __init__(self, dns_resolver: DNSResolver):
        self.resolver = dns_resolver

    def analyze(self, domain: str) -> Dict:
        """Analyze email security records with detailed assessment"""
        spf_analysis = self._analyze_spf(domain)
        dmarc_analysis = self._analyze_dmarc(domain)
        dkim_analysis = self._analyze_dkim(domain)
        mx_analysis = self._analyze_mx(domain)

        # Calculate email security score (0-10)
        score = 0
        max_score = 10

        # SPF scoring (3.3 points)
        if spf_analysis["configured"]:
            score += 2.5
            if spf_analysis.get("strict"):
                score += 0.8

        # DMARC scoring (3.3 points)
        if dmarc_analysis["configured"]:
            score += 2.0
            policy = dmarc_analysis.get("policy", "none")
            if policy == "reject":
                score += 1.3
            elif policy == "quarantine":
                score += 0.8

        # DKIM scoring (3.0 points)
        if dkim_analysis["configured"]:
            score += 3.0

        # MX records bonus (0.4 points)
        if mx_analysis["configured"]:
            score += 0.4

        return {
            "score": round(min(score, max_score), 1),
            "max_score": max_score,
            "spf": spf_analysis,
            "dmarc": dmarc_analysis,
            "dkim": dkim_analysis,
            "mx": mx_analysis,
            "recommendation": self._get_recommendation(
                spf_analysis, dmarc_analysis, dkim_analysis
            ),
            "risk_level": self._calculate_risk_level(score),
        }

    def _analyze_spf(self, domain: str) -> Dict:
        """Detailed SPF record analysis"""
        try:
            records = self.resolver.query(domain, "TXT")
            spf_record = None

            for record in records:
                if record.startswith("v=spf1"):
                    spf_record = record
                    break

            if not spf_record:
                return {
                    "configured": False,
                    "record": None,
                    "issues": ["SPF record not found"],
                }

            # Parse SPF mechanisms
            mechanisms = spf_record.split()[1:]

            # Check for common issues
            issues = []
            if len(mechanisms) > 10:
                issues.append(
                    "Too many DNS lookups (>10) - may cause validation failures"
                )

            all_mechanism = None
            for m in mechanisms:
                if "all" in m:
                    all_mechanism = m

            if not all_mechanism:
                issues.append('No "all" mechanism found')
            elif all_mechanism == "+all":
                issues.append("Permits all senders (+all) - extremely permissive")

            return {
                "configured": True,
                "record": spf_record,
                "mechanisms": mechanisms,
                "mechanism_count": len(mechanisms),
                "includes_all": all_mechanism is not None,
                "strict": "-all" in spf_record,
                "softfail": "~all" in spf_record,
                "all_mechanism": all_mechanism,
                "issues": issues,
            }
        except Exception as e:
            return {"configured": False, "record": None, "error": str(e)}

    def _analyze_dmarc(self, domain: str) -> Dict:
        """Detailed DMARC record analysis"""
        try:
            records = self.resolver.query(f"_dmarc.{domain}", "TXT")
            dmarc_record = None

            for record in records:
                if record.startswith("v=DMARC1"):
                    dmarc_record = record
                    break

            if not dmarc_record:
                return {
                    "configured": False,
                    "record": None,
                    "issues": ["DMARC record not found"],
                }

            # Parse DMARC policy
            policy = "none"
            if "p=reject" in dmarc_record:
                policy = "reject"
            elif "p=quarantine" in dmarc_record:
                policy = "quarantine"
            elif "p=none" in dmarc_record:
                policy = "none"

            # Extract percentage
            pct = "100"
            pct_match = re.search(r"pct=(\d+)", dmarc_record)
            if pct_match:
                pct = pct_match.group(1)

            # Extract reporting addresses
            rua = re.search(r"rua=([^;]+)", dmarc_record)
            ruf = re.search(r"ruf=([^;]+)", dmarc_record)

            # Check for issues
            issues = []
            if policy == "none":
                issues.append('Policy set to "none" - monitoring only, no enforcement')
            if int(pct) < 100:
                issues.append(f"Only {pct}% of emails are checked")
            if not rua:
                issues.append("No aggregate report address (rua) configured")

            return {
                "configured": True,
                "record": dmarc_record,
                "policy": policy,
                "percentage": f"{pct}%",
                "strict": policy == "reject",
                "rua": rua.group(1) if rua else None,
                "ruf": ruf.group(1) if ruf else None,
                "issues": issues,
            }
        except Exception as e:
            return {"configured": False, "record": None, "error": str(e)}

    def _analyze_dkim(self, domain: str) -> Dict:
        """Comprehensive DKIM analysis with extended selector list"""
        selectors = [
            "default",
            "google",
            "k1",
            "k2",
            "k3",
            "s1",
            "s2",
            "selector1",
            "selector2",
            "dkim",
            "mail",
            "mta",
            "smtp",
            "mandrill",
            "mailgun",
            "sendgrid",
            "amazonses",
            "dkim1",
            "dkim2",
            "key1",
            "key2",
            "mx",
            "email",
        ]

        found_selectors = []
        selector_details = {}

        for selector in selectors:
            try:
                records = self.resolver.query(f"{selector}._domainkey.{domain}", "TXT")
                if records:
                    found_selectors.append(selector)
                    selector_details[selector] = (
                        records[0] if records else "Record found"
                    )
            except:
                continue

        return {
            "configured": len(found_selectors) > 0,
            "selectors_found": found_selectors,
            "selector_details": selector_details,
            "total_checked": len(selectors),
            "issues": [] if len(found_selectors) > 0 else ["No DKIM selectors found"],
        }

    def _analyze_mx(self, domain: str) -> Dict:
        """Analyze MX records"""
        try:
            records = self.resolver.query(domain, "MX")
            mx_records = []

            for rdata in records:
                mx_records.append(
                    {
                        "priority": rdata.preference,
                        "host": str(rdata.exchange).rstrip("."),
                    }
                )

            # Sort by priority
            mx_records.sort(key=lambda x: x["priority"])

            return {
                "configured": len(mx_records) > 0,
                "records": mx_records,
                "count": len(mx_records),
            }
        except:
            return {"configured": False, "records": [], "count": 0}

    def _get_recommendation(self, spf: Dict, dmarc: Dict, dkim: Dict) -> str:
        """Generate detailed security recommendation"""
        if spf["configured"] and dmarc["configured"] and dkim["configured"]:
            if dmarc.get("policy") == "reject" and spf.get("strict"):
                return (
                    "✅ Excellent email security! All protections properly configured."
                )
            elif dmarc.get("policy") == "reject":
                return "⚠️ Good configuration. Consider using strict SPF (-all) for maximum protection."
            else:
                return '⚠️ Upgrade DMARC policy to "reject" for full protection against spoofing.'

        missing = []
        if not spf["configured"]:
            missing.append("SPF")
        if not dmarc["configured"]:
            missing.append("DMARC")
        if not dkim["configured"]:
            missing.append("DKIM")

        return f'🔴 Critical: Configure {", ".join(missing)} immediately to prevent email spoofing and phishing attacks.'

    def _calculate_risk_level(self, score: float) -> str:
        """Calculate overall email security risk level"""
        if score >= 8.5:
            return "LOW"
        elif score >= 6.0:
            return "MEDIUM"
        elif score >= 3.0:
            return "HIGH"
        else:
            return "CRITICAL"


class AdvancedSecurityScanner:
    """Main scanner orchestrator with comprehensive security assessment"""

    def __init__(self):
        self.dns_resolver = DNSResolver()
        self.email_analyzer = EmailSecurityAnalyzer(self.dns_resolver)
        self.timeout = 15

    def scan(self, domain: str) -> Dict:
        """Execute comprehensive security scan"""
        start_time = time.time()

        # Validate domain
        if not self._validate_domain(domain):
            return {"error": "Invalid domain format", "success": False}

        results = {
            "domain": domain,
            "scan_timestamp": datetime.now().isoformat(),
            "scanner_version": "2.0.0",
        }

        # Execute scans in parallel for maximum efficiency
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                "dns": executor.submit(self._scan_dns, domain),
                "ssl": executor.submit(self._scan_ssl, domain),
                "security_headers": executor.submit(
                    self._scan_security_headers, domain
                ),
                "email_security": executor.submit(self._scan_email_security, domain),
                "whois": executor.submit(self._scan_whois, domain),
                "geolocation": executor.submit(self._scan_geolocation, domain),
                "ports": executor.submit(self._scan_ports, domain),
                "http_methods": executor.submit(self._scan_http_methods, domain),
                "subdomain_takeover": executor.submit(
                    self._check_subdomain_takeover, domain
                ),
                "technology_detection": executor.submit(
                    self._detect_technologies, domain
                ),
            }

            for key, future in futures.items():
                try:
                    result = future.result(timeout=20)
                    results[key] = result
                except Exception as e:
                    logger.error(
                        f"{key} scan failed: {str(e)}\n{traceback.format_exc()}"
                    )
                    results[key] = {"error": str(e), "success": False}

        # Calculate overall security score
        results["overall_score"] = self._calculate_overall_score(results)
        results["risk_assessment"] = self._generate_risk_assessment(results)
        results["scan_duration"] = round(time.time() - start_time, 2)
        results["success"] = True

        return results

    def _validate_domain(self, domain: str) -> bool:
        """Validate domain format"""
        pattern = r"^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}$"
        return bool(re.match(pattern, domain))

    def _scan_dns(self, domain: str) -> Dict:
        """Comprehensive DNS records scan"""
        record_types = [
            "A",
            "AAAA",
            "MX",
            "NS",
            "TXT",
            "CNAME",
            "SOA",
            "CAA",
            "SRV",
            "PTR",
        ]
        records = {}

        for record_type in record_types:
            try:
                result = self.dns_resolver.query(domain, record_type)
                records[record_type] = result if result else []
            except Exception as e:
                records[record_type] = []
                logger.debug(f"DNS {record_type} query failed: {str(e)}")

        return {
            "records": records,
            "total_records": sum(len(v) for v in records.values()),
            "success": True,
        }

    def _scan_ssl(self, domain: str) -> Dict:
        """Scan SSL certificate"""
        try:
            data = SSLAnalyzer.analyze(domain)
            return {**data, "success": True}
        except Exception as e:
            return {"error": str(e), "success": False}

    def _scan_security_headers(self, domain: str) -> Dict:
        """Scan security headers"""
        try:
            data = SecurityHeadersAnalyzer.analyze(domain)
            return {**data, "success": True}
        except Exception as e:
            return {"error": str(e), "success": False}

    def _scan_email_security(self, domain: str) -> Dict:
        """Scan email security"""
        try:
            data = self.email_analyzer.analyze(domain)
            return {**data, "success": True}
        except Exception as e:
            return {"error": str(e), "success": False}

    def _scan_whois(self, domain: str) -> Dict:
        """Scan WHOIS information with enhanced data extraction"""
        try:
            # Perform WHOIS lookup
            w = whois.whois(domain)
            
            # Check if WHOIS data is actually available
            if not w:
                raise Exception("WHOIS lookup returned no data")
            
            def normalize_datetime(date_obj):
                """Normalize datetime to timezone-naive UTC"""
                if not date_obj:
                    return None
                
                # Handle list of dates
                if isinstance(date_obj, list):
                    date_obj = [d for d in date_obj if d]  # Filter out None values
                    if not date_obj:
                        return None
                    date_obj = date_obj[0]
                
                if not date_obj:
                    return None
                
                # If it's a string, parse it
                if isinstance(date_obj, str):
                    try:
                        date_obj = date_parser.parse(date_obj)
                    except:
                        return None
                
                # Ensure it's a datetime
                if not isinstance(date_obj, datetime):
                    return None
                
                # Convert to naive datetime (remove timezone)
                if date_obj.tzinfo is not None and date_obj.tzinfo.utcoffset(date_obj) is not None:
                    # Convert to UTC first, then make naive
                    utc_dt = date_obj.astimezone(pytz.UTC)
                    return utc_dt.replace(tzinfo=None)
                
                return date_obj
            
            def extract_string(value):
                """Extract string from various formats"""
                if not value:
                    return None
                if isinstance(value, list):
                    value = [v for v in value if v]
                    if not value:
                        return None
                    value = value[0]
                return str(value).strip() if value else None
            
            def extract_list(value):
                """Extract list from various formats"""
                if not value:
                    return []
                if isinstance(value, list):
                    return [str(v).strip().lower() for v in value if v]
                return [str(value).strip().lower()]
            
            # Get current time as naive datetime
            now = datetime.now()
            
            # Extract and normalize dates
            creation_dt = normalize_datetime(w.creation_date) if hasattr(w, 'creation_date') else None
            expiration_dt = normalize_datetime(w.expiration_date) if hasattr(w, 'expiration_date') else None
            updated_dt = normalize_datetime(w.updated_date) if hasattr(w, 'updated_date') else None
            
            # Extract registrar
            registrar = extract_string(w.registrar) if hasattr(w, 'registrar') else None
            
            # Extract name servers
            name_servers = extract_list(w.name_servers) if hasattr(w, 'name_servers') else []
            
            # Extract status
            status_list = extract_list(w.status) if hasattr(w, 'status') else []
            
            # Extract country
            country = extract_string(w.country) if hasattr(w, 'country') else None
            
            # Extract DNSSEC
            dnssec = extract_string(w.dnssec) if hasattr(w, 'dnssec') else None
            
            # Build result
            result = {
                'success': True
            }
            
            # Add registrar
            if registrar:
                result['registrar'] = registrar
            
            # Add dates and calculations
            if creation_dt:
                result['creation_date'] = creation_dt.strftime('%Y-%m-%d %H:%M:%S')
                age_days = (now - creation_dt).days
                result['age_days'] = age_days
                result['age_years'] = round(age_days / 365.25, 1)
            
            if expiration_dt:
                result['expiration_date'] = expiration_dt.strftime('%Y-%m-%d %H:%M:%S')
                result['days_until_expiration'] = (expiration_dt - now).days
            
            if updated_dt:
                result['updated_date'] = updated_dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Add name servers
            if name_servers:
                result['name_servers'] = name_servers
            
            # Add status
            if status_list:
                result['status'] = status_list
            
            # Add country
            if country:
                result['registrant_country'] = country
            
            # Add DNSSEC
            if dnssec:
                result['dnssec'] = dnssec
            
            # Check if we got meaningful data
            if not registrar and not creation_dt and not name_servers:
                raise Exception("WHOIS data incomplete or unavailable")
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"WHOIS lookup failed for {domain}: {error_msg}")
            return {
                'error': error_msg,
                'success': False
            }

    def _scan_geolocation(self, domain: str) -> Dict:
        """Get comprehensive geolocation data"""
        try:
            # Get IP first
            ip_list = self.dns_resolver.query(domain, "A")
            if not ip_list:
                return {"error": "No A records found", "success": False}

            ip = ip_list[0]

            # Try multiple geolocation APIs for better accuracy
            response = requests.get(f"http://ip-api.com/json/{ip}", timeout=5)

            if response.status_code == 200:
                data = response.json()
                return {
                    "ip": ip,
                    "country": data.get("country", "N/A"),
                    "country_code": data.get("countryCode", "N/A"),
                    "city": data.get("city", "N/A"),
                    "region": data.get("regionName", "N/A"),
                    "latitude": data.get("lat", "N/A"),
                    "longitude": data.get("lon", "N/A"),
                    "isp": data.get("isp", "N/A"),
                    "org": data.get("org", "N/A"),
                    "as": data.get("as", "N/A"),
                    "timezone": data.get("timezone", "N/A"),
                    "zip": data.get("zip", "N/A"),
                    "success": True,
                }
            else:
                return {"error": "Geolocation API failed", "success": False}
        except Exception as e:
            return {"error": str(e), "success": False}

    def _scan_ports(self, domain: str) -> Dict:
        """Comprehensive port scan with service detection"""
        common_ports = {
            20: "FTP-DATA",
            21: "FTP",
            22: "SSH",
            23: "Telnet",
            25: "SMTP",
            53: "DNS",
            80: "HTTP",
            110: "POP3",
            111: "RPC",
            135: "MSRPC",
            139: "NetBIOS",
            143: "IMAP",
            161: "SNMP",
            443: "HTTPS",
            445: "SMB",
            587: "SMTP",
            993: "IMAPS",
            995: "POP3S",
            1433: "MSSQL",
            1521: "Oracle",
            3306: "MySQL",
            3389: "RDP",
            5432: "PostgreSQL",
            5900: "VNC",
            6379: "Redis",
            8080: "HTTP-Proxy",
            8443: "HTTPS-Alt",
            27017: "MongoDB",
        }

        open_ports = []
        dangerous_ports = [21, 23, 135, 139, 445, 3389, 5900]
        sensitive_ports = [22, 1433, 3306, 5432, 6379, 27017]

        def check_port(port, service):
            try:
                with socket.create_connection((domain, port), timeout=2) as sock:
                    return {
                        "port": port,
                        "service": service,
                        "state": "open",
                        "danger_level": (
                            "high"
                            if port in dangerous_ports
                            else "medium" if port in sensitive_ports else "low"
                        ),
                    }
            except:
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            futures = [
                executor.submit(check_port, p, s) for p, s in common_ports.items()
            ]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    open_ports.append(result)

        # Calculate risk level
        risk = "LOW"
        high_risk_count = sum(1 for p in open_ports if p["danger_level"] == "high")
        medium_risk_count = sum(1 for p in open_ports if p["danger_level"] == "medium")

        if high_risk_count > 0:
            risk = "CRITICAL"
        elif medium_risk_count > 2:
            risk = "HIGH"
        elif medium_risk_count > 0 or len(open_ports) > 5:
            risk = "MEDIUM"

        return {
            "open_ports": sorted(open_ports, key=lambda x: x["port"]),
            "total_open": len(open_ports),
            "high_risk_ports": high_risk_count,
            "medium_risk_ports": medium_risk_count,
            "risk_level": risk,
            "success": True,
        }

    def _scan_http_methods(self, domain: str) -> Dict:
        """Check allowed HTTP methods"""
        try:
            response = requests.options(f"https://{domain}", timeout=5, verify=False)
            allowed = response.headers.get("Allow", "N/A")

            dangerous = ["PUT", "DELETE", "TRACE", "CONNECT", "PATCH"]
            methods = (
                [m.strip() for m in allowed.split(",")] if allowed != "N/A" else []
            )
            found_dangerous = [m for m in dangerous if m in methods]

            risk = "LOW"
            if any(m in ["TRACE", "CONNECT"] for m in found_dangerous):
                risk = "CRITICAL"
            elif any(m in ["PUT", "DELETE"] for m in found_dangerous):
                risk = "HIGH"
            elif "PATCH" in found_dangerous:
                risk = "MEDIUM"

            return {
                "allowed": allowed,
                "methods": methods,
                "dangerous": found_dangerous,
                "risk": risk,
                "total_methods": len(methods),
                "success": True,
            }
        except Exception as e:
            return {"error": str(e), "success": False}

    def _check_subdomain_takeover(self, domain: str) -> Dict:
        """Check for potential subdomain takeover vulnerabilities"""
        try:
            vulnerable_patterns = [
                "NoSuchBucket",
                "No Such Account",
                "Repository not found",
                "Project not found",
                "This page is reserved",
                "There isn't a GitHub Pages site here",
                "Fastly error: unknown domain",
                "The gods are wise",
                "Whatever you were looking for doesn't currently exist",
            ]

            try:
                response = requests.get(f"https://{domain}", timeout=5, verify=False)
                content = response.text.lower()

                found_patterns = []
                for pattern in vulnerable_patterns:
                    if pattern.lower() in content:
                        found_patterns.append(pattern)

                is_vulnerable = len(found_patterns) > 0

                return {
                    "vulnerable": is_vulnerable,
                    "patterns_found": found_patterns,
                    "risk": "HIGH" if is_vulnerable else "LOW",
                    "success": True,
                }
            except:
                return {
                    "vulnerable": False,
                    "patterns_found": [],
                    "risk": "LOW",
                    "success": True,
                }
        except Exception as e:
            return {"error": str(e), "success": False}

    def _detect_technologies(self, domain: str) -> Dict:
        """Detect web technologies and frameworks"""
        try:
            response = requests.get(f"https://{domain}", timeout=10, verify=False)
            headers = response.headers
            content = response.text[:5000]  # First 5KB

            technologies = {
                "web_server": [],
                "cms": [],
                "javascript_frameworks": [],
                "analytics": [],
                "cdn": [],
                "other": [],
            }

            # Web server detection
            server = headers.get("Server", "")
            if server:
                technologies["web_server"].append(server)

            # CMS detection
            if "wp-content" in content or "wordpress" in content.lower():
                technologies["cms"].append("WordPress")
            if "drupal" in content.lower():
                technologies["cms"].append("Drupal")
            if "joomla" in content.lower():
                technologies["cms"].append("Joomla")

            # Framework detection
            if "react" in content.lower():
                technologies["javascript_frameworks"].append("React")
            if "vue" in content.lower():
                technologies["javascript_frameworks"].append("Vue.js")
            if "angular" in content.lower():
                technologies["javascript_frameworks"].append("Angular")
            if "jquery" in content.lower():
                technologies["javascript_frameworks"].append("jQuery")

            # Analytics detection
            if "google-analytics" in content.lower() or "gtag" in content:
                technologies["analytics"].append("Google Analytics")
            if "hotjar" in content.lower():
                technologies["analytics"].append("Hotjar")

            # CDN detection
            if "cloudflare" in str(headers).lower():
                technologies["cdn"].append("Cloudflare")
            if "akamai" in str(headers).lower():
                technologies["cdn"].append("Akamai")

            # Other technologies
            powered_by = headers.get("X-Powered-By", "")
            if powered_by:
                technologies["other"].append(f"Powered by: {powered_by}")

            return {
                "technologies": technologies,
                "total_detected": sum(len(v) for v in technologies.values()),
                "success": True,
            }
        except Exception as e:
            return {"error": str(e), "success": False}

    def _calculate_overall_score(self, results: Dict) -> float:
        """Calculate comprehensive overall security score"""
        scores = []
        weights = []

        # Security headers (weight: 3)
        if results.get("security_headers", {}).get("success"):
            scores.append(results["security_headers"].get("score", 0))
            weights.append(3)

        # Email security (weight: 2.5)
        if results.get("email_security", {}).get("success"):
            scores.append(results["email_security"].get("score", 0))
            weights.append(2.5)

        # SSL/TLS (weight: 3)
        if results.get("ssl", {}).get("success"):
            ssl_score = results["ssl"].get("ssl_score", 0)
            scores.append(ssl_score)
            weights.append(3)

        # Port security (weight: 1.5)
        if results.get("ports", {}).get("success"):
            port_risk = results["ports"].get("risk_level", "LOW")
            port_score = {"LOW": 10, "MEDIUM": 7, "HIGH": 4, "CRITICAL": 2}.get(
                port_risk, 5
            )
            scores.append(port_score)
            weights.append(1.5)

        if not scores:
            return 0

        # Calculate weighted average
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        total_weight = sum(weights)

        return round(weighted_sum / total_weight, 1)

    def _generate_risk_assessment(self, results: Dict) -> Dict:
        """Generate comprehensive risk assessment"""
        critical_issues = []
        high_issues = []
        medium_issues = []
        low_issues = []

        # Check SSL
        if results.get("ssl", {}).get("success"):
            ssl = results["ssl"]
            if not ssl.get("valid"):
                critical_issues.append("Invalid or expired SSL certificate")
            elif ssl.get("days_remaining", 999) < 7:
                high_issues.append("SSL certificate expires in less than 7 days")

            if ssl.get("vulnerabilities"):
                for vuln in ssl["vulnerabilities"]:
                    if "SSLv2" in vuln or "SSLv3" in vuln:
                        critical_issues.append(vuln)
                    else:
                        medium_issues.append(vuln)

        # Check email security
        if results.get("email_security", {}).get("success"):
            email = results["email_security"]
            risk = email.get("risk_level", "HIGH")
            if risk == "CRITICAL":
                critical_issues.append("Critical email security misconfiguration")
            elif risk == "HIGH":
                high_issues.append("Poor email security configuration")

        # Check security headers
        if results.get("security_headers", {}).get("success"):
            headers = results["security_headers"]
            if headers.get("missing_critical"):
                critical_issues.extend(
                    [
                        f"Missing critical header: {h}"
                        for h in headers["missing_critical"][:2]
                    ]
                )

        # Check ports
        if results.get("ports", {}).get("success"):
            ports = results["ports"]
            if ports.get("risk_level") == "CRITICAL":
                critical_issues.append(
                    f"Critical: {ports.get('high_risk_ports', 0)} dangerous ports open"
                )
            elif ports.get("risk_level") in ["HIGH", "MEDIUM"]:
                high_issues.append(
                    f"{ports.get('total_open', 0)} ports open - review and secure"
                )

        # Check subdomain takeover
        if results.get("subdomain_takeover", {}).get("vulnerable"):
            critical_issues.append(
                "Potential subdomain takeover vulnerability detected"
            )

        # Overall risk level
        if critical_issues:
            overall_risk = "CRITICAL"
        elif high_issues:
            overall_risk = "HIGH"
        elif medium_issues:
            overall_risk = "MEDIUM"
        else:
            overall_risk = "LOW"

        return {
            "overall_risk": overall_risk,
            "critical_issues": critical_issues,
            "high_issues": high_issues,
            "medium_issues": medium_issues,
            "low_issues": low_issues,
            "total_issues": len(critical_issues)
            + len(high_issues)
            + len(medium_issues),
        }


# Initialize scanner
scanner = AdvancedSecurityScanner()


# Routes
@enscan.route("/")
def index():
    """Render main interface"""
    return render_template("enscan.html")

@enscan.route("/docs")
def documentation():
    """Render documentation page"""
    return render_template("documentation_si.html")


@enscan.route("/api/scan", methods=["POST"])
@rate_limit_api(requests_per_minute=5, requests_per_hour=50)  # Strict rate limit for resource-intensive scans
@validate_request({
    "domain": {
        "type": "string",
        "required": True,
        "max_length": 253
    }
}, strict=True)
def scan_endpoint():
    """
    Main scan endpoint
    OWASP: Rate limited, input validated, schema-based validation
    """
    try:
        # Get validated data from request context
        data = g.validated_data
        domain = InputValidator.validate_string(
            data.get("domain"), "domain", max_length=253, required=True
        )

        # Clean domain
        if domain.startswith(("http://", "https://")):
            domain = urlparse(domain).netloc
        domain = domain.replace("www.", "")

        # Execute scan
        result = scanner.scan(domain)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Scan endpoint failed: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": str(e), "success": False}), 500


@enscan.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "version": "2.0.0",
            "timestamp": datetime.now().isoformat(),
        }
    )

@enscan.route("/api/groq-reasoning", methods=["POST"])
@rate_limit_api(requests_per_minute=10, requests_per_hour=100)  # Rate limit for AI operations
@validate_request({
    "category": {
        "type": "string",
        "required": True,
        "max_length": 50,
        "allowed_values": ["overall", "ssl", "headers", "email", "ports", "risk"]
    },
    "scan_data": {
        "type": "dict",
        "required": True,
        "nested_schema": {}  # Flexible nested schema for scan data
    }
}, strict=True)
def groq_reasoning():
    """
    Server-side Groq API endpoint for security analysis reasoning
    OWASP: API keys must never be exposed to client-side
    All Groq operations are handled server-side only
    Rate limited and input validated
    """
    try:
        # Get validated data from request context (already validated by decorator)
        data = g.validated_data
        
        category = data.get('category')
        scan_data = data.get('scan_data')
        
        # Load Groq API key server-side only
        try:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from config import Config
            api_key = Config.GROQ_API_KEY
        except (ImportError, AttributeError):
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except ImportError:
                pass
            import os
            api_key = os.getenv('GROQ_API_KEY')
        
        if not api_key:
            return jsonify({
                "error": "Groq API key not configured",
                "message": "Please configure GROQ_API_KEY in environment variables",
                "success": False
            }), 503
        
        # Build prompt based on category
        prompt = build_reasoning_prompt(category, scan_data)
        
        # Call LLM router server-side
        try:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from core.llm_router import generate_text
            
            result = generate_text(
                prompt=prompt,
                app_name="enscan",
                task_type="security_analysis",
                system_prompt="You are a cybersecurity expert. Analyze security scan data and provide detailed reasoning and recommendations.",
                temperature=0.7,
                max_tokens=1000
            )
            
            reasoning = result.get("response", "").strip()
            if not reasoning:
                reasoning = "Unable to generate reasoning."
            
            return jsonify({
                "success": True,
                "reasoning": reasoning,
                "category": category
            })
        
        except Exception as llm_error:
            logger.error(f"LLM router error: {str(llm_error)}")
            return jsonify({
                "error": "Failed to generate reasoning",
                "message": "LLM service is currently unavailable. Please try again later.",
                "success": False
            }), 503
    
    except Exception as e:
        logger.error(f"Reasoning endpoint error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            "error": "Internal server error",
            "success": False
        }), 500

def build_reasoning_prompt(category: str, scan_data: dict) -> str:
    """Build reasoning prompt based on category"""
    domain = scan_data.get('domain', 'unknown')
    
    if category == "overall":
        return f"""Analyze this security scan for {domain} and provide detailed reasoning for the overall score of {scan_data.get('overall_score', 0)}/10.
Include:
- Why this score was given
- Main security strengths
- Critical weaknesses
- Specific recommendations

Data: SSL Score: {scan_data.get('ssl', {}).get('ssl_score', 0)}/10, Security Headers: {scan_data.get('security_headers', {}).get('grade', 'N/A')} ({scan_data.get('security_headers', {}).get('score', 0)}/10), Email Security: {scan_data.get('email_security', {}).get('score', 0)}/10, Open Ports: {scan_data.get('ports', {}).get('total_open', 0)}, Risk Level: {scan_data.get('risk_assessment', {}).get('overall_risk', 'UNKNOWN')}"""
    
    elif category == "ssl":
        ssl_data = scan_data.get('ssl', {})
        return f"""Explain the SSL/TLS score of {ssl_data.get('ssl_score', 0)}/10 for {domain}.
Details: Valid: {ssl_data.get('valid', False)}, Protocol: {ssl_data.get('protocol_version', 'Unknown')}, Cipher: {ssl_data.get('cipher_suite', {}).get('name', 'Unknown')}, Days Remaining: {ssl_data.get('days_remaining', 'N/A')}
Provide reasoning about security implications and recommendations."""
    
    elif category == "headers":
        headers_data = scan_data.get('security_headers', {})
        return f"""Explain the Security Headers grade "{headers_data.get('grade', 'N/A')}" and score {headers_data.get('score', 0)}/10 for {domain}.
Missing Critical: {', '.join(headers_data.get('missing_critical', []) or ['None'])}
Missing Important: {', '.join(headers_data.get('missing_important', []) or ['None'])}
Explain why this grade was given and what needs improvement."""
    
    elif category == "email":
        email_data = scan_data.get('email_security', {})
        return f"""Explain the Email Security score of {email_data.get('score', 0)}/10 and {email_data.get('risk_level', 'UNKNOWN')} risk level for {domain}.
SPF: {'Configured' if email_data.get('spf', {}).get('configured') else 'Not Configured'}
DMARC: {'Configured' if email_data.get('dmarc', {}).get('configured') else 'Not Configured'} (Policy: {email_data.get('dmarc', {}).get('policy', 'N/A')})
DKIM: {'Configured' if email_data.get('dkim', {}).get('configured') else 'Not Configured'}
Explain the security implications and recommendations."""
    
    elif category == "risk":
        risk_data = scan_data.get('risk_assessment', {})
        return f"""Analyze the overall risk assessment for {domain} showing {risk_data.get('overall_risk', 'UNKNOWN')} risk with {risk_data.get('total_issues', 0)} issues.
Critical Issues: {len(risk_data.get('critical_issues', []) or [])}
High Issues: {len(risk_data.get('high_issues', []) or [])}
Provide a comprehensive security analysis and prioritized action plan."""
    
    else:
        return f"Provide security analysis reasoning for {category} category based on the scan data for {domain}."

@enscan.route("/api/groq-key", methods=["GET"])
def get_groq_key():
    """
    SECURITY: This endpoint has been disabled to prevent API key exposure.
    API keys should NEVER be exposed to the client-side.
    All Groq API calls must be made server-side only.
    
    OWASP: Never expose API keys in client-side code or API responses.
    """
    logger.warning("Attempted access to deprecated /api/groq-key endpoint - API keys must remain server-side")
    return jsonify({
        "error": "API key endpoint disabled for security",
        "message": "API keys are server-side only. All Groq operations are handled server-side.",
        "success": False
    }), 403


# Application factory
# Blueprint is registered in server.py
# Socket timeout can be set in server.py if needed