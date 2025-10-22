from flask import Blueprint, request, jsonify, render_template
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS
import requests
import nmap
import re
import dns.resolver
import time
import logging
import socket
import ssl
import OpenSSL
import concurrent.futures
from datetime import datetime
from functools import lru_cache
from urllib.parse import urlparse
import os
import json
from pathlib import Path
import google.generativeai as genai
from bs4 import BeautifulSoup
import warnings

warnings.filterwarnings('ignore', message='Unverified HTTPS request')

# Create Blueprint instead of Flask app
webseeker = Blueprint('webseeker', __name__, url_prefix='/webseeker')


class Config:
    # Hardcode your API keys here
    VIRUSTOTAL_API_KEY = "cff4224acda1132a9e9398ea3499d63087bd9907df2b293f9e753b1bc186d205"
    IPINFO_API_KEY = "e8bc09f9d8bba4"
    ABUSEIPDB_API_KEY = "3e987939762bafb0a6c75d17374909beafdff50a5e6724ae1d25f66e6f775367d3e25862588e2495"
    GEMINI_API_KEY = "AIzaSyCMwpK-6Dr9X_MpcCyRR1PJcixg4pW55e8"
    
    CACHE_DURATION = 3600
    SCAN_TIMEOUT = 300
    MAX_CONCURRENT_SCANS = 3
    ALLOWED_SCAN_TYPES = ['quick', 'comprehensive', 'stealth']
    PORT_RANGES = {
        'quick': '20-25,53,80,110,143,443,465,587,993,995,3306,3389,5432,8080,8443',
        'comprehensive': '1-1024',
        'stealth': '21-23,25,53,80,110,111,135,139,143,443,445,993,995,1723,3306,3389,5900,8080'
    }

if Config.GEMINI_API_KEY:
    genai.configure(api_key=Config.GEMINI_API_KEY)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('webseeker.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AIAnalyzer:
    @staticmethod
    def analyze_website_content(target):
        """Enhanced website content analysis with better error handling"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            
            # Try HTTPS first, then HTTP
            url = None
            for protocol in ['https', 'http']:
                try:
                    url = f"{protocol}://{target}"
                    response = requests.get(url, headers=headers, timeout=15, verify=False, allow_redirects=True)
                    response.raise_for_status()
                    break
                except Exception as e:
                    logger.warning(f"Failed to fetch {protocol}://{target}: {str(e)}")
                    continue
            else:
                return {'success': False, 'error': 'Unable to fetch website content'}
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract comprehensive information
            title = soup.find('title')
            title_text = title.get_text().strip() if title else ""
            
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            description = meta_desc.get('content', '').strip() if meta_desc else ""
            
            # Get keywords
            meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
            keywords = meta_keywords.get('content', '').strip() if meta_keywords else ""
            
            # Extract headings
            headings = []
            for h in soup.find_all(['h1', 'h2', 'h3'])[:15]:
                text = h.get_text().strip()
                if text and len(text) > 2:
                    headings.append(text)
            
            # Get text content - improved extraction
            paragraphs = []
            for p in soup.find_all('p')[:20]:
                text = p.get_text().strip()
                if text and len(text) > 20:
                    paragraphs.append(text)
            
            content_sample = ' '.join(paragraphs)[:3000]
            
            # Extract links to understand site structure
            links = []
            for a in soup.find_all('a', href=True)[:30]:
                href = a.get('href', '')
                if href and not href.startswith('#'):
                    links.append(href)
            
            # Extract main content areas
            main_content = ""
            for tag in soup.find_all(['main', 'article', 'section'])[:5]:
                text = tag.get_text(separator=' ', strip=True)
                if len(text) > 50:
                    main_content += text[:500] + " "
            
            return {
                'title': title_text,
                'description': description,
                'keywords': keywords,
                'headings': headings,
                'content_sample': content_sample,
                'main_content': main_content.strip(),
                'links': links,
                'url': url,
                'success': True
            }
        except Exception as e:
            logger.error(f"Website content analysis error for {target}: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def calculate_security_score(scan_results):
        """Enhanced security scoring with detailed analysis"""
        score = 100
        risk_factors = []
        
        # SSL/TLS Analysis (25 points)
        ssl_info = scan_results.get("ssl_cert", {})
        if not ssl_info.get("success"):
            score -= 25
            risk_factors.append("No valid SSL/TLS certificate detected - unencrypted traffic")
        elif ssl_info.get("expired"):
            score -= 20
            risk_factors.append("SSL/TLS certificate has expired - security vulnerability")
        elif ssl_info.get("protocol") and "TLS" not in ssl_info.get("protocol", ""):
            score -= 10
            risk_factors.append("Outdated SSL/TLS protocol version detected")
        
        # Port Security Analysis (25 points)
        ports = scan_results.get("port_scan", {}).get("ports", [])
        open_ports = [p for p in ports if p.get("state") == "open"]
        
        # Critical danger ports
        critical_ports = [21, 23, 135, 139, 445]  # FTP, Telnet, NetBIOS, SMB
        high_risk_ports = [25, 110, 3389]  # SMTP, POP3, RDP
        
        critical_open = [p for p in open_ports if int(p.get("port", 0)) in critical_ports]
        high_risk_open = [p for p in open_ports if int(p.get("port", 0)) in high_risk_ports]
        
        if critical_open:
            score -= 20
            port_list = ', '.join([str(p.get('port')) for p in critical_open])
            risk_factors.append(f"Critical security risk: Dangerous ports exposed ({port_list})")
        
        if high_risk_open:
            score -= 10
            port_list = ', '.join([str(p.get('port')) for p in high_risk_open])
            risk_factors.append(f"High-risk ports accessible ({port_list})")
        
        if len(open_ports) > 10:
            score -= 10
            risk_factors.append(f"Excessive port exposure - {len(open_ports)} open ports detected")
        elif len(open_ports) > 5:
            score -= 5
        
        # Threat Intelligence Analysis (30 points)
        vt_stats = scan_results.get("virustotal", {}).get("domain_report", {}).get("last_analysis_stats", {})
        malicious = vt_stats.get("malicious", 0)
        suspicious = vt_stats.get("suspicious", 0)
        
        if malicious > 5:
            score -= 30
            risk_factors.append(f"CRITICAL: Flagged as malicious by {malicious} security vendors")
        elif malicious > 0:
            score -= 20
            risk_factors.append(f"Security threat detected: {malicious} vendor(s) report malicious activity")
        
        if suspicious > 5:
            score -= 15
            risk_factors.append(f"Suspicious activity reported by {suspicious} security sources")
        elif suspicious > 0:
            score -= 8
        
        # Abuse Database Analysis (20 points)
        abuse_info = scan_results.get("ip_info", {}).get("abuse", {})
        abuse_reports = abuse_info.get("total_reports", 0)
        confidence = abuse_info.get("confidence_score", 0)
        
        if abuse_reports > 100 or confidence > 90:
            score -= 20
            risk_factors.append(f"Severe abuse history: {abuse_reports} reports (confidence: {confidence}%)")
        elif abuse_reports > 50 or confidence > 75:
            score -= 15
            risk_factors.append(f"Significant abuse reports: {abuse_reports} incidents recorded")
        elif abuse_reports > 10 or confidence > 50:
            score -= 8
            risk_factors.append(f"Moderate abuse history: {abuse_reports} reports on record")
        
        return max(0, min(100, score)), risk_factors
    
    @staticmethod
    def determine_risk_level(score):
        if score >= 85:
            return "SAFE"
        elif score >= 70:
            return "LOW"
        elif score >= 50:
            return "MEDIUM"
        elif score >= 30:
            return "HIGH"
        else:
            return "CRITICAL"
    
    @staticmethod
    def generate_recommendations(risk_factors, scan_results):
        """Generate specific, actionable recommendations"""
        recommendations = []
        
        ssl_info = scan_results.get("ssl_cert", {})
        if not ssl_info.get("success"):
            recommendations.append("🔒 Implement HTTPS with a valid SSL/TLS certificate from a trusted CA (Let's Encrypt, DigiCert, etc.)")
        elif ssl_info.get("expired"):
            recommendations.append("⚠️ Immediately renew expired SSL certificate to restore secure connections")
        
        ports = scan_results.get("port_scan", {}).get("ports", [])
        critical_ports = [21, 23, 135, 139, 445]
        open_critical = [p for p in ports if p.get("state") == "open" and int(p.get("port", 0)) in critical_ports]
        
        if open_critical:
            recommendations.append("🚨 URGENT: Close critical security ports or restrict access via firewall rules")
            recommendations.append("🛡️ Implement network segmentation and access control lists (ACLs)")
        
        open_ports = [p for p in ports if p.get("state") == "open"]
        if len(open_ports) > 5:
            recommendations.append("🔍 Audit and minimize exposed services - follow principle of least privilege")
        
        vt_stats = scan_results.get("virustotal", {}).get("domain_report", {}).get("last_analysis_stats", {})
        if vt_stats.get("malicious", 0) > 0:
            recommendations.append("🦠 Conduct immediate malware scan and security audit")
            recommendations.append("🔧 Implement Web Application Firewall (WAF) and intrusion detection system")
        
        abuse_reports = scan_results.get("ip_info", {}).get("abuse", {}).get("total_reports", 0)
        if abuse_reports > 10:
            recommendations.append("📊 Review and remediate IP reputation issues")
            recommendations.append("✉️ Contact abuse@network-provider to investigate reported incidents")
        
        if not recommendations:
            recommendations.append("✅ Maintain current security posture with continuous monitoring")
            recommendations.append("📅 Schedule regular security assessments and penetration testing")
            recommendations.append("🔄 Keep all systems and software updated with latest security patches")
        
        return recommendations[:6]
    
    @staticmethod
    def generate_website_description(target, website_info, scan_results):
        """Enhanced AI-powered website analysis using Gemini with better fallback"""
        
        if not website_info.get('success'):
            return f"Unable to analyze {target} - website may be inaccessible, blocking automated requests, or down for maintenance. Manual review required."
        
        # Check if we have enough content to analyze
        has_content = (
            website_info.get('title') or 
            website_info.get('description') or 
            website_info.get('content_sample') or 
            website_info.get('main_content')
        )
        
        if not has_content:
            return f"{target} - Website is accessible but contains minimal content. May be under construction or require JavaScript rendering."
        
        # Try Gemini API if configured
        if Config.GEMINI_API_KEY:
            try:
                model = genai.GenerativeModel('gemini-2.5-pro')
                
                # Enhanced prompt with all available data
                prompt = f"""Analyze this website and provide a comprehensive 3-4 sentence professional description.

TARGET DOMAIN: {target}
Page Title: {website_info.get('title', 'N/A')}
Meta Description: {website_info.get('description', 'N/A')}
Keywords: {website_info.get('keywords', 'N/A')}
Main Headings: {', '.join(website_info.get('headings', [])[:8])}
Page Content Sample: {website_info.get('content_sample', '')[:1000]}
Main Content Areas: {website_info.get('main_content', '')[:500]}

Provide a clear, professional description that covers:
1. What type of website this is (specific category: e-commerce, news, blog, corporate, SaaS, etc.)
2. Primary purpose, services, or products offered
3. Target audience and industry/niche
4. Key features or notable characteristics

Write naturally in plain text - no formatting, no JSON, no markdown."""
                
                response = model.generate_content(prompt)
                description = response.text.strip()
                
                # Clean up response
                description = description.replace('```', '').replace('**', '').replace('*', '').strip()
                description = ' '.join(description.split())
                
                # Validate quality
                if len(description) >= 100 and 'unable' not in description.lower() and target not in description[:50]:
                    return description
                    
            except Exception as e:
                logger.warning(f"Gemini API error for {target}: {str(e)}")
        
        # Fallback to enhanced rule-based analysis
        return AIAnalyzer._enhanced_fallback_analysis(target, website_info)
    
    @staticmethod
    def _enhanced_fallback_analysis(target, website_info):
        """Comprehensive fallback analysis when AI is unavailable"""
        domain = target.lower()
        title = website_info.get('title', '').lower()
        description = website_info.get('description', '').lower()
        content = (website_info.get('content_sample', '') + ' ' + website_info.get('main_content', '')).lower()
        keywords = website_info.get('keywords', '').lower()
        headings = ' '.join(website_info.get('headings', [])).lower()
        
        combined_text = f"{domain} {title} {description} {content} {keywords} {headings}"
        
        # Enhanced category detection with confidence scoring
        categories = {
            'E-commerce Platform': {
                'keywords': ['shop', 'store', 'cart', 'buy', 'product', 'price', 'checkout', 'order', 'ecommerce', 'add to cart', 'shopping', 'purchase', 'payment'],
                'description': 'online retail platform offering products for purchase with shopping cart functionality'
            },
            'News & Media Outlet': {
                'keywords': ['news', 'article', 'journalist', 'breaking', 'latest', 'media', 'press', 'story', 'editorial', 'report', 'coverage'],
                'description': 'news organization providing journalism, articles, and current events coverage'
            },
            'Blog & Content Platform': {
                'keywords': ['blog', 'post', 'author', 'comment', 'read more', 'written by', 'published', 'category', 'tags'],
                'description': 'content publishing platform featuring articles, posts, and editorial content'
            },
            'Educational Institution': {
                'keywords': ['university', 'college', 'school', 'education', 'learn', 'course', 'student', 'academic', 'degree', 'faculty', 'campus'],
                'description': 'educational institution offering academic programs and learning resources'
            },
            'Corporate Website': {
                'keywords': ['company', 'business', 'enterprise', 'corporation', 'solutions', 'services', 'about us', 'careers', 'contact', 'team'],
                'description': 'corporate entity providing business services, solutions, and company information'
            },
            'Social Media Platform': {
                'keywords': ['social', 'connect', 'follow', 'share', 'community', 'profile', 'friend', 'network', 'post', 'feed'],
                'description': 'social networking platform enabling user connections and content sharing'
            },
            'Government Portal': {
                'keywords': ['gov', 'government', 'official', 'department', 'ministry', 'public service', 'citizen', 'policy'],
                'description': 'official government website providing public services and information'
            },
            'Technology & Software': {
                'keywords': ['software', 'tech', 'developer', 'api', 'code', 'programming', 'platform', 'app', 'cloud', 'saas'],
                'description': 'technology company offering software solutions, development tools, or technical services'
            },
            'Healthcare & Medical': {
                'keywords': ['health', 'medical', 'hospital', 'doctor', 'patient', 'care', 'clinic', 'treatment', 'medicine'],
                'description': 'healthcare provider or medical information resource'
            },
            'Financial Services': {
                'keywords': ['bank', 'finance', 'investment', 'trading', 'loan', 'credit', 'insurance', 'mortgage', 'financial'],
                'description': 'financial institution providing banking, investment, or related services'
            }
        }
        
        scores = {}
        for category, data in categories.items():
            score = sum(2 if kw in combined_text else 0 for kw in data['keywords'])
            # Bonus for keyword in title or domain
            score += sum(3 for kw in data['keywords'] if kw in title or kw in domain)
            scores[category] = score
        
        max_score = max(scores.values()) if scores else 0
        
        if max_score >= 3:
            detected_type = max(scores, key=scores.get)
            desc_template = categories[detected_type]['description']
            
            # Extract title if available for more context
            title_text = website_info.get('title', '').strip()
            if title_text and len(title_text) < 100:
                return f"{target} ({title_text}) is a {desc_template}. The site serves its target audience through its web presence and digital offerings."
            else:
                return f"{target} is identified as a {desc_template}. The platform provides relevant services and content to its user base."
        
        # Generic but informative fallback
        title_text = website_info.get('title', '').strip()
        if title_text:
            return f"{target} - {title_text}. This website provides online services and content to its visitors. Further manual analysis recommended for detailed categorization and purpose assessment."
        
        return f"{target} is an active website with limited publicly available categorization data. The site appears operational and may require manual review to determine its specific purpose, target audience, and primary functions."
    
    @staticmethod
    def generate_executive_summary(scan_results):
        """Comprehensive executive summary with enhanced AI insights"""
        target = scan_results.get("target", "unknown")
        
        website_info = AIAnalyzer.analyze_website_content(target)
        score, risk_factors = AIAnalyzer.calculate_security_score(scan_results)
        risk_level = AIAnalyzer.determine_risk_level(score)
        recommendations = AIAnalyzer.generate_recommendations(risk_factors, scan_results)
        website_description = AIAnalyzer.generate_website_description(target, website_info, scan_results)
        
        # Enhanced summary generation
        open_ports = len([p for p in scan_results.get("port_scan", {}).get("ports", []) if p.get("state") == "open"])
        
        if risk_level == "CRITICAL":
            summary = f"🚨 CRITICAL SECURITY ALERT: {target} has severe security vulnerabilities requiring immediate remediation. "
        elif risk_level == "HIGH":
            summary = f"⚠️ HIGH RISK DETECTED: {target} exhibits significant security concerns that demand urgent attention. "
        elif risk_level == "MEDIUM":
            summary = f"⚡ MODERATE RISK: {target} shows several security issues that should be addressed to improve posture. "
        elif risk_level == "LOW":
            summary = f"✓ LOW RISK: {target} maintains good security practices with minor areas for improvement. "
        else:
            summary = f"✅ SECURE: {target} demonstrates strong security configuration with minimal concerns. "
        
        summary += f"Comprehensive analysis identified {open_ports} accessible port(s) and evaluated {len(risk_factors) if risk_factors else 'no significant'} risk factor(s) across multiple security domains."
        
        key_findings = risk_factors if risk_factors else [
            "✓ No critical security vulnerabilities detected",
            "✓ Standard security practices implemented",
            "✓ No active malware or abuse reports"
        ]
        
        return {
            "executive_summary": summary,
            "website_description": website_description,
            "security_score": score,
            "risk_level": risk_level,
            "key_findings": key_findings[:5],
            "recommendations": recommendations
        }

class SecurityScanner:
    def __init__(self):
        try:
            self.nmap = nmap.PortScanner()
        except nmap.PortScannerError as e:
            logger.error(f"Nmap initialization error: {str(e)}")
            self.nmap = None
        
        self.scan_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=Config.MAX_CONCURRENT_SCANS
        )

    def validate_target(self, target):
        try:
            target = re.sub(r'^https?://', '', target)
            target = target.split('/')[0]
            target = target.split(':')[0]
            
            if not re.match(r'^[\w\-\.]+\.[a-zA-Z]{2,}$', target):
                return None, "Invalid domain format"
            
            try:
                socket.gethostbyname(target)
            except socket.gaierror:
                return None, "Domain cannot be resolved"
            
            return target, None
            
        except Exception as e:
            logger.error(f"Target validation error: {str(e)}")
            return None, f"Validation error: {str(e)}"

    def get_ip_info(self, target):
        """Enhanced IP information retrieval"""
        try:
            ip = socket.gethostbyname(target)
            
            ip_data = {
                'ip': ip,
                'hostname': target,
                'success': True
            }
            
            if Config.IPINFO_API_KEY:
                try:
                    response = requests.get(
                        f"https://ipinfo.io/{ip}/json?token={Config.IPINFO_API_KEY}",
                        timeout=10
                    )
                    if response.status_code == 200:
                        ip_data.update(response.json())
                except Exception as e:
                    logger.warning(f"IPInfo API error: {str(e)}")
            
            # Reverse DNS
            try:
                answers = dns.resolver.resolve(dns.reversename.from_address(ip), "PTR")
                ip_data['reverse_dns'] = str(answers[0]).rstrip('.')
            except Exception:
                try:
                    ip_data['reverse_dns'] = socket.getfqdn(ip)
                except Exception:
                    ip_data['reverse_dns'] = 'Not available'
            
            # WHOIS lookup
            try:
                import whois as whois_module
                whois_data = whois_module.whois(target)
                ip_data['whois'] = {
                    'registrar': str(whois_data.registrar) if hasattr(whois_data, 'registrar') and whois_data.registrar else 'N/A',
                    'creation_date': self._format_date(whois_data.creation_date) if hasattr(whois_data, 'creation_date') else 'N/A',
                    'expiration_date': self._format_date(whois_data.expiration_date) if hasattr(whois_data, 'expiration_date') else 'N/A'
                }
            except ImportError:
                logger.warning("python-whois not installed")
                ip_data['whois'] = {'registrar': 'N/A', 'creation_date': 'N/A', 'expiration_date': 'N/A'}
            except Exception as e:
                logger.warning(f"WHOIS error: {str(e)}")
                ip_data['whois'] = {'registrar': 'N/A', 'creation_date': 'N/A', 'expiration_date': 'N/A'}
            
            # ASN lookup
            try:
                asn_response = requests.get(f"https://ipapi.co/{ip}/json/", timeout=5)
                if asn_response.status_code == 200:
                    asn_data = asn_response.json()
                    ip_data['asn'] = asn_data.get('asn', 'N/A')
                    ip_data['network'] = asn_data.get('network', 'N/A')
                    ip_data['network_type'] = asn_data.get('network_type', 'N/A')
            except Exception as e:
                logger.warning(f"ASN lookup error: {str(e)}")
                ip_data['asn'] = 'N/A'
                ip_data['network'] = 'N/A'
                ip_data['network_type'] = 'N/A'
            
            # Abuse reports
            if Config.ABUSEIPDB_API_KEY:
                try:
                    abuse_response = requests.get(
                        "https://api.abuseipdb.com/api/v2/check",
                        params={'ipAddress': ip, 'maxAgeInDays': 90},
                        headers={'Key': Config.ABUSEIPDB_API_KEY},
                        timeout=5
                    )
                    if abuse_response.status_code == 200:
                        abuse_data = abuse_response.json().get('data', {})
                        ip_data['abuse'] = {
                            'total_reports': abuse_data.get('totalReports', 0),
                            'confidence_score': abuse_data.get('abuseConfidenceScore', 0),
                            'last_reported': abuse_data.get('lastReportedAt', 'Never')
                        }
                except Exception as e:
                    logger.warning(f"AbuseIPDB error: {str(e)}")
                    ip_data['abuse'] = {'total_reports': 0, 'confidence_score': 0, 'last_reported': 'Never'}
            else:
                ip_data['abuse'] = {'total_reports': 0, 'confidence_score': 0, 'last_reported': 'Never'}
            
            return ip_data
                
        except Exception as e:
            logger.error(f"IP info error: {str(e)}")
            return {"error": str(e), "success": False}

    def _format_date(self, date_value):
        if not date_value:
            return 'N/A'
        if isinstance(date_value, list):
            date_value = date_value[0]
        try:
            return str(date_value).split()[0] if date_value else 'N/A'
        except:
            return 'N/A'

    def check_ssl(self, target):
        try:
            context = ssl.create_default_context()
            with socket.create_connection((target, 443), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=target) as ssock:
                    cert = ssock.getpeercert(True)
                    x509 = OpenSSL.crypto.load_certificate(
                        OpenSSL.crypto.FILETYPE_ASN1, cert
                    )
                    
                    issuer = {}
                    for key, value in x509.get_issuer().get_components():
                        issuer[key.decode('utf-8')] = value.decode('utf-8')
                    
                    return {
                        'issuer': issuer,
                        'version': x509.get_version(),
                        'serial_number': str(x509.get_serial_number()),
                        'not_before': x509.get_notBefore().decode('utf-8') if x509.get_notBefore() else None,
                        'not_after': x509.get_notAfter().decode('utf-8') if x509.get_notAfter() else None,
                        'expired': x509.has_expired(),
                        'protocol': ssock.version(),
                        'success': True
                    }
        except Exception as e:
            logger.error(f"SSL check error: {str(e)}")
            return {"error": str(e), "success": False}

    def scan_virustotal(self, target):
        if not Config.VIRUSTOTAL_API_KEY:
            return {
                "domain_report": {
                    "last_analysis_stats": {
                        "harmless": 0,
                        "malicious": 0,
                        "suspicious": 0,
                        "unrated": 0
                    }
                },
                "success": False,
                "error": "API key not configured"
            }
        
        try:
            headers = {
                "x-apikey": Config.VIRUSTOTAL_API_KEY,
                "accept": "application/json"
            }
            
            domain_response = requests.get(
                f"https://www.virustotal.com/api/v3/domains/{target}",
                headers=headers,
                timeout=15
            )
            
            if domain_response.status_code == 200:
                domain_data = domain_response.json()
                return {
                    'domain_report': domain_data.get('data', {}).get('attributes', {}),
                    'success': True
                }
            else:
                return {
                    "domain_report": {
                        "last_analysis_stats": {
                            "harmless": 0,
                            "malicious": 0,
                            "suspicious": 0,
                            "unrated": 0
                        }
                    },
                    "success": False,
                    "error": f"API returned status {domain_response.status_code}"
                }
                
        except Exception as e:
            logger.error(f"VirusTotal error: {str(e)}")
            return {
                "domain_report": {
                    "last_analysis_stats": {
                        "harmless": 0,
                        "malicious": 0,
                        "suspicious": 0,
                        "unrated": 0
                    }
                },
                "success": False,
                "error": str(e)
            }

    def perform_port_scan(self, target, scan_type='quick'):
        if not self.nmap:
            return {"error": "Nmap not available", "success": False}
        
        try:
            port_range = Config.PORT_RANGES.get(scan_type, Config.PORT_RANGES['quick'])
            
            scan_args = {
                'quick': '-T4 -sV --version-intensity 5',
                'comprehensive': '-T4 -sV -sC --version-all',
                'stealth': '-T2 -sS -Pn --min-rate 100'
            }
            
            logger.info(f"Starting {scan_type} port scan on {target}")
            self.nmap.scan(target, port_range, arguments=scan_args.get(scan_type, scan_args['quick']))

            processed_results = {
                'ports': [],
                'scan_stats': {
                    'start_time': datetime.now().isoformat(),
                    'elapsed': self.nmap.scanstats().get('elapsed', '0'),
                    'scan_type': scan_type
                },
                'success': True
            }

            if len(self.nmap.all_hosts()) > 0:
                host = self.nmap.all_hosts()[0]
                
                for proto in self.nmap[host].all_protocols():
                    ports = self.nmap[host][proto].keys()
                    for port in ports:
                        port_info = self.nmap[host][proto][port]
                        processed_results['ports'].append({
                            'port': str(port),
                            'protocol': proto,
                            'state': port_info.get('state', ''),
                            'service': port_info.get('name', ''),
                            'version': port_info.get('version', ''),
                            'product': port_info.get('product', '')
                        })

            return processed_results

        except Exception as e:
            logger.error(f"Port scan error: {str(e)}")
            return {"error": str(e), "success": False}

    def perform_full_scan(self, target, scan_type):
        """Perform all security checks"""
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    'ip_info': executor.submit(self.get_ip_info, target),
                    'virustotal': executor.submit(self.scan_virustotal, target),
                    'port_scan': executor.submit(self.perform_port_scan, target, scan_type),
                    'ssl_cert': executor.submit(self.check_ssl, target)
                }

                results = {
                    'timestamp': datetime.now().isoformat(),
                    'target': target,
                    'scan_type': scan_type,
                    'success': True
                }
                
                for key, future in futures.items():
                    try:
                        results[key] = future.result(timeout=Config.SCAN_TIMEOUT)
                    except concurrent.futures.TimeoutError:
                        logger.warning(f"{key} timed out for {target}")
                        results[key] = {"error": "Scan timed out", "success": False}
                    except Exception as e:
                        logger.error(f"{key} error for {target}: {str(e)}")
                        results[key] = {"error": str(e), "success": False}

                return results

        except Exception as e:
            logger.error(f"Full scan error for {target}: {str(e)}")
            return {"error": f"Scan failed: {str(e)}", "success": False}

# Initialize scanner and AI analyzer
scanner = SecurityScanner()
ai_analyzer = AIAnalyzer()

# Routes
@webseeker.route('/')
def index():
    return render_template("webseeker.html")

@webseeker.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON", "success": False}), 400
        
        target = data.get('domain', '').strip()
        scan_type = data.get('scanType', 'quick')

        if not target:
            return jsonify({"error": "No target specified", "success": False}), 400
        
        if scan_type not in Config.ALLOWED_SCAN_TYPES:
            return jsonify({"error": "Invalid scan type", "success": False}), 400

        normalized_target, error = scanner.validate_target(target)
        if error:
            return jsonify({"error": error, "success": False}), 400

        results = scanner.perform_full_scan(normalized_target, scan_type)
        
        if not results.get("success", False):
            return jsonify({"error": results.get("error", "Unknown error"), "success": False}), 500

        # Generate AI executive summary with accurate scoring
        executive_summary = ai_analyzer.generate_executive_summary(results)
        results['executive_summary'] = executive_summary

        return jsonify(results)

    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({"error": "Internal server error", "success": False}), 500

@webseeker.route('/api/quick-check', methods=['GET'])
def quick_check():
    """Lightweight domain check endpoint"""
    try:
        target = request.args.get('domain', '').strip()
        if not target:
            return jsonify({"error": "No target specified", "success": False}), 400

        normalized_target, error = scanner.validate_target(target)
        if error:
            return jsonify({"error": error, "success": False}), 400

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            ip_future = executor.submit(scanner.get_ip_info, normalized_target)
            vt_future = executor.submit(scanner.scan_virustotal, normalized_target)

            results = {
                'timestamp': datetime.now().isoformat(),
                'target': normalized_target,
                'ip_info': ip_future.result(timeout=10),
                'virustotal': vt_future.result(timeout=10),
                'success': True
            }

        return jsonify(results)

    except Exception as e:
        logger.error(f"Quick check error: {str(e)}")
        return jsonify({"error": str(e), "success": False}), 500

@webseeker.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({"error": "Rate limit exceeded. Please try again later.", "success": False}), 429

@webseeker.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({"error": "Internal server error", "success": False}), 500

@webseeker.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found", "success": False}), 404


if __name__ == '__main__':
    logger.info("Starting security scanner server...")
    webseeker.run(debug=False, host='0.0.0.0', port=5000)


