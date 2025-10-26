from flask import Flask, request, jsonify, render_template, Blueprint
import re
import dns.resolver
import requests
from urllib.parse import urlparse
import tldextract
import concurrent.futures
import whois
from datetime import datetime
import socket
import ssl
import OpenSSL
from typing import Dict, List, Union
import json
import hashlib
import ipaddress
from collections import defaultdict
import time

enscan = Blueprint('enscan', __name__, template_folder='templates')

@enscan.route('/')
def index():
    return render_template('enscan.html')

class AdvancedScanner:
    def __init__(self):
        self.timeout = 8
        self.dns_servers = [
            '8.8.8.8',      # Google
            '8.8.4.4',      # Google
            '1.1.1.1',      # Cloudflare
            '1.0.0.1',      # Cloudflare
            '208.67.222.222' # OpenDNS
        ]
    
    def is_valid_domain(self, domain: str) -> bool:
        pattern = r'^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}$'
        return bool(re.match(pattern, domain))
    
    def dns_query(self, domain: str, record_type: str) -> Union[List[str], str]:
        """Enhanced DNS query with multiple nameservers"""
        resolver = dns.resolver.Resolver(configure=False)
        resolver.nameservers = self.dns_servers
        resolver.timeout = 5
        resolver.lifetime = 5
        
        try:
            answers = resolver.resolve(domain, record_type)
            return [str(record).strip('"') for record in answers]
        except dns.resolver.NoAnswer:
            return f"No {record_type} records found"
        except dns.resolver.NXDOMAIN:
            return f"Domain does not exist"
        except dns.resolver.Timeout:
            return f"DNS query timeout"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_ssl_info(self, domain: str) -> Dict:
        """Enhanced SSL certificate analysis"""
        try:
            context = ssl.create_default_context()
            with socket.create_connection((domain, 443), timeout=self.timeout) as sock:
                with context.wrap_socket(sock, server_hostname=domain) as ssock:
                    cert_bin = ssock.getpeercert(binary_form=True)
                    x509 = OpenSSL.crypto.load_certificate(OpenSSL.crypto.FILETYPE_ASN1, cert_bin)
                    cert = ssock.getpeercert()
                    
                    # Enhanced certificate analysis
                    fingerprint_sha256 = hashlib.sha256(cert_bin).hexdigest()
                    fingerprint_sha1 = hashlib.sha1(cert_bin).hexdigest()
                    
                    # Get protocol version
                    protocol_version = ssock.version()
                    cipher = ssock.cipher()
                    
                    # Parse dates
                    not_before = datetime.strptime(cert['notBefore'], '%b %d %H:%M:%S %Y %Z')
                    not_after = datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                    days_remaining = (not_after - datetime.now()).days
                    
                    # Get extensions
                    extensions = {}
                    for i in range(x509.get_extension_count()):
                        ext = x509.get_extension(i)
                        extensions[ext.get_short_name().decode()] = str(ext)
                    
                    return {
                        "issuer": dict(x[0] for x in cert['issuer']),
                        "subject": dict(x[0] for x in cert['subject']),
                        "version": cert['version'],
                        "valid_from": cert['notBefore'],
                        "expires": cert['notAfter'],
                        "days_remaining": days_remaining,
                        "is_valid": days_remaining > 0,
                        "fingerprint_sha256": fingerprint_sha256,
                        "fingerprint_sha1": fingerprint_sha1,
                        "serial_number": format(x509.get_serial_number(), 'X'),
                        "signature_algorithm": x509.get_signature_algorithm().decode(),
                        "san": cert.get('subjectAltName', []),
                        "protocol_version": protocol_version,
                        "cipher_suite": {
                            "name": cipher[0],
                            "protocol": cipher[1],
                            "bits": cipher[2]
                        },
                        "public_key_bits": x509.get_pubkey().bits(),
                        "ocsp_stapling": "Must-Staple" in extensions.get('tlsfeature', ''),
                        "key_usage": extensions.get('keyUsage', 'N/A'),
                        "extended_key_usage": extensions.get('extendedKeyUsage', 'N/A')
                    }
        except Exception as e:
            return {"error": str(e)}
    
    def get_security_headers(self, domain: str) -> Dict:
        """Comprehensive security headers analysis"""
        try:
            # Try both HTTP and HTTPS
            urls = [f"https://{domain}", f"http://{domain}"]
            response = None
            
            for url in urls:
                try:
                    response = requests.get(url, timeout=self.timeout, allow_redirects=True, verify=True)
                    break
                except:
                    continue
            
            if not response:
                return {"error": "Could not connect to domain"}
            
            headers = response.headers
            
            # Comprehensive security header checks
            security_checks = {
                "Strict-Transport-Security": {
                    "value": headers.get("Strict-Transport-Security", "Not set"),
                    "weight": 2,
                    "description": "Forces HTTPS connections",
                    "recommendation": "Add 'Strict-Transport-Security: max-age=31536000; includeSubDomains; preload'"
                },
                "Content-Security-Policy": {
                    "value": headers.get("Content-Security-Policy", "Not set"),
                    "weight": 2,
                    "description": "Prevents XSS and injection attacks",
                    "recommendation": "Implement a strict CSP policy to control resource loading"
                },
                "X-Frame-Options": {
                    "value": headers.get("X-Frame-Options", "Not set"),
                    "weight": 1,
                    "description": "Prevents clickjacking",
                    "recommendation": "Set to 'DENY' or 'SAMEORIGIN'"
                },
                "X-Content-Type-Options": {
                    "value": headers.get("X-Content-Type-Options", "Not set"),
                    "weight": 1,
                    "description": "Prevents MIME sniffing",
                    "recommendation": "Set to 'nosniff'"
                },
                "X-XSS-Protection": {
                    "value": headers.get("X-XSS-Protection", "Not set"),
                    "weight": 1,
                    "description": "Legacy XSS protection",
                    "recommendation": "Set to '1; mode=block' (legacy browsers)"
                },
                "Referrer-Policy": {
                    "value": headers.get("Referrer-Policy", "Not set"),
                    "weight": 1,
                    "description": "Controls referrer information",
                    "recommendation": "Set to 'strict-origin-when-cross-origin' or 'no-referrer'"
                },
                "Permissions-Policy": {
                    "value": headers.get("Permissions-Policy", "Not set"),
                    "weight": 1,
                    "description": "Controls browser features",
                    "recommendation": "Define allowed features like camera, microphone, geolocation"
                },
                "Cross-Origin-Embedder-Policy": {
                    "value": headers.get("Cross-Origin-Embedder-Policy", "Not set"),
                    "weight": 1,
                    "description": "Prevents cross-origin resource loading",
                    "recommendation": "Set to 'require-corp' for enhanced security"
                },
                "Cross-Origin-Opener-Policy": {
                    "value": headers.get("Cross-Origin-Opener-Policy", "Not set"),
                    "weight": 1,
                    "description": "Isolates browsing context",
                    "recommendation": "Set to 'same-origin' to prevent attacks"
                },
                "Cross-Origin-Resource-Policy": {
                    "value": headers.get("Cross-Origin-Resource-Policy", "Not set"),
                    "weight": 1,
                    "description": "Protects resources from cross-origin access",
                    "recommendation": "Set to 'same-origin' or 'same-site'"
                }
            }
            
            # Calculate weighted security score
            total_weight = sum(h["weight"] for h in security_checks.values())
            earned_score = sum(h["weight"] for h in security_checks.values() if h["value"] != "Not set")
            security_score = round((earned_score / total_weight) * 10, 1)
            
            # Get security grade and reasoning
            grade_info = self._calculate_grade(security_score, security_checks)
            
            return {
                "headers": security_checks,
                "security_score": security_score,
                "max_score": 10,
                "grade": grade_info["grade"],
                "grade_reasoning": grade_info["reasoning"],
                "missing_headers": grade_info["missing_headers"],
                "critical_issues": grade_info["critical_issues"],
                "improvements": grade_info["improvements"],
                "server": headers.get("Server", "Not disclosed"),
                "powered_by": headers.get("X-Powered-By", "Not disclosed"),
                "final_url": response.url,
                "status_code": response.status_code,
                "redirect_chain": len(response.history),
                "cookies": {
                    "total": len(response.cookies),
                    "secure": sum(1 for c in response.cookies if c.secure),
                    "httponly": sum(1 for c in response.cookies if c.has_nonstandard_attr('HttpOnly'))
                }
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _calculate_grade(self, score: float, headers: Dict) -> Dict:
        """Calculate security grade with detailed reasoning"""
        missing_headers = [name for name, data in headers.items() if data["value"] == "Not set"]
        critical_missing = []
        important_missing = []
        
        for name, data in headers.items():
            if data["value"] == "Not set":
                if data["weight"] == 2:
                    critical_missing.append(name)
                else:
                    important_missing.append(name)
        
        # Determine grade
        if score >= 9:
            grade = "A+"
            reasoning = "Excellent security posture! Your site has implemented nearly all modern security headers."
        elif score >= 8:
            grade = "A"
            reasoning = "Very good security configuration with most important headers in place."
        elif score >= 7:
            grade = "B"
            reasoning = "Good security foundation, but missing some important protections."
        elif score >= 5:
            grade = "C"
            reasoning = "Basic security measures present, but significant gaps exist that should be addressed."
        elif score >= 3:
            grade = "D"
            reasoning = "Poor security configuration. Critical security headers are missing, leaving your site vulnerable."
        else:
            grade = "F"
            reasoning = "Critical security failure! Your site lacks essential security headers and is highly vulnerable to attacks."
        
        # Build critical issues list
        critical_issues = []
        if "Strict-Transport-Security" in critical_missing:
            critical_issues.append({
                "issue": "Missing HSTS Header",
                "severity": "HIGH",
                "impact": "Vulnerable to man-in-the-middle attacks and SSL stripping",
                "fix": "Add Strict-Transport-Security header with long max-age"
            })
        
        if "Content-Security-Policy" in critical_missing:
            critical_issues.append({
                "issue": "Missing Content Security Policy",
                "severity": "HIGH",
                "impact": "Vulnerable to XSS attacks, data injection, and resource hijacking",
                "fix": "Implement a strict CSP policy tailored to your site's needs"
            })
        
        if "X-Frame-Options" in important_missing:
            critical_issues.append({
                "issue": "Missing X-Frame-Options",
                "severity": "MEDIUM",
                "impact": "Vulnerable to clickjacking attacks",
                "fix": "Set X-Frame-Options to DENY or SAMEORIGIN"
            })
        
        if "X-Content-Type-Options" in important_missing:
            critical_issues.append({
                "issue": "Missing X-Content-Type-Options",
                "severity": "MEDIUM",
                "impact": "Vulnerable to MIME-type sniffing attacks",
                "fix": "Set X-Content-Type-Options to nosniff"
            })
        
        # Build improvements list
        improvements = []
        for header in missing_headers:
            improvements.append({
                "header": header,
                "priority": "High" if headers[header]["weight"] == 2 else "Medium",
                "recommendation": headers[header]["recommendation"]
            })
        
        return {
            "grade": grade,
            "reasoning": reasoning,
            "missing_headers": missing_headers,
            "critical_issues": critical_issues,
            "improvements": improvements,
            "headers_present": len(headers) - len(missing_headers),
            "headers_total": len(headers)
        }
    
    def get_domain_info(self, domain: str) -> Dict:
        """Enhanced WHOIS information"""
        try:
            w = whois.whois(domain)
            
            def format_date(date_obj):
                if isinstance(date_obj, datetime):
                    return date_obj.strftime("%Y-%m-%d %H:%M:%S")
                elif isinstance(date_obj, list) and date_obj:
                    return date_obj[0].strftime("%Y-%m-%d %H:%M:%S") if isinstance(date_obj[0], datetime) else str(date_obj[0])
                return str(date_obj) if date_obj else "N/A"
            
            creation = format_date(w.creation_date)
            expiration = format_date(w.expiration_date)
            updated = format_date(w.updated_date)
            
            # Calculate domain age
            age_days = "N/A"
            age_years = "N/A"
            if isinstance(w.creation_date, datetime):
                age_days = (datetime.now() - w.creation_date).days
                age_years = round(age_days / 365.25, 1)
            elif isinstance(w.creation_date, list) and w.creation_date:
                if isinstance(w.creation_date[0], datetime):
                    age_days = (datetime.now() - w.creation_date[0]).days
                    age_years = round(age_days / 365.25, 1)
            
            # Days until expiration
            days_until_expiration = "N/A"
            if isinstance(w.expiration_date, datetime):
                days_until_expiration = (w.expiration_date - datetime.now()).days
            elif isinstance(w.expiration_date, list) and w.expiration_date:
                if isinstance(w.expiration_date[0], datetime):
                    days_until_expiration = (w.expiration_date[0] - datetime.now()).days
            
            return {
                "registrar": w.registrar or "N/A",
                "creation_date": creation,
                "expiration_date": expiration,
                "updated_date": updated,
                "domain_age_days": age_days,
                "domain_age_years": age_years,
                "days_until_expiration": days_until_expiration,
                "name_servers": w.name_servers if hasattr(w, 'name_servers') and w.name_servers else [],
                "status": w.status if hasattr(w, 'status') and w.status else [],
                "emails": w.emails if hasattr(w, 'emails') and w.emails else [],
                "registrant_name": w.name if hasattr(w, 'name') else "N/A",
                "registrant_org": w.org if hasattr(w, 'org') else "N/A",
                "registrant_country": w.country if hasattr(w, 'country') else "N/A",
                "registrant_state": w.state if hasattr(w, 'state') else "N/A",
                "registrant_city": w.city if hasattr(w, 'city') else "N/A",
                "dnssec": w.dnssec if hasattr(w, 'dnssec') else "N/A"
            }
        except Exception as e:
            return {"error": str(e)}
    
    def check_email_security(self, domain: str) -> Dict:
        """Enhanced email security checks"""
        spf = self.dns_query(domain, 'TXT')
        dmarc = self.dns_query(f"_dmarc.{domain}", 'TXT')
        
        # Try multiple DKIM selectors
        dkim_selectors = ['default', 'google', 'k1', 'selector1', 'selector2', 'dkim']
        dkim_results = {}
        
        for selector in dkim_selectors:
            result = self.dns_query(f"{selector}._domainkey.{domain}", 'TXT')
            if isinstance(result, list):
                dkim_results[selector] = result
        
        # Parse SPF
        spf_record = None
        spf_mechanisms = []
        if isinstance(spf, list):
            for record in spf:
                if "v=spf1" in str(record):
                    spf_record = str(record)
                    spf_mechanisms = [m.strip() for m in spf_record.split() if m.strip() != "v=spf1"]
        
        # Parse DMARC
        dmarc_record = None
        dmarc_policy = "None"
        dmarc_pct = "N/A"
        if isinstance(dmarc, list):
            for record in dmarc:
                if "v=DMARC1" in str(record):
                    dmarc_record = str(record)
                    # Extract policy
                    if "p=reject" in dmarc_record:
                        dmarc_policy = "reject"
                    elif "p=quarantine" in dmarc_record:
                        dmarc_policy = "quarantine"
                    elif "p=none" in dmarc_record:
                        dmarc_policy = "none"
                    
                    # Extract percentage
                    import re
                    pct_match = re.search(r'pct=(\d+)', dmarc_record)
                    if pct_match:
                        dmarc_pct = pct_match.group(1) + "%"
        
        return {
            "spf": {
                "present": spf_record is not None,
                "record": spf_record or "No SPF record found",
                "mechanisms": spf_mechanisms,
                "mechanism_count": len(spf_mechanisms)
            },
            "dmarc": {
                "present": dmarc_record is not None,
                "record": dmarc_record or "No DMARC record found",
                "policy": dmarc_policy,
                "percentage": dmarc_pct
            },
            "dkim": {
                "selectors_found": list(dkim_results.keys()),
                "total_selectors_checked": len(dkim_selectors),
                "records": dkim_results
            },
            "email_security_score": sum([
                spf_record is not None,
                dmarc_record is not None,
                len(dkim_results) > 0
            ])
        }
    
    def check_subdomain_takeover(self, domain: str) -> Dict:
        """Enhanced subdomain takeover detection"""
        vulnerable_patterns = {
            'github.io': 'GitHub Pages',
            'herokuapp.com': 'Heroku',
            'azurewebsites.net': 'Azure',
            'cloudapp.net': 'Azure',
            'amazonaws.com': 'AWS',
            'bitbucket.io': 'Bitbucket',
            'shopify.com': 'Shopify',
            'tumblr.com': 'Tumblr',
            'wordpress.com': 'WordPress',
            'ghost.io': 'Ghost',
            'pantheonsite.io': 'Pantheon',
            'acquia-sites.com': 'Acquia'
        }
        
        try:
            cname_records = self.dns_query(domain, 'CNAME')
            vulnerabilities = []
            
            if isinstance(cname_records, list):
                for cname in cname_records:
                    cname_str = str(cname).lower()
                    for pattern, service in vulnerable_patterns.items():
                        if pattern in cname_str:
                            vulnerabilities.append({
                                "cname": str(cname),
                                "service": service,
                                "pattern": pattern
                            })
            
            if vulnerabilities:
                return {
                    "vulnerable": True,
                    "vulnerabilities": vulnerabilities,
                    "count": len(vulnerabilities),
                    "risk_level": "HIGH"
                }
            else:
                return {
                    "vulnerable": False,
                    "status": "No obvious subdomain takeover vulnerabilities detected"
                }
        except:
            return {"error": "Could not check subdomain takeover"}
    
    def get_geolocation(self, ip: str) -> Dict:
        """Enhanced geolocation with multiple providers"""
        try:
            # Primary: ip-api.com
            response = requests.get(f"http://ip-api.com/json/{ip}?fields=status,country,countryCode,region,regionName,city,zip,lat,lon,timezone,isp,org,as,asname,reverse,mobile,proxy,hosting", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    return {
                        "ip": ip,
                        "country": data.get("country", "N/A"),
                        "country_code": data.get("countryCode", "N/A"),
                        "region": data.get("regionName", "N/A"),
                        "city": data.get("city", "N/A"),
                        "zip": data.get("zip", "N/A"),
                        "latitude": data.get("lat", "N/A"),
                        "longitude": data.get("lon", "N/A"),
                        "timezone": data.get("timezone", "N/A"),
                        "isp": data.get("isp", "N/A"),
                        "org": data.get("org", "N/A"),
                        "as": data.get("as", "N/A"),
                        "asname": data.get("asname", "N/A"),
                        "reverse_dns": data.get("reverse", "N/A"),
                        "mobile": data.get("mobile", False),
                        "proxy": data.get("proxy", False),
                        "hosting": data.get("hosting", False)
                    }
        except:
            pass
        
        return {"error": "Geolocation lookup failed"}
    
    def check_open_ports(self, domain: str) -> Dict:
        """Enhanced port scanning"""
        common_ports = {
            20: "FTP Data",
            21: "FTP Control",
            22: "SSH",
            23: "Telnet",
            25: "SMTP",
            53: "DNS",
            80: "HTTP",
            110: "POP3",
            143: "IMAP",
            443: "HTTPS",
            465: "SMTPS",
            587: "SMTP (Submission)",
            993: "IMAPS",
            995: "POP3S",
            3306: "MySQL",
            3389: "RDP",
            5432: "PostgreSQL",
            5900: "VNC",
            6379: "Redis",
            8080: "HTTP Proxy",
            8443: "HTTPS Alt",
            27017: "MongoDB"
        }
        
        open_ports = []
        
        def check_port(port, name):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex((domain, port))
                sock.close()
                
                if result == 0:
                    # Try to get service banner
                    banner = "N/A"
                    try:
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.settimeout(2)
                        sock.connect((domain, port))
                        sock.send(b'\r\n')
                        banner = sock.recv(1024).decode('utf-8', errors='ignore').strip()[:100]
                        sock.close()
                    except:
                        pass
                    
                    return {
                        "port": port,
                        "service": name,
                        "status": "open",
                        "banner": banner
                    }
            except:
                pass
            return None
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(check_port, port, name) for port, name in common_ports.items()]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    open_ports.append(result)
        
        # Risk assessment
        risk_level = "LOW"
        dangerous_ports = [21, 23, 3389, 5900]  # Unencrypted or risky services
        
        if any(p['port'] in dangerous_ports for p in open_ports):
            risk_level = "HIGH"
        elif len(open_ports) > 5:
            risk_level = "MEDIUM"
        
        return {
            "open_ports": open_ports,
            "total_open": len(open_ports),
            "total_scanned": len(common_ports),
            "risk_level": risk_level
        }
    
    def check_http_methods(self, domain: str) -> Dict:
        """Enhanced HTTP methods check"""
        try:
            response = requests.options(f"https://{domain}", timeout=5)
            allowed_methods = response.headers.get('Allow', 'N/A')
            
            if allowed_methods == 'N/A':
                # Try HTTP if HTTPS fails
                response = requests.options(f"http://{domain}", timeout=5)
                allowed_methods = response.headers.get('Allow', 'Could not determine')
            
            dangerous_methods = ['PUT', 'DELETE', 'TRACE', 'CONNECT', 'PATCH']
            safe_methods = ['GET', 'POST', 'HEAD', 'OPTIONS']
            
            if allowed_methods not in ['N/A', 'Could not determine']:
                methods_list = [m.strip() for m in allowed_methods.split(',')]
                found_dangerous = [m for m in dangerous_methods if m in methods_list]
                found_safe = [m for m in safe_methods if m in methods_list]
            else:
                methods_list = []
                found_dangerous = []
                found_safe = []
            
            risk = "LOW"
            if found_dangerous:
                risk = "HIGH" if len(found_dangerous) > 2 else "MEDIUM"
            
            return {
                "allowed_methods": allowed_methods,
                "methods_list": methods_list,
                "dangerous_methods": found_dangerous if found_dangerous else "None detected",
                "safe_methods": found_safe,
                "risk_level": risk,
                "status": "Potentially dangerous methods enabled" if found_dangerous else "Safe"
            }
        except Exception as e:
            return {"error": str(e)}
    
    def check_technologies(self, domain: str) -> Dict:
        """Detect web technologies"""
        try:
            response = requests.get(f"https://{domain}", timeout=self.timeout, allow_redirects=True)
            headers = response.headers
            content = response.text[:10000]  # First 10KB
            
            technologies = {
                "server": headers.get("Server", "Not disclosed"),
                "powered_by": headers.get("X-Powered-By", "Not disclosed"),
                "frameworks": [],
                "cms": [],
                "analytics": [],
                "cdn": []
            }
            
            # Detect frameworks and CMS
            tech_patterns = {
                "WordPress": ["wp-content", "wp-includes"],
                "Drupal": ["/sites/default", "Drupal"],
                "Joomla": ["joomla", "/components/com_"],
                "React": ["react", "__NEXT_DATA__"],
                "Vue.js": ["vue", "v-cloak"],
                "Angular": ["ng-app", "ng-controller"],
                "jQuery": ["jquery"],
                "Bootstrap": ["bootstrap"],
                "Laravel": ["laravel"],
                "Django": ["csrfmiddlewaretoken"]
            }
            
            for tech, patterns in tech_patterns.items():
                if any(pattern.lower() in content.lower() for pattern in patterns):
                    if tech in ["WordPress", "Drupal", "Joomla"]:
                        technologies["cms"].append(tech)
                    else:
                        technologies["frameworks"].append(tech)
            
            # Detect analytics
            if "google-analytics" in content.lower() or "gtag" in content.lower():
                technologies["analytics"].append("Google Analytics")
            if "facebook" in content.lower() and "pixel" in content.lower():
                technologies["analytics"].append("Facebook Pixel")
            
            # Detect CDN
            cdn_headers = ["cloudflare", "akamai", "fastly", "cloudfront"]
            for cdn in cdn_headers:
                if any(cdn in str(v).lower() for v in headers.values()):
                    technologies["cdn"].append(cdn.title())
            
            return technologies
        except:
            return {"error": "Could not detect technologies"}
    
    def scan_domain(self, domain: str) -> Dict:
        """Main scanning function with all checks"""
        start_time = time.time()
        
        if not self.is_valid_domain(domain):
            return {"error": "Invalid domain format"}
        
        results = {
            "scan_time": None,
            "domain": domain
        }
        
        # DNS Records
        record_types = ['A', 'AAAA', 'NS', 'MX', 'TXT', 'SOA', 'CNAME', 'PTR', 'SRV', 'CAA']
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(record_types)) as executor:
            future_to_record = {
                executor.submit(self.dns_query, domain, record_type): record_type 
                for record_type in record_types
            }
            for future in concurrent.futures.as_completed(future_to_record):
                record_type = future_to_record[future]
                results[record_type] = future.result()
        
        # Get IP for additional checks
        ip_address = None
        if 'A' in results and isinstance(results['A'], list) and results['A']:
            ip_address = results['A'][0]
        
        # Parallel execution of remaining checks
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                'domain_info': executor.submit(self.get_domain_info, domain),
                'ssl_info': executor.submit(self.get_ssl_info, domain),
                'security_headers': executor.submit(self.get_security_headers, domain),
                'email_security': executor.submit(self.check_email_security, domain),
                'subdomain_takeover': executor.submit(self.check_subdomain_takeover, domain),
                'http_methods': executor.submit(self.check_http_methods, domain),
                'open_ports': executor.submit(self.check_open_ports, domain),
                'technologies': executor.submit(self.check_technologies, domain)
            }
            
            if ip_address:
                futures['geolocation'] = executor.submit(self.get_geolocation, ip_address)
            
            for key, future in futures.items():
                try:
                    results[key] = future.result()
                except Exception as e:
                    results[key] = {"error": str(e)}
        
        # Calculate overall security score
        scores = []
        if 'security_headers' in results and 'security_score' in results['security_headers']:
            scores.append(results['security_headers']['security_score'])
        if 'email_security' in results and 'email_security_score' in results['email_security']:
            scores.append(results['email_security']['email_security_score'] * 3.33)  # Scale to 10
        
        results['overall_security_score'] = round(sum(scores) / len(scores), 1) if scores else 0
        
        results['scan_time'] = round(time.time() - start_time, 2)
        
        return results

# Initialize scanner
scanner = AdvancedScanner()

@enscan.route('/api/scan', methods=['POST'])
def scan():
    try:
        data = request.json
        input_value = data.get('input', '').strip()
        
        if not input_value:
            return jsonify({"error": "Empty input provided"}), 400
        
        # Extract domain from URL if needed
        if input_value.startswith('http'):
            parsed = urlparse(input_value)
            input_value = parsed.netloc or parsed.path
        
        # Remove www. if present
        input_value = input_value.replace('www.', '')
        
        result = scanner.scan_domain(input_value)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def create_app():
    app = Flask(__name__)
    app.register_blueprint(enscan)
    return app

if __name__ == '__main__':
    socket.setdefaulttimeout(10)
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)