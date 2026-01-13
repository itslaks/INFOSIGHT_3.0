from flask import Flask, request, jsonify, render_template, Blueprint, g
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import nmap
import socket
from datetime import datetime
import concurrent.futures
import os
import validators
from urllib.parse import urlparse
from functools import wraps
from utils.security import rate_limit_strict, validate_request, InputValidator

portscanner = Blueprint('portscanner', __name__, template_folder='templates')

# Configure logging
import logging
logger = logging.getLogger(__name__)
logger.info("=" * 70)
logger.info("🔌 PortScanner - Initializing")
logger.info("=" * 70)

class Config:
    SCAN_TIMEOUT = 6000  # 10 minutes for longer scans
    MAX_CONCURRENT_SCANS = 5
    DEFAULT_PORTS = '1-1024'
    
    SCAN_TYPES = {
        'intense_scan': {'command': '-T4 -A -v -Pn', 'description': 'Comprehensive scan'},
        'service_version': {'command': '-sV -Pn', 'description': 'Service version detection'},
        'os_detection': {'command': '-O -Pn', 'description': 'OS detection'},
        'tcp_connect': {'command': '-sT -Pn', 'description': 'TCP connect scan'},
        'syn_scan': {'command': '-sS -Pn', 'description': 'SYN scan (requires root)'},
        'udp_scan': {'command': '-sU -Pn', 'description': 'UDP scan'},
        'aggressive_scan': {'command': '-A -Pn', 'description': 'Aggressive scan with version detection'},
        'list_scan': {'command': '-sL -Pn', 'description': 'Lists targets without scanning'},
        'null_scan': {'command': '-sN -Pn', 'description': 'Null scan with no flags'},
        'xmas_scan': {'command': '-sX -Pn', 'description': 'Xmas scan with FIN, PSH, URG flags'},
        'fin_scan': {'command': '-sF -Pn', 'description': 'FIN scan with only FIN flag'},
        'full_port_scan': {'command': '-p- -Pn', 'description': 'Scans all 65,535 ports'},
        'script_scan': {'command': '-sC -Pn', 'description': 'Default script scan'},
        'version_intensity': {'command': '--version-intensity 9 -Pn', 'description': 'Intense version detection'},
        'timing_aggressive': {'command': '-T4 -Pn', 'description': 'Aggressive timing template'},
        'timing_insane': {'command': '-T5 -Pn', 'description': 'Insane timing template'},
        'traceroute': {'command': '--traceroute -Pn', 'description': 'Trace path to host'},
        'fragment_scan': {'command': '-f -Pn', 'description': 'Fragment packets'},
        'idle_scan': {'command': '-sI -Pn', 'description': 'Idle scan'},
        'ack_scan': {'command': '-sA -Pn', 'description': 'ACK scan'},
        'window_scan': {'command': '-sW -Pn', 'description': 'Window scan'},
        'maimon_scan': {'command': '-sM -Pn', 'description': 'Maimon scan'},
        'sctp_init_scan': {'command': '-sY -Pn', 'description': 'SCTP INIT scan'},
        'sctp_cookie_scan': {'command': '-sZ -Pn', 'description': 'SCTP COOKIE-ECHO scan'}
    }

# Configure rate limiting
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100 per day", "10 per minute"]
)

def requires_root(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        root_required_scans = ['syn_scan']
        scan_type = request.json.get('scanType')
        if scan_type in root_required_scans and os.geteuid() != 0:
            return jsonify({
                'error': f'The {scan_type} requires root privileges',
                'success': False
            }), 403
        return f(*args, **kwargs)
    return decorated_function

class PortScanner:
    def __init__(self):
        self.nmap = nmap.PortScanner()
        self.scan_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=Config.MAX_CONCURRENT_SCANS
        )

    def validate_target(self, target):
        """Validate and normalize target input (supports IP, domain, and URL)"""
        try:
            # Remove protocol and path if URL
            if '//' in target:
                parsed = urlparse(target)
                target = parsed.netloc or parsed.path
            
            # Remove any remaining path components and query parameters
            target = target.split('/')[0].split('?')[0]
            
            # Check if IP address
            try:
                socket.inet_aton(target)
                return target, None
            except socket.error:
                pass
            
            # Check if domain
            if validators.domain(target) or target.startswith('localhost'):
                try:
                    socket.gethostbyname(target)
                    return target, None
                except socket.gaierror:
                    return None, "Domain cannot be resolved"
            
            return None, "Invalid target format. Please provide a valid IP address, domain, or URL."
            
        except Exception as e:
            return None, f"Validation error: {str(e)}"

    def perform_port_scan(self, target, scan_type):
        """Perform Nmap port scan with error handling"""
        try:
            if scan_type not in Config.SCAN_TYPES:
                return {"error": "Invalid scan type", "success": False}

            scan_args = Config.SCAN_TYPES[scan_type]['command']
            
            # Adjust ports based on scan type
            if scan_type == 'full_port_scan':
                ports = '1-65535'
            elif scan_type == 'quick_scan':
                ports = 'T:21-25,80,139,443,445,3389'  # Common ports
            else:
                ports = Config.DEFAULT_PORTS

            try:
                scan_results = self.nmap.scan(
                    hosts=target,
                    ports=ports,
                    arguments=scan_args,
                    timeout=Config.SCAN_TIMEOUT
                )
                
                if not scan_results or 'scan' not in scan_results:
                    return {"error": "Scan produced no results", "success": False}

            except nmap.PortScannerError as e:
                return {"error": f"Scan failed: {str(e)}", "success": False}
            except Exception as e:
                return {"error": f"Unexpected error: {str(e)}", "success": False}

            processed_results = {
                'scan_info': {
                    'target': target,
                    'scan_type': scan_type,
                    'description': Config.SCAN_TYPES[scan_type]['description'],
                    'command_used': f"nmap {scan_args} -p {ports} {target}",
                    'start_time': datetime.now().isoformat(),
                    'elapsed': self.nmap.scanstats().get('elapsed', '0'),
                    'total_hosts': self.nmap.scanstats().get('totalhosts', '0'),
                    'up_hosts': self.nmap.scanstats().get('uphosts', '0'),
                    'down_hosts': self.nmap.scanstats().get('downhosts', '0')
                },
                'hosts': {},
                'success': True
            }

            for host in self.nmap.all_hosts():
                host_data = {
                    'state': self.nmap[host].state(),
                    'protocols': {},
                    'hostnames': self.nmap[host].hostnames()
                }

                # Add OS detection results if available
                if hasattr(self.nmap[host], 'osmatch') and self.nmap[host].osmatch():
                    host_data['os_matches'] = self.nmap[host].osmatch()

                # Add traceroute if available
                if hasattr(self.nmap[host], 'traceroute'):
                    host_data['traceroute'] = self.nmap[host].traceroute()

                # Process each protocol
                for proto in self.nmap[host].all_protocols():
                    ports = self.nmap[host][proto].keys()
                    host_data['protocols'][proto] = {}
                    
                    for port in ports:
                        port_info = self.nmap[host][proto][port]
                        port_data = {
                            'state': port_info.get('state'),
                            'service': port_info.get('name'),
                            'product': port_info.get('product', ''),
                            'version': port_info.get('version', ''),
                            'extrainfo': port_info.get('extrainfo', ''),
                            'reason': port_info.get('reason', ''),
                            'cpe': port_info.get('cpe', [])
                        }

                        # Add script output if available
                        if 'script' in port_info:
                            port_data['scripts'] = port_info['script']

                        host_data['protocols'][proto][port] = port_data

                processed_results['hosts'][host] = host_data

            return processed_results

        except Exception as e:
            return {"error": str(e), "success": False}

# Initialize scanner
scanner = PortScanner()

@portscanner.route('/')
def index():
    return render_template('portscanner.html')

@portscanner.route('/api/scan-types', methods=['GET'])
def get_scan_types():
    """Return available scan types and their descriptions"""
    return jsonify({
        'scan_types': Config.SCAN_TYPES,
        'success': True
    })

@portscanner.route('/api/scan', methods=['POST'])
@rate_limit_strict(requests_per_minute=5, requests_per_hour=30)  # Strict limit for port scans
@validate_request({
    "target": {
        "type": "string",
        "required": True,
        "max_length": 253  # Max domain length
    },
    "scanType": {
        "type": "string",
        "required": False,
        "max_length": 50,
        "allowed_values": list(Config.SCAN_TYPES.keys())
    }
}, strict=True)
@requires_root
def port_scan():
    """
    Port scanner endpoint
    OWASP: Rate limited, input validated, schema-based validation
    """
    try:
        # Get validated data from request context
        data = g.validated_data
        
        target = InputValidator.validate_string(
            data.get('target'), 'target', max_length=253, required=True
        )
        scan_type = data.get('scanType', 'quick_scan')

        normalized_target, error = scanner.validate_target(target)
        if error:
            return jsonify({"error": error, "success": False}), 400

        results = scanner.perform_port_scan(normalized_target, scan_type)
        if not results.get('success', False):
            return jsonify(results), 400

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500

@portscanner.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({"error": "Rate limit exceeded", "success": False}), 429

# Blueprint is registered in server.py
# Limiter should be initialized in server.py if needed