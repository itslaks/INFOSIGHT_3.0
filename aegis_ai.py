from flask import Flask, Blueprint, render_template_string, jsonify, request, render_template
from datetime import datetime, timedelta
import random
import json
from collections import defaultdict
import hashlib

# Create Aegis Blueprint
aegis_ai = Blueprint('aegis_ai', __name__, template_folder='templates')

# In-memory storage (replace with database in production)
class AegisDataStore:
    def __init__(self):
        self.events = []
        self.users = {}
        self.ai_tools = {}
        self.alerts = []
        self.honeypots = []
        self.policies = []
        self.risk_scores = {}
        self._initialize_sample_data()
    
    def _initialize_sample_data(self):
        # Sample users
        self.users = {
            'user_001': {'name': 'John Doe', 'dept': 'Engineering', 'risk_score': 15},
            'user_002': {'name': 'Jane Smith', 'dept': 'Finance', 'risk_score': 78},
            'user_003': {'name': 'Bob Wilson', 'dept': 'Marketing', 'risk_score': 42},
            'user_004': {'name': 'Alice Brown', 'dept': 'HR', 'risk_score': 8},
            'user_005': {'name': 'Charlie Davis', 'dept': 'Operations', 'risk_score': 91}
        }
        
        # Sample AI tools
        self.ai_tools = {
            'openai_gpt4': {'name': 'OpenAI GPT-4', 'sanctioned': True, 'category': 'LLM'},
            'anthropic_claude': {'name': 'Anthropic Claude', 'sanctioned': True, 'category': 'LLM'},
            'huggingface_api': {'name': 'HuggingFace API', 'sanctioned': False, 'category': 'ML Platform'},
            'cohere_embed': {'name': 'Cohere Embeddings', 'sanctioned': True, 'category': 'Embeddings'},
            'chatgpt_web': {'name': 'ChatGPT Web', 'sanctioned': False, 'category': 'LLM'},
            'gemini_api': {'name': 'Google Gemini', 'sanctioned': True, 'category': 'LLM'},
            'mistral_api': {'name': 'Mistral AI', 'sanctioned': False, 'category': 'LLM'}
        }
        
        # Sample events
        risk_patterns = [
            ('user_002', 'chatgpt_web', 'Queried customer database with unsanctioned tool', 'critical'),
            ('user_005', 'mistral_api', 'Large data export to external LLM', 'critical'),
            ('user_003', 'huggingface_api', 'First-time AI tool usage detected', 'medium'),
            ('user_001', 'openai_gpt4', 'Normal approved usage', 'low'),
            ('user_002', 'honeypot_llm', 'Attempted to access honeypot AI service', 'critical'),
            ('user_004', 'anthropic_claude', 'Normal approved usage', 'low')
        ]
        
        for i, (user, tool, desc, severity) in enumerate(risk_patterns):
            event = {
                'id': f'evt_{i+1:03d}',
                'timestamp': (datetime.now() - timedelta(hours=random.randint(1, 48))).isoformat(),
                'user_id': user,
                'ai_tool': tool,
                'event_type': 'api_call' if 'api' in tool else 'web_access',
                'description': desc,
                'severity': severity,
                'data_sensitivity': random.choice(['low', 'medium', 'high', 'critical']),
                'prompt_tokens': random.randint(100, 5000),
                'response_tokens': random.randint(200, 8000),
                'source_ip': f'10.0.{random.randint(1,255)}.{random.randint(1,255)}',
                'destination': f'{tool}.example.com'
            }
            self.events.append(event)
        
        # Sample alerts
        self.alerts = [
            {
                'id': 'alert_001',
                'timestamp': datetime.now().isoformat(),
                'severity': 'critical',
                'title': 'Unsanctioned AI Tool Usage - Data Exfiltration Risk',
                'user_id': 'user_002',
                'description': 'User accessed ChatGPT web interface with company database queries',
                'recommendation': 'Block user access, review data sent, conduct security interview',
                'status': 'open'
            },
            {
                'id': 'alert_002',
                'timestamp': (datetime.now() - timedelta(hours=2)).isoformat(),
                'severity': 'critical',
                'title': 'Honeypot AI Service Accessed',
                'user_id': 'user_002',
                'description': 'User attempted to access decoy AI endpoint - possible insider threat',
                'recommendation': 'Immediate investigation required, restrict network access',
                'status': 'investigating'
            },
            {
                'id': 'alert_003',
                'timestamp': (datetime.now() - timedelta(hours=5)).isoformat(),
                'severity': 'high',
                'title': 'Abnormal AI Usage Pattern',
                'user_id': 'user_005',
                'description': 'User exported 50MB data to external LLM service',
                'recommendation': 'Review export logs, check for PII/sensitive data leakage',
                'status': 'open'
            },
            {
                'id': 'alert_004',
                'timestamp': (datetime.now() - timedelta(hours=8)).isoformat(),
                'severity': 'medium',
                'title': 'First-Time AI Tool Usage',
                'user_id': 'user_003',
                'description': 'Marketing user accessed HuggingFace API for first time',
                'recommendation': 'Monitor continued usage, verify business justification',
                'status': 'closed'
            }
        ]
        
        # Sample honeypots
        self.honeypots = [
            {
                'id': 'honey_001',
                'name': 'Internal LLM Beta',
                'endpoint': 'https://internal-llm-beta.company.local',
                'status': 'active',
                'hits': 3,
                'last_hit': (datetime.now() - timedelta(hours=2)).isoformat()
            },
            {
                'id': 'honey_002',
                'name': 'Free GPT Proxy',
                'endpoint': 'https://free-gpt.company.local',
                'status': 'active',
                'hits': 0,
                'last_hit': None
            }
        ]
        
        # Sample policies
        self.policies = [
            {
                'id': 'pol_001',
                'name': 'Block Unsanctioned AI Tools',
                'description': 'Prevent access to non-approved AI/ML services',
                'action': 'block',
                'enabled': True
            },
            {
                'id': 'pol_002',
                'name': 'Alert on High-Risk Data Queries',
                'description': 'Trigger alert when sensitive data is sent to AI tools',
                'action': 'alert',
                'enabled': True
            },
            {
                'id': 'pol_003',
                'name': 'Monitor First-Time AI Usage',
                'description': 'Log when users access AI tools for the first time',
                'action': 'log',
                'enabled': True
            }
        ]

data_store = AegisDataStore()

@aegis_ai.route('/')
def dashboard():
    return render_template('aegis.html')

@aegis_ai.route('/api/dashboard/stats')
def dashboard_stats():
    critical_alerts = len([a for a in data_store.alerts if a['severity'] == 'critical' and a['status'] == 'open'])
    high_risk_users = len([u for u, d in data_store.users.items() if d['risk_score'] > 70])
    total_events = len(data_store.events)
    unsanctioned_tools = len([t for t, d in data_store.ai_tools.items() if not d['sanctioned']])
    
    return jsonify({
        'critical_alerts': critical_alerts,
        'high_risk_users': high_risk_users,
        'total_events_24h': total_events,
        'unsanctioned_tools_detected': unsanctioned_tools,
        'honeypot_hits': sum(h['hits'] for h in data_store.honeypots)
    })

@aegis_ai.route('/api/alerts')
def get_alerts():
    return jsonify(data_store.alerts)

@aegis_ai.route('/api/events')
def get_events():
    limit = int(request.args.get('limit', 50))
    return jsonify(data_store.events[:limit])

@aegis_ai.route('/api/users')
def get_users():
    users_list = [
        {
            'user_id': uid,
            'name': data['name'],
            'dept': data['dept'],
            'risk_score': data['risk_score'],
            'status': 'critical' if data['risk_score'] > 80 else 'warning' if data['risk_score'] > 50 else 'normal'
        }
        for uid, data in data_store.users.items()
    ]
    return jsonify(sorted(users_list, key=lambda x: x['risk_score'], reverse=True))

@aegis_ai.route('/api/ai-tools')
def get_ai_tools():
    tools_list = [
        {
            'tool_id': tid,
            'name': data['name'],
            'category': data['category'],
            'sanctioned': data['sanctioned'],
            'usage_count': len([e for e in data_store.events if e['ai_tool'] == tid])
        }
        for tid, data in data_store.ai_tools.items()
    ]
    return jsonify(tools_list)

@aegis_ai.route('/api/honeypots')
def get_honeypots():
    return jsonify(data_store.honeypots)

@aegis_ai.route('/api/policies')
def get_policies():
    return jsonify(data_store.policies)

@aegis_ai.route('/api/risk-analysis')
def risk_analysis():
    # Analyze risk patterns
    user_activity = defaultdict(int)
    tool_usage = defaultdict(int)
    
    for event in data_store.events:
        user_activity[event['user_id']] += 1
        tool_usage[event['ai_tool']] += 1
    
    risk_trends = [
        {'hour': f'{i}:00', 'critical': random.randint(0, 5), 'high': random.randint(2, 10), 
         'medium': random.randint(5, 15), 'low': random.randint(10, 30)}
        for i in range(24)
    ]
    
    return jsonify({
        'risk_trends': risk_trends[-12:],  # Last 12 hours
        'top_risk_users': sorted(
            [{'user_id': uid, 'score': data['risk_score']} for uid, data in data_store.users.items()],
            key=lambda x: x['score'],
            reverse=True
        )[:5]
    })

@aegis_ai.route('/api/alert/<alert_id>/update', methods=['POST'])
def update_alert(alert_id):
    data = request.json
    for alert in data_store.alerts:
        if alert['id'] == alert_id:
            alert['status'] = data.get('status', alert['status'])
            return jsonify({'success': True, 'alert': alert})
    return jsonify({'success': False, 'error': 'Alert not found'}), 404


# Flask app setup (for testing)
if __name__ == '__main__':
    app = Flask(__name__)
    app.register_blueprint(aegis_ai, url_prefix='/aegis_ai')
    
    @app.route('/')
    def index():
        return '<h1>Aegis AI Server</h1><p>Access the dashboard at <a href="/aegis_ai">/aegis_ai</a></p>'
    
    print("🛡️  Aegis AI - AI/ML Security Monitoring Platform")
    print("=" * 60)
    print("Starting server...")
    print("Access dashboard at: http://localhost:5000/aegis_ai")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)