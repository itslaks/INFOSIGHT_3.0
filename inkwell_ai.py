from flask import Flask, request, jsonify, render_template, Blueprint
import os
from datetime import datetime
import json
import re
from typing import Dict, List, Tuple, Optional
import sqlite3

# Create Blueprint for InkWell AI (no url_prefix - server.py handles it)
inkwell_ai = Blueprint('inkwell_ai', __name__, template_folder='templates')

# Try to import and configure Gemini
try:
    import google.generativeai as genai
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyCMwpK-6Dr9X_MpcCyRR1PJcixg4pW55e8')
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    print("✓ Gemini API configured successfully for InkWell AI")
except Exception as e:
    print(f"⚠ Warning: Gemini not available for InkWell AI: {e}")
    gemini_model = None


class DatabaseManager:
    def __init__(self, db_name='prompt_optimizer.db'):
        self.db_name = db_name
        self.init_db()
    
    def init_db(self):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS optimizations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                original_prompt TEXT,
                optimized_prompt TEXT,
                gemini_enhanced TEXT,
                category TEXT,
                enhancements_applied TEXT,
                quality_score REAL,
                clarity_score REAL,
                completeness_score REAL,
                timestamp DATETIME,
                processing_time REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS favorites (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                original_prompt TEXT,
                enhanced_prompt TEXT,
                gemini_enhanced TEXT,
                timestamp DATETIME,
                UNIQUE(user_id, original_prompt)
            )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_id ON optimizations(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON optimizations(timestamp)')
        
        conn.commit()
        conn.close()
        print("✓ InkWell AI database initialized")
    
    def save_optimization(self, data: Dict):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO optimizations 
            (user_id, original_prompt, optimized_prompt, gemini_enhanced, category, 
             enhancements_applied, quality_score, clarity_score, completeness_score, 
             timestamp, processing_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data.get('user_id', 'anonymous'),
            data['original_prompt'],
            data['optimized_prompt'],
            data.get('gemini_enhanced', ''),
            data.get('category', 'general'),
            json.dumps(data.get('enhancements_applied', [])),
            data.get('quality_score', 0),
            data.get('clarity_score', 0),
            data.get('completeness_score', 0),
            datetime.now().isoformat(),
            data.get('processing_time', 0)
        ))
        conn.commit()
        conn.close()
    
    def get_user_history(self, user_id: str, limit: int = 50):
        conn = sqlite3.connect(self.db_name)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM optimizations 
            WHERE user_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (user_id, limit))
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results
    
    def add_favorite(self, user_id: str, original: str, enhanced: str, gemini_enhanced: str):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO favorites (user_id, original_prompt, enhanced_prompt, gemini_enhanced, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, original, enhanced, gemini_enhanced, datetime.now().isoformat()))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()
    
    def get_favorites(self, user_id: str):
        conn = sqlite3.connect(self.db_name)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM favorites 
            WHERE user_id = ? 
            ORDER BY timestamp DESC
        ''', (user_id,))
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results


db_manager = DatabaseManager()


class AdvancedPromptOptimizer:
    def __init__(self):
        self.transformation_rules = {
            'clarity': self._enhance_clarity,
            'structure': self._add_structure,
            'context': self._add_context,
            'specificity': self._enhance_specificity,
            'role_play': self._add_role_context,
            'output_format': self._specify_output_format,
            'constraints': self._add_constraints,
            'examples': self._request_examples,
            'tone_control': self._add_tone_control,
        }
    
    def _enhance_clarity(self, text: str) -> str:
        filler_patterns = [
            r'\b(basically|essentially|actually|just|really|very|quite|somehow|kind of|sort of)\b',
            r'\b(um|uh|like|you know)\b',
        ]
        for pattern in filler_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _add_structure(self, text: str) -> str:
        return f"""{text}

Please structure your response with:
- Clear introduction
- Main points (numbered if applicable)
- Supporting details
- Conclusion"""
    
    def _add_context(self, text: str) -> str:
        return f"""{text}

Provide context including:
- Relevant background information
- Current state and assumptions
- Key dependencies"""
    
    def _enhance_specificity(self, text: str) -> str:
        return f"""{text}

Be specific about:
- Exact requirements and metrics
- Timeframes and constraints
- Technologies or methodologies
- Target outcomes"""
    
    def _add_role_context(self, text: str) -> str:
        return f"""Act as an expert specialist with deep knowledge in this domain.

{text}

Provide authoritative insights using industry best practices."""
    
    def _specify_output_format(self, text: str) -> str:
        return f"""{text}

Format your response using:
- Clear markdown headers
- Bullet points for lists
- Code blocks where applicable
- Tables for structured data"""
    
    def _add_constraints(self, text: str) -> str:
        return f"""{text}

Please adhere to these constraints:
- Stay focused on the core topic
- Maintain professional tone
- Prioritize accuracy
- Be concise yet comprehensive"""
    
    def _request_examples(self, text: str) -> str:
        return f"""{text}

Include practical examples:
- Real-world use cases
- Code samples (if applicable)
- Before/after comparisons
- Edge cases"""
    
    def _add_tone_control(self, text: str) -> str:
        return f"""{text}

Use a professional yet approachable tone that is:
- Clear and accessible
- Technically accurate
- Engaging and informative"""
    
    def detect_category(self, text: str) -> str:
        categories = {
            'Programming': ['code', 'program', 'script', 'function', 'api', 'python', 'javascript'],
            'Writing': ['write', 'blog', 'article', 'content', 'story', 'essay'],
            'Analysis': ['analyze', 'data', 'research', 'evaluate', 'compare'],
            'Business': ['business', 'strategy', 'marketing', 'plan'],
            'Creative': ['create', 'design', 'creative', 'brainstorm'],
            'Education': ['explain', 'learn', 'teach', 'understand', 'tutorial'],
        }
        text_lower = text.lower()
        for category, keywords in categories.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        return 'General'
    
    def calculate_quality_metrics(self, original: str, optimized: str) -> Dict:
        metrics = {
            'clarity_score': min(100, 60 + len(optimized.split()) * 0.5),
            'completeness_score': min(100, 50 + (optimized.count('\n') * 10)),
            'structure_score': min(100, 50 + (optimized.count('-') + optimized.count('*')) * 5),
            'specificity_score': min(100, 50 + len(re.findall(r'\d+', optimized)) * 3),
        }
        metrics['overall_score'] = sum(metrics.values()) / len(metrics)
        return metrics
    
    def enhance_with_gemini(self, prompt: str, enhancements: List[str], level: str = 'moderate') -> Optional[str]:
        if not gemini_model:
            return None
        
        level_instructions = {
            'light': 'Make minor improvements while preserving original intent.',
            'moderate': 'Create a balanced, well-structured version with clear improvements.',
            'aggressive': 'Completely refine into a professional, detailed prompt with comprehensive structure.',
            'expert': 'Transform into an expert-level prompt with maximum clarity and detail.',
        }
        
        enhancement_desc = {
            'clarity': 'improve clarity',
            'structure': 'add structure',
            'context': 'add context',
            'specificity': 'increase specificity',
            'role_play': 'add expert role',
            'output_format': 'specify format',
            'constraints': 'define constraints',
            'examples': 'request examples',
            'tone_control': 'control tone',
        }
        
        focuses = ', '.join([enhancement_desc.get(e, e) for e in enhancements]) if enhancements else 'all aspects'
        
        gemini_prompt = f"""You are an expert prompt engineer. Transform this prompt to be more effective.

Original Prompt:
{prompt}

Enhancement Level: {level} - {level_instructions.get(level, level_instructions['moderate'])}
Focus on: {focuses}

Improve clarity, structure, and effectiveness while maintaining the core intent.

Provide ONLY the enhanced prompt without any explanation or preamble."""

        try:
            response = gemini_model.generate_content(gemini_prompt)
            return response.text.strip() if response.text else None
        except Exception as e:
            print(f"Gemini error: {e}")
            return None
    
    def optimize(self, user_input: str, enhancements: Optional[List[str]] = None, 
                 use_gemini: bool = True, enhancement_level: str = 'moderate') -> Tuple[str, Optional[str], Dict]:
        if not enhancements:
            enhancements = ['clarity', 'structure', 'specificity']
        
        optimized = user_input
        for enhancement in enhancements:
            if enhancement in self.transformation_rules:
                optimized = self.transformation_rules[enhancement](optimized)
        
        gemini_optimized = None
        if use_gemini and gemini_model:
            gemini_optimized = self.enhance_with_gemini(user_input, enhancements, enhancement_level)
        
        final_output = gemini_optimized if gemini_optimized else optimized
        metrics = self.calculate_quality_metrics(user_input, final_output)
        
        metadata = {
            'original_length': len(user_input),
            'optimized_length': len(optimized),
            'gemini_length': len(gemini_optimized) if gemini_optimized else 0,
            'enhancements_applied': enhancements,
            'category': self.detect_category(user_input),
            'enhancement_level': enhancement_level,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        return optimized, gemini_optimized, metadata


optimizer = AdvancedPromptOptimizer()


# API Routes
@inkwell_ai.route('/')
def index():
    return render_template('inkwell_ai.html')

@inkwell_ai.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'gemini_available': gemini_model is not None
    }), 200


@inkwell_ai.route('/api/optimize', methods=['POST'])                        
def optimize_prompt():
    try:
        data = request.json
        user_input = data.get('prompt', '').strip()
        enhancements = data.get('enhancements', ['clarity', 'structure', 'specificity'])
        use_gemini = data.get('use_gemini', True)
        enhancement_level = data.get('enhancement_level', 'moderate')
        user_id = data.get('user_id', 'anonymous')
        
        if not user_input or len(user_input) < 3:
            return jsonify({'error': 'Prompt too short (minimum 3 characters)'}), 400
        
        if len(user_input) > 5000:
            return jsonify({'error': 'Prompt too long (maximum 5000 characters)'}), 400
        
        optimized, gemini_optimized, metadata = optimizer.optimize(
            user_input, enhancements, use_gemini, enhancement_level
        )
        
        db_manager.save_optimization({
            'user_id': user_id,
            'original_prompt': user_input,
            'optimized_prompt': optimized,
            'gemini_enhanced': gemini_optimized or '',
            'category': metadata['category'],
            'enhancements_applied': metadata['enhancements_applied'],
            'quality_score': metadata['metrics']['overall_score'],
            'clarity_score': metadata['metrics']['clarity_score'],
            'completeness_score': metadata['metrics']['completeness_score'],
            'processing_time': 0
        })
        
        return jsonify({
            'success': True,
            'original_prompt': user_input,
            'optimized_prompt': optimized,
            'gemini_enhanced': gemini_optimized,
            'metadata': metadata
        }), 200
    
    except Exception as e:
        print(f"❌ Error in optimize_prompt: {e}")
        return jsonify({'error': str(e)}), 500


@inkwell_ai.route('/api/analyze', methods=['POST'])
def analyze_prompt():
    try:
        data = request.json
        prompt = data.get('prompt', '').strip()
        
        if not prompt:
            return jsonify({'error': 'Empty prompt'}), 400
        
        category = optimizer.detect_category(prompt)
        metrics = optimizer.calculate_quality_metrics(prompt, prompt)
        
        return jsonify({
            'category': category,
            'metrics': metrics,
            'word_count': len(prompt.split()),
            'character_count': len(prompt),
            'sentences': len(re.split(r'[.!?]+', prompt))
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@inkwell_ai.route('/api/history/<user_id>', methods=['GET'])
def get_history(user_id):
    try:
        limit = request.args.get('limit', 50, type=int)
        history = db_manager.get_user_history(user_id, limit)
        return jsonify({'history': history, 'count': len(history)}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@inkwell_ai.route('/api/favorites', methods=['POST'])
def add_favorite():
    try:
        data = request.json     
        user_id = data.get('user_id')
        original = data.get('original')
        enhanced = data.get('enhanced')
        gemini_enhanced = data.get('gemini_enhanced', '')
        
        if not user_id or not original:
            return jsonify({'error': 'Missing required fields'}), 400
        
        success = db_manager.add_favorite(user_id, original, enhanced, gemini_enhanced)
        
        return jsonify({
            'success': success, 
            'message': 'Added to favorites!' if success else 'Already in favorites'
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@inkwell_ai.route('/api/favorites/<user_id>', methods=['GET'])
def get_favorites(user_id):
    try:
        favorites = db_manager.get_favorites(user_id)
        return jsonify({'favorites': favorites, 'count': len(favorites)}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500