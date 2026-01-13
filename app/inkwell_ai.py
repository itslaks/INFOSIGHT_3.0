from flask import Flask, Blueprint, request, jsonify, send_from_directory, render_template, g
from flask_cors import CORS
import os
import re
import time
import json
import hashlib
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
import threading
from collections import defaultdict

# OWASP: Import security utilities for rate limiting and input validation
try:
    from utils.security import rate_limit_api, rate_limit_strict, validate_request, InputValidator
except ImportError:
    # Fallback if security utils not available
    def rate_limit_api(*args, **kwargs):
        def decorator(f):
            return f
        return decorator
    def rate_limit_strict(*args, **kwargs):
        def decorator(f):
            return f
        return decorator
    def validate_request(*args, **kwargs):
        def decorator(f):
            return f
        return decorator
    class InputValidator:
        @staticmethod
        def validate_string(value, name, max_length=397, required=False):
            if value is None and required:
                raise ValueError(f"{name} is required")
            if value and len(str(value)) > max_length:
                raise ValueError(f"{name} exceeds maximum length")
            return str(value)[:max_length] if value else ""

# Load environment variables
load_dotenv()

# Try to import Groq
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("⚠️  WARNING: groq package not installed")

# Try to import local LLM utilities
try:
    from utils.local_llm_utils import generate_with_ollama, check_ollama_available
    LOCAL_LLM_AVAILABLE = True
except ImportError:
    LOCAL_LLM_AVAILABLE = False
    print("⚠️  WARNING: local_llm_utils not available")

# Create blueprint for integration with main server
inkwell_ai = Blueprint('inkwell_ai', __name__, template_folder='templates')

# Initialize Groq client
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
groq_client = None
if GROQ_AVAILABLE and GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        print("✅ Groq AI initialized successfully!")
    except Exception as e:
        print(f"⚠️  WARNING: Failed to initialize Groq: {e}")
        groq_client = None
else:
    if not GROQ_AVAILABLE:
        print("⚠️  WARNING: groq package not installed!")
    elif not GROQ_API_KEY:
        print("⚠️  WARNING: GROQ_API_KEY not found in .env file!")


@dataclass
class PromptMetrics:
    clarity_score: float
    specificity_score: float
    structure_score: float
    actionability_score: float
    context_richness: float
    technical_depth: float
    creativity_score: float
    completeness_score: float
    readability_score: float
    overall_score: float
    word_count: int
    char_count: int
    sentence_count: int
    avg_sentence_length: float
    vocabulary_richness: float
    complexity_index: float


class DatabaseManager:
    def __init__(self, db_name='inkwell_ultimate.db'):
        self.db_name = db_name
        self.init_db()
    
    def init_db(self):
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            
            # Optimizations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS optimizations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_id TEXT,
                    original_prompt TEXT NOT NULL,
                    rule_based_prompt TEXT,
                    ai_enhanced_prompt TEXT,
                    best_version TEXT,
                    category TEXT,
                    tags TEXT,
                    clarity_score REAL,
                    specificity_score REAL,
                    structure_score REAL,
                    overall_score REAL,
                    improvements TEXT,
                    processing_time REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Templates table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS prompt_templates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    template_id TEXT UNIQUE,
                    name TEXT NOT NULL,
                    category TEXT,
                    template_text TEXT NOT NULL,
                    description TEXT,
                    usage_count INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # User profiles
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    total_optimizations INTEGER DEFAULT 0,
                    avg_quality_score REAL DEFAULT 0,
                    favorite_categories TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # A/B Tests
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ab_tests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id TEXT UNIQUE,
                    user_id TEXT,
                    original_prompt TEXT,
                    variations TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Batch jobs
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS batch_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT UNIQUE,
                    user_id TEXT,
                    status TEXT,
                    total_items INTEGER,
                    completed_items INTEGER DEFAULT 0,
                    results TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    completed_at DATETIME
                )
            ''')
            
            conn.commit()
            
        # Migrate existing database if needed
        self._migrate_database()
            
        # Insert default templates
        self._insert_default_templates()
        print("✅ Database initialized")
    
    def _migrate_database(self):
        """Migrate database schema for existing databases"""
        try:
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()
                
                # Get all table names
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
                # Migrate batch_jobs table
                if 'batch_jobs' in tables:
                    cursor.execute("PRAGMA table_info(batch_jobs)")
                    columns = [row[1] for row in cursor.fetchall()]
                    
                    if 'completed_items' not in columns:
                        print("🔄 Migrating batch_jobs table: adding completed_items column...")
                        try:
                            cursor.execute('ALTER TABLE batch_jobs ADD COLUMN completed_items INTEGER DEFAULT 0')
                            conn.commit()
                            print("✅ Migration completed: added completed_items column")
                        except sqlite3.OperationalError as e:
                            if 'duplicate column' not in str(e).lower():
                                raise
                
                # Migrate optimizations table
                if 'optimizations' in tables:
                    cursor.execute("PRAGMA table_info(optimizations)")
                    opt_columns = [row[1] for row in cursor.fetchall()]
                    
                    if 'ai_enhanced_prompt' not in opt_columns:
                        print("🔄 Migrating optimizations table: adding ai_enhanced_prompt column...")
                        try:
                            cursor.execute('ALTER TABLE optimizations ADD COLUMN ai_enhanced_prompt TEXT')
                            conn.commit()
                            print("✅ Added ai_enhanced_prompt column")
                        except sqlite3.OperationalError as e:
                            if 'duplicate column' not in str(e).lower():
                                raise
                    
                    if 'processing_time' not in opt_columns:
                        print("🔄 Migrating optimizations table: adding processing_time column...")
                        try:
                            cursor.execute('ALTER TABLE optimizations ADD COLUMN processing_time REAL')
                            conn.commit()
                            print("✅ Added processing_time column")
                        except sqlite3.OperationalError as e:
                            if 'duplicate column' not in str(e).lower():
                                raise
                        
        except Exception as e:
            print(f"⚠️  Migration error (non-critical): {e}")
    
    def _insert_default_templates(self):
        templates = [
            {
                'template_id': 'code_func_1',
                'name': 'Code Function Template',
                'category': 'Coding',
                'description': 'Template for creating code functions with best practices',
                'template_text': '''Create a {language} function named {name} that {purpose}.

Requirements:
- Input parameters: {inputs}
- Output: {outputs}
- Include error handling for edge cases
- Add comprehensive documentation
- Follow {language} best practices
- Include type hints/annotations where applicable'''
            },
            {
                'template_id': 'blog_article_1',
                'name': 'Blog Article Template',
                'category': 'Writing',
                'description': 'Professional blog article structure',
                'template_text': '''Write a {length} blog article about {topic} for {audience}.

Structure:
- Compelling headline that hooks the reader
- Engaging introduction with a problem statement
- 3-5 main points with supporting evidence
- Real-world examples and case studies
- Actionable takeaways
- Strong conclusion with call-to-action

Tone: {tone}
SEO Keywords: {keywords}'''
            },
            {
                'template_id': 'data_analysis_1',
                'name': 'Data Analysis Template',
                'category': 'Analysis',
                'description': 'Comprehensive data analysis framework',
                'template_text': '''Analyze the following data: {data_description}

Analysis Requirements:
- Descriptive statistics (mean, median, mode, std dev)
- Data distribution and patterns
- Correlation analysis
- Outlier detection
- Trend identification
- Key insights and findings
- Actionable recommendations
- Visualization suggestions: {viz_types}'''
            },
            {
                'template_id': 'tutorial_1',
                'name': 'Step-by-Step Tutorial',
                'category': 'Education',
                'description': 'Create comprehensive tutorials',
                'template_text': '''Create a detailed tutorial for {skill} aimed at {level} learners.

Include:
- Prerequisites and requirements
- Learning objectives
- Step-by-step instructions with screenshots
- Code examples (if applicable)
- Common mistakes and troubleshooting
- Practice exercises
- Additional resources for further learning'''
            },
            {
                'template_id': 'business_analysis_1',
                'name': 'Business Analysis Template',
                'category': 'Business',
                'description': 'Strategic business analysis framework',
                'template_text': '''Conduct a comprehensive business analysis of {subject}.

Framework: {framework} (SWOT/Porter's Five Forces/PESTEL)

Analysis Components:
- Current market position
- Competitive landscape
- Strengths and advantages
- Weaknesses and vulnerabilities
- Opportunities for growth
- Threats and risks
- Strategic recommendations
- Implementation roadmap'''
            },
            {
                'template_id': 'creative_writing_1',
                'name': 'Creative Writing Template',
                'category': 'Creative',
                'description': 'Creative storytelling framework',
                'template_text': '''Write a {genre} {format} about {theme}.

Story Elements:
- Setting: {setting}
- Main characters: {characters}
- Central conflict: {conflict}
- Plot structure: Beginning, Rising Action, Climax, Falling Action, Resolution
- Narrative style: {style}
- Target length: {length}
- Tone and atmosphere: {tone}'''
            }
        ]
        
        try:
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()
                for template in templates:
                    cursor.execute('''
                        INSERT OR IGNORE INTO prompt_templates 
                        (template_id, name, category, template_text, description)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (
                        template['template_id'],
                        template['name'],
                        template['category'],
                        template['template_text'],
                        template['description']
                    ))
                conn.commit()
        except Exception as e:
            print(f"⚠️  Error inserting templates: {e}")
    
    def save_optimization(self, data: Dict, user_id: str):
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO optimizations 
                (user_id, original_prompt, rule_based_prompt, ai_enhanced_prompt,
                 best_version, category, tags, clarity_score, specificity_score,
                 structure_score, overall_score, improvements, processing_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id, data['original'], data.get('rule_based', ''),
                data.get('ai_enhanced', ''), data['best_version'], data['category'],
                json.dumps(data.get('tags', [])), data['metrics']['clarity_score'],
                data['metrics']['specificity_score'], data['metrics']['structure_score'],
                data['metrics']['overall_score'], json.dumps(data.get('improvements', [])),
                data.get('processing_time', 0)
            ))
            conn.commit()
            return cursor.lastrowid
    
    def get_user_insights(self, user_id: str, days: int = 30) -> Dict:
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        with sqlite3.connect(self.db_name) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Overall stats
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_optimizations,
                    AVG(overall_score) as avg_quality_score,
                    AVG(processing_time) as avg_processing_time
                FROM optimizations
                WHERE user_id = ? AND timestamp >= ?
            ''', (user_id, cutoff))
            stats = dict(cursor.fetchone())
            
            # Category distribution
            cursor.execute('''
                SELECT category, COUNT(*) as count, AVG(overall_score) as avg_score
                FROM optimizations
                WHERE user_id = ? AND timestamp >= ?
                GROUP BY category
                ORDER BY count DESC
            ''', (user_id, cutoff))
            categories = [dict(row) for row in cursor.fetchall()]
            
            # Recent optimizations
            cursor.execute('''
                SELECT original_prompt, best_version, overall_score, timestamp
                FROM optimizations
                WHERE user_id = ? AND timestamp >= ?
                ORDER BY timestamp DESC
                LIMIT 10
            ''', (user_id, cutoff))
            recent = [dict(row) for row in cursor.fetchall()]
            
            return {
                'user_id': user_id,
                'period_days': days,
                'statistics': stats,
                'category_distribution': categories,
                'recent_optimizations': recent
            }
    
    def get_templates(self, category: str = None):
        with sqlite3.connect(self.db_name) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if category:
                cursor.execute('SELECT * FROM prompt_templates WHERE category = ?', (category,))
            else:
                cursor.execute('SELECT * FROM prompt_templates')
            
            return [dict(row) for row in cursor.fetchall()]
    
    def save_ab_test(self, test_id: str, user_id: str, original: str, variations: List):
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO ab_tests (test_id, user_id, original_prompt, variations)
                VALUES (?, ?, ?, ?)
            ''', (test_id, user_id, original, json.dumps(variations)))
            conn.commit()
    
    def create_batch_job(self, job_id: str, user_id: str, total_items: int):
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO batch_jobs (job_id, user_id, status, total_items)
                VALUES (?, ?, 'processing', ?)
            ''', (job_id, user_id, total_items))
            conn.commit()
    
    def update_batch_job(self, job_id: str, completed: int, results: List, status: str = 'processing'):
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE batch_jobs 
                SET completed_items = ?, results = ?, status = ?,
                    completed_at = CASE WHEN ? = 'completed' THEN CURRENT_TIMESTAMP ELSE NULL END
                WHERE job_id = ?
            ''', (completed, json.dumps(results), status, status, job_id))
            conn.commit()


class UltimatePromptOptimizer:
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.groq = groq_client
        
        self.strategies = {
            'clarity': self._enhance_clarity,
            'structure': self._add_structure,
            'context': self._enrich_context,
            'specificity': self._increase_specificity,
            'role_definition': self._define_role,
            'examples': self._include_examples,
        }
    
    def analyze_prompt(self, prompt: str) -> PromptMetrics:
        """Comprehensive prompt analysis"""
        words = prompt.split()
        sentences = re.split(r'[.!?]+', prompt)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        word_count = len(words)
        char_count = len(prompt)
        sentence_count = max(len(sentences), 1)
        avg_sentence_length = word_count / sentence_count
        unique_words = len(set(word.lower() for word in words))
        vocabulary_richness = unique_words / max(word_count, 1)
        
        # Calculate scores
        filler_words = len(re.findall(r'\b(basically|essentially|actually|just|really|very|quite|somehow|kind of|sort of|like|you know)\b', prompt, re.IGNORECASE))
        clarity_score = max(0, min(100, 85 - (filler_words * 3) - (max(0, avg_sentence_length - 20) * 2)))
        
        numbers = len(re.findall(r'\d+', prompt))
        specific_terms = len(re.findall(r'\b(specific|exactly|precisely|detailed|particular|explicit)\b', prompt, re.IGNORECASE))
        specificity_score = min(100, 50 + (numbers * 5) + (specific_terms * 8))
        
        structure_markers = prompt.count('\n') + prompt.count('- ') + prompt.count('* ') + prompt.count(':')
        structure_score = min(100, 40 + (structure_markers * 6))
        
        action_verbs = len(re.findall(r'\b(create|build|write|analyze|design|develop|implement|explain|describe|compare|evaluate|generate|optimize|solve)\b', prompt, re.IGNORECASE))
        actionability_score = min(100, 50 + (action_verbs * 10))
        
        context_markers = len(re.findall(r'\b(background|context|because|given|considering|assuming|goal|objective|purpose|requirement|constraint)\b', prompt, re.IGNORECASE))
        context_richness = min(100, 40 + (context_markers * 12))
        
        technical_terms = len(re.findall(r'\b(algorithm|architecture|framework|methodology|implementation|optimization|scalability|performance)\b', prompt, re.IGNORECASE))
        technical_depth = min(100, 30 + (technical_terms * 15))
        
        questions = prompt.count('?')
        creative_terms = len(re.findall(r'\b(creative|innovative|unique|novel|explore|brainstorm|imagine|alternative)\b', prompt, re.IGNORECASE))
        creativity_score = min(100, 40 + (questions * 8) + (creative_terms * 10))
        
        has_task = bool(re.search(r'\b(create|write|analyze|design|build)\b', prompt, re.IGNORECASE))
        has_context = len(prompt.split()) > 15
        has_format = bool(re.search(r'\b(format|structure|include|provide)\b', prompt, re.IGNORECASE))
        completeness_score = (has_task * 40) + (has_context * 30) + (has_format * 30)
        
        avg_word_length = sum(len(word) for word in words) / max(word_count, 1)
        readability_score = max(0, min(100, 100 - (avg_sentence_length * 1.5) - (avg_word_length * 4)))
        
        complexity_index = (avg_sentence_length * 0.4) + (vocabulary_richness * 30) + (technical_depth * 0.3)
        
        overall_score = (
            clarity_score * 0.15 +
            specificity_score * 0.15 +
            structure_score * 0.1 +
            actionability_score * 0.15 +
            context_richness * 0.15 +
            completeness_score * 0.15 +
            readability_score * 0.1 +
            technical_depth * 0.05
        )
        
        return PromptMetrics(
            clarity_score=round(clarity_score, 2),
            specificity_score=round(specificity_score, 2),
            structure_score=round(structure_score, 2),
            actionability_score=round(actionability_score, 2),
            context_richness=round(context_richness, 2),
            technical_depth=round(technical_depth, 2),
            creativity_score=round(creativity_score, 2),
            completeness_score=round(completeness_score, 2),
            readability_score=round(readability_score, 2),
            overall_score=round(overall_score, 2),
            word_count=word_count,
            char_count=char_count,
            sentence_count=sentence_count,
            avg_sentence_length=round(avg_sentence_length, 2),
            vocabulary_richness=round(vocabulary_richness, 3),
            complexity_index=round(complexity_index, 2)
        )
    
    def _enhance_clarity(self, text: str) -> str:
        patterns = [
            (r'\b(basically|essentially|actually|just|really|very|quite|somehow|kind of|sort of)\b', ''),
            (r'\s+', ' '),
        ]
        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text.strip()
    
    def _add_structure(self, text: str) -> str:
        return f"""{text}

Please organize your response with:
1. Clear introduction stating the main objective
2. Core content broken into logical sections
3. Supporting details with examples
4. Summary of key takeaways"""
    
    def _enrich_context(self, text: str) -> str:
        return f"""Context and Background:
{text}

Additional Context Requirements:
- Relevant background information
- Current state and assumptions
- Dependencies and constraints
- Success criteria and goals"""
    
    def _increase_specificity(self, text: str) -> str:
        return f"""{text}

Specific Requirements:
- Exact metrics and measurements
- Concrete timeframes and deadlines
- Specific technologies or methodologies
- Measurable outcomes and deliverables
- Detailed constraints and limitations"""
    
    def _define_role(self, text: str) -> str:
        return f"""You are a world-class expert with deep knowledge and extensive experience in this domain.

{text}

Provide authoritative, professional insights leveraging best practices and cutting-edge knowledge."""
    
    def _include_examples(self, text: str) -> str:
        return f"""{text}

Include Practical Examples:
- Real-world use cases demonstrating the concept
- Concrete code samples or demonstrations
- Before/after comparisons showing improvements
- Edge cases and how to handle them
- Common pitfalls and how to avoid them"""
    
    def ai_enhance_groq(self, prompt: str, strategies: List[str], level: str) -> Optional[str]:
        """AI enhancement using Groq with local LLM fallback"""
        level_instructions = {
            'light': 'Make minimal improvements while preserving the original intent. Focus only on clarity.',
            'moderate': 'Create a well-structured, professional prompt with balanced improvements.',
            'aggressive': 'Completely transform into an expert-level prompt with comprehensive detail.',
            'expert': 'Transform into a masterclass prompt with maximum clarity, depth, and sophistication.'
        }
        
        strategy_desc = {
            'clarity': 'eliminate ambiguity and filler',
            'structure': 'add clear organization',
            'context': 'enrich with context',
            'specificity': 'make highly specific',
            'role_definition': 'define expert role',
            'examples': 'include examples'
        }
        
        focus = ', '.join([strategy_desc.get(s, s) for s in strategies])
        
        enhancement_prompt = f"""You are an elite prompt engineering expert. Transform this prompt into a highly effective version.

Original Prompt:
{prompt}

Enhancement Level: {level} - {level_instructions.get(level, level_instructions['moderate'])}
Focus Areas: {focus}

Requirements:
- Maintain core intent while dramatically improving effectiveness
- Apply advanced prompting techniques
- Ensure clarity, specificity, and actionability
- Add appropriate structure and formatting
- Include relevant context and constraints

Provide ONLY the enhanced prompt, no explanations."""
        
        system_prompt = "You are an expert prompt engineer. Provide only the optimized prompt without explanations."
        
        # Try Groq first
        if self.groq:
            try:
                completion = self.groq.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": enhancement_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=2048
                )
                
                result = completion.choices[0].message.content.strip()
                if result:
                    return result
            except Exception as e:
                error_str = str(e).lower()
                # Check if we should fallback
                if any(keyword in error_str for keyword in [
                    "resource exhausted", "quota", "rate limit", "429", 
                    "503", "500", "timeout", "unavailable", "error", "api key"
                ]):
                    print(f"⚠️  Groq API error, falling back to local LLM: {e}")
                else:
                    print(f"⚠️  Groq API error: {e}")
        
        # Fallback to local LLM
        if LOCAL_LLM_AVAILABLE:
            try:
                result, success = generate_with_ollama(
                    prompt=enhancement_prompt,
                    system_prompt=system_prompt,
                    temperature=0.7,
                    max_tokens=2048
                )
                if success and result:
                    print("✅ Using local LLM for enhancement")
                    return result.strip()
            except Exception as e:
                print(f"⚠️  Local LLM error: {e}")
        
        return None
    
    def rule_based_optimize(self, prompt: str, strategies: List[str]) -> str:
        """Apply rule-based optimizations"""
        optimized = prompt
        for strategy in strategies:
            if strategy in self.strategies:
                optimized = self.strategies[strategy](optimized)
        return optimized
    
    def optimize(self, prompt: str, strategies: List[str], level: str = 'moderate', use_ai: bool = True) -> Dict:
        """Complete optimization pipeline"""
        start_time = time.time()
        
        # Analyze original
        original_metrics = self.analyze_prompt(prompt)
        category, tags = self._detect_category_tags(prompt)
        
        # Rule-based
        rule_based = self.rule_based_optimize(prompt, strategies)
        rule_metrics = self.analyze_prompt(rule_based)
        
        # AI enhancement
        ai_enhanced = None
        if use_ai and self.groq:
            ai_enhanced = self.ai_enhance_groq(prompt, strategies, level)
        
        # Determine best version
        versions = [
            (prompt, original_metrics),
            (rule_based, rule_metrics)
        ]
        
        if ai_enhanced:
            ai_metrics = self.analyze_prompt(ai_enhanced)
            versions.append((ai_enhanced, ai_metrics))
        
        best_version, best_metrics = max(versions, key=lambda x: x[1].overall_score)
        
        improvements = self._calculate_improvements(original_metrics, best_metrics)
        processing_time = time.time() - start_time
        
        return {
            'success': True,
            'original': prompt,
            'rule_based': rule_based,
            'ai_enhanced': ai_enhanced or '',
            'best_version': best_version,
            'category': category,
            'tags': tags,
            'metrics': asdict(best_metrics),
            'original_metrics': asdict(original_metrics),
            'improvements': improvements,
            'processing_time': round(processing_time, 3),
            'ai_used': ai_enhanced is not None
        }
    
    def generate_variations(self, prompt: str, num_variations: int = 3) -> List[Dict]:
        """Generate A/B test variations"""
        variations = []
        
        strategy_sets = [
            ['clarity', 'structure', 'specificity'],
            ['context', 'examples', 'role_definition'],
            ['clarity', 'specificity', 'examples'],
            ['structure', 'context', 'role_definition']
        ]
        
        for i in range(min(num_variations, len(strategy_sets))):
            result = self.optimize(prompt, strategy_sets[i], 'moderate', use_ai=True)
            variations.append({
                'variation_id': f"var_{i+1}",
                'prompt': result['best_version'],
                'strategies': strategy_sets[i],
                'score': result['metrics']['overall_score'],
                'improvements': result['improvements']
            })
        
        return variations
    
    def _detect_category_tags(self, prompt: str) -> Tuple[str, List[str]]:
        """Detect category and tags"""
        prompt_lower = prompt.lower()
        
        categories = {
            'coding': ['code', 'program', 'function', 'python', 'javascript', 'api'],
            'writing': ['write', 'article', 'blog', 'content', 'essay'],
            'analysis': ['analyze', 'data', 'research', 'evaluate'],
            'business': ['business', 'strategy', 'marketing', 'sales'],
            'education': ['explain', 'teach', 'learn', 'tutorial']
        }
        
        scores = defaultdict(int)
        for cat, keywords in categories.items():
            for kw in keywords:
                if kw in prompt_lower:
                    scores[cat] += 1
        
        category = max(scores.items(), key=lambda x: x[1])[0] if scores else 'general'
        
        tags = []
        tag_patterns = {
            'detailed': r'\b(detailed|comprehensive|thorough)\b',
            'beginner': r'\b(beginner|simple|basic)\b',
            'advanced': r'\b(advanced|expert|complex)\b',
            'example': r'\b(example|sample|demo)\b'
        }
        
        for tag, pattern in tag_patterns.items():
            if re.search(pattern, prompt_lower):
                tags.append(tag)
        
        return category, tags
    
    def _calculate_improvements(self, original: PromptMetrics, optimized: PromptMetrics) -> List[str]:
        """Calculate improvements"""
        improvements = []
        
        if optimized.clarity_score > original.clarity_score + 10:
            improvements.append(f"Clarity improved by {optimized.clarity_score - original.clarity_score:.1f}%")
        
        if optimized.specificity_score > original.specificity_score + 10:
            improvements.append(f"Specificity improved by {optimized.specificity_score - original.specificity_score:.1f}%")
        
        if optimized.structure_score > original.structure_score + 10:
            improvements.append(f"Structure improved by {optimized.structure_score - original.structure_score:.1f}%")
        
        if optimized.context_richness > original.context_richness + 10:
            improvements.append(f"Context enriched by {optimized.context_richness - original.context_richness:.1f}%")
        
        overall_improvement = optimized.overall_score - original.overall_score
        if overall_improvement > 5:
            improvements.append(f"Overall quality improved by {overall_improvement:.1f}%")
        
        return improvements if improvements else ["Prompt optimized successfully"]


# Initialize
db = DatabaseManager()
optimizer = UltimatePromptOptimizer(db)


# ============================================================================
# API ROUTES
# ============================================================================

@inkwell_ai.route('/')
def index():
    return render_template('inkwell_ai.html')


@inkwell_ai.route('/favicon.ico')
def favicon():
    """Return 204 No Content for favicon requests"""
    from flask import Response
    return Response(status=204)


@inkwell_ai.route('/api/health')
@rate_limit_api(requests_per_minute=60, requests_per_hour=600)  # OWASP: Rate limit health checks
def health():
    """OWASP: Rate limited health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'groq_available': groq_client is not None,
        'local_llm_available': LOCAL_LLM_AVAILABLE and check_ollama_available() if LOCAL_LLM_AVAILABLE else False,
        'timestamp': datetime.now().isoformat()
    })


@inkwell_ai.route('/api/optimize', methods=['POST'])
@rate_limit_api(requests_per_minute=10, requests_per_hour=100)  # OWASP: Rate limit optimization endpoint
@validate_request({  # OWASP: Input validation with schema
    "prompt": {"type": "string", "required": True, "max_length": 397},  # OWASP: Max 397 chars as specified
    "strategies": {"type": "list", "required": False, "max_items": 10, 
                   "item_schema": {"type": "string", "max_length": 50}},
    "level": {"type": "string", "required": False, "max_length": 50,
              "allowed_values": ["light", "moderate", "aggressive"]},
    "use_ai": {"type": "bool", "required": False},
    "user_id": {"type": "string", "required": False, "max_length": 100}
})
def optimize_endpoint():
    """OWASP: Validated and rate-limited prompt optimization endpoint"""
    try:
        # Use validated data from request context (set by validate_request decorator)
        data = g.validated_data
        prompt = data.get('prompt', '').strip()
        
        # OWASP: Additional validation (already validated by decorator, but double-check)
        if not prompt or len(prompt) < 5:
            return jsonify({'error': 'Prompt too short (minimum 5 characters)'}), 400
        
        strategies = data.get('strategies', ['clarity', 'structure', 'specificity'])
        level = data.get('level', 'moderate')
        use_ai = data.get('use_ai', True)
        user_id = data.get('user_id', 'anonymous')
        
        result = optimizer.optimize(prompt, strategies, level, use_ai)
        
        # Save to database
        db.save_optimization(result, user_id)
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@inkwell_ai.route('/api/analyze', methods=['POST'])
@rate_limit_api(requests_per_minute=20, requests_per_hour=200)  # OWASP: Rate limit analysis endpoint
@validate_request({  # OWASP: Input validation with schema
    "prompt": {"type": "string", "required": True, "max_length": 397}  # OWASP: Max 397 chars as specified
})
def analyze_endpoint():
    """OWASP: Validated and rate-limited prompt analysis endpoint"""
    try:
        # Use validated data from request context
        data = g.validated_data
        prompt = data.get('prompt', '').strip()
        
        metrics = optimizer.analyze_prompt(prompt)
        category, tags = optimizer._detect_category_tags(prompt)
        
        # Recommendations
        recommendations = []
        if metrics.clarity_score < 70:
            recommendations.append("Consider removing filler words and ambiguous language")
        if metrics.specificity_score < 70:
            recommendations.append("Add more specific details, numbers, and concrete examples")
        if metrics.structure_score < 60:
            recommendations.append("Improve structure with bullet points, sections, or numbered lists")
        if metrics.context_richness < 60:
            recommendations.append("Provide more context about goals, constraints, and requirements")
        
        return jsonify({
            'success': True,
            'metrics': asdict(metrics),
            'category': category,
            'tags': tags,
            'recommendations': recommendations,
            'quality_level': 'excellent' if metrics.overall_score >= 80 else 'good' if metrics.overall_score >= 60 else 'needs_improvement'
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@inkwell_ai.route('/api/variations', methods=['POST'])
@rate_limit_api(requests_per_minute=10, requests_per_hour=100)  # OWASP: Rate limit variations endpoint
@validate_request({  # OWASP: Input validation with schema
    "prompt": {"type": "string", "required": True, "max_length": 397},  # OWASP: Max 397 chars as specified
    "num_variations": {"type": "int", "required": False, "min_value": 1, "max_value": 4},
    "user_id": {"type": "string", "required": False, "max_length": 100}
})
def variations_endpoint():
    """OWASP: Validated and rate-limited A/B test variations endpoint"""
    try:
        # Use validated data from request context
        data = g.validated_data
        prompt = data.get('prompt', '').strip()
        num_variations = min(data.get('num_variations', 3), 4)
        user_id = data.get('user_id', 'anonymous')
        
        variations = optimizer.generate_variations(prompt, num_variations)
        
        # Save A/B test
        test_id = hashlib.md5(f"{user_id}{time.time()}".encode()).hexdigest()[:16]
        db.save_ab_test(test_id, user_id, prompt, variations)
        
        return jsonify({
            'success': True,
            'test_id': test_id,
            'original': prompt,
            'variations': variations,
            'recommendation': f"Test these {len(variations)} variations to find the most effective prompt"
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@inkwell_ai.route('/api/insights/<user_id>')
@rate_limit_api(requests_per_minute=30, requests_per_hour=300)  # OWASP: Rate limit insights endpoint
def insights_endpoint(user_id):
    """OWASP: Rate limited and validated insights endpoint"""
    try:
        # OWASP: Validate user_id and days parameter
        user_id = InputValidator.validate_string(user_id, 'user_id', max_length=100, required=True)
        days = InputValidator.validate_integer(
            request.args.get('days', 30, type=int),
            'days',
            min_value=1,
            max_value=365,
            required=False
        )
        insights = db.get_user_insights(user_id, days)
        
        return jsonify({
            'success': True,
            'insights': insights
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@inkwell_ai.route('/api/templates', methods=['GET'])
@rate_limit_api(requests_per_minute=30, requests_per_hour=300)  # OWASP: Rate limit templates endpoint
def get_templates_endpoint():
    """OWASP: Rate limited and validated templates retrieval"""
    try:
        # OWASP: Validate category parameter
        category = None
        if request.args.get('category'):
            category = InputValidator.validate_string(
                request.args.get('category'),
                'category',
                max_length=100,
                required=False
            )
        templates = db.get_templates(category)
        
        return jsonify({
            'success': True,
            'templates': templates,
            'total': len(templates)
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@inkwell_ai.route('/api/templates', methods=['POST'])
@rate_limit_api(requests_per_minute=10, requests_per_hour=100)  # OWASP: Rate limit template creation
@validate_request({  # OWASP: Input validation with schema
    "name": {"type": "string", "required": True, "max_length": 397},  # OWASP: Max 397 chars as specified
    "category": {"type": "string", "required": False, "max_length": 100},
    "template_text": {"type": "string", "required": True, "max_length": 10000},  # Longer for templates
    "description": {"type": "string", "required": False, "max_length": 500}
})
def create_template_endpoint():
    """OWASP: Validated and rate-limited template creation endpoint"""
    try:
        # Use validated data from request context
        data = g.validated_data
        
        template_id = hashlib.md5(f"{data.get('name')}{time.time()}".encode()).hexdigest()[:16]
        
        with sqlite3.connect(db.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO prompt_templates (template_id, name, category, template_text, description)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                template_id,
                data.get('name'),
                data.get('category', 'General'),
                data.get('template_text'),
                data.get('description', '')
            ))
            conn.commit()
        
        return jsonify({
            'success': True,
            'template_id': template_id,
            'message': 'Template created successfully'
        }), 201
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@inkwell_ai.route('/api/batch', methods=['POST'])
@rate_limit_strict(requests_per_minute=5, requests_per_hour=30)  # OWASP: Strict rate limit for batch operations
@validate_request({  # OWASP: Input validation with schema
    "prompts": {
        "type": "list",
        "required": True,
        "max_items": 50,  # OWASP: Limit batch size
        "item_schema": {"type": "string", "max_length": 397}  # OWASP: Max 397 chars per prompt
    },
    "level": {"type": "string", "required": False, "max_length": 50,
              "allowed_values": ["light", "moderate", "aggressive"]},
    "use_ai": {"type": "bool", "required": False},
    "user_id": {"type": "string", "required": False, "max_length": 100}
})
def batch_endpoint():
    """OWASP: Validated and strictly rate-limited batch optimization endpoint"""
    try:
        # Use validated data from request context
        data = g.validated_data
        prompts = data.get('prompts', [])
        
        level = data.get('level', 'moderate')
        use_ai = data.get('use_ai', True)
        user_id = data.get('user_id', 'anonymous')
        
        job_id = hashlib.md5(f"{user_id}{time.time()}".encode()).hexdigest()[:16]
        
        # Create job
        db.create_batch_job(job_id, user_id, len(prompts))
        
        # Process in background thread
        def process_batch():
            results = []
            try:
                for i, prompt in enumerate(prompts):
                    try:
                        result = optimizer.optimize(prompt, ['clarity', 'structure', 'specificity'], level, use_ai)
                        results.append({
                            'index': i,
                            'success': True,
                            'original': prompt,
                            'optimized': result['best_version'],
                            'score': result['metrics']['overall_score']
                        })
                    except Exception as e:
                        results.append({
                            'index': i,
                            'success': False,
                            'error': str(e)
                        })
                    
                    # Update progress (with error handling)
                    try:
                        db.update_batch_job(job_id, i + 1, results)
                    except Exception as db_error:
                        print(f"⚠️  Error updating batch job progress: {db_error}")
                        # Continue processing even if DB update fails
                
                # Mark complete
                try:
                    db.update_batch_job(job_id, len(prompts), results, 'completed')
                except Exception as db_error:
                    print(f"⚠️  Error marking batch job as complete: {db_error}")
                    # Try to update status at least
                    try:
                        with sqlite3.connect(db.db_name) as conn:
                            cursor = conn.cursor()
                            cursor.execute('UPDATE batch_jobs SET status = ? WHERE job_id = ?', ('completed', job_id))
                            conn.commit()
                    except:
                        pass
            except Exception as e:
                print(f"⚠️  Critical error in batch processing: {e}")
                # Try to mark as failed
                try:
                    with sqlite3.connect(db.db_name) as conn:
                        cursor = conn.cursor()
                        cursor.execute('UPDATE batch_jobs SET status = ? WHERE job_id = ?', ('failed', job_id))
                        conn.commit()
                except:
                    pass
        
        thread = threading.Thread(target=process_batch)
        thread.start()
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': f'Processing {len(prompts)} prompts in background',
            'status_url': f'/inkwell_ai/api/batch/{job_id}'
        }), 202
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@inkwell_ai.route('/api/batch/<job_id>')
@rate_limit_api(requests_per_minute=60, requests_per_hour=600)  # OWASP: Rate limit status polling
def batch_status_endpoint(job_id):
    """OWASP: Rate limited and validated batch job status endpoint"""
    try:
        # OWASP: Validate job_id parameter
        job_id = InputValidator.validate_string(job_id, 'job_id', max_length=100, required=True)
        with sqlite3.connect(db.db_name) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Check if completed_items column exists
            cursor.execute("PRAGMA table_info(batch_jobs)")
            columns = [row[1] for row in cursor.fetchall()]
            has_completed_items = 'completed_items' in columns
            
            if has_completed_items:
                cursor.execute('SELECT * FROM batch_jobs WHERE job_id = ?', (job_id,))
            else:
                # Fallback query without completed_items
                cursor.execute('''
                    SELECT id, job_id, user_id, status, total_items, results, 
                           created_at, completed_at
                    FROM batch_jobs WHERE job_id = ?
                ''', (job_id,))
            
            job = cursor.fetchone()
            
            if not job:
                return jsonify({'success': False, 'error': 'Job not found'}), 404
            
            job_dict = dict(job)
            
            # Handle missing completed_items column
            if 'completed_items' not in job_dict:
                # Calculate from results if available
                if job_dict.get('results'):
                    try:
                        results = json.loads(job_dict['results'])
                        job_dict['completed_items'] = len(results) if isinstance(results, list) else 0
                    except:
                        job_dict['completed_items'] = 0
                else:
                    job_dict['completed_items'] = 0
            
            # Parse results JSON
            if job_dict.get('results'):
                try:
                    job_dict['results'] = json.loads(job_dict['results'])
                except:
                    job_dict['results'] = []
            else:
                job_dict['results'] = []
            
            return jsonify({
                'success': True,
                'job': job_dict
            }), 200
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# For standalone running (if needed)
if __name__ == '__main__':
    from flask import Flask
    app = Flask(__name__, template_folder='templates')
    CORS(app)
    app.register_blueprint(inkwell_ai, url_prefix='/inkwell_ai')
    
    print("\n" + "="*60)
    print("🚀 InkWell AI Ultimate - Starting...")
    print("="*60)
    print(f"✅ Database: Initialized")
    print(f"{'✅' if groq_client else '⚠️ '} Groq AI: {'Connected' if groq_client else 'Not configured (add GROQ_API_KEY to .env)'}")
    if LOCAL_LLM_AVAILABLE:
        local_available = check_ollama_available()
        print(f"{'✅' if local_available else '⚠️ '} Local LLM: {'Available' if local_available else 'Not running (start Ollama/llama.cpp server)'}")
    print("="*60)
    print("🌐 Server running on http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, port=5000, threaded=True)