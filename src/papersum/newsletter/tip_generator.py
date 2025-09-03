from typing import List, Dict, Optional, Set
import logging
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
import re
from collections import Counter

from ..intelligence.preference_learner import UserInterest
from ..parse.pdf_extractor import ExtractedPaper
from .coding_tips import CodingTip, TipCategory, DifficultyLevel, CodingTipsManager


@dataclass
class TipTemplate:
    """Template for generating coding tips"""
    title_template: str
    content_template: str
    category: TipCategory
    difficulty: DifficultyLevel
    code_template: Optional[str] = None
    required_keywords: List[str] = None
    tags_template: List[str] = None


class AutoTipGenerator:
    """Automatically generates coding tips based on trends and user interests"""
    
    def __init__(self, tips_manager: Optional[CodingTipsManager] = None):
        self.logger = logging.getLogger(__name__)
        self.tips_manager = tips_manager or CodingTipsManager()
        
        # Templates for auto-generating tips
        self.tip_templates = self._create_tip_templates()
        
        # Technology trending patterns
        self.tech_trends = {
            "transformers": {
                "keywords": ["transformer", "attention", "bert", "gpt", "t5"],
                "frameworks": ["huggingface", "transformers", "torch"],
                "applications": ["nlp", "language model", "text generation"]
            },
            "pytorch_optimization": {
                "keywords": ["pytorch", "optimization", "mps", "apple silicon"],
                "frameworks": ["torch", "pytorch", "accelerate"],
                "applications": ["training", "inference", "gpu"]
            },
            "vector_databases": {
                "keywords": ["embeddings", "vector", "similarity", "faiss"],
                "frameworks": ["faiss", "chromadb", "pinecone", "weaviate"],
                "applications": ["search", "retrieval", "rag"]
            },
            "async_python": {
                "keywords": ["async", "asyncio", "fastapi", "concurrent"],
                "frameworks": ["fastapi", "asyncio", "aiohttp"],
                "applications": ["web", "api", "concurrent"]
            }
        }
    
    def _create_tip_templates(self) -> List[TipTemplate]:
        """Create templates for generating tips"""
        return [
            TipTemplate(
                title_template="Optimize {library} for {use_case}",
                content_template="When working with {library} for {use_case}, consider using {optimization} to improve {metric}. This is especially important when dealing with {context}.",
                category=TipCategory.PERFORMANCE,
                difficulty=DifficultyLevel.INTERMEDIATE,
                code_template="""# {optimization} example for {library}
{code_snippet}

# This improves {metric} by {improvement}""",
                required_keywords=["optimization", "performance"],
                tags_template=["performance", "optimization", "{library}"]
            ),
            
            TipTemplate(
                title_template="Modern {framework} Patterns for {domain}",
                content_template="The {framework} ecosystem has evolved significantly. For {domain} applications, the recommended pattern is to use {pattern} which provides {benefits}.",
                category=TipCategory.TOOLS_AND_LIBRARIES,
                difficulty=DifficultyLevel.INTERMEDIATE,
                code_template="""# Modern {framework} pattern
{code_example}

# Benefits: {benefits}""",
                required_keywords=["pattern", "framework"],
                tags_template=["{framework}", "patterns", "best-practices"]
            ),
            
            TipTemplate(
                title_template="{model_type} Model Deployment Best Practices",
                content_template="When deploying {model_type} models in production, key considerations include {considerations}. Use {deployment_strategy} for optimal performance and reliability.",
                category=TipCategory.ML_MODELS,
                difficulty=DifficultyLevel.ADVANCED,
                code_template="""# Production {model_type} deployment
{deployment_code}

# Key benefits: {benefits}""",
                required_keywords=["deployment", "production", "model"],
                tags_template=["deployment", "production", "{model_type}"]
            ),
            
            TipTemplate(
                title_template="Data Processing with {tool} for {data_type}",
                content_template="For {data_type} processing, {tool} offers {advantages}. The key pattern is to {approach} which handles {challenges} effectively.",
                category=TipCategory.DATA_SCIENCE,
                difficulty=DifficultyLevel.INTERMEDIATE,
                code_template="""# Efficient {data_type} processing
{processing_code}""",
                required_keywords=["data", "processing"],
                tags_template=["data-processing", "{tool}", "{data_type}"]
            )
        ]
    
    def generate_tips_from_papers(
        self, 
        papers: List[ExtractedPaper], 
        user_interests: List[UserInterest],
        count: int = 3
    ) -> List[CodingTip]:
        """Generate coding tips based on research papers and user interests"""
        
        if not papers:
            return []
        
        # Extract technical topics from papers
        paper_topics = self._extract_technical_topics(papers)
        
        # Match with user interests
        relevant_topics = self._match_topics_to_interests(paper_topics, user_interests)
        
        # Generate tips
        generated_tips = []
        
        for topic_data in relevant_topics[:count]:
            tip = self._generate_tip_from_topic(topic_data)
            if tip:
                generated_tips.append(tip)
        
        self.logger.info(f"Generated {len(generated_tips)} tips from {len(papers)} papers")
        return generated_tips
    
    def _extract_technical_topics(self, papers: List[ExtractedPaper]) -> List[Dict]:
        """Extract technical topics and patterns from papers"""
        
        topics = []
        
        for paper in papers:
            # Combine title and abstract for analysis
            text = f"{paper.title} {paper.abstract}".lower()
            
            # Extract programming/technical keywords
            tech_keywords = self._find_technical_keywords(text)
            
            # Identify frameworks and libraries
            frameworks = self._identify_frameworks(text)
            
            # Extract methodology keywords
            methods = self._extract_methods(text)
            
            # Determine primary domain
            domain = self._classify_technical_domain(text, tech_keywords)
            
            if tech_keywords or frameworks:
                topics.append({
                    'paper_title': paper.title,
                    'keywords': tech_keywords,
                    'frameworks': frameworks,
                    'methods': methods,
                    'domain': domain,
                    'text_sample': text[:200]
                })
        
        return topics
    
    def _find_technical_keywords(self, text: str) -> List[str]:
        """Find programming and technical keywords"""
        
        tech_patterns = {
            # Programming languages
            'python': r'\bpython\b',
            'pytorch': r'\bpytorch\b|\btorch\b',
            'tensorflow': r'\btensorflow\b|\btf\b',
            'javascript': r'\bjavascript\b|\bjs\b|\bnode\.?js\b',
            'rust': r'\brust\b',
            'go': r'\bgo\b|\bgolang\b',
            
            # ML/AI techniques
            'transformer': r'\btransformer\b',
            'attention': r'\battention\b',
            'embedding': r'\bembedding\b|\bembeddings\b',
            'neural_network': r'\bneural\s+network\b',
            'deep_learning': r'\bdeep\s+learning\b',
            'reinforcement_learning': r'\breinforcement\s+learning\b',
            
            # Development concepts
            'api': r'\bapi\b|\brest\b|\brestful\b',
            'database': r'\bdatabase\b|\bdb\b|\bsql\b',
            'containerization': r'\bdocker\b|\bcontainer\b',
            'microservices': r'\bmicroservice\b',
            'async': r'\basync\b|\basynchronous\b',
            'optimization': r'\boptimization\b|\boptimize\b'
        }
        
        found_keywords = []
        for keyword, pattern in tech_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                found_keywords.append(keyword)
        
        return found_keywords
    
    def _identify_frameworks(self, text: str) -> List[str]:
        """Identify frameworks and libraries mentioned"""
        
        frameworks = [
            'fastapi', 'flask', 'django', 'react', 'vue', 'angular',
            'pytorch', 'tensorflow', 'keras', 'scikit-learn', 'pandas',
            'numpy', 'matplotlib', 'seaborn', 'jupyter', 'docker',
            'kubernetes', 'redis', 'postgresql', 'mongodb', 'elasticsearch'
        ]
        
        found = []
        for framework in frameworks:
            if framework in text:
                found.append(framework)
        
        return found
    
    def _extract_methods(self, text: str) -> List[str]:
        """Extract methodology and approach keywords"""
        
        method_keywords = [
            'supervised learning', 'unsupervised learning', 'reinforcement learning',
            'classification', 'regression', 'clustering', 'optimization',
            'fine-tuning', 'transfer learning', 'few-shot', 'zero-shot',
            'preprocessing', 'feature engineering', 'data augmentation',
            'cross-validation', 'hyperparameter tuning', 'ensemble methods'
        ]
        
        found_methods = []
        for method in method_keywords:
            if method in text:
                found_methods.append(method)
        
        return found_methods
    
    def _classify_technical_domain(self, text: str, keywords: List[str]) -> str:
        """Classify the technical domain of the content"""
        
        domain_indicators = {
            'machine_learning': ['learning', 'model', 'neural', 'training', 'prediction'],
            'web_development': ['api', 'web', 'server', 'client', 'frontend', 'backend'],
            'data_science': ['data', 'analysis', 'visualization', 'statistics', 'pandas'],
            'devops': ['deployment', 'container', 'docker', 'kubernetes', 'ci/cd'],
            'software_engineering': ['architecture', 'design', 'pattern', 'testing', 'debugging']
        }
        
        domain_scores = {}
        for domain, indicators in domain_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text or indicator in keywords)
            domain_scores[domain] = score
        
        return max(domain_scores, key=domain_scores.get) if domain_scores else 'general'
    
    def _match_topics_to_interests(
        self, 
        topics: List[Dict], 
        user_interests: List[UserInterest]
    ) -> List[Dict]:
        """Match extracted topics to user interests"""
        
        if not user_interests:
            return topics  # Return all if no specific interests
        
        matched_topics = []
        
        for topic in topics:
            relevance_score = 0.0
            
            # Calculate relevance to user interests
            for interest in user_interests:
                # Check keyword overlap
                topic_keywords = set(topic.get('keywords', []) + topic.get('frameworks', []))
                interest_keywords = set(kw.lower() for kw in interest.keywords)
                
                overlap = len(topic_keywords & interest_keywords)
                if overlap > 0:
                    # Weight by interest confidence
                    relevance_score += (overlap / len(interest_keywords)) * interest.confidence_score
            
            if relevance_score > 0.1 or not user_interests:  # Include if relevant or no interests
                topic['relevance_score'] = relevance_score
                matched_topics.append(topic)
        
        # Sort by relevance
        matched_topics.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        return matched_topics
    
    def _generate_tip_from_topic(self, topic_data: Dict) -> Optional[CodingTip]:
        """Generate a specific tip from topic data"""
        
        keywords = topic_data.get('keywords', [])
        frameworks = topic_data.get('frameworks', [])
        domain = topic_data.get('domain', 'general')
        
        if not keywords and not frameworks:
            return None
        
        # Select appropriate template based on content
        template = self._select_template(domain, keywords, frameworks)
        if not template:
            return None
        
        # Fill template with topic data
        try:
            tip = self._fill_template(template, topic_data)
            return tip
        except Exception as e:
            self.logger.warning(f"Failed to generate tip from topic: {e}")
            return None
    
    def _select_template(self, domain: str, keywords: List[str], frameworks: List[str]) -> Optional[TipTemplate]:
        """Select best template for the topic"""
        
        # Match templates to domain and content
        if 'optimization' in keywords or 'performance' in keywords:
            return [t for t in self.tip_templates if t.category == TipCategory.PERFORMANCE][0]
        elif frameworks:
            return [t for t in self.tip_templates if t.category == TipCategory.TOOLS_AND_LIBRARIES][0]
        elif domain == 'machine_learning':
            return [t for t in self.tip_templates if t.category == TipCategory.ML_MODELS][0]
        elif domain == 'data_science':
            return [t for t in self.tip_templates if t.category == TipCategory.DATA_SCIENCE][0]
        else:
            return random.choice(self.tip_templates)
    
    def _fill_template(self, template: TipTemplate, topic_data: Dict) -> CodingTip:
        """Fill template with actual data"""
        
        keywords = topic_data.get('keywords', [])
        frameworks = topic_data.get('frameworks', [])
        domain = topic_data.get('domain', 'general')
        
        # Select primary framework/library
        primary_tech = frameworks[0] if frameworks else (keywords[0] if keywords else 'Python')
        
        # Generate specific content
        replacements = {
            'library': primary_tech,
            'framework': primary_tech,
            'use_case': self._generate_use_case(domain, keywords),
            'optimization': self._generate_optimization_suggestion(primary_tech),
            'metric': self._select_metric(domain),
            'context': self._generate_context(keywords),
            'pattern': self._generate_pattern_name(primary_tech),
            'benefits': self._generate_benefits(primary_tech, domain),
            'model_type': self._generate_model_type(keywords),
            'deployment_strategy': self._generate_deployment_strategy(primary_tech),
            'considerations': self._generate_considerations(domain),
            'tool': primary_tech,
            'data_type': self._generate_data_type(keywords),
            'advantages': self._generate_advantages(primary_tech),
            'approach': self._generate_approach(domain),
            'challenges': self._generate_challenges(domain)
        }
        
        # Fill in title
        title = template.title_template.format(**replacements)
        
        # Fill in content
        content = template.content_template.format(**replacements)
        
        # Generate code example if template has one
        code_example = None
        if template.code_template:
            code_example = self._generate_code_example(template, replacements, topic_data)
        
        # Generate tags
        tags = []
        if template.tags_template:
            tags = [tag.format(**replacements) if '{' in tag else tag for tag in template.tags_template]
        
        # Add technology-specific tags
        tags.extend(keywords[:3])
        tags.extend(frameworks[:2])
        
        return CodingTip(
            title=title,
            content=content,
            category=template.category,
            difficulty=template.difficulty,
            code_example=code_example,
            tags=list(set(tags))  # Remove duplicates
        )
    
    def _generate_use_case(self, domain: str, keywords: List[str]) -> str:
        use_cases = {
            'machine_learning': ['model training', 'inference', 'feature extraction'],
            'web_development': ['API development', 'web applications', 'microservices'],
            'data_science': ['data analysis', 'visualization', 'statistical modeling'],
            'devops': ['deployment', 'container orchestration', 'monitoring']
        }
        
        domain_cases = use_cases.get(domain, ['general applications'])
        
        # Try to match keywords to specific use cases
        for keyword in keywords:
            if 'model' in keyword:
                return 'model deployment'
            elif 'api' in keyword:
                return 'API development'
            elif 'data' in keyword:
                return 'data processing'
        
        return random.choice(domain_cases)
    
    def _generate_optimization_suggestion(self, tech: str) -> str:
        optimizations = {
            'pytorch': 'MPS backend acceleration',
            'torch': 'model.half() for memory efficiency',
            'pandas': 'chunking and dtype optimization',
            'fastapi': 'async endpoints with connection pooling',
            'default': 'caching and parallel processing'
        }
        
        return optimizations.get(tech.lower(), optimizations['default'])
    
    def _select_metric(self, domain: str) -> str:
        metrics = {
            'machine_learning': 'training speed',
            'web_development': 'response time',
            'data_science': 'processing throughput',
            'performance': 'execution speed'
        }
        return metrics.get(domain, 'performance')
    
    def _generate_context(self, keywords: List[str]) -> str:
        if any('large' in k or 'big' in k for k in keywords):
            return 'large datasets or models'
        elif any('real' in k or 'production' in k for k in keywords):
            return 'production environments'
        else:
            return 'resource-constrained environments'
    
    def _generate_pattern_name(self, tech: str) -> str:
        patterns = {
            'fastapi': 'dependency injection with async context managers',
            'pytorch': 'model factory pattern with device abstraction',
            'pandas': 'pipeline pattern with method chaining',
            'default': 'builder pattern with fluent interface'
        }
        return patterns.get(tech.lower(), patterns['default'])
    
    def _generate_benefits(self, tech: str, domain: str) -> str:
        benefits = {
            ('pytorch', 'machine_learning'): 'faster training and lower memory usage',
            ('fastapi', 'web_development'): 'better concurrency and automatic API documentation',
            ('pandas', 'data_science'): 'improved memory efficiency and processing speed',
            'default': 'better performance and maintainability'
        }
        
        key = (tech.lower(), domain)
        return benefits.get(key, benefits['default'])
    
    def _generate_model_type(self, keywords: List[str]) -> str:
        if 'transformer' in keywords:
            return 'Transformer'
        elif any('neural' in k for k in keywords):
            return 'Neural Network'
        elif any('tree' in k or 'forest' in k for k in keywords):
            return 'Tree-based'
        else:
            return 'Machine Learning'
    
    def _generate_deployment_strategy(self, tech: str) -> str:
        strategies = {
            'pytorch': 'TorchServe with Docker containers',
            'fastapi': 'Uvicorn with Nginx reverse proxy',
            'default': 'containerized microservice architecture'
        }
        return strategies.get(tech.lower(), strategies['default'])
    
    def _generate_considerations(self, domain: str) -> str:
        considerations = {
            'machine_learning': 'model size, inference latency, and memory usage',
            'web_development': 'load balancing, security, and rate limiting',
            'data_science': 'data privacy, processing scale, and result reproducibility'
        }
        return considerations.get(domain, 'scalability, security, and maintainability')
    
    def _generate_data_type(self, keywords: List[str]) -> str:
        if any('text' in k or 'nlp' in k for k in keywords):
            return 'text data'
        elif any('image' in k or 'vision' in k for k in keywords):
            return 'image data'
        elif any('time' in k or 'series' in k for k in keywords):
            return 'time series data'
        else:
            return 'structured data'
    
    def _generate_advantages(self, tech: str) -> str:
        advantages = {
            'pytorch': 'dynamic computation graphs and excellent debugging support',
            'fastapi': 'automatic API documentation and built-in data validation',
            'pandas': 'powerful data manipulation and built-in statistical functions'
        }
        return advantages.get(tech.lower(), 'robust ecosystem and community support')
    
    def _generate_approach(self, domain: str) -> str:
        approaches = {
            'machine_learning': 'implement preprocessing pipelines',
            'web_development': 'use async/await patterns',
            'data_science': 'leverage vectorized operations'
        }
        return approaches.get(domain, 'follow established patterns')
    
    def _generate_challenges(self, domain: str) -> str:
        challenges = {
            'machine_learning': 'memory limitations and computational complexity',
            'web_development': 'concurrent requests and data consistency',
            'data_science': 'missing data and outliers'
        }
        return challenges.get(domain, 'scalability and performance bottlenecks')
    
    def _generate_code_example(self, template: TipTemplate, replacements: Dict, topic_data: Dict) -> str:
        """Generate code example for the tip"""
        
        keywords = topic_data.get('keywords', [])
        frameworks = topic_data.get('frameworks', [])
        primary_tech = frameworks[0] if frameworks else (keywords[0] if keywords else 'python')
        
        # Generate appropriate code based on technology
        if primary_tech.lower() == 'pytorch':
            return self._generate_pytorch_code()
        elif primary_tech.lower() == 'fastapi':
            return self._generate_fastapi_code()
        elif primary_tech.lower() == 'pandas':
            return self._generate_pandas_code()
        else:
            return self._generate_generic_code(primary_tech)
    
    def _generate_pytorch_code(self) -> str:
        return """import torch

# Check for optimal device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Load model with optimization
model = YourModel().to(device)
if device.type == 'mps':
    model = model.half()  # Use float16 for memory efficiency

# Optimized inference
with torch.no_grad():
    output = model(input_data.to(device))"""
    
    def _generate_fastapi_code(self) -> str:
        return """from fastapi import FastAPI, Depends
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await initialize_resources()
    yield
    # Shutdown
    await cleanup_resources()

app = FastAPI(lifespan=lifespan)

@app.get("/items/{item_id}")
async def read_item(item_id: int, db: Session = Depends(get_db)):
    return await get_item(db, item_id)"""
    
    def _generate_pandas_code(self) -> str:
        return """import pandas as pd

# Optimized data loading
df = pd.read_csv(
    'large_file.csv',
    chunksize=10000,
    dtype={'id': 'int32', 'value': 'float32'},
    usecols=['id', 'value', 'category']
)

# Efficient processing pipeline
result = (
    df.groupby('category')
    .agg({'value': ['mean', 'std']})
    .round(3)
)"""
    
    def _generate_generic_code(self, tech: str) -> str:
        return f"""# Example using {tech}

def optimized_function(data):
    \"\"\"Optimized implementation for {tech}\"\"\"
    # Add your optimization logic here
    result = process_data(data)
    return result

# Usage
result = optimized_function(input_data)"""
    
    def generate_trending_tips(self, trending_topics: List[str], count: int = 2) -> List[CodingTip]:
        """Generate tips based on trending topics"""
        
        generated_tips = []
        
        for topic in trending_topics[:count]:
            tip = self._create_trending_tip(topic)
            if tip:
                generated_tips.append(tip)
        
        return generated_tips
    
    def _create_trending_tip(self, topic: str) -> Optional[CodingTip]:
        """Create a tip for a specific trending topic"""
        
        topic_lower = topic.lower()
        
        # Predefined tips for common trending topics
        trending_tip_data = {
            'apple silicon optimization': {
                'title': 'Apple Silicon GPU Acceleration with PyTorch MPS',
                'content': 'Apple Silicon Macs offer significant GPU acceleration through the Metal Performance Shaders (MPS) backend. Use torch.backends.mps.is_available() to detect support and .half() for memory efficiency.',
                'category': TipCategory.PERFORMANCE,
                'difficulty': DifficultyLevel.ADVANCED,
                'code': '''import torch

# Check MPS availability
if torch.backends.mps.is_available():
    device = torch.device("mps")
    model = model.to(device).half()
    
    # Optimized inference
    with torch.no_grad():
        output = model(input.to(device))
        torch.mps.empty_cache()  # Free memory''',
                'tags': ['pytorch', 'apple-silicon', 'mps', 'optimization']
            },
            
            'local ai model': {
                'title': 'Running AI Models Locally with Hugging Face',
                'content': 'Deploy AI models locally for privacy and cost control. Use transformers library with proper caching and device optimization for efficient local inference.',
                'category': TipCategory.ML_MODELS,
                'difficulty': DifficultyLevel.INTERMEDIATE,
                'code': '''from transformers import AutoTokenizer, AutoModel

# Load model locally
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(
    model_name, 
    cache_dir="./models"
)
model = AutoModel.from_pretrained(
    model_name,
    cache_dir="./models",
    local_files_only=False  # Download if not cached
)''',
                'tags': ['transformers', 'local-ai', 'huggingface', 'privacy']
            },
            
            'type-safe python': {
                'title': 'Type-Safe Python with Pydantic V2',
                'content': 'Pydantic V2 offers significant performance improvements and better type safety. Use BaseModel for data validation and Field for advanced constraints.',
                'category': TipCategory.PYTHON_BEST_PRACTICES,
                'difficulty': DifficultyLevel.INTERMEDIATE,
                'code': '''from pydantic import BaseModel, Field, validator
from typing import List, Optional

class UserConfig(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(regex=r'^[^@]+@[^@]+\\.[^@]+$')
    preferences: List[str] = Field(default_factory=list)
    
    @validator('preferences')
    def validate_prefs(cls, v):
        return [pref.lower().strip() for pref in v]''',
                'tags': ['pydantic', 'type-safety', 'validation', 'python']
            }
        }
        
        # Find matching trending topic
        for key, tip_data in trending_tip_data.items():
            if key in topic_lower:
                return CodingTip(
                    title=tip_data['title'],
                    content=tip_data['content'],
                    category=tip_data['category'],
                    difficulty=tip_data['difficulty'],
                    code_example=tip_data['code'],
                    tags=tip_data['tags']
                )
        
        # Fallback: create generic tip for unknown trending topic
        return CodingTip(
            title=f"Exploring {topic}",
            content=f"{topic} is gaining popularity in the development community. Consider exploring this technology to stay current with industry trends.",
            category=TipCategory.TOOLS_AND_LIBRARIES,
            difficulty=DifficultyLevel.BEGINNER,
            tags=[topic.lower().replace(' ', '-')]
        )
    
    def generate_personalized_tips(
        self,
        user_interests: List[UserInterest],
        recent_papers: List[ExtractedPaper],
        trending_topics: List[str],
        total_tips: int = 3
    ) -> List[CodingTip]:
        """Generate a personalized set of coding tips"""
        
        all_tips = []
        
        # 1. Generate tips from papers (40% of tips)
        paper_tip_count = max(1, total_tips * 2 // 5)
        paper_tips = self.generate_tips_from_papers(recent_papers, user_interests, paper_tip_count)
        all_tips.extend(paper_tips)
        
        # 2. Generate tips from trending topics (30% of tips) 
        trending_tip_count = max(1, total_tips * 3 // 10)
        trending_tips = self.generate_trending_tips(trending_topics, trending_tip_count)
        all_tips.extend(trending_tips)
        
        # 3. Get curated tips from database/fallback (30% of tips)
        curated_count = total_tips - len(all_tips)
        if curated_count > 0:
            # Match difficulty to user experience level
            difficulty = self._estimate_user_difficulty(user_interests)
            
            # Get categories based on interests
            categories = self._map_interests_to_categories(user_interests)
            
            curated_tips = self.tips_manager.get_tips_for_user(
                difficulty_level=difficulty,
                categories=categories,
                count=curated_count
            )
            all_tips.extend(curated_tips)
        
        # Ensure we don't exceed requested count
        if len(all_tips) > total_tips:
            # Prioritize by type: paper-based > trending > curated
            all_tips = all_tips[:total_tips]
        
        self.logger.info(f"Generated {len(all_tips)} personalized tips")
        return all_tips
    
    def _estimate_user_difficulty(self, user_interests: List[UserInterest]) -> DifficultyLevel:
        """Estimate user's technical level from their interests"""
        
        if not user_interests:
            return DifficultyLevel.INTERMEDIATE
        
        # Count advanced keywords across interests
        advanced_keywords = {
            'optimization', 'architecture', 'scalability', 'distributed',
            'microservice', 'kubernetes', 'performance', 'algorithm',
            'neural network', 'transformer', 'reinforcement learning'
        }
        
        advanced_count = 0
        total_keywords = 0
        
        for interest in user_interests:
            total_keywords += len(interest.keywords)
            for keyword in interest.keywords:
                if any(adv in keyword.lower() for adv in advanced_keywords):
                    advanced_count += 1
        
        if total_keywords == 0:
            return DifficultyLevel.INTERMEDIATE
        
        advanced_ratio = advanced_count / total_keywords
        
        if advanced_ratio > 0.3:
            return DifficultyLevel.ADVANCED
        elif advanced_ratio > 0.1:
            return DifficultyLevel.INTERMEDIATE
        else:
            return DifficultyLevel.BEGINNER
    
    def _map_interests_to_categories(self, user_interests: List[UserInterest]) -> List[TipCategory]:
        """Map user interests to tip categories"""
        
        categories = []
        
        for interest in user_interests:
            interest_text = f"{interest.topic} {' '.join(interest.keywords)}".lower()
            
            if any(kw in interest_text for kw in ['machine learning', 'model', 'neural', 'ai']):
                categories.append(TipCategory.ML_MODELS)
            if any(kw in interest_text for kw in ['data', 'analysis', 'science']):
                categories.append(TipCategory.DATA_SCIENCE)
            if any(kw in interest_text for kw in ['api', 'web', 'server']):
                categories.append(TipCategory.SOFTWARE_ARCHITECTURE)
            if any(kw in interest_text for kw in ['performance', 'optimization', 'speed']):
                categories.append(TipCategory.PERFORMANCE)
            if any(kw in interest_text for kw in ['python', 'programming', 'code']):
                categories.append(TipCategory.PYTHON_BEST_PRACTICES)
        
        # Remove duplicates and ensure we have at least one category
        unique_categories = list(set(categories))
        
        if not unique_categories:
            unique_categories = [TipCategory.PYTHON_BEST_PRACTICES, TipCategory.TOOLS_AND_LIBRARIES]
        
        return unique_categories