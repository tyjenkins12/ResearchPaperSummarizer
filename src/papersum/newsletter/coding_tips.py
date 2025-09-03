from typing import List, Dict, Optional
import random
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from sqlalchemy.orm import Session

from ..database.models import CodingTip as DBCodingTip


class TipCategory(Enum):
    """Categories of coding tips"""
    PYTHON_BEST_PRACTICES = "python_best_practices"
    ML_MODELS = "ml_models"
    DATA_SCIENCE = "data_science"
    SOFTWARE_ARCHITECTURE = "software_architecture"
    PERFORMANCE = "performance"
    TOOLS_AND_LIBRARIES = "tools_and_libraries"
    DEBUGGING = "debugging"
    TESTING = "testing"


class DifficultyLevel(Enum):
    """Difficulty levels for tips"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


@dataclass
class CodingTip:
    """Structure for a coding tip"""
    title: str
    content: str
    category: TipCategory
    difficulty: DifficultyLevel
    code_example: Optional[str] = None
    links: List[str] = None
    tags: List[str] = None


class CodingTipsManager:
    """Manages coding tips with database integration"""
    
    def __init__(self, db_session: Optional[Session] = None):
        self.db_session = db_session
        self.fallback_tips = self._get_curated_tips()
    
    def _get_curated_tips(self) -> List[CodingTip]:
        """Curated fallback tips if database is not available"""
        return [
            CodingTip(
                title="Use Type Hints for Better Code Documentation",
                content="Type hints make your Python code more readable and help catch bugs early. Modern IDEs and tools like mypy can verify types automatically.",
                category=TipCategory.PYTHON_BEST_PRACTICES,
                difficulty=DifficultyLevel.INTERMEDIATE,
                code_example="""def process_papers(papers: List[str]) -> Dict[str, int]:
    \"\"\"Process papers and return word counts\"\"\"
    result = {}
    for paper in papers:
        result[paper] = len(paper.split())
    return result""",
                tags=["typing", "documentation", "python"]
            ),
            
            CodingTip(
                title="Sentence-BERT for Semantic Similarity",
                content="Sentence-BERT creates embeddings that capture semantic meaning, perfect for document similarity, clustering, and search. Much better than traditional TF-IDF for understanding context.",
                category=TipCategory.ML_MODELS,
                difficulty=DifficultyLevel.INTERMEDIATE,
                code_example="""from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
texts = ["AI research paper", "Machine learning study"]
embeddings = model.encode(texts)
similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]""",
                tags=["nlp", "embeddings", "similarity"],
                links=["https://www.sbert.net/"]
            ),
            
            CodingTip(
                title="Python Project Structure: src/ Layout", 
                content="Use the src/ layout for Python projects to avoid import issues and make packaging easier. This separates source code from tests and configuration.",
                category=TipCategory.SOFTWARE_ARCHITECTURE,
                difficulty=DifficultyLevel.BEGINNER,
                code_example="""project/
├── src/
│   └── mypackage/
│       ├── __init__.py
│       ├── module1.py
│       └── subpackage/
├── tests/
├── pyproject.toml
└── README.md""",
                tags=["project-structure", "packaging", "python"]
            ),
            
            CodingTip(
                title="Apple Silicon MPS Optimization",
                content="Use PyTorch's MPS backend for GPU acceleration on Apple Silicon. Check device availability and use appropriate data types for optimal performance.",
                category=TipCategory.PERFORMANCE,
                difficulty=DifficultyLevel.ADVANCED,
                code_example="""import torch

# Check for MPS availability
if torch.backends.mps.is_available():
    device = torch.device("mps")
    model = model.to(device)
    model.half()  # Use float16 for memory efficiency
else:
    device = torch.device("cpu")""",
                tags=["pytorch", "apple-silicon", "gpu", "optimization"]
            ),
            
            CodingTip(
                title="Efficient Data Loading with Pandas",
                content="Use chunking and data type optimization when loading large datasets. Specify dtypes explicitly to save memory and improve performance.",
                category=TipCategory.DATA_SCIENCE,
                difficulty=DifficultyLevel.INTERMEDIATE,
                code_example="""# Memory-efficient data loading
df = pd.read_csv(
    'large_file.csv',
    chunksize=10000,
    dtype={'id': 'int32', 'score': 'float32'},
    usecols=['id', 'score', 'text']  # Only load needed columns
)

for chunk in df:
    process_chunk(chunk)""",
                tags=["pandas", "memory", "performance"]
            ),
            
            CodingTip(
                title="Ruff: The Fast Python Linter",
                content="Ruff replaces multiple Python tools (flake8, isort, black) with a single, extremely fast Rust-based linter. Configure it in pyproject.toml for consistent code style.",
                category=TipCategory.TOOLS_AND_LIBRARIES,
                difficulty=DifficultyLevel.BEGINNER,
                code_example="""# pyproject.toml
[tool.ruff]
line-length = 88
select = ["E", "F", "I", "N", "W"]
ignore = ["E501"]

# Run with:
# ruff check .
# ruff format .""",
                tags=["linting", "code-quality", "tools"]
            )
        ]
    
    def add_tip_to_database(
        self,
        title: str,
        content: str,
        category: str,
        difficulty: str,
        code_example: Optional[str] = None,
        tags: Optional[List[str]] = None,
        links: Optional[List[str]] = None
    ) -> bool:
        """Add new tip to database (primary method for adding tips)"""
        if not self.db_session:
            return False
        
        try:
            db_tip = DBCodingTip(
                title=title,
                content=content,
                category=category,
                difficulty_level=difficulty,
                code_example=code_example,
                tags=tags or [],
                external_links=links or [],
                created_at=datetime.utcnow()
            )
            
            self.db_session.add(db_tip)
            self.db_session.commit()
            return True
            
        except Exception as e:
            self.db_session.rollback()
            raise e
    
    def bulk_import_tips(self, tips_file_path: str) -> int:
        """Import tips from JSON/YAML file (for bulk operations)"""
        import json
        
        with open(tips_file_path, 'r') as f:
            tips_data = json.load(f)
        
        count = 0
        for tip_data in tips_data:
            if self.add_tip_to_database(**tip_data):
                count += 1
        
        return count
    
    def get_tips_for_user(
        self,
        difficulty_level: DifficultyLevel,
        categories: Optional[List[TipCategory]] = None,
        count: int = 2,
        exclude_recent: bool = True
    ) -> List[CodingTip]:
        """Get coding tips appropriate for user's level and interests"""
        
        # Try database first
        if self.db_session:
            return self._get_tips_from_database(difficulty_level, categories, count, exclude_recent)
        
        # Fallback to curated tips
        return self._get_tips_from_fallback(difficulty_level, categories, count)
    
    def _get_tips_from_database(
        self,
        difficulty_level: DifficultyLevel,
        categories: Optional[List[TipCategory]],
        count: int,
        exclude_recent: bool
    ) -> List[CodingTip]:
        """Get tips from database with smart filtering"""
        query = self.db_session.query(DBCodingTip).filter(
            DBCodingTip.difficulty_level == difficulty_level.value,
            DBCodingTip.is_active == True
        )
        
        if categories:
            category_names = [cat.value for cat in categories]
            query = query.filter(DBCodingTip.category.in_(category_names))
        
        if exclude_recent:
            # Avoid tips used in last 4 weeks
            four_weeks_ago = datetime.utcnow().timestamp() - (4 * 7 * 24 * 3600)
            query = query.filter(
                (DBCodingTip.last_used == None) |
                (DBCodingTip.last_used < four_weeks_ago)
            )
        
        # Order by least recently used, then randomly
        tips = query.order_by(DBCodingTip.last_used.asc().nullsfirst()).limit(count * 2).all()
        
        if len(tips) <= count:
            selected = tips
        else:
            selected = random.sample(tips, count)
        
        # Update last_used timestamp
        for tip in selected:
            tip.last_used = datetime.utcnow()
            tip.times_included += 1
        
        self.db_session.commit()
        
        # Convert to dataclass format
        return [self._db_tip_to_dataclass(tip) for tip in selected]
    
    def _get_tips_from_fallback(
        self,
        difficulty_level: DifficultyLevel,
        categories: Optional[List[TipCategory]],
        count: int
    ) -> List[CodingTip]:
        """Get tips from fallback curated list"""
        filtered_tips = [
            tip for tip in self.fallback_tips
            if tip.difficulty == difficulty_level
        ]
        
        if categories:
            filtered_tips = [
                tip for tip in filtered_tips
                if tip.category in categories
            ]
        
        if len(filtered_tips) <= count:
            return filtered_tips
        
        return random.sample(filtered_tips, count)
    
    def _db_tip_to_dataclass(self, db_tip: DBCodingTip) -> CodingTip:
        """Convert database tip to dataclass"""
        return CodingTip(
            title=db_tip.title,
            content=db_tip.content,
            category=TipCategory(db_tip.category),
            difficulty=DifficultyLevel(db_tip.difficulty_level),
            code_example=db_tip.code_example,
            tags=db_tip.tags,
            links=db_tip.external_links
        )
    
    def seed_initial_tips(self) -> int:
        """Seed database with initial curated tips (run once)"""
        if not self.db_session:
            return 0
        
        count = 0
        for tip in self.fallback_tips:
            try:
                self.add_tip_to_database(
                    title=tip.title,
                    content=tip.content,
                    category=tip.category.value,
                    difficulty=tip.difficulty.value,
                    code_example=tip.code_example,
                    tags=tip.tags,
                    links=tip.links
                )
                count += 1
            except Exception:
                continue  # Skip duplicates
        
        return count
    
    def get_trending_topics(self) -> List[str]:
        """Get trending coding topics (could be enhanced with web scraping)"""
        return [
            "Apple Silicon optimization",
            "Local AI model deployment", 
            "Type-safe Python with Pydantic",
            "FastAPI async patterns",
            "Transformer model fine-tuning",
            "Vector databases and embeddings",
            "Docker multi-stage builds",
            "GitHub Actions CI/CD",
            "Python 3.13 new features",
            "Rust-based Python tools"
        ]


class TipContentGenerator:
    """Generates new coding tips automatically (future enhancement)"""
    
    def __init__(self):
        self.tip_manager = CodingTipsManager()
    
    def generate_from_trending_topics(self) -> List[CodingTip]:
        """Generate tips based on current trending topics"""
        # This could use web scraping + AI to generate new tips
        # For now, returns trending topic suggestions
        trending = self.tip_manager.get_trending_topics()
        
        suggestions = []
        for topic in trending[:3]:
            suggestions.append(CodingTip(
                title=f"Trending: {topic}",
                content=f"Learn more about {topic} - a trending topic in the development community.",
                category=TipCategory.TOOLS_AND_LIBRARIES,
                difficulty=DifficultyLevel.INTERMEDIATE,
                tags=[topic.lower().replace(" ", "-")]
            ))
        
        return suggestions