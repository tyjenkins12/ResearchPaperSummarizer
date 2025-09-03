from typing import List, Dict, Optional, Tuple, Set
import logging
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from dataclasses import dataclass
from urllib.parse import urlencode
import time
import re
from pathlib import Path

from ..intelligence.preference_learner import UserInterest
from ..parse.pdf_extractor import ExtractedPaper, PaperSection
from ..config.settings import settings


@dataclass
class ArxivPaper:
    """Structure for arXiv paper metadata"""
    id: str
    title: str
    authors: List[str]
    abstract: str
    categories: List[str]
    published: datetime
    updated: datetime
    pdf_url: str
    entry_id: str
    relevance_score: float = 0.0


class ArxivAPI:
    """Interface to arXiv API for paper discovery"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.base_url = "http://export.arxiv.org/api/query"
        self.rate_limit_delay = 3  # seconds between requests (arXiv policy)
        self.max_results_per_query = 1000
        
        # Common ML/CS categories for broader searches
        self.default_categories = [
            "cs.AI",    # Artificial Intelligence
            "cs.CL",    # Computation and Language
            "cs.CV",    # Computer Vision
            "cs.LG",    # Machine Learning
            "cs.NE",    # Neural and Evolutionary Computing
            "stat.ML",  # Machine Learning (Statistics)
            "cs.CR",    # Cryptography and Security
            "cs.DB",    # Databases
            "cs.SE",    # Software Engineering
        ]
    
    def search_papers(
        self,
        query: str = "",
        categories: Optional[List[str]] = None,
        max_results: int = 100,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[ArxivPaper]:
        """Search arXiv for papers matching criteria"""
        
        # Build search query
        search_terms = []
        
        if query:
            # Search in title and abstract
            search_terms.append(f'(ti:"{query}" OR abs:"{query}")')
        
        if categories:
            cat_queries = [f'cat:{cat}' for cat in categories]
            if len(cat_queries) == 1:
                search_terms.append(cat_queries[0])
            else:
                search_terms.append(f'({" OR ".join(cat_queries)})')
        
        # Date filtering
        if start_date:
            date_str = start_date.strftime('%Y%m%d%H%M%S')
            search_terms.append(f'submittedDate:[{date_str}0000 TO *]')
        
        if end_date:
            date_str = end_date.strftime('%Y%m%d%H%M%S')
            search_terms.append(f'submittedDate:[* TO {date_str}0000]')
        
        # Combine search terms
        if search_terms:
            full_query = " AND ".join(search_terms)
        else:
            # Default to recent papers in ML categories
            cat_list = " OR ".join([f"cat:{cat}" for cat in self.default_categories])
            full_query = f"({cat_list})"
        
        return self._execute_search(full_query, max_results)
    
    def _execute_search(self, query: str, max_results: int) -> List[ArxivPaper]:
        """Execute search against arXiv API"""
        
        papers = []
        start = 0
        batch_size = min(max_results, 100)  # arXiv recommends max 100 per request
        
        while len(papers) < max_results and start < self.max_results_per_query:
            params = {
                'search_query': query,
                'start': start,
                'max_results': batch_size,
                'sortBy': 'submittedDate',
                'sortOrder': 'descending'
            }
            
            url = f"{self.base_url}?{urlencode(params)}"
            
            try:
                self.logger.info(f"Fetching papers {start}-{start+batch_size} from arXiv...")
                self.logger.debug(f"Query URL: {url}")
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                batch_papers = self._parse_arxiv_response(response.text)
                
                if not batch_papers:
                    # Log the first bit of response for debugging
                    self.logger.debug(f"No papers parsed from response: {response.text[:200]}...")
                    break  # No more results
                
                papers.extend(batch_papers)
                start += batch_size
                
                # Rate limiting
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                self.logger.error(f"Error fetching arXiv papers: {e}")
                break
        
        self.logger.info(f"Found {len(papers)} papers from arXiv")
        return papers[:max_results]
    
    def _parse_arxiv_response(self, xml_content: str) -> List[ArxivPaper]:
        """Parse arXiv API XML response"""
        
        try:
            root = ET.fromstring(xml_content)
            
            # arXiv uses Atom namespace
            ns = {'atom': 'http://www.w3.org/2005/Atom',
                  'arxiv': 'http://arxiv.org/schemas/atom'}
            
            papers = []
            
            for entry in root.findall('atom:entry', ns):
                try:
                    paper = self._parse_entry(entry, ns)
                    if paper:
                        papers.append(paper)
                except Exception as e:
                    self.logger.warning(f"Failed to parse arXiv entry: {e}")
                    continue
            
            return papers
            
        except ET.ParseError as e:
            self.logger.error(f"Failed to parse arXiv XML response: {e}")
            return []
    
    def _parse_entry(self, entry, namespaces) -> Optional[ArxivPaper]:
        """Parse individual arXiv entry"""
        
        try:
            # Extract basic fields
            title = entry.find('atom:title', namespaces)
            title = title.text.strip().replace('\n', ' ') if title is not None else ""
            
            abstract = entry.find('atom:summary', namespaces)
            abstract = abstract.text.strip().replace('\n', ' ') if abstract is not None else ""
            
            entry_id = entry.find('atom:id', namespaces)
            entry_id = entry_id.text if entry_id is not None else ""
            
            # Extract arXiv ID from entry ID
            arxiv_id = entry_id.split('/')[-1] if entry_id else ""
            
            # Extract authors
            authors = []
            for author in entry.findall('atom:author', namespaces):
                name = author.find('atom:name', namespaces)
                if name is not None:
                    authors.append(name.text.strip())
            
            # Extract categories
            categories = []
            for category in entry.findall('atom:category', namespaces):
                term = category.get('term', '')
                if term:
                    categories.append(term)
            
            # Extract dates
            published = entry.find('atom:published', namespaces)
            published = self._parse_date(published.text) if published is not None else datetime.utcnow()
            
            updated = entry.find('atom:updated', namespaces)
            updated = self._parse_date(updated.text) if updated is not None else published
            
            # Build PDF URL
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            
            return ArxivPaper(
                id=arxiv_id,
                title=title,
                authors=authors,
                abstract=abstract,
                categories=categories,
                published=published,
                updated=updated,
                pdf_url=pdf_url,
                entry_id=entry_id
            )
            
        except Exception as e:
            self.logger.warning(f"Error parsing arXiv entry: {e}")
            return None
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse arXiv date format"""
        try:
            # arXiv uses format: 2024-01-15T14:30:45Z
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except:
            return datetime.utcnow()
    
    def get_recent_papers(self, days_back: int = 7, max_results: int = 200) -> List[ArxivPaper]:
        """Get papers published in the last N days"""
        
        # For now, get recent papers without strict date filtering
        # since arXiv's date handling can be complex with timezones
        return self.search_papers(
            categories=self.default_categories,
            max_results=max_results
        )
    
    def search_by_keywords(
        self,
        keywords: List[str],
        max_results: int = 50,
        days_back: int = 30
    ) -> List[ArxivPaper]:
        """Search for papers by keywords in title/abstract"""
        
        if not keywords:
            return []
        
        # Build keyword query (search in title OR abstract)
        keyword_query = " OR ".join([f'"{keyword}"' for keyword in keywords[:5]])
        
        # Remove strict date filtering for more reliable results
        return self.search_papers(
            query=keyword_query,
            categories=self.default_categories,
            max_results=max_results
        )


class PaperMonitor:
    """Monitors arXiv for papers matching user interests"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.arxiv_api = ArxivAPI()
        self.seen_papers: Set[str] = set()
        
        # Load previously seen papers to avoid duplicates
        self._load_seen_papers()
    
    def _load_seen_papers(self):
        """Load previously processed paper IDs"""
        cache_file = settings.models.cache_dir / "seen_papers.txt"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    self.seen_papers = set(line.strip() for line in f)
                self.logger.info(f"Loaded {len(self.seen_papers)} previously seen papers")
            except Exception as e:
                self.logger.warning(f"Could not load seen papers cache: {e}")
    
    def _save_seen_papers(self):
        """Save seen paper IDs to cache"""
        try:
            cache_file = settings.models.cache_dir / "seen_papers.txt"
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(cache_file, 'w') as f:
                for paper_id in sorted(self.seen_papers):
                    f.write(f"{paper_id}\n")
        except Exception as e:
            self.logger.warning(f"Could not save seen papers cache: {e}")
    
    def discover_papers_for_interests(
        self,
        user_interests: List[UserInterest],
        days_back: int = 7,
        max_papers_per_interest: int = 10
    ) -> List[ArxivPaper]:
        """Discover new papers matching user interests"""
        
        if not user_interests:
            self.logger.info("No user interests provided, getting recent papers from popular categories")
            recent_papers = self.arxiv_api.get_recent_papers(days_back=days_back, max_results=50)
            return [paper for paper in recent_papers if paper.id not in self.seen_papers]
        
        all_discovered_papers = []
        
        for interest in user_interests:
            self.logger.info(f"Searching for papers matching interest: {interest.topic}")
            
            # Search using interest keywords
            papers = self.arxiv_api.search_by_keywords(
                keywords=interest.keywords[:3],  # Use top 3 keywords
                max_results=max_papers_per_interest,
                days_back=days_back
            )
            
            # Filter out already seen papers
            new_papers = [paper for paper in papers if paper.id not in self.seen_papers]
            
            # Set initial relevance score based on interest confidence
            for paper in new_papers:
                paper.relevance_score = interest.confidence_score
                self.seen_papers.add(paper.id)
            
            all_discovered_papers.extend(new_papers)
            
            # Rate limiting between searches
            time.sleep(2)
        
        # Remove duplicates and sort by relevance
        unique_papers = {}
        for paper in all_discovered_papers:
            if paper.id not in unique_papers:
                unique_papers[paper.id] = paper
            else:
                # Keep paper with higher relevance score
                if paper.relevance_score > unique_papers[paper.id].relevance_score:
                    unique_papers[paper.id] = paper
        
        discovered = list(unique_papers.values())
        discovered.sort(key=lambda p: p.relevance_score, reverse=True)
        
        # Save updated seen papers cache
        self._save_seen_papers()
        
        self.logger.info(f"Discovered {len(discovered)} new papers matching user interests")
        return discovered
    
    def get_trending_papers(
        self,
        days_back: int = 3,
        min_citations_boost: int = 5
    ) -> List[ArxivPaper]:
        """Get trending papers (high activity, discussions, etc.)"""
        
        # Get very recent papers
        recent_papers = self.arxiv_api.get_recent_papers(
            days_back=days_back,
            max_results=100
        )
        
        # Filter for papers with indicators of high interest
        trending = []
        for paper in recent_papers:
            if paper.id in self.seen_papers:
                continue
            
            # Simple trending indicators
            title_lower = paper.title.lower()
            abstract_lower = paper.abstract.lower()
            
            trend_score = 0
            
            # Boost for trending keywords
            trending_keywords = [
                'gpt', 'llm', 'large language model', 'transformer', 'attention',
                'multimodal', 'vision-language', 'diffusion', 'generative',
                'few-shot', 'zero-shot', 'in-context', 'reasoning',
                'alignment', 'reinforcement learning', 'neural networks'
            ]
            
            for keyword in trending_keywords:
                if keyword in title_lower:
                    trend_score += 2
                elif keyword in abstract_lower:
                    trend_score += 1
            
            # Boost for multiple authors (collaborative work often more impactful)
            if len(paper.authors) >= 3:
                trend_score += 1
            
            # Boost for cross-category papers
            if len(paper.categories) >= 2:
                trend_score += 1
            
            if trend_score >= 2:
                paper.relevance_score = min(1.0, trend_score / 10)
                trending.append(paper)
                self.seen_papers.add(paper.id)
        
        trending.sort(key=lambda p: p.relevance_score, reverse=True)
        self.logger.info(f"Found {len(trending)} trending papers")
        
        return trending[:20]  # Top 20 trending
    
    def monitor_specific_authors(
        self,
        author_names: List[str],
        days_back: int = 30
    ) -> List[ArxivPaper]:
        """Monitor specific authors for new papers"""
        
        all_papers = []
        
        for author in author_names:
            # Clean author name for search
            clean_name = re.sub(r'[^\w\s]', '', author).strip()
            
            try:
                papers = self.arxiv_api.search_papers(
                    query=f'au:"{clean_name}"',
                    max_results=20,
                    start_date=datetime.utcnow() - timedelta(days=days_back)
                )
                
                new_papers = [paper for paper in papers if paper.id not in self.seen_papers]
                
                # High relevance for specific author follows
                for paper in new_papers:
                    paper.relevance_score = 0.9
                    self.seen_papers.add(paper.id)
                
                all_papers.extend(new_papers)
                time.sleep(2)  # Rate limiting
                
            except Exception as e:
                self.logger.warning(f"Failed to search for author {author}: {e}")
        
        self.logger.info(f"Found {len(all_papers)} new papers from monitored authors")
        return all_papers


class PaperRelevanceScorer:
    """Scores paper relevance for newsletter inclusion"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def score_papers(
        self,
        papers: List[ArxivPaper],
        user_interests: List[UserInterest]
    ) -> List[ArxivPaper]:
        """Score papers based on user interests and industry importance"""
        
        if not user_interests:
            # Without user interests, use basic industry importance
            return self._score_by_industry_importance(papers)
        
        scored_papers = []
        
        for paper in papers:
            # Calculate personal relevance
            personal_score = self._calculate_personal_relevance(paper, user_interests)
            
            # Calculate industry importance
            industry_score = self._calculate_industry_importance(paper)
            
            # Combined score (weighted: 70% personal, 30% industry)
            final_score = (0.7 * personal_score) + (0.3 * industry_score)
            
            paper.relevance_score = final_score
            scored_papers.append(paper)
        
        # Sort by relevance score
        scored_papers.sort(key=lambda p: p.relevance_score, reverse=True)
        
        return scored_papers
    
    def _calculate_personal_relevance(
        self,
        paper: ArxivPaper,
        user_interests: List[UserInterest]
    ) -> float:
        """Calculate how relevant paper is to user's personal interests"""
        
        paper_text = f"{paper.title} {paper.abstract}".lower()
        max_relevance = 0.0
        
        for interest in user_interests:
            # Count keyword matches
            keyword_matches = sum(
                1 for keyword in interest.keywords
                if keyword.lower() in paper_text
            )
            
            # Calculate match ratio
            if interest.keywords:
                match_ratio = keyword_matches / len(interest.keywords)
                
                # Weight by interest confidence and recency
                recency_weight = 1.0
                if interest.last_seen:
                    days_ago = (datetime.utcnow() - interest.last_seen).days
                    recency_weight = max(0.1, 1.0 - (days_ago / 365))
                
                relevance = match_ratio * interest.confidence_score * recency_weight
                max_relevance = max(max_relevance, relevance)
        
        return min(1.0, max_relevance)
    
    def _calculate_industry_importance(self, paper: ArxivPaper) -> float:
        """Calculate industry importance based on various signals"""
        
        title_lower = paper.title.lower()
        abstract_lower = paper.abstract.lower()
        
        importance_score = 0.0
        
        # High-impact keywords
        impact_keywords = {
            'breakthrough': 0.3,
            'state-of-the-art': 0.2,
            'sota': 0.2,
            'novel': 0.15,
            'efficient': 0.1,
            'scalable': 0.1,
            'real-time': 0.1,
            'production': 0.15,
            'deployment': 0.1
        }
        
        for keyword, weight in impact_keywords.items():
            if keyword in title_lower:
                importance_score += weight
            elif keyword in abstract_lower:
                importance_score += weight * 0.5
        
        # High-impact research areas
        impact_areas = {
            'large language model': 0.2,
            'multimodal': 0.15,
            'reinforcement learning': 0.1,
            'computer vision': 0.1,
            'natural language processing': 0.1,
            'machine learning': 0.05,
            'artificial intelligence': 0.05
        }
        
        for area, weight in impact_areas.items():
            if area in title_lower:
                importance_score += weight
            elif area in abstract_lower:
                importance_score += weight * 0.5
        
        # Boost for multiple categories (interdisciplinary work)
        if len(paper.categories) >= 2:
            importance_score += 0.1
        
        # Boost for multiple authors (collaborative research)
        if len(paper.authors) >= 3:
            importance_score += 0.05
        
        return min(1.0, importance_score)
    
    def _score_by_industry_importance(self, papers: List[ArxivPaper]) -> List[ArxivPaper]:
        """Fallback scoring when no user interests available"""
        
        for paper in papers:
            paper.relevance_score = self._calculate_industry_importance(paper)
        
        papers.sort(key=lambda p: p.relevance_score, reverse=True)
        return papers


class ArxivToPaperConverter:
    """Converts arXiv papers to ExtractedPaper format"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def convert_arxiv_paper(self, arxiv_paper: ArxivPaper) -> ExtractedPaper:
        """Convert ArxivPaper to ExtractedPaper format"""
        
        # Create mock sections from abstract
        sections = []
        if arxiv_paper.abstract:
            sections.append(PaperSection(
                title="Abstract",
                content=arxiv_paper.abstract,
                section_type="abstract"
            ))
        
        return ExtractedPaper(
            title=arxiv_paper.title,
            authors=arxiv_paper.authors,
            abstract=arxiv_paper.abstract,
            sections=sections,
            full_text=f"{arxiv_paper.title}\n\n{arxiv_paper.abstract}",
            metadata={
                'arxiv_id': arxiv_paper.id,
                'categories': arxiv_paper.categories,
                'published_date': arxiv_paper.published.isoformat(),
                'pdf_url': arxiv_paper.pdf_url,
                'relevance_score': arxiv_paper.relevance_score
            }
        )


class ArxivMonitoringService:
    """High-level service for monitoring arXiv papers"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.paper_monitor = PaperMonitor()
        self.relevance_scorer = PaperRelevanceScorer()
        self.converter = ArxivToPaperConverter()
    
    def discover_relevant_papers(
        self,
        user_interests: List[UserInterest],
        max_papers: int = 20,
        days_back: int = 7,
        include_trending: bool = True
    ) -> List[ExtractedPaper]:
        """Main method to discover relevant papers for newsletter"""
        
        self.logger.info(f"Discovering papers for {len(user_interests)} user interests")
        
        all_arxiv_papers = []
        
        # 1. Get papers matching user interests
        if user_interests:
            interest_papers = self.paper_monitor.discover_papers_for_interests(
                user_interests=user_interests,
                days_back=days_back,
                max_papers_per_interest=max_papers // len(user_interests) + 2
            )
            all_arxiv_papers.extend(interest_papers)
        
        # 2. Get trending papers if enabled
        if include_trending:
            trending_papers = self.paper_monitor.get_trending_papers(days_back=days_back)
            all_arxiv_papers.extend(trending_papers)
        
        # 3. Score all papers for relevance
        if user_interests:
            scored_papers = self.relevance_scorer.score_papers(all_arxiv_papers, user_interests)
        else:
            scored_papers = self.relevance_scorer._score_by_industry_importance(all_arxiv_papers)
        
        # 4. Select top papers and convert to ExtractedPaper format
        top_papers = scored_papers[:max_papers]
        extracted_papers = []
        
        for arxiv_paper in top_papers:
            try:
                extracted_paper = self.converter.convert_arxiv_paper(arxiv_paper)
                extracted_papers.append(extracted_paper)
            except Exception as e:
                self.logger.warning(f"Failed to convert paper {arxiv_paper.id}: {e}")
        
        self.logger.info(f"Successfully discovered {len(extracted_papers)} relevant papers")
        return extracted_papers
    
    def get_weekly_paper_batch(
        self,
        user_interests: List[UserInterest],
        target_count: int = 5
    ) -> List[ExtractedPaper]:
        """Get a weekly batch of papers for newsletter"""
        
        return self.discover_relevant_papers(
            user_interests=user_interests,
            max_papers=target_count * 2,  # Get more, then select best
            days_back=7,
            include_trending=True
        )[:target_count]
    
    def monitor_authors_and_keywords(
        self,
        author_names: List[str],
        monitor_keywords: List[str],
        days_back: int = 14
    ) -> List[ExtractedPaper]:
        """Monitor specific authors and keywords"""
        
        all_papers = []
        
        # Monitor authors
        if author_names:
            author_papers = self.paper_monitor.monitor_specific_authors(
                author_names=author_names,
                days_back=days_back
            )
            all_papers.extend(author_papers)
        
        # Monitor keywords
        if monitor_keywords:
            keyword_papers = self.paper_monitor.arxiv_api.search_by_keywords(
                keywords=monitor_keywords,
                max_results=30,
                days_back=days_back
            )
            new_keyword_papers = [
                paper for paper in keyword_papers 
                if paper.id not in self.paper_monitor.seen_papers
            ]
            all_papers.extend(new_keyword_papers)
        
        # Convert to ExtractedPaper format
        extracted_papers = []
        for arxiv_paper in all_papers:
            try:
                extracted_paper = self.converter.convert_arxiv_paper(arxiv_paper)
                extracted_papers.append(extracted_paper)
                self.paper_monitor.seen_papers.add(arxiv_paper.id)
            except Exception as e:
                self.logger.warning(f"Failed to convert monitored paper {arxiv_paper.id}: {e}")
        
        return extracted_papers