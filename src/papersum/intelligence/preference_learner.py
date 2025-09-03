from typing import List, Dict, Set, Tuple, Optional
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
from collections import Counter, defaultdict

from ..config.settings import settings
from ..parse.pdf_extractor import ExtractedPaper
from ..database.models import User, Paper, ResearchTopic


@dataclass
class UserInterest:
    """Represents a learned user interest"""
    topic: str
    keywords: List[str]
    embedding: np.ndarray
    confidence_score: float
    paper_count: int
    last_seen: datetime


@dataclass 
class PaperProfile:
    """Extracted profile from a research paper"""
    title_keywords: List[str]
    abstract_keywords: List[str]
    research_areas: List[str]
    technical_keywords: List[str]
    methodology: List[str]
    embedding: np.ndarray


class PreferenceLearner:
    """Learns user research preferences from uploaded papers"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Load sentence transformer for semantic understanding
        self.encoder = SentenceTransformer(
            settings.models.sentence_transformer_model,
            cache_folder=str(settings.models.cache_dir)
        )
        
        # Research domain keywords for classification
        self.research_domains = {
            'machine_learning': [
                'neural network', 'deep learning', 'supervised learning', 'unsupervised learning',
                'reinforcement learning', 'gradient descent', 'backpropagation', 'overfitting',
                'regularization', 'cross-validation', 'feature selection', 'ensemble methods'
            ],
            'natural_language_processing': [
                'nlp', 'language model', 'transformer', 'bert', 'gpt', 'attention mechanism',
                'tokenization', 'embedding', 'sentiment analysis', 'named entity recognition',
                'machine translation', 'question answering', 'text summarization'
            ],
            'computer_vision': [
                'cnn', 'convolutional', 'image classification', 'object detection', 'segmentation',
                'face recognition', 'opencv', 'image processing', 'feature extraction',
                'gan', 'generative adversarial', 'style transfer'
            ],
            'data_science': [
                'data analysis', 'statistics', 'visualization', 'pandas', 'numpy', 'matplotlib',
                'exploratory data analysis', 'correlation', 'regression', 'clustering',
                'dimensionality reduction', 'time series', 'anomaly detection'
            ],
            'software_engineering': [
                'architecture', 'design patterns', 'microservices', 'api', 'database',
                'testing', 'ci/cd', 'devops', 'containerization', 'scalability',
                'performance optimization', 'code quality'
            ],
            'ai_research': [
                'artificial intelligence', 'agi', 'reasoning', 'knowledge representation',
                'expert systems', 'logic programming', 'planning', 'search algorithms',
                'multi-agent systems', 'cognitive science'
            ]
        }
    
    def extract_paper_profile(self, paper: ExtractedPaper) -> PaperProfile:
        """Extract key characteristics from a paper"""
        
        # Extract keywords from different sections
        title_keywords = self._extract_keywords(paper.title)
        abstract_keywords = self._extract_keywords(paper.abstract) if paper.abstract else []
        
        # Classify research areas
        combined_text = f"{paper.title} {paper.abstract}".lower()
        research_areas = self._classify_research_areas(combined_text)
        
        # Extract technical terms and methodologies
        technical_keywords = self._extract_technical_terms(combined_text)
        methodology = self._extract_methodology_keywords(combined_text)
        
        # Create semantic embedding of the paper
        paper_text = f"{paper.title}. {paper.abstract}"
        if not paper_text.strip():
            paper_text = paper.full_text[:1000] if paper.full_text else "empty paper"
        
        embedding = self.encoder.encode([paper_text])[0]
        
        return PaperProfile(
            title_keywords=title_keywords,
            abstract_keywords=abstract_keywords,
            research_areas=research_areas,
            technical_keywords=technical_keywords,
            methodology=methodology,
            embedding=embedding
        )
    
    def _extract_keywords(self, text: str, min_length: int = 3) -> List[str]:
        """Extract meaningful keywords from text"""
        if not text:
            return []
        
        # Clean and tokenize
        text = re.sub(r'[^\w\s-]', ' ', text.lower())
        words = text.split()
        
        # Filter out common words and short words
        stopwords = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'among', 'this', 'that', 'these',
            'those', 'is', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'can', 'must', 'shall'
        }
        
        keywords = [
            word for word in words 
            if len(word) >= min_length and word not in stopwords
        ]
        
        # Count frequency and return most common
        keyword_counts = Counter(keywords)
        return [word for word, count in keyword_counts.most_common(20)]
    
    def _classify_research_areas(self, text: str) -> List[str]:
        """Classify paper into research domains"""
        areas = []
        
        for domain, keywords in self.research_domains.items():
            # Count how many domain keywords appear in text
            matches = sum(1 for keyword in keywords if keyword in text)
            
            # If significant overlap, include this domain
            if matches >= 2 or matches / len(keywords) > 0.1:
                areas.append(domain)
        
        return areas
    
    def _extract_technical_terms(self, text: str) -> List[str]:
        """Extract technical terms and model names"""
        technical_patterns = [
            r'\b[A-Z]{2,}(?:-[A-Z0-9]+)*\b',  # Acronyms like CNN, BERT, GPT-3
            r'\b[a-z]+-\d+\b',                # Model versions like bert-base, t5-small
            r'\b(?:algorithm|model|method|approach|technique|framework)\b',
            r'\b(?:neural|deep|machine|artificial|computer|data)\s+\w+',
        ]
        
        technical_terms = []
        for pattern in technical_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            technical_terms.extend(matches)
        
        return list(set(technical_terms))[:15]  # Limit to top 15
    
    def _extract_methodology_keywords(self, text: str) -> List[str]:
        """Extract methodology and approach keywords"""
        methodology_keywords = [
            'supervised', 'unsupervised', 'semi-supervised', 'reinforcement',
            'classification', 'regression', 'clustering', 'optimization',
            'training', 'validation', 'testing', 'evaluation', 'benchmark',
            'dataset', 'preprocessing', 'feature engineering', 'hyperparameter',
            'fine-tuning', 'transfer learning', 'few-shot', 'zero-shot'
        ]
        
        found_methods = []
        for keyword in methodology_keywords:
            if keyword in text:
                found_methods.append(keyword)
        
        return found_methods
    
    def learn_from_papers(self, papers: List[ExtractedPaper]) -> List[UserInterest]:
        """Learn user interests from a collection of papers"""
        if not papers:
            return []
        
        self.logger.info(f"Learning preferences from {len(papers)} papers...")
        
        # Extract profiles from all papers
        paper_profiles = [self.extract_paper_profile(paper) for paper in papers]
        
        # Collect all embeddings for clustering
        embeddings = np.array([profile.embedding for profile in paper_profiles])
        
        # Cluster papers to find interest areas
        num_clusters = min(len(papers) // 2 + 1, 8)  # Reasonable number of clusters
        if len(papers) >= 3:
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            clusters = kmeans.fit_predict(embeddings)
        else:
            clusters = [0] * len(papers)  # Single cluster for few papers
        
        # Generate interests for each cluster
        interests = []
        for cluster_id in set(clusters):
            cluster_papers = [
                paper_profiles[i] for i, c in enumerate(clusters) if c == cluster_id
            ]
            
            if cluster_papers:
                interest = self._create_interest_from_cluster(cluster_papers, cluster_id)
                interests.append(interest)  # Fixed: was 'interest.append(interest)'
        
        self.logger.info(f"Identified {len(interests)} interest areas")
        return interests
    
    def _create_interest_from_cluster(self, cluster_papers: List[PaperProfile], cluster_id: int) -> UserInterest:
        """Create a user interest from a cluster of similar papers"""
        
        # Aggregate keywords across cluster
        all_keywords = []
        all_research_areas = []
        all_technical_terms = []
        
        for paper in cluster_papers:
            all_keywords.extend(paper.title_keywords + paper.abstract_keywords)
            all_research_areas.extend(paper.research_areas)
            all_technical_terms.extend(paper.technical_keywords)
        
        # Find most common terms
        keyword_counts = Counter(all_keywords)
        top_keywords = [word for word, _ in keyword_counts.most_common(10)]
        
        research_area_counts = Counter(all_research_areas)
        primary_area = research_area_counts.most_common(1)[0][0] if research_area_counts else "general"
        
        # Create cluster centroid embedding
        cluster_embeddings = [paper.embedding for paper in cluster_papers]  # Fixed: was 'paper.emvedding'
        centroid_embedding = np.mean(cluster_embeddings, axis=0)
        
        # Generate topic name from most common keywords
        if top_keywords:
            topic_name = f"{primary_area}: {', '.join(top_keywords[:3])}"
        else:
            topic_name = f"Interest Area {cluster_id + 1}"
        
        # Calculate confidence based on cluster coherence
        if len(cluster_embeddings) > 1:
            # Calculate average pairwise similarity within cluster
            similarities = []
            for i, emb1 in enumerate(cluster_embeddings):
                for j, emb2 in enumerate(cluster_embeddings[i+1:], i+1):
                    sim = cosine_similarity([emb1], [emb2])[0][0]
                    similarities.append(sim)
            confidence_score = np.mean(similarities) if similarities else 0.5
        else:
            confidence_score = 0.8  # Single paper gets high confidence
        
        return UserInterest(
            topic=topic_name,
            keywords=top_keywords,
            embedding=centroid_embedding,
            confidence_score=confidence_score,
            paper_count=len(cluster_papers),
            last_seen=datetime.utcnow()
        )
    
    def update_interests_from_feedback(
        self, 
        interests: List[UserInterest], 
        paper_ratings: Dict[str, int],
        papers_db: Dict[str, ExtractedPaper]
    ) -> List[UserInterest]:
        """Update interest confidence scores based on user feedback on recommended papers"""
        
        if not paper_ratings or not papers_db:
            return interests
        
        # Analyze highly rated papers to boost matching interests
        for paper_id, rating in paper_ratings.items():
            if rating >= 4 and paper_id in papers_db:  # High rating (4-5 stars)
                highly_rated_paper = papers_db[paper_id]
                paper_embedding = self.encoder.encode([f"{highly_rated_paper.title}. {highly_rated_paper.abstract}"])[0]
                
                # Find which interests this paper matches best
                for interest in interests:
                    similarity = cosine_similarity([paper_embedding], [interest.embedding])[0][0]
                    
                    if similarity > 0.6:  # Strong match
                        # Boost confidence score
                        boost = 0.1 * (rating - 3)  # +0.1 for 4 stars, +0.2 for 5 stars
                        interest.confidence_score = min(1.0, interest.confidence_score + boost)
                        interest.last_seen = datetime.utcnow()
                        
                        self.logger.info(f"Boosted interest '{interest.topic}' by {boost:.2f}")
            
            elif rating <= 2 and paper_id in papers_db:  # Low rating (1-2 stars)
                poorly_rated_paper = papers_db[paper_id]
                paper_embedding = self.encoder.encode([f"{poorly_rated_paper.title}. {poorly_rated_paper.abstract}"])[0]
                
                # Reduce confidence for matching interests
                for interest in interests:
                    similarity = cosine_similarity([paper_embedding], [interest.embedding])[0][0]
                    
                    if similarity > 0.6:  # Strong match but low rating
                        # Reduce confidence score
                        reduction = 0.05 * (3 - rating)  # -0.1 for 2 stars, -0.15 for 1 star
                        interest.confidence_score = max(0.1, interest.confidence_score - reduction)
                        
                        self.logger.info(f"Reduced interest '{interest.topic}' by {reduction:.2f}")
        
        return interests
    
    def get_interest_keywords(self, interests: List[UserInterest], top_n: int = 50) -> List[str]:
        """Get top keywords across all user interests for paper matching"""
        all_keywords = []
        
        for interest in interests:
            # Weight keywords by interest confidence and recency
            recency_weight = 1.0
            if interest.last_seen:
                days_ago = (datetime.utcnow() - interest.last_seen).days
                recency_weight = max(0.1, 1.0 - (days_ago / 365))  # Decay over year
            
            weight = interest.confidence_score * recency_weight
            
            # Add keywords multiple times based on weight
            repetitions = max(1, int(weight * 10))
            all_keywords.extend(interest.keywords * repetitions)
        
        # Return most frequent weighted keywords
        keyword_counts = Counter(all_keywords)
        return [word for word, _ in keyword_counts.most_common(top_n)]
    
    def calculate_paper_similarity(
        self, 
        candidate_paper: ExtractedPaper, 
        user_interests: List[UserInterest]
    ) -> float:
        """Calculate how well a candidate paper matches user interests"""
        
        if not user_interests:
            return 0.0
        
        # Create embedding for candidate paper
        paper_text = f"{candidate_paper.title}. {candidate_paper.abstract}"
        if not paper_text.strip():
            return 0.0
        
        paper_embedding = self.encoder.encode([paper_text])[0]
        
        # Calculate similarity to each user interest
        similarities = []
        for interest in user_interests:
            sim = cosine_similarity([paper_embedding], [interest.embedding])[0][0]
            
            # Weight by interest confidence and recency
            recency_weight = 1.0
            if interest.last_seen:
                days_ago = (datetime.utcnow() - interest.last_seen).days
                recency_weight = max(0.1, 1.0 - (days_ago / 365))
            
            weighted_sim = sim * interest.confidence_score * recency_weight
            similarities.append(weighted_sim)
        
        # Return highest similarity (best match across all interests)
        return max(similarities) if similarities else 0.0
    
    def identify_trending_interests(
        self, 
        recent_papers: List[ExtractedPaper], 
        time_window_days: int = 30
    ) -> List[str]:
        """Identify trending topics from recently uploaded papers"""
        
        if not recent_papers:
            return []
        
        # Extract keywords from recent papers
        recent_keywords = []
        for paper in recent_papers:
            profile = self.extract_paper_profile(paper)
            recent_keywords.extend(profile.title_keywords + profile.abstract_keywords)
        
        # Find most frequent recent keywords
        keyword_counts = Counter(recent_keywords)
        trending = [word for word, count in keyword_counts.most_common(10) if count >= 2]
        
        return trending
    
    def suggest_research_areas(self, user_interests: List[UserInterest]) -> List[str]:
        """Suggest new research areas user might be interested in"""
        
        if not user_interests:
            return ["machine_learning", "data_science", "software_engineering"]
        
        # Count current research domains
        current_domains = Counter()
        for interest in user_interests:
            for domain, keywords in self.research_domains.items():
                overlap = len(set(interest.keywords) & set(keywords))
                if overlap > 0:
                    current_domains[domain] += overlap
        
        # Suggest related domains
        suggestions = []
        for domain in current_domains:
            if domain == 'machine_learning':
                suggestions.extend(['natural_language_processing', 'computer_vision'])
            elif domain == 'natural_language_processing':
                suggestions.extend(['machine_learning', 'ai_research'])
            elif domain == 'computer_vision':
                suggestions.extend(['machine_learning', 'data_science'])
            elif domain == 'data_science':
                suggestions.extend(['machine_learning', 'software_engineering'])
        
        # Remove duplicates and current domains
        suggestions = list(set(suggestions) - set(current_domains.keys()))
        return suggestions[:5]
    
    def create_user_profile_summary(self, user_interests: List[UserInterest]) -> Dict[str, any]:
        """Create a comprehensive summary of user's research profile"""
        
        if not user_interests:
            return {"profile_status": "no_interests_learned"}
        
        # Aggregate statistics
        total_papers = sum(interest.paper_count for interest in user_interests)
        avg_confidence = np.mean([interest.confidence_score for interest in user_interests])
        
        # Group by research domain
        domain_breakdown = defaultdict(list)
        for interest in user_interests:
            for domain, keywords in self.research_domains.items():
                overlap = len(set(interest.keywords) & set(keywords))
                if overlap > 0:
                    domain_breakdown[domain].append({
                        'topic': interest.topic,
                        'confidence': interest.confidence_score,
                        'papers': interest.paper_count
                    })
        
        # Get top interests by confidence
        top_interests = sorted(user_interests, key=lambda x: x.confidence_score, reverse=True)[:5]
        
        return {
            "profile_status": "active",
            "total_papers_analyzed": total_papers,
            "interests_identified": len(user_interests),
            "average_confidence": avg_confidence,
            "top_interests": [
                {
                    "topic": interest.topic,
                    "keywords": interest.keywords[:5],
                    "confidence": interest.confidence_score,
                    "paper_count": interest.paper_count
                }
                for interest in top_interests
            ],
            "research_domains": dict(domain_breakdown),
            "last_updated": datetime.utcnow().isoformat()
        }


class InterestEvolution:
    """Tracks how user interests change over time"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def track_interest_changes(
        self,
        old_interests: List[UserInterest],
        new_interests: List[UserInterest]
    ) -> Dict[str, any]:
        """Compare interest profiles over time to detect shifts"""
        
        if not old_interests:
            return {"status": "initial_learning", "new_interests": len(new_interests)}
        
        # Calculate similarity between old and new interest sets
        old_embeddings = np.array([interest.embedding for interest in old_interests])
        new_embeddings = np.array([interest.embedding for interest in new_interests])
        
        # Find best matches between old and new interests
        similarity_matrix = cosine_similarity(old_embeddings, new_embeddings)
        
        stable_interests = 0
        emerging_interests = []
        declining_interests = []
        
        for i, old_interest in enumerate(old_interests):
            best_match_idx = np.argmax(similarity_matrix[i])
            best_similarity = similarity_matrix[i][best_match_idx]
            
            if best_similarity > 0.7:  # High similarity - stable interest
                stable_interests += 1
            else:  # Interest may be declining
                declining_interests.append(old_interest.topic)
        
        # Identify truly new interests
        for j, new_interest in enumerate(new_interests):
            best_old_match = np.max(similarity_matrix[:, j])
            if best_old_match < 0.5:  # Low similarity to any old interest
                emerging_interests.append(new_interest.topic)
        
        return {
            "stable_interests": stable_interests,
            "emerging_interests": emerging_interests,
            "declining_interests": declining_interests,
            "total_shift_score": 1.0 - (stable_interests / max(len(old_interests), 1))
        }