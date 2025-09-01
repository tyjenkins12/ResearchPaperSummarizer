# Utilizing Sentence-BERT for semantic understanding of research paper content

from typing import List, Dict, Tuple
import logging
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from spacy.lan.en import English
from ..config.settings import settings
from ..parse.pdf_extractor import ExtractedPaper, PaperSection


@dataclass
class ScoredSentence:
    text: str
    score: float
    section: str
    position: int


class ExtractiveSummarizer:

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        self.sentence_model = SentenceTransformer(
            settings.models.sentence_transformer_model,
            cache_folder = str(settings.models.cache_dir)
        )

        self.nlp = English()
        self.nlp.add_pipe('sentencizer')

    def _split_into_sentences(self, text: str) -> List[str]:
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 20]

    def _calculate_sentence_scores(self, sentences: List[str], target_length: int = 3) -> List[float]:
        if not sentences:
            return []

        # Finding the center focus of the document to score sentences based on their similarity to centroid
        
        embeddings = self.sentence_model.encode(sentences)
        doc_centroid = np.mean(embeddings, axis = 0).reshape(1, -1)
        similarities = cosine_similarity(embeddings, doc_centroid).flatten()
        position_weights = np.exp(-np.arrange(len(sentences)) * 0.1)
        final_scores = similarities * 0.7 + position_weights * 0.3

        return final_scores.tolist()

    def _calculate_sections_scores(self, sections: List[PaperSection]) -> Dict[str, float]:
        section_weights = {
            'abstract': 1.0,
            'introduction': 0.8,
            'conclusion': 0.9,
            'results': 0.7,
            'methods': 0.5,
            'background': 0.4,
            'other': 0.3
        }
        return {section.title: section_weights.get(section.section_type, 0.3)for section in sections}

    def summarize_paper(self, paper: ExtractedPaper, num_sentences: int = 5) -> str:
        all_scored_sentences = []

        if paper.abstract:
            abstract_sentences = self._split_into_sentences(paper.abstract)
            abstract_scores = self._calculate_sentence_scores(abstract_sentences)

            for i, (sentence, score) in enumerate(zip(abstract_sentences, abstract_scores)):
                all_scored_sentences.append(ScoredSentence(
                    text = sentence,
                    score = score * 1.2,
                    section = "abstract",
                    position = i
                ))

        section_weights = self._calculate_section_scores(paper.sections)

        for section in paper.sections:
            if not section.content:
                continue

            sentences = self._split_into_sentences(section.content)
            scores = self._calculate_sentence_scores(sentences)
            section_weight = section_weights.get(section.title, 0.3)

            for i, (sentence, score) in enumerate(zip(sentences, scores)):
                all_scored_sentences.append(ScoredSentence(
                    text = sentence,
                    score = score * section_weight,
                    section = section.section_type,
                    position = i
                ))

        # Sorting the sentences by score and selecting the highest scoring

        top_sentences = sorted(all_scored_sentences, key = lambda x: x.score, reverse = True)[:num_sentences]

        top_sentences.sort(key = lambda x: (x.section, x.position))

        return ' '.join([sent.text for sent in top_sentences])

    def summarize_text(self, text: str, num_sentences: int = 5) -> str:
        sentences = self._split_into_sentences(text)
        if len(sentences) <= num_sentences:
            return text

        scores = self._calculate_sentences_scores(sentences, num_sentences)

        sentence_scores = list(zip(sentences, scores, ranges(len(sentences))))
        top_sentences = sorted(sentences_scores, ket = lambda x: x[1], reverse = True)[:num_sentences]

        top_sentences.sort(key = lambda x: x[2])

        return ' '.join([sent[0] for sent in top_sentences])