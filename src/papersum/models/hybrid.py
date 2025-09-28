from typing import List, Dict, Optional
import logging
from dataclasses import dataclass
from enum import Enum
from .extractive import ExtractiveSummarizer, ScoredSentence
from .abstractive import AbstractiveSummarizer, LightweightSummarizer
from ..parse.pdf_extractor import ExtractedPaper, PaperSection


class SummaryStrategy(Enum):
    EXTRACT_THEN_ABSTRACT = "extract_then_abstract"
    PARALLEL_COMBINE = "parallel_combine"
    SECTION_WISE = "section_wise"
    ADAPTIVE = "adaptive"


@dataclass
class HybridSummary:
    extractive_summary: str
    abstractive_summary: str
    hybrid_summary: str
    strategy_used: str
    section_summaries: Dict[str, str]
    metadata: Dict[str, any]


class HybridSummarizer:
    def __init__(self, use_lightweight: bool = False):
        self.logger = logging.getLogger(__name__)
        self.extractive = ExtractiveSummarizer()
        if use_lightweight:
            self.abstractive = LightweightSummarizer()
            self.logger.info("Using lightweight abstractive model for Apple Silicon")
        else:
            self.abstractive = AbstractiveSummarizer()

    def _adaptive_strategy(self, paper: ExtractedPaper) -> SummaryStrategy:
        text_length = len(paper.full_text)
        section_count = len(paper.sections)

        # Short papers will use extract then abstract approach
        if text_length < 5000:
            return SummaryStrategy.EXTRACT_THEN_ABSTRACT

        # For well structured paper, section-wise
        if section_count >= 4 and paper.abstract:
            return SummaryStrategy.SECTION_WISE

        # For long and unstructured, parallel
        if text_length > 20000:
            return SummaryStrategy.PARALLEL_COMBINE

        # Default to extract then abstract
        return SummaryStrategy.EXTRACT_THEN_ABSTRACT

    def _extract_then_abstract(self, paper: ExtractedPaper, target_length: int) -> str:
        extract_count = min(target_length // 20, 10)
        extractive_summary = self.extractive.summarize_paper(paper, extract_count)

        # Return early if extractive is short enough
        if len(extractive_summary.split()) <= target_length:
            return extractive_summary

        # Otherwise, use abstraction to refine the extraction
        refined_summary = self.abstractive.summarize_text(
            extractive_summary,
            target_length = target_length
        )

        return refined_summary

    def _parallel_combine(self, paper: ExtractedPaper, target_length: int) -> str:
        extractive_summary = self.extractive.summarize_paper(
            paper,
            num_sentences = target_length // 25
        )

        abstractive_summary = self.abstractive.summarize_paper(
            paper,
            target_length = target_length
        )

        ext_sentences = extractive_summary.split('. ')
        abs_sentences = abstractive_summary.split('. ')

        combined = []
        max_len = max(len(ext_sentences), len(abs_sentences))

        for i in range(max_len):
            if i < len(abs_sentences) and abs_sentences[i].strip():
                combined.append(abs_sentences[i])
            if i < len(ext_sentences) and ext_sentences[i].strip():
                combined.append(ext_sentences[i])

        result = '. '.join(combined[:target_length // 15])
        return result + '.' if not result.endswith('.') else result

    def _section_wise_summary(self, paper: ExtractedPaper, target_length: int) -> str:
        section_summaries = {}
        section_weights = {
            'abstract': 0.3,
            'introduction': 0.15,
            'conclusion': 0.25,
            'results': 0.2,
            'methods': 0.1
        }

        for section in paper.sections:
            if not section.content or len(section.content) < 100:
                continue
            
            weight = section_weights.get(section.section_type, 0.05)
            section_target = int(target_length * weight)

            if section_target < 30:
                continue

            if len(section.content) < 2000:
                summary = self.extractive.summarize_text(
                    section.content,
                    num_sentences = max(1, section_target // 20)
                )
            else:
                summary = self.abstractive.summarize_text(
                    section.content,
                    target_length = section_target
                )

            section_summaries[section.section_type] = summary

        ordered_sections = ['abstract', 'introduction', 'methods', 'results', 'conclusion']
        final_parts = []

        for section_type in ordered_sections:
            if section_type in section_summaries:
                final_parts.append(section_summaries[section_type])

        return ' '.join(final_parts)

    def summarize_paper_full(self, paper: ExtractedPaper, target_length: int = 200, strategy: Optional[SummaryStrategy] = None) -> HybridSummary:
        if strategy is None:
            strategy = self._adaptive_strategy(paper)

        self.logger.info(f"Using strategy: {strategy.value}")

        extractive_summary = self.extractive.summarize_paper(paper, target_length // 20)
        abstractive_summary = self.abstractive.summarize_paper(paper, target_length)

        if strategy == SummaryStrategy.EXTRACT_THEN_ABSTRACT:
            hybrid_summary = self._extract_then_abstract(paper, target_length)
        elif strategy == SummaryStrategy.PARALLEL_COMBINE:
            hybrid_summary = self._parallel_combine(paper, target_length)
        elif strategy == SummaryStrategy.SECTION_WISE:
            hybrid_summary = self._section_wise_summary(paper, target_length)
        else:
            hybrid_summary = self._extract_then_abstract(paper, target_length)

        section_summaries = self.extractive.summarize_sections(paper.sections)

        return HybridSummary(
            extractive_summary = extractive_summary,
            abstractive_summary = abstractive_summary,
            hybrid_summary = hybrid_summary,
            strategy_used = strategy.value,
            section_summaries = section_summaries,
            metadata = {
                "paper_length": len(paper.full_text),
                "sections_count": len(paper.sections),
                "has_abstract": bool(paper.abstract),
                "device_used": str(self.abstractive.device)
            }
        )

    def summarize_text_adaptive(self, text: str, target_length: int = 200) -> str:
        #Adaptive summarization of raw text
        text_length = len(text)

        if text_length < 2000:
            return self.abstractive.summarize_text(text, target_length)
        elif text_length < 10000:
            extracted = self.extractive.summarize_text(text, target_length // 15)
            return self.abstractive.summarize_text(extracted, target_length)
        else:
            extracted = self.extractive.summarize_text(text, target_length // 10)
            if len(extracted.split()) <= target_length:
                return extracted
            return self.abstractive.summarize_text(extracted, target_length)
