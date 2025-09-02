from typing import Optional, List
import logging
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BartTokenizer,
    BartForConditionalGeneration,
    T5Tokenizer,
    T5ForConditionalGeneration
)
from ..config.settings import settings
from ..parse.pdf_extractor import ExtractedPaper


class AbstractiveSummarizer:

    def __init__(self, model_name: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name or settings.models.abstractive_model
        self.device = self._get_optimal_device()
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _get_optimal_device(self) -> torch.device:
        if torch.backends.mps.is_available():
            self.logger.info("Using Apple Silicon MPS acceleration")
            return torch.device("mps")
        elif torch.cuda.is_availavle():
            return torch.device("cuda")
        else:
            self.logger.info("Using CPU")
            return torch.device("cpu")

    def _load_model(self):
        try:
            self.logger.info(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir = str(settings.models.cache_dir)
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                cache_dir = str(settings.models.cache_dir),
                torch_dtype = torch.float16 if self.device.type == "mps" else torch.float32,
                low_cpu_mem_usage = True
            )
            self.model.to(self.device)
            self.model.eval()

            if hasattr(self.model.config, 'use_memory_efficient_attention'):
                self.model.config.use_memory_efficient_attention = True

            self.logger.info(f"Model loaded successfully on {self.device}")

        except Exception as e:
            self.logger.error(f"Failed to load model {self.model_name}: {e}")
            raise

    def _chunk_text(self, text: str, max_chunk_length: int = 800) -> List[str]:
        tokens = self.tokenizer.encode(text, truncation = False)

        if len(tokens) <= max_chunk_length:
            return [text]

        sentences = text.split('. ')
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_tokens = len(self.tokenizer.encode(sentence + '. '))

            if current_length + sentence_tokens > max_chunk_length and current_chunk:
                chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sentence]
                current_length = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_length += sentence_tokens

        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')

        return chunks

    def _generate_summary_chunk(self, text: str) -> str:
        inputs = self.tokenizer(
            text,
            max_length = 800,
            truncation = True,
            padding = True,
            return_tensors = "pt"
        ).to(self.device)

        with torch.no_grad():
            if self.device.type == "mps":
                with torch.autocast(device_type="cpu", dtype=torch.float16):
                    summary_ids = self.model.generate(
                        **inputs,
                        max_new_tokens = min(settings.models.max_summary_length, 256),
                        min_length = settings.models.min_summary_length,
                        num_beams = 3,
                        length_penalty = 1.5,
                        no_repeat_ngram_size = 3,
                        repetition_penalty = 1.1,
                        early_stopping = True,
                        do_sample = False,
                        use_cache = True
                    )
            else:
                summary_ids = self.model.generate(
                    **inputs,
                    max_length = settings.models.max_summary_length,
                    min_length = settings.models.min_summary_length,
                    num_beams = 4,
                    length_penalty = 2.0,
                    no_repeat_ngram_size = 3,
                    repetition_penalty = 1.2,
                    early_stopping = True,
                    do_sample = False
                )

        if self.device.type == "mps":
            torch.mps.empty_cache()

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens = True)
        return summary.strip()

    def summarize_paper(self, paper: ExtractedPaper, target_length: int = 200) -> str:
        content_parts = []

        if paper.abstract:
            content_parts.append(paper.abstract)

        for section in paper.sections:
            if section.section_type in ['conclusion', 'introduction', 'results']:
                if section.content:
                    limited_content = section.content[:1000]
                    content_parts.append(limited_content)

        combined_text = ' '.join(content_parts)
        return self.summarize_text(combined_text, target_length)

    def summarize_text(self, text: str, target_length: int = 200) -> str:
        if not text.strip():
            return "No content to summarize."

        if self.device.type == "mps":
            max_chunk = 600
            settings.models.max_summary_length = min(300, target_length * 2)
        else:
            max_chunk = 900
            settings.models.max_summary_length = target_length * 3

        chunks = self._chunk_text(text, max_chunk_length = max_chunk)

        if len(chunks) == 1:
            return self._generate_summary_chunk(chunks[0])

        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            self.logger.info(f"Processing chunk {i + 1} / {len(chunks)} on {self.device}")

            try:
                chunk_summary = self._generate_summary_chunk(chunk)
                chunk_summaries.append(chunk_summary)

                if self.device.type == "mps":
                    torch.mps.empty_cache()

            except torch.OutOfMemoryError:
                self.logger.warning(f"OOM on chunk {i + 1}, skipping")
                continue

        if not chunk_summaries:
            return "Summary generation failed due to memory constraints."

        combined = ' '.join(chunk_summaries)

        if len(combined) > 1000:
            return self._generate_summary_chunk(combined)
        else:
            return combined


class LightweightSummarizer(AbstractiveSummarizer):

    def __init__(self):
        super().__init__(model_name = "t5-small")

    def _load_model(self):
        try:
            self.logger.info("Loading lightweight T5-small model for Apple Silicon")
            self.tokenizer = T5Tokenizer.from_pretrained(
                "t5-small",
                cache_dir = str(settings.models.cache_dir)
            )

            self.model = T5ForConditionalGeneration.from_pretrained(
                "t5-small",
                cache_dir = str(settings.models.cache_dir),
                torch_dtype = torch.float16 if self.device.type == "mps" else torch.float32,
                low_cpu_mem_usage = True
            )

            self.model.to(self.device)
            self.model.eval()

            self.logger.info(f"Lightweight model ready on {self.device}")

        except Exception as e:
            self.logger.error(f"Failed to load lightweight model: {e}")
            raise

    def _generate_summary_chunks(self, text: str) -> str:
        input_text = f"summarize: {text}"
        inputs = self.tokenizer(
            input_text,
            max_length = 512,
            truncation = True,
            return_tensors = "pt"
        ).to(self.device)

        with torch.no_grad():
            summary_ids = self.model.generate(
                **inputs,
                max_new_tokens = 150,
                min_length = 30,
                num_beams = 2,
                length_penalty = 1.0,
                no_repeat_ngram_size = 2,
                early_stopping = True,
                do_sample = False
            )

        if self.device.type == "mps":
            torch.mps.empty_cache()

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens = True)
        return summary.strip()