from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class ModelConfig(BaseModel):
    sentence_transformer_model: str = "all-MiniLM-L6-v2"

    abstractive_model: str = "facebook/bart-large-cnn"
    max_summary_length: int = 512
    min_summary_length: int = 100

    cache_dir: Path = Path.home() /".cache" / "papersum" / "models"


class GrobidConfig(BaseModel):
    host: str = "localhost"
    port: int = 8070
    timeout: int = 60


class AppSettings(BaseSettings):
    app_name: str = "Research Paper Summarizer"
    version: str = "2.0.0"
    debug: bool = False

    host: str = "localhost"
    port: int = 8000

    models: ModelConfig = Field(default_factory = ModelConfig)
    grobid: GrobidConfig = Field(default_factory = GrobidConfig)

    max_file_size_mb: int = 50
    supported_formats: list[str] = ["pdf"]

    upload_dir: Path = Path("data/uploads")
    output_dir: Path = Path("data/outputs")

    class Config:
        env_prefix = "PAPERSUM_"
        env_nested_delimiter = "__"


settings = AppSettings()