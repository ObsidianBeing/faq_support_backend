"""
Configuration management for FAQ Support Bot
Loads settings from environment variables with sensible defaults
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Create a .env file in the project root to configure.
    """
    
    # API Keys
    OPENAI_API_KEY: str
    ANTHROPIC_API_KEY: Optional[str] = None
    PINECONE_API_KEY: str
    
    # Pinecone Configuration
    PINECONE_ENVIRONMENT: str = "us-east-1-aws"
    PINECONE_INDEX_NAME: str = "faq-support-bot"
    PINECONE_DIMENSION: int = 1536  # OpenAI ada-002 dimension
    PINECONE_METRIC: str = "cosine"
    
    # LLM Configuration
    LLM_PROVIDER: str = "openai"  # openai or anthropic
    LLM_MODEL: str = "gpt-4o-mini"  # or gpt-4, claude-3-5-sonnet-20241022, etc.
    LLM_TEMPERATURE: float = 0.3
    LLM_MAX_TOKENS: int = 1000
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSION: int = 1536
    
    # Document Processing
    DEFAULT_CHUNK_SIZE: int = 800
    DEFAULT_CHUNK_OVERLAP: int = 200
    
    # Retrieval Configuration
    RETRIEVAL_TOP_K: int = 5
    
    # Confidence & Escalation
    ESCALATION_THRESHOLD: float = 0.6
    ESCALATION_ENDPOINT: str = "https://api.example.com/escalate"
    ESCALATION_TIMEOUT: int = 10  # seconds
    
    # Vector Score Thresholds
    MIN_VECTOR_SCORE: float = 0.5  # Minimum similarity score to consider
    HIGH_CONFIDENCE_VECTOR_SCORE: float = 0.85
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()