"""
Production settings with all features enabled
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Create a .env file in the project root to configure.
    """
    
    # ========================================================================
    # API Keys
    # ========================================================================
    OPENAI_API_KEY: str
    ANTHROPIC_API_KEY: Optional[str] = None
    PINECONE_API_KEY: str
    
    # ========================================================================
    # Pinecone Configuration
    # ========================================================================
    PINECONE_ENVIRONMENT: str = "us-east-1"
    PINECONE_INDEX_NAME: str = "faq-support-bot"
    PINECONE_MEMORY_INDEX_NAME: str = "conversation-memory"
    PINECONE_CLOUD: str = "aws"
    PINECONE_DIMENSION: int = 1536
    PINECONE_METRIC: str = "cosine"
    
    # ========================================================================
    # Redis Configuration
    # ========================================================================
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    REDIS_MAX_CONNECTIONS: int = 50
    
    # ========================================================================
    # PostgreSQL Configuration
    # ========================================================================
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "faq_bot"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_MIN_POOL_SIZE: int = 10
    POSTGRES_MAX_POOL_SIZE: int = 50
    
    # ========================================================================
    # LLM Configuration
    # ========================================================================
    LLM_PROVIDER: str = "openai"  # openai or anthropic
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.3
    LLM_MAX_TOKENS: int = 1000
    
    # ========================================================================
    # Embedding Configuration
    # ========================================================================
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSION: int = 1536
    
    # ========================================================================
    # Document Processing
    # ========================================================================
    DEFAULT_CHUNK_SIZE: int = 800
    DEFAULT_CHUNK_OVERLAP: int = 200
    MAX_UPLOAD_SIZE_MB: int = 50
    DOCUMENT_FRESHNESS_DAYS: int = 30
    
    # ========================================================================
    # Multi-Tier Memory Configuration
    # ========================================================================
    # L1: Working Memory (Redis - 1 hour)
    MAX_WORKING_MEMORY_TURNS: int = 10
    WORKING_MEMORY_TTL_SECONDS: int = 3600
    
    # L2: Session Memory (Redis - 24 hours)
    SESSION_MEMORY_TTL_SECONDS: int = 86400
    SESSION_SUMMARY_INTERVAL: int = 10  # Summarize every N turns
    
    # L3: User Memory (PostgreSQL - 90 days)
    USER_HISTORY_RETENTION_DAYS: int = 90
    USER_PROFILE_UPDATE_THRESHOLD: int = 5
    
    # L4: Semantic Memory (Pinecone - Forever)
    ENABLE_SEMANTIC_MEMORY: bool = True
    
    # Context Window
    CONTEXT_WINDOW_SIZE: int = 3
    ENABLE_QUERY_ENHANCEMENT: bool = True
    
    # ========================================================================
    # Retrieval Configuration
    # ========================================================================
    RETRIEVAL_TOP_K: int = 10
    MIN_VECTOR_SCORE: float = 0.5
    HIGH_CONFIDENCE_VECTOR_SCORE: float = 0.85
    
    # Hybrid Search
    HYBRID_SEARCH_ENABLED: bool = True
    ENABLE_BM25_SEARCH: bool = True
    ENABLE_ENTITY_SEARCH: bool = True
    ENABLE_RERANKING: bool = True
    ENABLE_DIVERSITY_FILTER: bool = True
    
    # Retrieval Weights (must sum to 1.0)
    VECTOR_WEIGHT: float = 0.6
    BM25_WEIGHT: float = 0.2
    ENTITY_WEIGHT: float = 0.1
    RECENCY_WEIGHT: float = 0.1
    
    # Re-ranking
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANKING_TOP_K: int = 20
    
    # Query Expansion
    ENABLE_QUERY_EXPANSION: bool = True
    MAX_QUERY_EXPANSIONS: int = 3
    
    # ========================================================================
    # Semantic Caching
    # ========================================================================
    SEMANTIC_CACHE_ENABLED: bool = True
    CACHE_TTL_SECONDS: int = 3600
    CACHE_SIMILARITY_THRESHOLD: float = 0.95
    CACHE_MAX_SIZE: int = 10000
    
    # ========================================================================
    # Confidence & Escalation
    # ========================================================================
    ESCALATION_THRESHOLD: float = 0.6
    ESCALATION_ENDPOINT: str = "https://api.example.com/escalate"
    ESCALATION_TIMEOUT: int = 10
    
    # ========================================================================
    # Quality & Learning
    # ========================================================================
    ENABLE_QUALITY_SCORING: bool = True
    ENABLE_FEEDBACK_COLLECTION: bool = True
    FEEDBACK_LEARNING_ENABLED: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"


# Global settings instance
settings = Settings()