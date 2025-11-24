"""
Production-ready implementation draft 1
"""

from .document_processor import DocumentProcessor
from .vector_store import VectorStoreService
from .llm_service import LLMService
from .confidence_calculator import ConfidenceCalculator
from .escalation_service import EscalationService

# New production services
from .exceptions import *
from .document_manager import DocumentManager
from .multi_tier_memory import MultiTierMemory
from .advanced_retrieval import HybridRetriever, SemanticCache

__all__ = [
    # Original services
    'DocumentProcessor',
    'VectorStoreService',
    'LLMService',
    'ConfidenceCalculator',
    'EscalationService',
    
    # New services
    'DocumentManager',
    'MultiTierMemory',
    'HybridRetriever',
    'SemanticCache',
    
    # Exceptions
    'DocumentProcessingError',
    'FileReadError',
    'UnsupportedFormatError',
    'ExtractionError',
    'ChunkingError',
    'VectorStoreError',
    'EmbeddingError',
    'RetrievalError',
    'MemoryError',
    'CacheError',
]