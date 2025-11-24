"""
Custom exceptions
Provides specific error types for better error handling
"""


class DocumentProcessingError(Exception):
    """Base exception for document processing errors."""
    pass


class FileReadError(DocumentProcessingError):
    """Failed to read uploaded file."""
    pass


class UnsupportedFormatError(DocumentProcessingError):
    """File format not supported."""
    pass


class ExtractionError(DocumentProcessingError):
    """Failed to extract text from document."""
    pass


class ChunkingError(DocumentProcessingError):
    """Failed to chunk document."""
    pass


class VectorStoreError(DocumentProcessingError):
    """Failed to store in vector database."""
    pass


class EmbeddingError(Exception):
    """Failed to generate embeddings."""
    pass


class RetrievalError(Exception):
    """Failed to retrieve documents."""
    pass


class MemoryError(Exception):
    """Failed to access memory system."""
    pass


class CacheError(Exception):
    """Failed to access cache."""
    pass