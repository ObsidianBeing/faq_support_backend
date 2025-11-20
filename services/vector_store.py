"""
Vector Store Service
Manages Pinecone vector database operations
"""

from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
import openai
from config import settings


class VectorStoreService:
    """
    Handles all vector database operations using Pinecone.
    """
    
    def __init__(self):
        self.pc = None
        self.index = None
        self.openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
    
    async def initialize(self):
        """
        Initialize Pinecone connection and create index if needed.
        """
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        
        # Check if index exists, create if not
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        
        if settings.PINECONE_INDEX_NAME not in existing_indexes:
            print(f"Creating new Pinecone index: {settings.PINECONE_INDEX_NAME}")
            self.pc.create_index(
                name=settings.PINECONE_INDEX_NAME,
                dimension=settings.PINECONE_DIMENSION,
                metric=settings.PINECONE_METRIC,
                spec=ServerlessSpec(
                    cloud='aws',
                    region=settings.PINECONE_ENVIRONMENT
                )
            )
        
        self.index = self.pc.Index(settings.PINECONE_INDEX_NAME)
    
    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for text using OpenAI.
        """
        response = self.openai_client.embeddings.create(
            model=settings.EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    
    async def store_chunks(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """
        Generate embeddings and store chunks in Pinecone.
        
        Returns:
            List of vector IDs
        """
        vectors_to_upsert = []
        vector_ids = []
        
        for chunk in chunks:
            # Generate unique ID for this chunk
            doc_id = chunk['metadata']['doc_id']
            chunk_num = chunk['metadata']['chunk_num']
            vector_id = f"{doc_id}_{chunk_num}"
            
            # Generate embedding
            embedding = self._generate_embedding(chunk['text'])
            
            # Prepare vector with metadata
            vector = {
                'id': vector_id,
                'values': embedding,
                'metadata': {
                    **chunk['metadata'],
                    'text': chunk['text']  # Store text in metadata for retrieval
                }
            }
            
            vectors_to_upsert.append(vector)
            vector_ids.append(vector_id)
        
        # Upsert vectors in batches
        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            self.index.upsert(vectors=batch)
        
        return vector_ids
    
    async def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks using vector similarity.
        
        Returns:
            List of matching chunks with scores and metadata
        """
        # Generate query embedding
        query_embedding = self._generate_embedding(query)
        
        # Search Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # Format results
        formatted_results = []
        for match in results.matches:
            if match.score >= settings.MIN_VECTOR_SCORE:
                formatted_results.append({
                    'id': match.id,
                    'score': match.score,
                    'text': match.metadata.get('text', ''),
                    'metadata': {
                        k: v for k, v in match.metadata.items()
                        if k != 'text'  # Exclude text from metadata dict
                    }
                })
        
        return formatted_results
    
    async def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all indexed documents.
        """
        stats = self.index.describe_index_stats()
        
        # Extract unique document IDs from namespaces
        documents = []
        
        # Query to get sample vectors and extract doc_ids
        # This is a workaround since Pinecone doesn't have a direct list API
        try:
            # Fetch some vectors to extract unique doc_ids
            sample_results = self.index.query(
                vector=[0.0] * settings.PINECONE_DIMENSION,
                top_k=10000,
                include_metadata=True
            )
            
            doc_map = {}
            for match in sample_results.matches:
                doc_id = match.metadata.get('doc_id')
                if doc_id and doc_id not in doc_map:
                    doc_map[doc_id] = {
                        'doc_id': doc_id,
                        'filename': match.metadata.get('filename', 'unknown'),
                        'upload_time': match.metadata.get('upload_time', 'unknown'),
                        'file_type': match.metadata.get('file_type', 'unknown')
                    }
            
            documents = list(doc_map.values())
        except Exception as e:
            print(f"Error listing documents: {e}")
        
        return documents
    
    async def delete_document(self, doc_id: str) -> int:
        """
        Delete all vectors for a specific document.
        
        Returns:
            Number of vectors deleted
        """
        # Pinecone requires vector IDs to delete
        # We need to find all vectors with this doc_id
        
        # Query to find all vectors for this doc
        try:
            results = self.index.query(
                vector=[0.0] * settings.PINECONE_DIMENSION,
                top_k=10000,
                include_metadata=True,
                filter={'doc_id': doc_id}
            )
            
            vector_ids = [match.id for match in results.matches]
            
            if vector_ids:
                # Delete in batches
                batch_size = 1000
                for i in range(0, len(vector_ids), batch_size):
                    batch = vector_ids[i:i + batch_size]
                    self.index.delete(ids=batch)
            
            return len(vector_ids)
        except Exception as e:
            print(f"Error deleting document: {e}")
            return 0
    
    async def get_document_count(self) -> int:
        """
        Get total number of vectors in index.
        """
        try:
            stats = self.index.describe_index_stats()
            return stats.total_vector_count
        except:
            return 0
    
    async def health_check(self) -> bool:
        """
        Check if Pinecone connection is healthy.
        """
        try:
            self.index.describe_index_stats()
            return True
        except:
            return False