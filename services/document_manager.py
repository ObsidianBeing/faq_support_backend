"""
Document Manager Service
Maintains document registry in Redis for fast access
Avoids expensive Pinecone queries for document listing
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import redis
from config import settings


class DocumentManager:
    """
    Maintain separate document catalog in Redis.
    Much faster than querying Pinecone for document lists.
    """
    
    def __init__(self):
        self.redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=2,  # Separate DB for document catalog
            decode_responses=True
        )
    
    async def register_document(
        self,
        doc_id: str,
        filename: str,
        metadata: Dict[str, Any],
        vector_ids: List[str]
    ):
        """
        Register newly uploaded document in catalog.
        
        Args:
            doc_id: Unique document identifier
            filename: Original filename
            metadata: Document metadata (file type, pages, etc.)
            vector_ids: List of Pinecone vector IDs for this document
        """
        doc_info = {
            'doc_id': doc_id,
            'filename': filename,
            'upload_time': datetime.utcnow().isoformat(),
            'file_type': metadata.get('file_type', 'unknown'),
            'total_chunks': len(vector_ids),
            'vector_ids': json.dumps(vector_ids),
            'total_pages': metadata.get('total_pages', 0),
            'last_accessed': datetime.utcnow().isoformat(),
            'status': 'active'
        }
        
        # Store in Redis hash
        self.redis_client.hset(f"doc:{doc_id}", mapping=doc_info)
        
        # Add to set of all document IDs
        self.redis_client.sadd("all_docs", doc_id)
        
        # Index by filename for quick lookup
        self.redis_client.set(f"filename:{filename}", doc_id)
        
        # Update document count
        self.redis_client.incr("doc_count")
    
    async def list_documents(self) -> List[Dict[str, Any]]:
        """
        Fast document listing from Redis (no Pinecone query needed).
        
        Returns:
            List of document metadata dictionaries
        """
        doc_ids = self.redis_client.smembers("all_docs")
        
        documents = []
        for doc_id in doc_ids:
            doc_data = self.redis_client.hgetall(f"doc:{doc_id}")
            if doc_data:
                # Parse JSON fields
                doc_data['vector_ids'] = json.loads(doc_data.get('vector_ids', '[]'))
                doc_data['total_chunks'] = int(doc_data.get('total_chunks', 0))
                doc_data['total_pages'] = int(doc_data.get('total_pages', 0))
                documents.append(doc_data)
        
        # Sort by upload time (newest first)
        documents.sort(key=lambda x: x.get('upload_time', ''), reverse=True)
        
        return documents
    
    async def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get single document metadata.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Document metadata or None if not found
        """
        doc_data = self.redis_client.hgetall(f"doc:{doc_id}")
        
        if doc_data:
            # Parse JSON fields
            doc_data['vector_ids'] = json.loads(doc_data.get('vector_ids', '[]'))
            doc_data['total_chunks'] = int(doc_data.get('total_chunks', 0))
            doc_data['total_pages'] = int(doc_data.get('total_pages', 0))
            
            # Update last accessed time
            self.redis_client.hset(
                f"doc:{doc_id}",
                "last_accessed",
                datetime.utcnow().isoformat()
            )
            
            return doc_data
        
        return None
    
    async def get_document_by_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Get document by filename.
        
        Args:
            filename: Original filename
            
        Returns:
            Document metadata or None if not found
        """
        doc_id = self.redis_client.get(f"filename:{filename}")
        
        if doc_id:
            return await self.get_document(doc_id)
        
        return None
    
    async def delete_document(
        self,
        doc_id: str,
        pinecone_index
    ) -> bool:
        """
        Delete document from both catalog and Pinecone.
        
        Args:
            doc_id: Document identifier
            pinecone_index: Pinecone index instance for vector deletion
            
        Returns:
            True if deleted, False if not found
        """
        doc_data = await self.get_document(doc_id)
        
        if not doc_data:
            return False
        
        # Delete vectors from Pinecone (batch delete)
        vector_ids = doc_data['vector_ids']
        if vector_ids:
            batch_size = 1000
            for i in range(0, len(vector_ids), batch_size):
                batch = vector_ids[i:i+batch_size]
                pinecone_index.delete(ids=batch)
        
        # Delete from Redis catalog
        self.redis_client.delete(f"doc:{doc_id}")
        self.redis_client.srem("all_docs", doc_id)
        
        # Delete filename index
        if doc_data.get('filename'):
            self.redis_client.delete(f"filename:{doc_data['filename']}")
        
        # Decrement document count
        self.redis_client.decr("doc_count")
        
        return True
    
    async def update_document_status(self, doc_id: str, status: str):
        """
        Update document status (active, archived, processing, error).
        
        Args:
            doc_id: Document identifier
            status: New status
        """
        self.redis_client.hset(f"doc:{doc_id}", "status", status)
    
    async def get_document_count(self) -> int:
        """
        Get total number of indexed documents.
        
        Returns:
            Document count
        """
        count = self.redis_client.get("doc_count")
        return int(count) if count else 0
    
    async def search_documents(self, query: str) -> List[Dict[str, Any]]:
        """
        Simple text search in document filenames.
        
        Args:
            query: Search query
            
        Returns:
            List of matching documents
        """
        all_docs = await self.list_documents()
        
        query_lower = query.lower()
        matches = [
            doc for doc in all_docs
            if query_lower in doc.get('filename', '').lower()
        ]
        
        return matches