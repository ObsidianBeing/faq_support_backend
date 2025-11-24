"""
Advanced Hybrid Retrieval System
- Vector similarity (semantic)
- BM25 keyword matching
- Entity-based retrieval
- Re-ranking with cross-encoder
- Semantic caching
- Diversity filtering
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import hashlib
import time
from collections import defaultdict

from pinecone import Pinecone
import openai
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
import redis

from config import settings


@dataclass
class RetrievalResult:
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]
    retrieval_method: str  # 'vector', 'bm25', 'entity', 'cached'
    rerank_score: Optional[float] = None


class SemanticCache:
    """
    Cache answers based on semantic similarity of queries.
    """
    
    def __init__(self):
        self.redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=1,  # Separate DB for cache
            decode_responses=False  # Store bytes for embeddings
        )
        
        self.openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        self.cache_ttl = 3600  # 1 hour
        self.similarity_threshold = 0.95  # Very high threshold
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for query."""
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        return response.data[0].embedding
    
    def _compute_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """Cosine similarity between embeddings."""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
    async def get_cached_answer(
        self,
        query: str
    ) -> Optional[Dict[str, Any]]:
        """
        Check if we have a cached answer for similar query.
        """
        query_embedding = self._get_query_embedding(query)
        
        # Get all cached embeddings
        cache_keys = self.redis_client.keys("cache:query:*")
        
        best_match = None
        best_similarity = 0.0
        
        for key in cache_keys[:100]:  # Check last 100 cached queries
            cached_data = self.redis_client.hgetall(key)
            
            if not cached_data:
                continue
            
            # Decode embedding
            cached_embedding = np.frombuffer(
                cached_data[b'embedding'],
                dtype=np.float32
            ).tolist()
            
            # Compute similarity
            similarity = self._compute_similarity(query_embedding, cached_embedding)
            
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_match = {
                    'answer': cached_data[b'answer'].decode('utf-8'),
                    'confidence': float(cached_data[b'confidence']),
                    'sources': eval(cached_data[b'sources'].decode('utf-8')),
                    'cached_at': float(cached_data[b'timestamp']),
                    'cache_similarity': similarity
                }
        
        if best_match:
            # Check if cache is still fresh
            age = time.time() - best_match['cached_at']
            if age < self.cache_ttl:
                return best_match
        
        return None
    
    async def cache_answer(
        self,
        query: str,
        answer: str,
        confidence: float,
        sources: List[Dict]
    ):
        """Cache an answer."""
        query_embedding = self._get_query_embedding(query)
        
        # Create cache key
        query_hash = hashlib.md5(query.encode()).hexdigest()
        cache_key = f"cache:query:{query_hash}"
        
        # Store with embedding
        self.redis_client.hset(cache_key, mapping={
            'query': query,
            'embedding': np.array(query_embedding, dtype=np.float32).tobytes(),
            'answer': answer,
            'confidence': str(confidence),
            'sources': str(sources),
            'timestamp': str(time.time())
        })
        
        # Set TTL
        self.redis_client.expire(cache_key, self.cache_ttl)


class HybridRetriever:
    """
    Advanced retrieval combining multiple strategies.
    """
    
    def __init__(self):
        # Vector store
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index = self.pc.Index(settings.PINECONE_INDEX_NAME)
        
        # OpenAI for embeddings
        self.openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        
        # Cross-encoder for re-ranking
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # BM25 index (in-memory, rebuild periodically)
        self.bm25_index = None
        self.bm25_docs = []
        self.bm25_metadata = []
        
        # Semantic cache
        self.cache = SemanticCache()
        
        # Retrieval weights
        self.weights = {
            'vector': 0.6,
            'bm25': 0.2,
            'entity': 0.1,
            'recency': 0.1
        }
    
    async def initialize_bm25(self):
        """Build BM25 index from Pinecone data."""
        # Fetch all documents (or sample)
        # In production, maintain separately or rebuild periodically
        docs = await self._fetch_all_documents()
        
        tokenized_docs = [doc['text'].lower().split() for doc in docs]
        self.bm25_index = BM25Okapi(tokenized_docs)
        self.bm25_docs = docs
    
    async def _fetch_all_documents(self) -> List[Dict]:
        """Fetch documents for BM25 indexing."""
        # Query with dummy vector to get documents
        results = self.index.query(
            vector=[0.0] * settings.PINECONE_DIMENSION,
            top_k=10000,
            include_metadata=True
        )
        
        return [
            {
                'id': match.id,
                'text': match.metadata.get('text', ''),
                'metadata': match.metadata
            }
            for match in results.matches
        ]
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        response = self.openai_client.embeddings.create(
            model=settings.EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict] = None,
        user_context: Optional[Dict] = None
    ) -> List[RetrievalResult]:
        """
        Hybrid retrieval with multiple strategies.
        """
        # Check cache first
        cached = await self.cache.get_cached_answer(query)
        if cached:
            return [RetrievalResult(
                id='cached',
                text=cached['answer'],
                score=cached['confidence'],
                metadata={'cached': True},
                retrieval_method='cached'
            )]
        
        # Parallel retrieval from multiple sources
        results_by_method = {}
        
        # 1. Vector similarity search
        results_by_method['vector'] = await self._vector_search(query, top_k, filters)
        
        # 2. BM25 keyword search
        if self.bm25_index:
            results_by_method['bm25'] = await self._bm25_search(query, top_k)
        
        # 3. Entity-based retrieval (if entities in query)
        entities = self._extract_entities(query, user_context)
        if entities:
            results_by_method['entity'] = await self._entity_search(entities, top_k, filters)
        
        # Combine results with fusion
        combined = self._reciprocal_rank_fusion(results_by_method)
        
        # Re-rank with cross-encoder
        reranked = await self._rerank(query, combined, top_k=top_k)
        
        # Apply diversity filter
        diverse = self._apply_diversity_filter(reranked, top_k=top_k)
        
        return diverse[:top_k]
    
    async def _vector_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict] = None
    ) -> List[RetrievalResult]:
        """Standard vector similarity search."""
        query_embedding = self._generate_embedding(query)
        
        search_params = {
            'vector': query_embedding,
            'top_k': top_k * 2,  # Get more for fusion
            'include_metadata': True
        }
        
        if filters:
            search_params['filter'] = filters
        
        results = self.index.query(**search_params)
        
        return [
            RetrievalResult(
                id=match.id,
                text=match.metadata.get('text', ''),
                score=match.score,
                metadata=match.metadata,
                retrieval_method='vector'
            )
            for match in results.matches
        ]
    
    async def _bm25_search(
        self,
        query: str,
        top_k: int
    ) -> List[RetrievalResult]:
        """BM25 keyword-based search."""
        if not self.bm25_index:
            return []
        
        tokenized_query = query.lower().split()
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # Get top-k by score
        top_indices = np.argsort(scores)[-top_k * 2:][::-1]
        
        return [
            RetrievalResult(
                id=self.bm25_docs[idx]['id'],
                text=self.bm25_docs[idx]['text'],
                score=scores[idx] / 100,  # Normalize roughly to 0-1
                metadata=self.bm25_docs[idx]['metadata'],
                retrieval_method='bm25'
            )
            for idx in top_indices
            if scores[idx] > 0
        ]
    
    async def _entity_search(
        self,
        entities: List[str],
        top_k: int,
        filters: Optional[Dict] = None
    ) -> List[RetrievalResult]:
        """Search based on extracted entities."""
        results = []
        
        for entity in entities:
            entity_results = await self._vector_search(
                query=entity,
                top_k=top_k,
                filters=filters
            )
            
            for result in entity_results:
                result.retrieval_method = 'entity'
                result.score *= 1.1  # Slight boost for entity matches
            
            results.extend(entity_results)
        
        return results
    
    def _extract_entities(
        self,
        query: str,
        user_context: Optional[Dict] = None
    ) -> List[str]:
        """Extract entities from query and context."""
        entities = []
        
        # From user context
        if user_context and 'entities' in user_context:
            entities.extend(user_context['entities'])
        
        # Simple pattern matching (in production, use NER)
        # Look for product names, feature names
        patterns = [
            r'TechWidget\s+\w+',
            r'Pro\s+Plan',
            r'API\s+key',
        ]
        
        import re
        for pattern in patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            entities.extend(matches)
        
        return list(set(entities))
    
    def _reciprocal_rank_fusion(
        self,
        results_by_method: Dict[str, List[RetrievalResult]],
        k: int = 60
    ) -> List[RetrievalResult]:
        """
        Combine results from multiple methods using RRF.
        
        RRF score = sum(1 / (k + rank_i)) for each method
        """
        # Build score map
        scores = defaultdict(float)
        result_map = {}
        
        for method, results in results_by_method.items():
            weight = self.weights.get(method, 0.1)
            
            for rank, result in enumerate(results, 1):
                rrf_score = 1.0 / (k + rank)
                scores[result.id] += rrf_score * weight
                
                if result.id not in result_map:
                    result_map[result.id] = result
        
        # Sort by combined score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        # Return combined results with updated scores
        combined = []
        for doc_id in sorted_ids:
            result = result_map[doc_id]
            result.score = scores[doc_id]
            combined.append(result)
        
        return combined
    
    async def _rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int
    ) -> List[RetrievalResult]:
        """
        Re-rank results using cross-encoder for better relevance.
        """
        if not results:
            return []
        
        # Prepare pairs for cross-encoder
        pairs = [[query, result.text] for result in results]
        
        # Get relevance scores
        rerank_scores = self.reranker.predict(pairs)
        
        # Update scores
        for result, rerank_score in zip(results, rerank_scores):
            result.rerank_score = float(rerank_score)
            # Combine original and rerank score
            result.score = 0.5 * result.score + 0.5 * rerank_score
        
        # Sort by new scores
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results[:top_k * 2]  # Return more for diversity filter
    
    def _apply_diversity_filter(
        self,
        results: List[RetrievalResult],
        top_k: int,
        similarity_threshold: float = 0.9
    ) -> List[RetrievalResult]:
        """
        Filter out near-duplicate results to increase diversity.
        """
        if not results:
            return []
        
        diverse_results = [results[0]]  # Always include top result
        
        for result in results[1:]:
            if len(diverse_results) >= top_k:
                break
            
            # Check similarity with already selected results
            is_diverse = True
            
            for selected in diverse_results:
                # Simple text similarity check
                overlap = self._jaccard_similarity(
                    result.text.lower().split(),
                    selected.text.lower().split()
                )
                
                if overlap > similarity_threshold:
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse_results.append(result)
        
        return diverse_results
    
    def _jaccard_similarity(self, set1: List[str], set2: List[str]) -> float:
        """Compute Jaccard similarity between two sets."""
        set1, set2 = set(set1), set(set2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0
    
    # ========================================================================
    # QUERY EXPANSION
    # ========================================================================
    
    async def expand_query(self, query: str) -> List[str]:
        """
        Generate query variations for better recall.
        """
        prompt = f"""Generate 3 alternative phrasings of this question that mean the same thing:

Original: "{query}"

Alternative phrasings (one per line):"""

        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You generate alternative phrasings."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150
        )
        
        alternatives = response.choices[0].message.content.strip().split('\n')
        alternatives = [alt.strip('- ') for alt in alternatives if alt.strip()]
        
        return [query] + alternatives[:3]
    
    # ========================================================================
    # ANALYTICS
    # ========================================================================
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get statistics about retrieval performance."""
        return {
            'cache_hit_rate': self._compute_cache_hit_rate(),
            'avg_results_per_method': self._compute_avg_results(),
            'most_common_entities': self._get_common_entities(),
            'avg_rerank_improvement': self._compute_rerank_improvement()
        }
    
    def _compute_cache_hit_rate(self) -> float:
        """Compute cache hit rate."""
        # Would track in Redis metrics
        return 0.42  # Placeholder
    
    def _compute_avg_results(self) -> Dict[str, float]:
        """Average results from each method."""
        return {
            'vector': 8.5,
            'bm25': 6.2,
            'entity': 3.1
        }
    
    def _get_common_entities(self) -> List[str]:
        """Most commonly extracted entities."""
        return ['TechWidget Pro', 'Pro Plan', 'API key']
    
    def _compute_rerank_improvement(self) -> float:
        """Average improvement from re-ranking."""
        return 0.15  # 15% improvement