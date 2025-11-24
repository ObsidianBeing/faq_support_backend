"""
FAQ Support Bot - Production-Ready Main API
Full implementation with all production features
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import os
import uvicorn
from datetime import datetime
import uuid
import asyncio
import time

from services.document_processor import DocumentProcessor
from services.vector_store import VectorStoreService
from services.llm_service import LLMService
from services.confidence_calculator import ConfidenceCalculator
from services.escalation_service import EscalationService

# New production services
from services.document_manager import DocumentManager
from services.multi_tier_memory import MultiTierMemory
from services.advanced_retrieval import HybridRetriever
from services.exceptions import *

from config import settings

# Initialize FastAPI
app = FastAPI(
    title="FAQ Support Bot API - Production",
    description="AI-powered FAQ bot with multi-tier memory and hybrid retrieval",
    version="1.0.0-production"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
doc_processor = DocumentProcessor()
vector_store = VectorStoreService()
llm_service = LLMService()
confidence_calc = ConfidenceCalculator()
escalation_service = EscalationService()

# New production services
doc_manager = DocumentManager()
memory = MultiTierMemory()
hybrid_retriever = HybridRetriever()


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User's question")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for history")
    user_id: Optional[str] = Field("anonymous", description="User identifier")
    use_context: bool = Field(True, description="Use conversation history")


class ChatResponse(BaseModel):
    answer_text: str
    confidence: float
    confidence_label: str
    source_reference: List[Dict[str, Any]]
    escalated_to_human: bool
    escalation_request_id: Optional[str] = None
    conversation_id: str
    timestamp: str
    # Production additions
    cached: bool = False
    query_enhancement: Optional[Dict[str, Any]] = None
    retrieval_method: Optional[str] = None
    response_time_ms: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    services: Dict[str, Any]
    uptime_seconds: int


# Global uptime tracker
START_TIME = time.time()


# ============================================================================
# DOCUMENT MANAGEMENT
# ============================================================================

@app.post("/api/v1/documents/upload", tags=["Document Management"])
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document (PDF, Markdown, or TXT).
    Now with proper error handling and document catalog.
    """
    start_time = time.time()
    
    # Validate file size
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max: {settings.MAX_UPLOAD_SIZE_MB}MB"
        )
    
    # Validate file type
    file_ext = os.path.splitext(file.filename)[1].lower()
    allowed = ['.pdf', '.md', '.txt', '.markdown']
    
    if file_ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed)}"
        )
    
    try:
        # Read file
        try:
            content = await file.read()
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to read file: {str(e)}"
            )
        
        doc_id = str(uuid.uuid4())
        
        # Extract text
        try:
            extracted_data = doc_processor.extract_text(
                content=content,
                filename=file.filename,
                file_type=file_ext
            )
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=f"Text extraction failed: {str(e)}"
            )
        
        # Chunk document
        try:
            chunks = doc_processor.chunk_document(
                text=extracted_data['text'],
                metadata={
                    'doc_id': doc_id,
                    'filename': file.filename,
                    'upload_time': datetime.utcnow().isoformat(),
                    **extracted_data['metadata']
                }
            )
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=f"Chunking failed: {str(e)}"
            )
        
        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="No chunks generated. Document may be empty or unreadable."
            )
        
        # Store in vector database
        try:
            vector_ids = await vector_store.store_chunks(chunks)
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Vector store unavailable: {str(e)}"
            )
        
        # Register in document catalog
        await doc_manager.register_document(
            doc_id=doc_id,
            filename=file.filename,
            metadata=extracted_data['metadata'],
            vector_ids=vector_ids
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return JSONResponse(
            status_code=201,
            content={
                "success": True,
                "document_id": doc_id,
                "filename": file.filename,
                "chunks_created": len(chunks),
                "total_vectors": len(vector_ids),
                "file_size_bytes": file_size,
                "processing_time_ms": round(processing_time, 2),
                "message": "Document uploaded and indexed successfully"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )


@app.get("/api/v1/documents/list", tags=["Document Management"])
async def list_documents():
    """
    List all indexed documents (fast - uses Redis catalog).
    """
    try:
        documents = await doc_manager.list_documents()
        return {
            "success": True,
            "documents": documents,
            "total_count": len(documents)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/documents/{doc_id}", tags=["Document Management"])
async def delete_document(doc_id: str):
    """
    Delete a document and all its vectors.
    """
    try:
        success = await doc_manager.delete_document(doc_id, vector_store.index)
        
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {
            "success": True,
            "document_id": doc_id,
            "message": "Document deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# CHAT ENDPOINT (PRODUCTION)
# ============================================================================

@app.post("/api/v1/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Production chat endpoint with:
    - Multi-tier memory (4 levels)
    - Hybrid retrieval (Vector + BM25 + Re-ranking)
    - Semantic caching
    - User profiling
    - Proper error handling
    """
    start_time = time.time()
    
    try:
        conversation_id = request.conversation_id or str(uuid.uuid4())
        user_id = request.user_id or "anonymous"
        
        # Get comprehensive context from multi-tier memory
        context = None
        if request.use_context:
            try:
                context = await memory.get_comprehensive_context(
                    conversation_id=conversation_id,
                    user_id=user_id
                )
            except Exception as e:
                print(f"Memory retrieval failed: {e}")
                context = None
        
        # Hybrid retrieval with caching
        try:
            search_results = await hybrid_retriever.retrieve(
                query=request.question,
                top_k=settings.RETRIEVAL_TOP_K,
                user_context=context
            )
            
            # Check if result was cached
            was_cached = (
                len(search_results) > 0 and
                search_results[0].retrieval_method == 'cached'
            )
            
        except Exception as e:
            print(f"Retrieval failed: {e}")
            # Fallback to simple vector search
            search_results = await vector_store.search(
                query=request.question,
                top_k=settings.RETRIEVAL_TOP_K
            )
            was_cached = False
        
        # Handle no results
        if not search_results:
            answer = "I couldn't find relevant information to answer your question."
            confidence = 0.0
            
            # Store in memory
            if context is not None:
                try:
                    await memory.add_to_working_memory(
                        conversation_id=conversation_id,
                        turn={
                            "question": request.question,
                            "answer": answer,
                            "confidence": 0.0,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    )
                except:
                    pass
            
            # Escalate
            escalation_result = await escalation_service.escalate(
                question=request.question,
                retrieved_snippets=[],
                attempted_answer=answer,
                confidence_score=0.0
            )
            
            response_time = (time.time() - start_time) * 1000
            
            return ChatResponse(
                answer_text=answer,
                confidence=0.0,
                confidence_label="none",
                source_reference=[],
                escalated_to_human=True,
                escalation_request_id=escalation_result.get('escalation_id'),
                conversation_id=conversation_id,
                timestamp=datetime.utcnow().isoformat(),
                cached=False,
                response_time_ms=round(response_time, 2)
            )
        
        # Format search results for LLM
        context_chunks = [
            {
                'id': r.id if hasattr(r, 'id') else r.get('id'),
                'text': r.text if hasattr(r, 'text') else r.get('text'),
                'score': r.score if hasattr(r, 'score') else r.get('score'),
                'metadata': r.metadata if hasattr(r, 'metadata') else r.get('metadata', {})
            }
            for r in search_results
        ]
        
        # Get conversation context for LLM
        conversation_context = ""
        if context and context.get('working_memory'):
            working_memory = context['working_memory'][-3:]  # Last 3 turns
            context_parts = []
            for turn in working_memory:
                context_parts.append(f"User: {turn['question']}")
                context_parts.append(f"Assistant: {turn['answer'][:100]}...")
            conversation_context = "\n".join(context_parts)
        
        # Generate answer
        llm_response = await llm_service.generate_answer(
            question=request.question,
            context_chunks=context_chunks,
            conversation_context=conversation_context
        )
        
        # Calculate confidence
        confidence_score = confidence_calc.calculate_confidence(
            vector_scores=[c['score'] for c in context_chunks],
            llm_confidence=llm_response.get('confidence', 0.5),
            answer_text=llm_response['answer']
        )
        
        confidence_label = confidence_calc.get_confidence_label(confidence_score)
        
        # Source attribution
        source_references = _format_source_references(context_chunks)
        
        # Escalation decision
        should_escalate = confidence_score < settings.ESCALATION_THRESHOLD
        escalation_id = None
        
        if should_escalate:
            escalation_result = await escalation_service.escalate(
                question=request.question,
                retrieved_snippets=[c['text'] for c in context_chunks],
                answer=llm_response['answer'],
                confidence_score=confidence_score
            )
            escalation_id = escalation_result.get('escalation_id')
        
        # Store in memory
        if context is not None:
            try:
                await memory.add_to_working_memory(
                    conversation_id=conversation_id,
                    turn={
                        "question": request.question,
                        "answer": llm_response['answer'],
                        "confidence": confidence_score,
                        "sources": source_references,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
            except Exception as e:
                print(f"Failed to store in memory: {e}")
        
        response_time = (time.time() - start_time) * 1000
        
        return ChatResponse(
            answer_text=llm_response['answer'],
            confidence=round(confidence_score, 3),
            confidence_label=confidence_label,
            source_reference=source_references,
            escalated_to_human=should_escalate,
            escalation_request_id=escalation_id,
            conversation_id=conversation_id,
            timestamp=datetime.utcnow().isoformat(),
            cached=was_cached,
            response_time_ms=round(response_time, 2)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Chat processing failed: {str(e)}"
        )


def _format_source_references(search_results: List[Dict]) -> List[Dict[str, Any]]:
    """Format source references from search results."""
    references = []
    for idx, result in enumerate(search_results, 1):
        metadata = result.get('metadata', {})
        text = result.get('text', '')
        references.append({
            "rank": idx,
            "score": round(result.get('score', 0), 3),
            "source_file": metadata.get('filename', 'unknown'),
            "section": metadata.get('section', 'N/A'),
            "chunk_id": result.get('id', 'unknown'),
            "text_snippet": text[:200] + "..." if len(text) > 200 else text
        })
    return references


# ============================================================================
# HEALTH CHECK (PRODUCTION)
# ============================================================================

@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Comprehensive health check that actually tests services.
    """
    try:
        # Check all services in parallel
        vector_health, llm_health, redis_health, postgres_health = await asyncio.gather(
            vector_store.health_check(),
            asyncio.to_thread(llm_service.health_check),
            _check_redis_health(),
            _check_postgres_health(),
            return_exceptions=True
        )
        
        # Determine overall status
        all_healthy = all(
            (not isinstance(h, Exception) and h.get('status') == 'healthy')
            for h in [vector_health, llm_health, redis_health, postgres_health]
        )
        
        response = {
            "status": "healthy" if all_healthy else "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "vector_store": vector_health if not isinstance(vector_health, Exception) else {'status': 'error', 'error': str(vector_health)},
                "llm": llm_health if not isinstance(llm_health, Exception) else {'status': 'error', 'error': str(llm_health)},
                "redis": redis_health if not isinstance(redis_health, Exception) else {'status': 'error', 'error': str(redis_health)},
                "postgres": postgres_health if not isinstance(postgres_health, Exception) else {'status': 'error', 'error': str(postgres_health)}
            },
            "uptime_seconds": int(time.time() - START_TIME)
        }
        
        status_code = 200 if all_healthy else 503
        
        return JSONResponse(status_code=status_code, content=response)
        
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "services": {},
                "uptime_seconds": int(time.time() - START_TIME)
            }
        )


async def _check_redis_health():
    """Check Redis connection."""
    try:
        import redis
        client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=0
        )
        
        start = time.time()
        client.ping()
        latency = (time.time() - start) * 1000
        
        info = client.info()
        
        return {
            'status': 'healthy',
            'latency_ms': round(latency, 2),
            'connected_clients': info.get('connected_clients'),
            'used_memory_mb': round(info.get('used_memory', 0) / 1024 / 1024, 2)
        }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e)
        }


async def _check_postgres_health():
    """Check PostgreSQL connection."""
    try:
        if memory.pg_pool is None:
            return {'status': 'not_initialized'}
        
        start = time.time()
        async with memory.pg_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        latency = (time.time() - start) * 1000
        
        return {
            'status': 'healthy',
            'latency_ms': round(latency, 2),
            'pool_size': memory.pg_pool.get_size(),
            'pool_free': memory.pg_pool.get_idle_size()
        }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e)
        }


# ============================================================================
# STARTUP/SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize all services on startup."""
    print("🚀 Starting FAQ Support Bot API (Production)...")
    print()
    
    # Initialize vector store
    print("📊 Initializing vector store...")
    await vector_store.initialize()
    print("✅ Vector store ready")
    
    # Initialize memory system
    print("🧠 Initializing multi-tier memory...")
    try:
        await memory.initialize()
        print("✅ Multi-tier memory ready")
    except Exception as e:
        print(f"⚠️  Memory initialization failed: {e}")
        print("   Continuing without memory features...")
    
    # Initialize hybrid retriever
    print("🔍 Initializing hybrid retrieval...")
    try:
        await hybrid_retriever.initialize_bm25()
        print("✅ Hybrid retrieval ready")
    except Exception as e:
        print(f"⚠️  Hybrid retrieval initialization failed: {e}")
        print("   Falling back to vector-only search...")
    
    print()
    print("=" * 60)
    print("🎉 FAQ Support Bot API is ready!")
    print(f"📝 LLM Provider: {settings.LLM_PROVIDER} ({settings.LLM_MODEL})")
    print(f"💾 Vector Store: Pinecone ({settings.PINECONE_INDEX_NAME})")
    print(f"🗄️  Memory: PostgreSQL + Redis (4-tier)")
    print(f"🔍 Retrieval: Hybrid (Vector + BM25 + Re-rank + Cache)")
    print(f"📈 Escalation: {settings.ESCALATION_ENDPOINT}")
    print("=" * 60)
    print()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("👋 Shutting down FAQ Support Bot API...")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )