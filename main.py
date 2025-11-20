"""
FAQ Support Bot - Main FastAPI Application
Production-ready chatbot with document ingestion, vector search, and escalation
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import os
import uvicorn
from datetime import datetime
import uuid

from services.document_processor import DocumentProcessor
from services.vector_store import VectorStoreService
from services.llm_service import LLMService
from services.confidence_calculator import ConfidenceCalculator
from services.escalation_service import EscalationService
from config import settings

app = FastAPI(
    title="FAQ Support Bot API",
    description="AI-powered FAQ bot with confidence-based escalation",
    version="1.0.0"
)

# Initialize services
doc_processor = DocumentProcessor()
vector_store = VectorStoreService()
llm_service = LLMService()
confidence_calc = ConfidenceCalculator()
escalation_service = EscalationService()


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ConfigureChunkingRequest(BaseModel):
    chunk_size: int = Field(default=800, ge=100, le=2000, description="Token size for text chunks")
    chunk_overlap: int = Field(default=200, ge=0, le=500, description="Overlap between chunks")


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User's question")
    conversation_id: Optional[str] = Field(None, description="Optional conversation tracking ID")


class ChatResponse(BaseModel):
    answer_text: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    confidence_label: str
    source_reference: List[Dict[str, Any]]
    escalated_to_human: bool
    escalation_request_id: Optional[str] = None
    conversation_id: str
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    vector_store_connected: bool
    llm_service_available: bool
    documents_indexed: int


# ============================================================================
# PHASE 1: DOCUMENT INGESTION
# ============================================================================

@app.post("/api/v1/documents/upload", tags=["Document Management"])
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload and process a document (PDF, Markdown, or TXT).
    Extracts text, chunks it, and stores in vector database.
    """
    try:
        # Validate file type
        allowed_extensions = ['.pdf', '.md', '.txt', '.markdown']
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Read file content
        content = await file.read()
        
        # Process document
        doc_id = str(uuid.uuid4())
        
        # Extract text and metadata
        extracted_data = doc_processor.extract_text(
            content=content,
            filename=file.filename,
            file_type=file_ext
        )
        
        # Chunk the document
        chunks = doc_processor.chunk_document(
            text=extracted_data['text'],
            metadata={
                'doc_id': doc_id,
                'filename': file.filename,
                'upload_time': datetime.utcnow().isoformat(),
                **extracted_data['metadata']
            }
        )
        
        # Store in vector database
        vector_ids = await vector_store.store_chunks(chunks)
        
        return JSONResponse(
            status_code=201,
            content={
                "success": True,
                "document_id": doc_id,
                "filename": file.filename,
                "chunks_created": len(chunks),
                "vector_ids": vector_ids[:5],  # Return first 5 IDs as sample
                "total_vectors": len(vector_ids),
                "message": "Document uploaded and indexed successfully"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")


@app.post("/api/v1/documents/configure-chunking", tags=["Configuration"])
async def configure_chunking(config: ConfigureChunkingRequest):
    """
    Configure chunking parameters for document processing.
    """
    try:
        doc_processor.configure_chunking(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        
        return {
            "success": True,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "message": "Chunking configuration updated"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/documents/list", tags=["Document Management"])
async def list_documents():
    """
    List all indexed documents.
    """
    try:
        documents = await vector_store.list_documents()
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
    Delete a document and all its vectors from the index.
    """
    try:
        deleted_count = await vector_store.delete_document(doc_id)
        
        if deleted_count == 0:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {
            "success": True,
            "document_id": doc_id,
            "vectors_deleted": deleted_count,
            "message": "Document deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# PHASE 3-6: QUERY, RETRIEVAL, ANSWER GENERATION, CONFIDENCE & ESCALATION
# ============================================================================

@app.post("/api/v1/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Main chat endpoint. Handles question answering with confidence-based escalation.
    
    Process:
    1. Vector similarity search for relevant chunks
    2. Generate answer using LLM with retrieved context
    3. Calculate confidence score (vector similarity + LLM confidence)
    4. Escalate to human if confidence below threshold
    5. Return structured response with source attribution
    """
    try:
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        # PHASE 3: Vector Similarity Search
        search_results = await vector_store.search(
            query=request.question,
            top_k=settings.RETRIEVAL_TOP_K
        )
        
        if not search_results:
            # No relevant documents found
            return ChatResponse(
                answer_text="I couldn't find any relevant information in the documentation to answer your question.",
                confidence=0.0,
                confidence_label="none",
                source_reference=[],
                escalated_to_human=True,
                escalation_request_id=await _escalate_query(
                    question=request.question,
                    retrieved_chunks=[],
                    answer="No relevant information found",
                    confidence=0.0
                ),
                conversation_id=conversation_id,
                timestamp=datetime.utcnow().isoformat()
            )
        
        # PHASE 4: Answer Generation
        llm_response = await llm_service.generate_answer(
            question=request.question,
            context_chunks=search_results
        )
        
        # PHASE 5: Confidence Calculation
        confidence_score = confidence_calc.calculate_confidence(
            vector_scores=[chunk['score'] for chunk in search_results],
            llm_confidence=llm_response.get('confidence', 0.5),
            answer_text=llm_response['answer']
        )
        
        confidence_label = confidence_calc.get_confidence_label(confidence_score)
        
        # PHASE 6: Source Attribution
        source_references = _format_source_references(search_results)
        
        # Escalation Decision
        should_escalate = confidence_score < settings.ESCALATION_THRESHOLD
        escalation_id = None
        
        if should_escalate:
            escalation_id = await _escalate_query(
                question=request.question,
                retrieved_chunks=search_results,
                answer=llm_response['answer'],
                confidence=confidence_score
            )
        
        return ChatResponse(
            answer_text=llm_response['answer'],
            confidence=round(confidence_score, 3),
            confidence_label=confidence_label,
            source_reference=source_references,
            escalated_to_human=should_escalate,
            escalation_request_id=escalation_id,
            conversation_id=conversation_id,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


async def _escalate_query(
    question: str,
    retrieved_chunks: List[Dict],
    answer: str,
    confidence: float
) -> str:
    """
    Internal function to handle escalation to human support.
    """
    try:
        escalation_id = await escalation_service.escalate(
            question=question,
            retrieved_snippets=[chunk['text'] for chunk in retrieved_chunks],
            attempted_answer=answer,
            confidence_score=confidence
        )
        return escalation_id
    except Exception as e:
        # Log error but don't fail the request
        print(f"Escalation failed: {str(e)}")
        return f"escalation_failed_{uuid.uuid4()}"


def _format_source_references(search_results: List[Dict]) -> List[Dict[str, Any]]:
    """
    Format source references from search results.
    """
    references = []
    for idx, result in enumerate(search_results, 1):
        metadata = result.get('metadata', {})
        references.append({
            "rank": idx,
            "score": round(result['score'], 3),
            "source_file": metadata.get('filename', 'unknown'),
            "section": metadata.get('section', 'N/A'),
            "chunk_id": result.get('id', 'unknown'),
            "text_snippet": result['text'][:200] + "..." if len(result['text']) > 200 else result['text']
        })
    return references


# ============================================================================
# CONFIGURATION & HEALTH ENDPOINTS
# ============================================================================

@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint to verify all services are operational.
    """
    try:
        vector_store_ok = await vector_store.health_check()
        llm_ok = llm_service.health_check()
        doc_count = await vector_store.get_document_count()
        
        return HealthResponse(
            status="healthy" if (vector_store_ok and llm_ok) else "degraded",
            vector_store_connected=vector_store_ok,
            llm_service_available=llm_ok,
            documents_indexed=doc_count
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            vector_store_connected=False,
            llm_service_available=False,
            documents_indexed=0
        )


@app.get("/api/v1/config", tags=["Configuration"])
async def get_configuration():
    """
    Get current system configuration.
    """
    return {
        "llm_provider": settings.LLM_PROVIDER,
        "llm_model": settings.LLM_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
        "chunk_size": doc_processor.chunk_size,
        "chunk_overlap": doc_processor.chunk_overlap,
        "retrieval_top_k": settings.RETRIEVAL_TOP_K,
        "escalation_threshold": settings.ESCALATION_THRESHOLD,
        "escalation_endpoint": settings.ESCALATION_ENDPOINT
    }


@app.post("/api/v1/config/llm", tags=["Configuration"])
async def update_llm_config(
    provider: str = "openai",
    model: Optional[str] = None
):
    """
    Update LLM provider and model.
    """
    try:
        llm_service.configure(provider=provider, model=model)
        return {
            "success": True,
            "provider": provider,
            "model": llm_service.model,
            "message": "LLM configuration updated"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# STARTUP/SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """
    Initialize services on startup.
    """
    print("🚀 Starting FAQ Support Bot API...")
    await vector_store.initialize()
    print("✅ Vector store initialized")
    print(f"✅ LLM Provider: {settings.LLM_PROVIDER} ({settings.LLM_MODEL})")
    print(f"✅ Escalation endpoint: {settings.ESCALATION_ENDPOINT}")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Cleanup on shutdown.
    """
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