# FAQ Support Bot API

Production-ready AI-powered FAQ support bot with document ingestion, vector search, confidence-based escalation, and source attribution.

## 🚀 Features

- **Multi-format document support**: PDF, Markdown, and TXT
- **Vector-based semantic search**: Using Pinecone for efficient retrieval
- **Configurable LLM providers**: OpenAI (default) or Anthropic
- **Hybrid confidence scoring**: Combines vector similarity, LLM confidence, and answer quality
- **Automatic escalation**: Low-confidence queries escalated to human support
- **Source attribution**: All answers include references to source documents
- **Configurable chunking**: Adjust chunk size and overlap via API
- **RESTful API**: Fully HTTP-based, tested with any HTTP client

## 📋 Prerequisites

- Python 3.9+
- OpenAI API key
- Pinecone API key
- (Optional) Anthropic API key

## 🛠️ Installation

### 1. Clone and Setup

```bash
# Create project directory
mkdir faq-support-bot
cd faq-support-bot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```env
OPENAI_API_KEY=sk-your-openai-key-here
PINECONE_API_KEY=your-pinecone-key-here

# Optional
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
```

### 3. Project Structure

```
faq-support-bot/
├── main.py                 # FastAPI application
├── config.py              # Configuration management
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (create this)
├── .env.example          # Environment template
├── services/
│   ├── __init__.py
│   ├── document_processor.py    # Document parsing & chunking
│   ├── vector_store.py          # Pinecone operations
│   ├── llm_service.py           # LLM interactions
│   ├── confidence_calculator.py # Confidence scoring
│   └── escalation_service.py    # Human escalation
└── README.md
```

Create the services directory:

```bash
mkdir services
touch services/__init__.py
```

## 🚀 Quick Start

### 1. Start the Server

```bash
python main.py
```

The API will be available at `http://localhost:8000`

API documentation: `http://localhost:8000/docs`

### 2. Upload a Document

```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your-faq-document.pdf"
```

### 3. Ask Questions

```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I reset my password?"
  }'
```

## 📚 API Endpoints

### Document Management

#### Upload Document
```http
POST /api/v1/documents/upload
Content-Type: multipart/form-data

file: <PDF/MD/TXT file>
```

**Response:**
```json
{
  "success": true,
  "document_id": "uuid",
  "filename": "faq.pdf",
  "chunks_created": 15,
  "vector_ids": ["id1", "id2", ...],
  "total_vectors": 15
}
```

#### List Documents
```http
GET /api/v1/documents/list
```

#### Delete Document
```http
DELETE /api/v1/documents/{doc_id}
```

#### Configure Chunking
```http
POST /api/v1/documents/configure-chunking
Content-Type: application/json

{
  "chunk_size": 1000,
  "chunk_overlap": 250
}
```

### Chat Interface

#### Ask Question
```http
POST /api/v1/chat
Content-Type: application/json

{
  "question": "Your question here",
  "conversation_id": "optional-uuid"
}
```

**Response:**
```json
{
  "answer_text": "Based on the documentation...",
  "confidence": 0.85,
  "confidence_label": "high",
  "source_reference": [
    {
      "rank": 1,
      "score": 0.92,
      "source_file": "faq.pdf",
      "section": "Password Reset",
      "chunk_id": "doc_123_chunk_5",
      "text_snippet": "To reset your password..."
    }
  ],
  "escalated_to_human": false,
  "escalation_request_id": null,
  "conversation_id": "uuid",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Configuration

#### Get Configuration
```http
GET /api/v1/config
```

#### Update LLM Provider
```http
POST /api/v1/config/llm?provider=openai&model=gpt-4
```

### System

#### Health Check
```http
GET /api/v1/health
```

**Response:**
```json
{
  "status": "healthy",
  "vector_store_connected": true,
  "llm_service_available": true,
  "documents_indexed": 150
}
```

## 🧪 Testing with HTTP Clients

### Using cURL

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Upload document
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@docs/faq.pdf"

# Ask question
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the return policies?"}'
```

### Using HTTPie

```bash
# Health check
http GET localhost:8000/api/v1/health

# Upload document
http --form POST localhost:8000/api/v1/documents/upload file@docs/faq.pdf

# Ask question
http POST localhost:8000/api/v1/chat question="What are the return policies?"
```

### Using Postman

1. Import OpenAPI spec from `http://localhost:8000/openapi.json`
2. Use the interactive documentation at `http://localhost:8000/docs`

## 🔧 Configuration Options

### LLM Providers

**OpenAI (Default):**
```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini  # or gpt-4, gpt-3.5-turbo
```

**Anthropic:**
```env
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-5-sonnet-20241022
```

### Chunking Strategy

Adjust via environment or API:
```env
DEFAULT_CHUNK_SIZE=800      # Tokens per chunk
DEFAULT_CHUNK_OVERLAP=200   # Overlap between chunks
```

### Confidence Thresholds

```env
ESCALATION_THRESHOLD=0.6           # Below this = escalate
MIN_VECTOR_SCORE=0.5              # Minimum similarity
HIGH_CONFIDENCE_VECTOR_SCORE=0.85  # High-quality match
```

### Retrieval

```env
RETRIEVAL_TOP_K=5  # Number of chunks to retrieve
```

## 📊 How It Works

### 1. Document Ingestion
- Upload PDF/MD/TXT files
- Extract text and metadata
- Split into chunks with configurable size
- Generate embeddings using OpenAI
- Store in Pinecone vector database

### 2. Query Processing
- User asks a question
- Generate query embedding
- Vector similarity search in Pinecone
- Retrieve top-K most relevant chunks

### 3. Answer Generation
- Construct prompt with retrieved context
- Call LLM (OpenAI or Anthropic)
- LLM generates answer with confidence rating

### 4. Confidence Calculation
**Hybrid scoring (0.0 - 1.0):**
- **40%** Vector similarity scores (semantic match quality)
- **40%** LLM self-reported confidence
- **20%** Answer quality heuristics (length, specificity, uncertainty)

**Confidence Labels:**
- `high` (0.8+): Strong match, clear answer
- `medium` (0.6-0.8): Good match, some uncertainty
- `low` (0.3-0.6): Weak match, unclear answer
- `very_low` (0.0-0.3): No match or completely uncertain

### 5. Escalation
If confidence < threshold (default 0.6):
- Call escalation API endpoint
- Send structured payload with question, snippets, answer, confidence
- Mark response as escalated
- Return escalation ID to user

### 6. Source Attribution
Every answer includes:
- Source filename
- Section/page reference
- Relevance score
- Text snippet
- Chunk identifier

## 🎯 Confidence Calculation Details

### Vector Confidence Component (40%)
- Average of top 3 similarity scores
- Bonus for low variance (consistency)
- Bonus for multiple high-quality matches (>0.85)

### LLM Confidence Component (40%)
- Model explicitly rates its own confidence
- Extracted from response format `[CONFIDENCE: X.XX]`
- Fallback heuristics if not provided

### Answer Quality Component (20%)
Heuristics based on:
- Answer length (penalize too short/long)
- Uncertainty phrases ("I don't know", "unclear")
- Hedging words ("might", "possibly")
- Specificity indicators ("according to", "specifically")

## 🔄 Escalation Flow

```
┌─────────────┐
│ User Query  │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│ Vector Search   │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ LLM Generation  │
└──────┬──────────┘
       │
       ▼
┌──────────────────────┐
│ Confidence < 0.6?    │
└──────┬───────────────┘
       │
       ├─── YES ──▶ POST /escalate
       │              {
       │                "question": "...",
       │                "snippets": [...],
       │                "confidence": 0.45
       │              }
       │
       └─── NO ───▶ Return answer
```

## 🚨 Current Limitations

1. **Pinecone Index Management**
   - Currently creates a single index for all documents
   - No namespace separation
   - **Future**: Multi-tenant support with namespaces

2. **Document Listing**
   - Uses query workaround (limited to 10k vectors)
   - **Future**: Maintain separate document metadata store

3. **Conversation Memory**
   - No multi-turn conversation tracking
   - Each query is independent
   - **Future**: Add conversation history storage

4. **Chunking Strategy**
   - Fixed token-based chunking
   - No semantic chunking
   - **Future**: Implement recursive character splitting with semantic boundaries

5. **Escalation Endpoint**
   - Currently a dummy endpoint
   - **Future**: Integration with ticketing systems (Zendesk, Jira, etc.)

6. **Rate Limiting**
   - No built-in rate limiting
   - **Future**: Add per-user/IP rate limits

7. **Authentication**
   - No authentication implemented
   - **Future**: JWT/API key authentication

8. **Caching**
   - No caching of embeddings or responses
   - **Future**: Redis cache for frequent queries

## 🔐 Production Considerations

### Security
- Add API key authentication
- Implement rate limiting
- Validate file uploads (size, content)
- Sanitize user inputs
- Use HTTPS in production

### Performance
- Implement caching (Redis)
- Batch embedding generation
- Connection pooling for Pinecone
- Async operations throughout

### Monitoring
- Add structured logging
- Implement metrics (Prometheus)
- Error tracking (Sentry)
- Performance monitoring (APM)

### Deployment
- Use production ASGI server (Gunicorn + Uvicorn)
- Container deployment (Docker)
- Orchestration (Kubernetes)
- Load balancing
- Auto-scaling

## 📝 Example Use Cases

1. **Product FAQ Bot**: Upload product manual, answer customer questions
2. **Internal Knowledge Base**: Company policies, procedures, documentation
3. **Technical Support**: Software documentation, troubleshooting guides
4. **Legal Document Q&A**: Terms of service, contracts, compliance docs
5. **Educational Content**: Course materials, study guides, references

## 🤝 Contributing

This is a demonstration project. For production use, consider:
- Adding comprehensive tests (pytest)
- Implementing CI/CD pipelines
- Adding monitoring and observability
- Implementing proper error recovery
- Adding data validation and sanitization

## 📄 License

MIT License - feel free to use and modify for your needs.

## 🆘 Support

For issues or questions:
1. Check the `/api/v1/health` endpoint
2. Review logs for error messages
3. Verify API keys in `.env`
4. Ensure Pinecone index is created
5. Test with simple queries first

## 🎉 Quick Test

```bash
# 1. Start server
python main.py

# 2. Check health
curl http://localhost:8000/api/v1/health

# 3. Upload test document
echo "FAQ: How to reset password? Answer: Click 'Forgot Password' on login page." > test.txt
curl -X POST http://localhost:8000/api/v1/documents/upload -F "file=@test.txt"

# 4. Ask question
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I reset my password?"}'
```

You should see a high-confidence response with source attribution!