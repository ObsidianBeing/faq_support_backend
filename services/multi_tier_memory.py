"""
Multi-Tier Memory System
L1: Working Memory (Redis) - Last 10 turns, 1 hour
L2: Session Memory (Redis) - Full session, 24 hours
L3: User Memory (PostgreSQL) - Cross-session, 90 days
L4: Semantic Memory (Pinecone) - Forever
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import redis
import asyncpg
from pinecone import Pinecone
import openai
from config import settings


@dataclass
class UserProfile:
    user_id: str
    technical_level: str  # beginner, intermediate, expert
    preferred_language: str
    response_style: str  # concise, detailed, friendly
    common_topics: List[str]
    avg_confidence_threshold: float
    created_at: datetime
    updated_at: datetime


class MultiTierMemory:
    """
    Hierarchical memory system with automatic summarization.
    """
    
    def __init__(self):
        # L1 & L2: Redis
        self.redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=0,
            decode_responses=True
        )
        
        # L3: PostgreSQL (async)
        self.pg_pool = None
        
        # L4: Pinecone for semantic memory
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.memory_index = self.pc.Index(settings.PINECONE_MEMORY_INDEX_NAME)

        
        # OpenAI for summarization
        self.openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
    
    async def initialize(self):
        """Initialize database connections."""
        self.pg_pool = await asyncpg.create_pool(
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT,
            database=settings.POSTGRES_DB,
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD,
            min_size=10,
            max_size=50
        )
        
        await self._create_tables()
    
    async def _create_tables(self):
        """Create necessary PostgreSQL tables."""
        async with self.pg_pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id VARCHAR(255) PRIMARY KEY,
                    technical_level VARCHAR(50),
                    preferred_language VARCHAR(10),
                    response_style VARCHAR(50),
                    common_topics JSONB,
                    avg_confidence_threshold FLOAT,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                );
                
                CREATE TABLE IF NOT EXISTS conversation_sessions (
                    session_id VARCHAR(255) PRIMARY KEY,
                    user_id VARCHAR(255),
                    started_at TIMESTAMP,
                    ended_at TIMESTAMP,
                    total_turns INT,
                    topics JSONB,
                    summary TEXT,
                    avg_confidence FLOAT,
                    avg_satisfaction FLOAT
                );
                
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    turn_id VARCHAR(255) PRIMARY KEY,
                    session_id VARCHAR(255),
                    user_id VARCHAR(255),
                    timestamp TIMESTAMP,
                    question TEXT,
                    answer TEXT,
                    confidence FLOAT,
                    sources JSONB,
                    feedback_rating INT,
                    feedback_comment TEXT
                );
                
                CREATE INDEX IF NOT EXISTS idx_user_sessions 
                    ON conversation_sessions(user_id, started_at DESC);
                
                CREATE INDEX IF NOT EXISTS idx_session_turns 
                    ON conversation_turns(session_id, timestamp);
            """)
    
    # ========================================================================
    # L1: WORKING MEMORY (Redis, 1 hour TTL)
    # ========================================================================
    
    def get_working_memory(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get last 10 turns from working memory."""
        key = f"working_memory:{conversation_id}"
        data = self.redis_client.get(key)
        
        if data:
            return json.loads(data)
        return []
    
    def add_to_working_memory(
        self,
        conversation_id: str,
        turn: Dict[str, Any]
    ):
        """Add turn to working memory."""
        key = f"working_memory:{conversation_id}"
        
        # Get existing
        turns = self.get_working_memory(conversation_id)
        turns.append(turn)
        
        # Keep only last 10
        turns = turns[-10:]
        
        # Store with 1 hour TTL
        self.redis_client.setex(
            key,
            3600,
            json.dumps(turns)
        )
    
    # ========================================================================
    # L2: SESSION MEMORY (Redis, 24 hour TTL)
    # ========================================================================
    
    async def get_session_summary(self, conversation_id: str) -> Optional[Dict]:
        """Get compressed session summary."""
        key = f"session_summary:{conversation_id}"
        data = self.redis_client.get(key)
        
        if data:
            return json.loads(data)
        return None
    
    async def create_session_summary(
        self,
        conversation_id: str,
        turns: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Use LLM to create dense summary of conversation.
        Compress 100 turns into 2KB summary.
        """
        if not turns:
            return {}
        
        # Format conversation for summarization
        conversation_text = self._format_for_summarization(turns)
        
        prompt = f"""Analyze this customer support conversation and create a structured summary:

Conversation:
{conversation_text}

Provide a JSON summary with:
1. main_topic: Primary subject discussed
2. key_entities: Products/features mentioned
3. issues_raised: List of problems discussed
4. resolution_status: "resolved", "pending", "escalated"
5. user_sentiment: "positive", "neutral", "negative"
6. action_items: Any follow-ups needed
7. context_for_next_session: Brief context if conversation continues

Output only valid JSON, no markdown."""

        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a conversation analyzer. Output only JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        summary = json.loads(response.choices[0].message.content)
        
        # Store with 24 hour TTL
        key = f"session_summary:{conversation_id}"
        self.redis_client.setex(
            key,
            86400,  # 24 hours
            json.dumps(summary)
        )
        
        return summary
    
    def _format_for_summarization(self, turns: List[Dict]) -> str:
        """Format turns for LLM summarization."""
        lines = []
        for i, turn in enumerate(turns, 1):
            lines.append(f"Turn {i}:")
            lines.append(f"Q: {turn['question']}")
            lines.append(f"A: {turn['answer'][:200]}...")
            lines.append("")
        
        return "\n".join(lines)
    
    # ========================================================================
    # L3: USER MEMORY (PostgreSQL, 90 days)
    # ========================================================================
    
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Retrieve user profile from long-term storage."""
        async with self.pg_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM user_profiles WHERE user_id = $1",
                user_id
            )
            
            if row:
                return UserProfile(
                    user_id=row['user_id'],
                    technical_level=row['technical_level'],
                    preferred_language=row['preferred_language'],
                    response_style=row['response_style'],
                    common_topics=row['common_topics'],
                    avg_confidence_threshold=row['avg_confidence_threshold'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                )
            return None
    
    async def create_or_update_user_profile(
        self,
        user_id: str,
        profile_data: Dict[str, Any]
    ):
        """Create or update user profile."""
        async with self.pg_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO user_profiles 
                (user_id, technical_level, preferred_language, response_style, 
                 common_topics, avg_confidence_threshold, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, NOW())
                ON CONFLICT (user_id) 
                DO UPDATE SET
                    technical_level = EXCLUDED.technical_level,
                    preferred_language = EXCLUDED.preferred_language,
                    response_style = EXCLUDED.response_style,
                    common_topics = EXCLUDED.common_topics,
                    avg_confidence_threshold = EXCLUDED.avg_confidence_threshold,
                    updated_at = NOW()
            """,
                user_id,
                profile_data.get('technical_level', 'intermediate'),
                profile_data.get('preferred_language', 'en'),
                profile_data.get('response_style', 'balanced'),
                json.dumps(profile_data.get('common_topics', [])),
                profile_data.get('avg_confidence_threshold', 0.6)
            )
    
    async def store_conversation_session(
        self,
        session_id: str,
        user_id: str,
        summary: Dict[str, Any],
        turns: List[Dict[str, Any]]
    ):
        """Store completed session in PostgreSQL."""
        async with self.pg_pool.acquire() as conn:
            # Store session
            await conn.execute("""
                INSERT INTO conversation_sessions 
                (session_id, user_id, started_at, ended_at, total_turns, 
                 topics, summary, avg_confidence)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
                session_id,
                user_id,
                turns[0]['timestamp'],
                turns[-1]['timestamp'],
                len(turns),
                json.dumps(summary.get('key_entities', [])),
                summary.get('context_for_next_session', ''),
                sum(t.get('confidence', 0) for t in turns) / len(turns)
            )
            
            # Store individual turns
            for turn in turns:
                await conn.execute("""
                    INSERT INTO conversation_turns 
                    (turn_id, session_id, user_id, timestamp, question, 
                     answer, confidence, sources)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                    turn.get('turn_id', f"{session_id}_{turn['timestamp']}"),
                    session_id,
                    user_id,
                    turn['timestamp'],
                    turn['question'],
                    turn['answer'],
                    turn.get('confidence', 0.0),
                    json.dumps(turn.get('sources', []))
                )
    
    async def get_user_history(
        self,
        user_id: str,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get user's conversation history."""
        async with self.pg_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT session_id, started_at, topics, summary, avg_confidence
                FROM conversation_sessions
                WHERE user_id = $1 
                  AND started_at > NOW() - INTERVAL '$2 days'
                ORDER BY started_at DESC
                LIMIT 20
            """, user_id, days)
            
            return [dict(row) for row in rows]
    
    # ========================================================================
    # L4: SEMANTIC MEMORY (Pinecone, Forever)
    # ========================================================================
    
    async def store_semantic_memory(
        self,
        conversation_id: str,
        summary: Dict[str, Any]
    ):
        """Store conversation in semantic memory for future retrieval."""
        # Create embedding of summary
        summary_text = self._create_summary_text(summary)
        
        embedding_response = self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=summary_text
        )
        embedding = embedding_response.data[0].embedding
        
        # Store in Pinecone
        self.memory_index.upsert(vectors=[{
            'id': f"memory_{conversation_id}",
            'values': embedding,
            'metadata': {
                'conversation_id': conversation_id,
                'main_topic': summary.get('main_topic', ''),
                'entities': summary.get('key_entities', []),
                'sentiment': summary.get('user_sentiment', ''),
                'resolution': summary.get('resolution_status', ''),
                'timestamp': datetime.utcnow().isoformat()
            }
        }])
    
    async def retrieve_similar_conversations(
        self,
        current_summary: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Find similar past conversations for context."""
        # Create embedding
        embedding_response = self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=current_summary
        )
        embedding = embedding_response.data[0].embedding
        
        # Search semantic memory
        results = self.memory_index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        return [
            {
                'conversation_id': match.metadata['conversation_id'],
                'topic': match.metadata['main_topic'],
                'similarity': match.score
            }
            for match in results.matches
        ]
    
    def _create_summary_text(self, summary: Dict) -> str:
        """Convert summary to searchable text."""
        parts = [
            f"Topic: {summary.get('main_topic', '')}",
            f"Entities: {', '.join(summary.get('key_entities', []))}",
            f"Issues: {', '.join(summary.get('issues_raised', []))}",
            f"Sentiment: {summary.get('user_sentiment', '')}",
            f"Resolution: {summary.get('resolution_status', '')}"
        ]
        return " | ".join(parts)
    
    # ========================================================================
    # AUTOMATIC PROFILE LEARNING
    # ========================================================================
    
    async def learn_from_interactions(
        self,
        user_id: str,
        turns: List[Dict[str, Any]]
    ):
        """
        Automatically learn user preferences from interactions.
        """
        # Analyze question complexity → technical level
        avg_question_length = sum(len(t['question'].split()) for t in turns) / len(turns)
        technical_terms = self._count_technical_terms(turns)
        
        technical_level = 'beginner'
        if avg_question_length > 15 and technical_terms > 5:
            technical_level = 'expert'
        elif avg_question_length > 10 or technical_terms > 2:
            technical_level = 'intermediate'
        
        # Extract common topics
        topics = []
        for turn in turns:
            if 'metadata' in turn and 'topics' in turn['metadata']:
                topics.extend(turn['metadata']['topics'])
        
        common_topics = list(set(topics))
        
        # Analyze preferred response style
        feedback = [t.get('feedback_rating', 0) for t in turns if 'feedback_rating' in t]
        avg_satisfaction = sum(feedback) / len(feedback) if feedback else 0
        
        # Determine response style based on satisfaction correlation
        # (simplified - in production, use ML)
        response_style = 'balanced'
        if avg_satisfaction > 4.5:
            response_style = 'current_style'  # They like current style
        
        # Update profile
        await self.create_or_update_user_profile(user_id, {
            'technical_level': technical_level,
            'common_topics': common_topics,
            'response_style': response_style,
            'avg_confidence_threshold': 0.6
        })
    
    def _count_technical_terms(self, turns: List[Dict]) -> int:
        """Count technical terms in questions."""
        technical_keywords = {
            'api', 'endpoint', 'authentication', 'oauth', 'jwt', 'ssl',
            'database', 'query', 'index', 'schema', 'deployment', 'server',
            'configuration', 'integration', 'webhook', 'callback'
        }
        
        count = 0
        for turn in turns:
            question_lower = turn['question'].lower()
            count += sum(1 for term in technical_keywords if term in question_lower)
        
        return count
    
    # ========================================================================
    # UNIFIED MEMORY RETRIEVAL
    # ========================================================================
    
    async def get_comprehensive_context(
        self,
        conversation_id: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get all relevant context from all memory tiers.
        """
        context = {
            'working_memory': [],
            'session_summary': None,
            'user_profile': None,
            'user_history': [],
            'similar_conversations': []
        }
        
        # L1: Working memory (last 10 turns)
        context['working_memory'] = self.get_working_memory(conversation_id)
        
        # L2: Session summary
        context['session_summary'] = await self.get_session_summary(conversation_id)
        
        if user_id:
            # L3: User profile and history
            context['user_profile'] = await self.get_user_profile(user_id)
            context['user_history'] = await self.get_user_history(user_id, days=30)
            
            # L4: Similar conversations
            if context['session_summary']:
                summary_text = self._create_summary_text(context['session_summary'])
                context['similar_conversations'] = await self.retrieve_similar_conversations(
                    summary_text,
                    top_k=3
                )
        
        return context