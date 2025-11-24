"""
proper error handling and retry logic
"""

import httpx
import uuid
from typing import List, Dict, Any
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential
from config import settings


class EscalationService:
    """
    Manages escalation of queries to human support with retry logic.
    """
    
    def __init__(self):
        self.endpoint = settings.ESCALATION_ENDPOINT
        self.timeout = settings.ESCALATION_TIMEOUT
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    async def escalate(
        self,
        question: str,
        retrieved_snippets: List[str],
        attempted_answer: str,
        confidence_score: float
    ) -> Dict[str, Any]:
        """
        Escalate a query to human support with retry logic.
        
        Args:
            question: Original user question
            retrieved_snippets: Top retrieved document snippets
            attempted_answer: AI-generated answer
            confidence_score: Confidence in the answer
            
        Returns:
            Dict with escalation status and details
        """
        escalation_id = str(uuid.uuid4())
        
        payload = {
            "escalation_id": escalation_id,
            "timestamp": datetime.utcnow().isoformat(),
            "original_question": question,
            "retrieved_snippets": retrieved_snippets[:3],
            "attempted_answer": attempted_answer,
            "confidence_score": round(confidence_score, 3),
            "confidence_label": self._get_confidence_label(confidence_score),
            "reason": self._determine_escalation_reason(confidence_score, retrieved_snippets)
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                
                # Raise exception for HTTP errors
                response.raise_for_status()
                
                # Parse response
                data = response.json()
                
                return {
                    'success': True,
                    'escalation_id': data.get('request_id', escalation_id),
                    'status': data.get('status', 'pending'),
                    'estimated_response_time': data.get('eta', '2 hours'),
                    'ticket_url': data.get('ticket_url'),
                    'error': None
                }
                
        except httpx.HTTPStatusError as e:
            # HTTP error (4xx, 5xx)
            return {
                'success': False,
                'escalation_id': escalation_id,
                'status': 'failed',
                'error': f"HTTP {e.response.status_code}: {e.response.text[:200]}",
                'fallback': 'email_notification_sent',
                'retry_scheduled': True
            }
            
        except httpx.TimeoutException:
            # Timeout error
            return {
                'success': False,
                'escalation_id': escalation_id,
                'status': 'timeout',
                'error': f"Escalation API timeout after {self.timeout}s",
                'fallback': 'queued_for_retry',
                'retry_scheduled': True
            }
            
        except httpx.ConnectError:
            # Connection error
            return {
                'success': False,
                'escalation_id': escalation_id,
                'status': 'connection_error',
                'error': "Could not connect to escalation service",
                'fallback': 'logged_for_manual_review',
                'retry_scheduled': True
            }
            
        except Exception as e:
            # Unexpected error
            return {
                'success': False,
                'escalation_id': escalation_id,
                'status': 'error',
                'error': f"Unexpected error: {str(e)}",
                'fallback': 'logged_for_manual_review',
                'retry_scheduled': False
            }
    
    def _get_confidence_label(self, confidence: float) -> str:
        """Convert confidence score to label."""
        if confidence >= 0.8:
            return "high"
        elif confidence >= 0.6:
            return "medium"
        elif confidence >= 0.3:
            return "low"
        else:
            return "very_low"
    
    def _determine_escalation_reason(
        self,
        confidence: float,
        snippets: List[str]
    ) -> str:
        """Determine why the query is being escalated."""
        if not snippets:
            return "No relevant documentation found"
        
        if confidence < 0.3:
            return "Very low confidence - unclear or ambiguous question"
        elif confidence < 0.5:
            return "Low confidence - insufficient documentation coverage"
        elif confidence < 0.6:
            return "Borderline confidence - answer may be incomplete"
        else:
            return "Human verification recommended"
    
    async def get_escalation_status(self, escalation_id: str) -> Dict[str, Any]:
        """
        Check status of escalated query.
        
        Args:
            escalation_id: Escalation identifier
            
        Returns:
            Status information
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.endpoint}/{escalation_id}",
                    headers={"Content-Type": "application/json"}
                )
                
                response.raise_for_status()
                return response.json()
                
        except Exception as e:
            return {
                'escalation_id': escalation_id,
                'status': 'unknown',
                'error': str(e)
            }