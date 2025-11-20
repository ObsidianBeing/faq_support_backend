"""
Escalation Service
Handles escalation of low-confidence queries to human support
"""

import httpx
import uuid
from typing import List, Optional
from datetime import datetime
from config import settings


class EscalationService:
    """
    Manages escalation of queries to human support via API.
    """
    
    def __init__(self):
        self.endpoint = settings.ESCALATION_ENDPOINT
        self.timeout = settings.ESCALATION_TIMEOUT
    
    async def escalate(
        self,
        question: str,
        retrieved_snippets: List[str],
        attempted_answer: str,
        confidence_score: float
    ) -> str:
        """
        Escalate a query to human support.
        
        Sends structured data to the escalation endpoint.
        
        Returns:
            Escalation request ID
        """
        escalation_id = str(uuid.uuid4())
        
        payload = {
            "escalation_id": escalation_id,
            "timestamp": datetime.utcnow().isoformat(),
            "original_question": question,
            "retrieved_snippets": retrieved_snippets[:3],  # Top 3 snippets
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
                
                if response.status_code in [200, 201, 202]:
                    # Successful escalation
                    response_data = response.json()
                    return response_data.get('request_id', escalation_id)
                else:
                    # Log error but return generated ID
                    print(f"Escalation endpoint returned status {response.status_code}")
                    return escalation_id
                    
        except httpx.TimeoutException:
            print(f"Escalation timeout after {self.timeout}s")
            return escalation_id
        except Exception as e:
            print(f"Escalation failed: {str(e)}")
            return escalation_id
    
    def _get_confidence_label(self, confidence: float) -> str:
        """
        Convert confidence score to label.
        """
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
        """
        Determine why the query is being escalated.
        """
        if not snippets:
            return "No relevant documentation found"
        
        if confidence < 0.3:
            return "Very low confidence - unclear or ambiguous question"
        elif confidence < 0.6:
            return "Low confidence - insufficient or unclear documentation"
        else:
            return "Borderline confidence - human verification recommended"