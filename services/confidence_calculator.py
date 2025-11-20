"""
Confidence Calculator Service
Combines vector similarity scores and LLM confidence for final confidence rating
"""

from typing import List
from config import settings


class ConfidenceCalculator:
    """
    Calculates confidence scores using multiple signals.
    """
    
    def __init__(self):
        self.high_threshold = settings.HIGH_CONFIDENCE_VECTOR_SCORE
        self.min_threshold = settings.MIN_VECTOR_SCORE
    
    def calculate_confidence(
        self,
        vector_scores: List[float],
        llm_confidence: float,
        answer_text: str
    ) -> float:
        """
        Calculate final confidence score combining multiple signals.
        
        Formula:
        - Vector score component (40%): Average of top vector similarity scores
        - LLM confidence component (40%): Model's self-reported confidence
        - Answer quality component (20%): Heuristics on answer text
        
        Returns:
            Float between 0.0 and 1.0
        """
        # Component 1: Vector Similarity Score (40% weight)
        vector_confidence = self._calculate_vector_confidence(vector_scores)
        
        # Component 2: LLM Self-Reported Confidence (40% weight)
        llm_conf_normalized = max(0.0, min(1.0, llm_confidence))
        
        # Component 3: Answer Quality Heuristics (20% weight)
        quality_confidence = self._calculate_answer_quality(answer_text)
        
        # Weighted combination
        final_confidence = (
            vector_confidence * 0.4 +
            llm_conf_normalized * 0.4 +
            quality_confidence * 0.2
        )
        
        return max(0.0, min(1.0, final_confidence))
    
    def _calculate_vector_confidence(self, scores: List[float]) -> float:
        """
        Calculate confidence based on vector similarity scores.
        
        Strategy:
        - High scores (>0.85) indicate strong semantic match
        - Multiple high scores indicate consistent evidence
        - Low variance indicates agreement across sources
        """
        if not scores:
            return 0.0
        
        # Take top 3 scores for analysis
        top_scores = sorted(scores, reverse=True)[:3]
        
        # Average of top scores
        avg_score = sum(top_scores) / len(top_scores)
        
        # Bonus for consistency (low variance)
        if len(top_scores) > 1:
            variance = sum((s - avg_score) ** 2 for s in top_scores) / len(top_scores)
            consistency_bonus = max(0, 0.1 - variance)  # Up to 0.1 bonus
        else:
            consistency_bonus = 0
        
        # Bonus for multiple high-quality matches
        high_quality_matches = sum(1 for s in top_scores if s > self.high_threshold)
        match_bonus = high_quality_matches * 0.05  # 0.05 per high-quality match
        
        confidence = avg_score + consistency_bonus + match_bonus
        
        return max(0.0, min(1.0, confidence))
    
    def _calculate_answer_quality(self, answer_text: str) -> float:
        """
        Heuristic-based confidence from answer characteristics.
        
        Factors:
        - Length (too short or too long may indicate issues)
        - Presence of uncertainty phrases
        - Presence of specific details
        - Coherence indicators
        """
        answer_lower = answer_text.lower()
        confidence = 0.7  # Start with neutral
        
        # Length checks
        word_count = len(answer_text.split())
        
        if word_count < 10:
            confidence -= 0.2  # Very short answers may be incomplete
        elif word_count > 300:
            confidence -= 0.1  # Very long answers may be unfocused
        elif 30 <= word_count <= 150:
            confidence += 0.1  # Ideal length range
        
        # Uncertainty phrases (negative indicators)
        uncertainty_phrases = [
            "i don't know", "i'm not sure", "i cannot find",
            "there is no information", "it's unclear", "not mentioned",
            "doesn't specify", "cannot determine"
        ]
        
        for phrase in uncertainty_phrases:
            if phrase in answer_lower:
                confidence -= 0.3
                break  # Only penalize once
        
        # Hedging phrases (moderate negative)
        hedging_phrases = [
            "might", "possibly", "perhaps", "may be", "could be",
            "it seems", "appears to", "likely"
        ]
        
        hedge_count = sum(1 for phrase in hedging_phrases if phrase in answer_lower)
        confidence -= hedge_count * 0.05  # Small penalty per hedge
        
        # Positive indicators (specificity)
        specific_indicators = [
            "according to", "the document states", "as mentioned",
            "specifically", "in section", "on page"
        ]
        
        for indicator in specific_indicators:
            if indicator in answer_lower:
                confidence += 0.1
                break  # Only bonus once
        
        return max(0.0, min(1.0, confidence))
    
    def get_confidence_label(self, confidence: float) -> str:
        """
        Convert numeric confidence to human-readable label.
        """
        if confidence >= 0.8:
            return "high"
        elif confidence >= 0.6:
            return "medium"
        elif confidence >= 0.3:
            return "low"
        else:
            return "very_low"
    
    def should_escalate(self, confidence: float) -> bool:
        """
        Determine if query should be escalated based on confidence.
        """
        return confidence < settings.ESCALATION_THRESHOLD