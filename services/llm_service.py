"""
LLM Service
Handles answer generation using OpenAI or Anthropic
"""

from typing import List, Dict, Any
import openai
import anthropic
import re
from config import settings


class LLMService:
    """
    Manages LLM interactions for answer generation.
    Supports OpenAI and Anthropic with configurable models.
    """
    
    def __init__(self):
        self.provider = settings.LLM_PROVIDER.lower()
        self.model = settings.LLM_MODEL
        self.temperature = settings.LLM_TEMPERATURE
        self.max_tokens = settings.LLM_MAX_TOKENS
        
        # Initialize clients
        self.openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        self.anthropic_client = None
        
        if settings.ANTHROPIC_API_KEY:
            self.anthropic_client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    
    def configure(self, provider: str, model: str = None):
        """
        Update LLM provider and model.
        """
        provider = provider.lower()
        
        if provider not in ['openai', 'anthropic']:
            raise ValueError("Provider must be 'openai' or 'anthropic'")
        
        if provider == 'anthropic' and not self.anthropic_client:
            raise ValueError("Anthropic API key not configured")
        
        self.provider = provider
        
        if model:
            self.model = model
        else:
            # Set default model for provider
            if provider == 'openai':
                self.model = 'gpt-4o-mini'
            else:
                self.model = 'claude-3-5-sonnet-20241022'
    
    async def generate_answer(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate answer using retrieved context chunks.
        
        Returns:
            Dict with 'answer' and 'confidence' keys
        """
        # Build context from chunks
        context = self._build_context(context_chunks)
        
        # Create prompt
        prompt = self._create_prompt(question, context)
        
        # Call appropriate LLM
        if self.provider == 'openai':
            return await self._call_openai(prompt, question)
        else:
            return await self._call_anthropic(prompt, question)
    
    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Build context string from retrieved chunks.
        """
        context_parts = []
        
        for idx, chunk in enumerate(chunks, 1):
            metadata = chunk.get('metadata', {})
            source = metadata.get('filename', 'Unknown')
            section = metadata.get('section', 'N/A')
            
            context_parts.append(
                f"[Source {idx}: {source} - {section}]\n{chunk['text']}\n"
            )
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, question: str, context: str) -> str:
        """
        Create a structured prompt for the LLM.
        """
        prompt = f"""You are a helpful FAQ support assistant. Your task is to answer user questions based ONLY on the provided documentation context.

IMPORTANT INSTRUCTIONS:
1. Answer the question using ONLY information from the provided context
2. If the context doesn't contain enough information to answer confidently, say so clearly
3. Reference specific sources when providing information
4. Be concise and direct
5. At the end of your answer, provide a confidence rating from 0.0 to 1.0 in the format: [CONFIDENCE: X.XX]
   - 0.0-0.4: Low confidence (context lacks clear information)
   - 0.5-0.7: Medium confidence (some relevant information but incomplete)
   - 0.8-1.0: High confidence (context clearly answers the question)

CONTEXT:
{context}

USER QUESTION:
{question}

YOUR ANSWER:"""
        
        return prompt
    
    async def _call_openai(self, prompt: str, question: str) -> Dict[str, Any]:
        """
        Call OpenAI API for answer generation.
        """
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful FAQ support assistant. Always provide a confidence score at the end of your response."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            answer_text = response.choices[0].message.content.strip()
            
            # Extract confidence from answer
            confidence = self._extract_confidence(answer_text)
            
            # Remove confidence marker from answer
            answer_text = re.sub(r'\[CONFIDENCE:?\s*[\d.]+\]', '', answer_text).strip()
            
            return {
                'answer': answer_text,
                'confidence': confidence
            }
            
        except Exception as e:
            return {
                'answer': f"Error generating answer: {str(e)}",
                'confidence': 0.0
            }
    
    async def _call_anthropic(self, prompt: str, question: str) -> Dict[str, Any]:
        """
        Call Anthropic API for answer generation.
        """
        try:
            response = self.anthropic_client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            answer_text = response.content[0].text.strip()
            
            # Extract confidence from answer
            confidence = self._extract_confidence(answer_text)
            
            # Remove confidence marker from answer
            answer_text = re.sub(r'\[CONFIDENCE:?\s*[\d.]+\]', '', answer_text).strip()
            
            return {
                'answer': answer_text,
                'confidence': confidence
            }
            
        except Exception as e:
            return {
                'answer': f"Error generating answer: {str(e)}",
                'confidence': 0.0
            }
    
    def _extract_confidence(self, text: str) -> float:
        """
        Extract confidence score from LLM response.
        """
        # Look for [CONFIDENCE: X.XX] pattern
        match = re.search(r'\[CONFIDENCE:?\s*([\d.]+)\]', text, re.IGNORECASE)
        
        if match:
            try:
                confidence = float(match.group(1))
                return max(0.0, min(1.0, confidence))  # Clamp between 0 and 1
            except ValueError:
                pass
        
        # Fallback: analyze response for confidence indicators
        text_lower = text.lower()
        
        if any(phrase in text_lower for phrase in [
            "i don't know", "i'm not sure", "cannot find", "no information",
            "unclear", "not mentioned", "doesn't specify"
        ]):
            return 0.3
        
        if any(phrase in text_lower for phrase in [
            "might", "possibly", "perhaps", "may be"
        ]):
            return 0.5
        
        # Default medium confidence if no indicators
        return 0.6
    
    def health_check(self) -> dict:
        """
        Health check for the LLM provider.
        Returns a standardized dict with status and optional error.
        """
        try:
            if self.provider == "openai":
                # Minimal non-billing OpenAI call to validate API access
                self.openai_client.models.list()
                return {"status": "healthy"}
    
            elif self.provider == "anthropic" and self.anthropic_client:
                # You may optionally add a test call
                return {"status": "healthy"}
    
            return {"status": "unhealthy", "error": "Unsupported or uninitialized provider"}
    
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
