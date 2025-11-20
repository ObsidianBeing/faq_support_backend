"""
Document Processing Service
Handles extraction, parsing, and chunking of documents
"""

import re
from typing import Dict, List, Any
from io import BytesIO
import tiktoken

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

from config import settings


class DocumentProcessor:
    """
    Processes documents in various formats and chunks them for vector storage.
    """
    
    def __init__(self):
        self.chunk_size = settings.DEFAULT_CHUNK_SIZE
        self.chunk_overlap = settings.DEFAULT_CHUNK_OVERLAP
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def configure_chunking(self, chunk_size: int, chunk_overlap: int):
        """
        Update chunking parameters.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_text(self, content: bytes, filename: str, file_type: str) -> Dict[str, Any]:
        """
        Extract text from various document formats.
        
        Returns:
            Dict with 'text' and 'metadata' keys
        """
        if file_type == '.pdf':
            return self._extract_from_pdf(content, filename)
        elif file_type in ['.md', '.markdown', '.txt']:
            return self._extract_from_text(content, filename)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    def _extract_from_pdf(self, content: bytes, filename: str) -> Dict[str, Any]:
        """
        Extract text from PDF files.
        """
        if PyPDF2 is None:
            raise ImportError("PyPDF2 is required for PDF processing. Install with: pip install PyPDF2")
        
        pdf_file = BytesIO(content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text_parts = []
        metadata = {
            'filename': filename,
            'total_pages': len(pdf_reader.pages),
            'file_type': 'pdf'
        }
        
        for page_num, page in enumerate(pdf_reader.pages, 1):
            page_text = page.extract_text()
            if page_text.strip():
                text_parts.append(f"\n--- Page {page_num} ---\n{page_text}")
        
        full_text = "\n".join(text_parts)
        
        return {
            'text': full_text,
            'metadata': metadata
        }
    
    def _extract_from_text(self, content: bytes, filename: str) -> Dict[str, Any]:
        """
        Extract text from plain text or Markdown files.
        """
        try:
            text = content.decode('utf-8')
        except UnicodeDecodeError:
            text = content.decode('latin-1')
        
        # Extract sections from Markdown
        sections = self._extract_markdown_sections(text)
        
        metadata = {
            'filename': filename,
            'file_type': 'text/markdown',
            'sections_found': len(sections)
        }
        
        return {
            'text': text,
            'metadata': metadata
        }
    
    def _extract_markdown_sections(self, text: str) -> List[str]:
        """
        Extract section headers from Markdown.
        """
        # Match markdown headers (# Header, ## Header, etc.)
        header_pattern = re.compile(r'^#{1,6}\s+(.+)$', re.MULTILINE)
        sections = header_pattern.findall(text)
        return sections
    
    def chunk_document(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk document into smaller pieces with overlap.
        Uses token-based chunking for more accurate sizing.
        
        Returns:
            List of chunk dictionaries with text and metadata
        """
        # Tokenize the entire text
        tokens = self.tokenizer.encode(text)
        
        chunks = []
        start_idx = 0
        chunk_num = 0
        
        while start_idx < len(tokens):
            # Get chunk tokens
            end_idx = start_idx + self.chunk_size
            chunk_tokens = tokens[start_idx:end_idx]
            
            # Decode back to text
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            # Clean up chunk text
            chunk_text = chunk_text.strip()
            
            if chunk_text:
                # Try to identify section from text
                section = self._identify_section(chunk_text)
                
                chunk_metadata = {
                    **metadata,
                    'chunk_num': chunk_num,
                    'chunk_size': len(chunk_tokens),
                    'section': section,
                    'start_token': start_idx,
                    'end_token': end_idx
                }
                
                chunks.append({
                    'text': chunk_text,
                    'metadata': chunk_metadata
                })
                
                chunk_num += 1
            
            # Move to next chunk with overlap
            start_idx += (self.chunk_size - self.chunk_overlap)
        
        return chunks
    
    def _identify_section(self, text: str) -> str:
        """
        Try to identify which section this chunk belongs to.
        Looks for markdown headers or page markers.
        """
        # Look for markdown headers
        header_match = re.search(r'^#{1,6}\s+(.+)$', text, re.MULTILINE)
        if header_match:
            return header_match.group(1).strip()
        
        # Look for page markers
        page_match = re.search(r'--- Page (\d+) ---', text)
        if page_match:
            return f"Page {page_match.group(1)}"
        
        # Look for first sentence/line as fallback
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if len(line) > 10 and len(line) < 100:
                return line[:80] + "..." if len(line) > 80 else line
        
        return "Unknown Section"