import logging as log
import re
import unicodedata
from typing import List, Tuple, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

log.basicConfig(
    level=log.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def preprocess_vietnamese_text(text: str) -> str:
    """
    Clean and preprocess Vietnamese text extracted from PDFs.
    Handles common issues like:
    - Excessive spaces between characters
    - Broken Vietnamese diacritics  
    - Redundant whitespace
    - Newlines in the middle of words
    """
    if not text:
        return text
    
    # Normalize unicode (NFC form for Vietnamese)
    text = unicodedata.normalize('NFC', text)
    
    # Vietnamese characters
    vn_vowels = 'aáàảãạăắằẳẵặâấầẩẫậeéèẻẽẹêếềểễệiíìỉĩịoóòỏõọôốồổỗộơớờởỡợuúùủũụưứừửữựyýỳỷỹỵ'
    vn_vowels_upper = vn_vowels.upper()
    vn_all = vn_vowels + vn_vowels_upper + 'đĐ'
    all_letters = 'a-zA-Z' + vn_all
    
    # Step 1: Fix newlines in middle of words
    text = re.sub(r'([' + all_letters + r'])\n([' + all_letters + r'])', r'\1\2', text)
    
    # Step 2: Normalize tabs and multiple spaces
    text = re.sub(r'\t', ' ', text)
    text = re.sub(r' {2,}', ' ', text)
    
    # Step 3: Fix spaced-out single characters
    # Pattern: sequence of single letters separated by single spaces
    # E.g., "t h ự c" -> "thực", "G i ớ i" -> "Giới"
    
    # Process line by line for better control
    lines = text.split('\n')
    fixed_lines = []
    
    for line in lines:
        # Split into words (space-separated)
        words = line.split(' ')
        
        # Check if line has many single-char "words"
        single_chars = sum(1 for w in words if len(w) == 1 and re.match(r'[' + all_letters + r']', w))
        
        if len(words) > 3 and single_chars / len(words) > 0.4:
            # This line likely has spaced-out characters
            # Join consecutive single chars, keep multi-char words as boundaries
            result = []
            current_syllable = []
            
            for word in words:
                if len(word) == 1 and re.match(r'[' + all_letters + r']', word):
                    current_syllable.append(word)
                else:
                    # Multi-char word or punctuation - acts as boundary
                    if current_syllable:
                        result.append(''.join(current_syllable))
                        current_syllable = []
                    if word:  # Don't add empty strings
                        result.append(word)
            
            # Don't forget last syllable
            if current_syllable:
                result.append(''.join(current_syllable))
            
            fixed_lines.append(' '.join(result))
        else:
            # Line looks normal, just clean it
            fixed_lines.append(line)
    
    text = '\n'.join(fixed_lines)
    
    # Step 4: Handle remaining isolated spaced chars (edge cases)
    # Remove space between a single char and following single char
    text = re.sub(r'\b([' + all_letters + r']) ([' + all_letters + r'])\b', r'\1\2', text)
    
    # Step 5: Add space before uppercase after lowercase (restore word boundaries)
    text = re.sub(r'([a-z' + vn_vowels + r'đ])([A-Z' + vn_vowels_upper + r'Đ])', r'\1 \2', text)
    
    # Step 6: Clean up multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    
    # Step 7: Clean multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Step 8: Fix punctuation spacing
    text = re.sub(r' +([.,;:!?\]\)])', r'\1', text)
    text = re.sub(r'([.,;:!?])([' + all_letters + r'])', r'\1 \2', text)
    
    # Step 9: Fix bracket spacing
    text = re.sub(r'\[ +', '[', text)
    text = re.sub(r' +\]', ']', text)
    text = re.sub(r'\( +', '(', text)
    text = re.sub(r' +\)', ')', text)
    
    # Step 10: Strip lines
    lines = text.split('\n')
    lines = [line.strip() for line in lines]
    text = '\n'.join(lines)
    
    return text.strip()


class SemanticChunker:
    """
    Advanced semantic chunking with section detection.
    Attempts to preserve semantic boundaries (headers, paragraphs, sections).
    """
    
    # Common section header patterns
    HEADER_PATTERNS = [
        r'^#{1,6}\s+.+$',  # Markdown headers
        r'^\d+\.\s+.+$',   # Numbered sections (1. Introduction)
        r'^\d+\.\d+\s+.+$',  # Sub-numbered (1.1 Background)
        r'^[A-Z][^a-z]*$',  # ALL CAPS HEADERS
        r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*:$',  # Title Case With Colon:
        r'^\*\*[^*]+\*\*$',  # **Bold Headers**
        r'^Chapter\s+\d+',  # Chapter headers
        r'^Section\s+\d+',  # Section headers
        r'^Part\s+[A-Z\d]+',  # Part headers
        r'^Chương\s+\d+',  # Chapter headers
        r'^Mục\s+\d+',  # Section headers
        r'^Phần\s+[A-Z\d]+',  # Part headers
        r'^Điều\s+\d+',  # Article headers
    ]
    
    def __init__(self, max_chunk_size: int = 1000, min_chunk_size: int = 200, overlap: int = 100):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap = overlap
        self.header_regex = re.compile('|'.join(self.HEADER_PATTERNS), re.MULTILINE)
    
    def _is_header(self, line: str) -> bool:
        """Check if a line is likely a section header."""
        line = line.strip()
        if not line:
            return False
        # Check against header patterns
        if self.header_regex.match(line):
            return True
        # Short lines ending with colon could be headers
        if len(line) < 50 and line.endswith(':'):
            return True
        return False
    
    def _split_into_sections(self, text: str) -> List[Tuple[Optional[str], str]]:
        """Split text into sections based on headers."""
        lines = text.split('\n')
        sections = []
        current_header = None
        current_content = []
        
        for line in lines:
            if self._is_header(line):
                # Save previous section
                if current_content:
                    content = '\n'.join(current_content).strip()
                    if content:
                        sections.append((current_header, content))
                current_header = line.strip()
                current_content = []
            else:
                current_content.append(line)
        
        # Don't forget the last section
        if current_content:
            content = '\n'.join(current_content).strip()
            if content:
                sections.append((current_header, content))
        
        return sections
    
    def _chunk_section(self, header: Optional[str], content: str) -> List[dict]:
        """Chunk a section while preserving context."""
        chunks = []
        
        # If content is small enough, keep it as one chunk
        if len(content) <= self.max_chunk_size:
            chunks.append({
                'header': header,
                'content': content,
                'is_complete_section': True
            })
            return chunks
        
        # Split on sentence boundaries first
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            if current_length + sentence_len > self.max_chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'header': header,
                    'content': chunk_text,
                    'is_complete_section': False
                })
                
                # Start new chunk with overlap
                overlap_sentences = []
                overlap_len = 0
                for s in reversed(current_chunk):
                    if overlap_len + len(s) <= self.overlap:
                        overlap_sentences.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break
                
                current_chunk = overlap_sentences + [sentence]
                current_length = overlap_len + sentence_len
            else:
                current_chunk.append(sentence)
                current_length += sentence_len
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size or not chunks:
                chunks.append({
                    'header': header,
                    'content': chunk_text,
                    'is_complete_section': len(chunks) == 0
                })
            elif chunks:
                # Merge with previous chunk if too small
                chunks[-1]['content'] += ' ' + chunk_text
        
        return chunks
    
    def chunk_text(self, text: str, page_number: int = 1) -> List[dict]:
        """Chunk text with semantic awareness."""
        sections = self._split_into_sections(text)
        all_chunks = []
        
        for header, content in sections:
            section_chunks = self._chunk_section(header, content)
            for chunk in section_chunks:
                chunk['page_number'] = page_number
                all_chunks.append(chunk)
        
        log.info(f"Created {len(all_chunks)} semantic chunks from page {page_number}")
        return all_chunks


class TextProcessor:
    """Original chunking processor using RecursiveCharacterTextSplitter."""
    
    def __init__(self, chunk_size=750, chunk_overlap=200, preprocess: bool = True):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        self.preprocess = preprocess

    def split_documents(self, raw_documents: List[Document], apply_preprocess: bool = None) -> List[Document]:
        """Chunk documents into smaller pieces."""
        should_preprocess = apply_preprocess if apply_preprocess is not None else self.preprocess
        
        try:
            # Preprocess text if enabled
            if should_preprocess:
                preprocessed_docs = []
                for doc in raw_documents:
                    cleaned_text = preprocess_vietnamese_text(doc.page_content)
                    preprocessed_docs.append(Document(
                        page_content=cleaned_text,
                        metadata=doc.metadata
                    ))
                raw_documents = preprocessed_docs
            
            splitted_docs = self.splitter.split_documents(raw_documents)
            log.info(f"Split {len(raw_documents)} documents into {len(splitted_docs)} chunks.")
            return splitted_docs
        except Exception as e:
            log.error(f"Error splitting documents: {e}")
            return []


class HybridChunker:
    """
    Combines semantic chunking with fallback to RecursiveCharacterTextSplitter.
    Best for documents with varied structure.
    Includes text preprocessing to fix common PDF extraction issues.
    """
    
    def __init__(self, 
                 max_chunk_size: int = 1000, 
                 min_chunk_size: int = 200, 
                 overlap: int = 100,
                 use_semantic: bool = True,
                 preprocess: bool = True):
        self.semantic_chunker = SemanticChunker(max_chunk_size, min_chunk_size, overlap)
        self.text_processor = TextProcessor(chunk_size=max_chunk_size, chunk_overlap=overlap)
        self.use_semantic = use_semantic
        self.preprocess = preprocess
        self.max_chunk_size = max_chunk_size
    
    def chunk_documents(self, documents: List[Document]) -> List[dict]:
        """
        Chunk documents using hybrid approach.
        Returns list of dicts with content, metadata, and structural info.
        """
        all_chunks = []
        
        for doc in documents:
            page_num = doc.metadata.get('page', 1)
            text = doc.page_content
            
            # Preprocess text to fix spacing and character issues
            if self.preprocess:
                original_len = len(text)
                text = preprocess_vietnamese_text(text)
                if len(text) != original_len:
                    log.info(f"Preprocessed page {page_num}: {original_len} -> {len(text)} chars")
            
            if self.use_semantic:
                # Try semantic chunking first
                chunks = self.semantic_chunker.chunk_text(text, page_num)
                
                # Validate chunks - fall back to simple splitting if needed
                valid_chunks = []
                for chunk in chunks:
                    if len(chunk['content']) > self.max_chunk_size * 1.5:
                        # Chunk too large, use text processor
                        sub_doc = Document(page_content=chunk['content'], metadata={'page': page_num})
                        sub_chunks = self.text_processor.split_documents([sub_doc])
                        for sc in sub_chunks:
                            valid_chunks.append({
                                'header': chunk.get('header'),
                                'content': sc.page_content,
                                'page_number': page_num,
                                'is_complete_section': False
                            })
                    else:
                        valid_chunks.append(chunk)
                
                all_chunks.extend(valid_chunks)
            else:
                # Fall back to simple text processing
                # Create a new document with preprocessed text
                preprocessed_doc = Document(page_content=text, metadata=doc.metadata)
                split_docs = self.text_processor.split_documents([preprocessed_doc])
                for sd in split_docs:
                    all_chunks.append({
                        'header': None,
                        'content': sd.page_content,
                        'page_number': page_num,
                        'is_complete_section': False
                    })
        
        log.info(f"HybridChunker created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks