# rag_service.py - Updated with Ensemble Retriever and Enhanced Caching

import os
import asyncio
import nest_asyncio
from concurrent.futures import ThreadPoolExecutor,as_completed
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re
import requests
from io import BytesIO
import threading
import hashlib
from uuid import uuid4
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
import time

# Environment setup
from dotenv import load_dotenv
load_dotenv()
os.environ["LLAMA_CLOUD_API_KEY"] = os.getenv("LLAMA_CLOUD_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["VOYAGE_API_KEY"] = os.getenv("VOYAGE_API_KEY")

# Async setup
nest_asyncio.apply()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread pool for CPU-intensive operations
executor = ThreadPoolExecutor(max_workers=4)

try:
    import fitz  # PyMuPDF
    import nltk
    from sentence_transformers import SentenceTransformer
    # Download required NLTK data (run once)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
except ImportError as e:
    logger.warning(f"Some dependencies not available: {e}")

@dataclass
class Document:
    """Document object similar to langchain_core.documents.Document"""
    page_content: str
    metadata: Dict[str, Any]

@dataclass
class Chunk:
    """Represents a document chunk with metadata"""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    start_char: int
    end_char: int

class PDFParser:
    """
    Optimized PDF Parser using PyMuPDF for extracting text and tables from PDFs loaded from URLs.
    Optimized for RAG applications with semantic chunking in mind.
    """
    
    def __init__(self, 
                 table_strategy: str = "lines",
                 min_words_vertical: int = 3,
                 min_words_horizontal: int = 1,
                 snap_tolerance: float = 3.0,
                 max_workers: int = 4):
        """
        Initialize the PDF parser with table detection parameters.
        
        Args:
            table_strategy: Strategy for table detection ("lines", "lines_strict", "text")
            min_words_vertical: Minimum words to establish virtual column boundary
            min_words_horizontal: Minimum words to establish virtual row boundary
            snap_tolerance: Tolerance for snapping lines together
            max_workers: Maximum number of worker threads for parallel processing
        """
        self.table_strategy = table_strategy
        self.min_words_vertical = min_words_vertical
        self.min_words_horizontal = min_words_horizontal
        self.snap_tolerance = snap_tolerance
        self.max_workers = max_workers
        
        # Pre-compile regex patterns for better performance
        self._whitespace_pattern = re.compile(r'\s+')
        self._artifact_pattern = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]')
        
        # Thread-local storage for PDF documents to avoid threading issues
        self._local = threading.local()
    
    def fetch_pdf_from_url(self, url: str) -> bytes:
        """
        Fetch PDF content from a URL with optimized settings.
        
        Args:
            url: URL of the PDF file
            
        Returns:
            PDF content as bytes
            
        Raises:
            requests.RequestException: If URL fetch fails
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept-Encoding': 'gzip, deflate',  # Enable compression
                'Connection': 'keep-alive'
            }
            
            # Use a session for connection pooling and increased timeout
            with requests.Session() as session:
                session.headers.update(headers)
                response = session.get(url, timeout=900, stream=True)
                response.raise_for_status()
                
                # Read content in chunks for better memory usage
                content = BytesIO()
                for chunk in response.iter_content(chunk_size=8192):
                    content.write(chunk)
                
                pdf_bytes = content.getvalue()
                
                # Verify it's a PDF
                if not pdf_bytes.startswith(b'%PDF'):
                    raise ValueError("URL does not point to a valid PDF file")
                    
                return pdf_bytes
                
        except requests.RequestException as e:
            raise requests.RequestException(f"Failed to fetch PDF from URL: {e}")
    
    def clean_text(self, text: str) -> str:
        """
        Clean extracted text for better semantic processing using pre-compiled patterns.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Use pre-compiled regex patterns for better performance
        text = self._whitespace_pattern.sub(' ', text)
        text = text.strip()
        
        # Remove common PDF artifacts
        text = self._artifact_pattern.sub('', text)
        
        return text
    
    def extract_text_from_page(self, page) -> str:
        """
        Extract clean text from a PDF page.
        
        Args:
            page: PyMuPDF page object
            
        Returns:
            Cleaned text content
        """
        # Extract text with layout preservation
        text = page.get_text("text")
        return self.clean_text(text)
    
    def extract_tables_from_page(self, page) -> List[str]:
        """
        Extract tables from a PDF page and convert to markdown format.
        
        Args:
            page: PyMuPDF page object
            
        Returns:
            List of tables in markdown format
        """
        tables_markdown = []
        
        try:
            # Find tables using the specified strategy
            table_finder = page.find_tables(
                strategy=self.table_strategy,
                min_words_vertical=self.min_words_vertical,
                min_words_horizontal=self.min_words_horizontal,
                snap_tolerance=self.snap_tolerance
            )
            
            # Pre-allocate list if we know the size
            if hasattr(table_finder, 'tables'):
                tables_count = len(table_finder.tables)
                if tables_count > 0:
                    tables_markdown = []  # Will append as we go
                    
                    # Extract each table as markdown
                    for table_idx, table in enumerate(table_finder.tables):
                        try:
                            # Get markdown representation optimized for LLM/RAG
                            markdown_table = table.to_markdown()
                            
                            if markdown_table and markdown_table.strip():
                                # Use f-string for better performance than concatenation
                                formatted_table = f"\n**Table {table_idx + 1}** (Rows: {table.row_count}, Columns: {table.col_count})\n\n{markdown_table}\n"
                                tables_markdown.append(formatted_table)
                                
                        except Exception as e:
                            print(f"Warning: Failed to extract table {table_idx + 1}: {e}")
                            continue
                    
        except Exception as e:
            print(f"Warning: Table detection failed on page: {e}")
        
        return tables_markdown
    
    def process_single_page(self, page_data: tuple) -> Document:
        """
        Process a single page (used for parallel processing).
        
        Args:
            page_data: Tuple of (page_num, page, pdf_metadata, url)
            
        Returns:
            Document object
        """
        page_num, page, pdf_metadata, url = page_data
        
        try:
            # Extract text
            text = self.extract_text_from_page(page)
            
            # Extract tables
            tables = self.extract_tables_from_page(page)
            
            # Create document object
            doc = self.create_page_document(
                page_num=page_num,
                text=text,
                tables=tables,
                pdf_metadata=pdf_metadata,
                url=url
            )
            
            return doc
            
        except Exception as e:
            print(f"Error processing page {page_num + 1}: {e}")
            # Return empty document to maintain page order
            return Document(
                page_content="",
                metadata={
                    "source": url,
                    "page": page_num + 1,
                    "page_index": page_num,
                    "error": str(e),
                    **pdf_metadata
                }
            )
    
    def create_page_document(self, 
                           page_num: int, 
                           text: str, 
                           tables: List[str], 
                           pdf_metadata: Dict[str, Any],
                           url: str) -> Document:
        """
        Create a Document object for a single page with optimized string operations.
        
        Args:
            page_num: Page number (0-indexed)
            text: Extracted text content
            tables: List of tables in markdown format
            pdf_metadata: PDF metadata
            url: Source URL
            
        Returns:
            Document object
        """
        # Use list for efficient string building
        page_content_parts = []
        
        if text:
            page_content_parts.append(f"**Page {page_num + 1} Content:**\n\n{text}")
        
        if tables:
            page_content_parts.append(f"\n**Tables on Page {page_num + 1}:**\n")
            page_content_parts.extend(tables)
        
        # Join once for better performance
        page_content = "".join(page_content_parts)
        
        # Create comprehensive metadata for RAG
        text_length = len(text)
        metadata = {
            "source": url,
            "page": page_num + 1,
            "page_index": page_num,
            "content_type": "pdf_page",
            "has_tables": len(tables) > 0,
            "table_count": len(tables),
            "text_length": text_length,
            "total_content_length": len(page_content),
            **pdf_metadata
        }
        
        return Document(page_content=page_content, metadata=metadata)
    
    def parse_pdf_from_url(self, url: str, use_parallel: bool = True) -> List[Document]:
        """
        Parse a PDF from URL and return list of Document objects with optional parallel processing.
        
        Args:
            url: URL of the PDF file
            use_parallel: Whether to use parallel processing for pages
            
        Returns:
            List of Document objects, one per page
            
        Raises:
            Exception: If PDF parsing fails
        """
        documents = []
        
        try:
            # Fetch PDF content
            pdf_content = self.fetch_pdf_from_url(url)
            
            # Open PDF document
            pdf_doc = fitz.open(stream=pdf_content, filetype="pdf")
            
            # Extract PDF metadata once
            pdf_metadata = {
                "title": pdf_doc.metadata.get("title", ""),
                "author": pdf_doc.metadata.get("author", ""),
                "subject": pdf_doc.metadata.get("subject", ""),
                "creator": pdf_doc.metadata.get("creator", ""),
                "producer": pdf_doc.metadata.get("producer", ""),
                "creation_date": pdf_doc.metadata.get("creationDate", ""),
                "modification_date": pdf_doc.metadata.get("modDate", ""),
                "total_pages": pdf_doc.page_count
            }
            
            if use_parallel and pdf_doc.page_count > 3:
                # Parallel processing for larger documents
                documents = self._process_pages_parallel(pdf_doc, pdf_metadata, url)
            else:
                # Sequential processing for smaller documents or when parallel is disabled
                documents = self._process_pages_sequential(pdf_doc, pdf_metadata, url)
            
            pdf_doc.close()
            
        except Exception as e:
            if 'pdf_doc' in locals():
                pdf_doc.close()
            raise Exception(f"Failed to parse PDF: {e}")
        
        return documents
    
    def _process_pages_sequential(self, pdf_doc, pdf_metadata: Dict[str, Any], url: str) -> List[Document]:
        """Process pages sequentially."""
        documents = []
        
        for page_num in range(pdf_doc.page_count):
            try:
                page = pdf_doc[page_num]
                doc = self.process_single_page((page_num, page, pdf_metadata, url))
                documents.append(doc)
                print(f"Processed page {page_num + 1}/{pdf_doc.page_count}")
            except Exception as e:
                print(f"Error processing page {page_num + 1}: {e}")
                continue
        
        return documents
    
    def _process_pages_parallel(self, pdf_doc, pdf_metadata: Dict[str, Any], url: str) -> List[Document]:
        """Process pages in parallel."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        documents = [None] * pdf_doc.page_count  # Pre-allocate list
        
        # Prepare page data for parallel processing
        page_data_list = []
        for page_num in range(pdf_doc.page_count):
            page = pdf_doc[page_num]
            page_data_list.append((page_num, page, pdf_metadata, url))
        
        # Process pages in parallel
        with ThreadPoolExecutor(max_workers=min(self.max_workers, pdf_doc.page_count)) as executor:
            # Submit all tasks
            future_to_page = {
                executor.submit(self.process_single_page, page_data): page_data[0] 
                for page_data in page_data_list
            }
            
            # Collect results as they complete
            completed_count = 0
            for future in as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    doc = future.result()
                    documents[page_num] = doc  # Maintain page order
                    completed_count += 1
                    print(f"Processed page {page_num + 1}/{pdf_doc.page_count} ({completed_count} completed)")
                except Exception as e:
                    print(f"Error processing page {page_num + 1}: {e}")
        
        # Filter out None values (failed pages)
        return [doc for doc in documents if doc is not None]

class FinancialPolicyChunker:
    """
    Optimized chunking strategy for financial policy documents.
    Combines hierarchical and semantic chunking for optimal RAG performance.
    """
    
    def __init__(self, 
                 target_chunk_size: int = 1000,
                 max_chunk_size: int = 1500,
                 min_chunk_size: int = 200,
                 overlap_size: int = 100,
                 semantic_threshold: float = 0.3,
                 use_semantic_model: bool = False):
        """
        Initialize the chunker with optimal parameters for financial documents.
        """
        self.target_chunk_size = target_chunk_size
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_size = overlap_size
        self.semantic_threshold = semantic_threshold
        self.use_semantic_model = use_semantic_model
        
        # Initialize semantic model if requested
        self.semantic_model = None
        if use_semantic_model:
            try:
                self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            except:
                print("Warning: Could not load semantic model, falling back to rule-based chunking")
        
        # Common section patterns in financial policies
        self.section_patterns = [
            r'^(\d+\.?\d*)\s+(PREAMBLE|DEFINITIONS|COVERAGE|EXCLUSIONS|CONDITIONS|CLAIMS|PREMIUM)',
            r'^(\d+\.?\d*)\s+([A-Z][A-Z\s&-]+)$',  # Numbered sections
            r'^([A-Z][A-Z\s&-]+):?\s*$',  # All caps headings
            r'^\*\*([^*]+)\*\*$',  # Bold headings from markdown
            r'^(Table of Benefits|Benefits|Features|Plans):',  # Table headers
            r'^(Optional covers|Add-ons|Discounts):',  # Policy sections
        ]
        
        # Sentence boundary patterns
        self.sentence_endings = r'[.!?]\s+(?=[A-Z])'
        
        # Table detection patterns
        self.table_patterns = [
            r'\*\*Table \d+\*\*',
            r'\|.*\|.*\|',  # Markdown table rows
            r'^\s*\|[-:]+\|',  # Table separators
        ]
    
    def identify_document_structure(self, text: str) -> List[Dict[str, Any]]:
        """
        Identify hierarchical structure in the document.
        """
        sections = []
        lines = text.split('\n')
        current_pos = 0
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                current_pos += len(line) + 1
                continue
            
            # Check for section headers
            section_match = None
            section_type = "content"
            
            for pattern in self.section_patterns:
                match = re.match(pattern, line_stripped, re.IGNORECASE)
                if match:
                    section_match = match
                    if any(keyword in line_stripped.upper() for keyword in 
                          ['DEFINITION', 'COVERAGE', 'EXCLUSION', 'CLAIM', 'PREMIUM', 'BENEFIT']):
                        section_type = "policy_section"
                    else:
                        section_type = "header"
                    break
            
            # Check for tables
            if any(re.search(pattern, line_stripped) for pattern in self.table_patterns):
                section_type = "table"
            
            if section_match or section_type == "table":
                sections.append({
                    'start_pos': current_pos,
                    'line_num': i,
                    'header': line_stripped,
                    'type': section_type,
                    'level': self._determine_header_level(line_stripped)
                })
            
            current_pos += len(line) + 1
        
        # Add document end
        sections.append({
            'start_pos': len(text),
            'line_num': len(lines),
            'header': 'END',
            'type': 'end',
            'level': 0
        })
        
        return sections
    
    def _determine_header_level(self, header: str) -> int:
        """Determine the hierarchical level of a header"""
        # Check for numbered sections (1.1, 2.3.4, etc.)
        number_match = re.match(r'^(\d+(?:\.\d+)*)', header.strip())
        if number_match:
            return len(number_match.group(1).split('.'))
        
        # Check for special markers
        if header.startswith('**') or header.isupper():
            return 1
        
        return 2
    
    def extract_hierarchical_chunks(self, text: str, page_metadata: Dict[str, Any]) -> List[Chunk]:
        """Extract chunks based on document hierarchy."""
        sections = self.identify_document_structure(text)
        chunks = []
        
        for i in range(len(sections) - 1):
            current_section = sections[i]
            next_section = sections[i + 1]
            
            # Extract section content
            section_start = current_section['start_pos']
            section_end = next_section['start_pos']
            section_content = text[section_start:section_end].strip()
            
            if len(section_content) < self.min_chunk_size:
                continue
            
            # Create metadata for this section
            section_metadata = {
                **page_metadata,
                'chunk_type': 'hierarchical',
                'section_header': current_section['header'],
                'section_type': current_section['type'],
                'section_level': current_section['level'],
                'hierarchical_path': self._build_hierarchical_path(sections, i)
            }
            
            # If section is too large, apply semantic chunking
            if len(section_content) > self.max_chunk_size:
                sub_chunks = self.apply_semantic_chunking(
                    section_content, 
                    section_metadata,
                    section_start
                )
                chunks.extend(sub_chunks)
            else:
                chunk_id = self._generate_chunk_id(section_content, section_metadata)
                chunk = Chunk(
                    content=section_content,
                    metadata=section_metadata,
                    chunk_id=chunk_id,
                    start_char=section_start,
                    end_char=section_end
                )
                chunks.append(chunk)
        
        return chunks
    
    def _build_hierarchical_path(self, sections: List[Dict], current_idx: int) -> str:
        """Build a hierarchical path for the current section"""
        path_parts = []
        current_level = sections[current_idx]['level']
        
        # Look backwards to find parent sections
        for i in range(current_idx, -1, -1):
            section = sections[i]
            if section['level'] < current_level:
                path_parts.insert(0, section['header'])
                current_level = section['level']
        
        path_parts.append(sections[current_idx]['header'])
        return ' > '.join(path_parts)
    
    def apply_semantic_chunking(self, text: str, base_metadata: Dict[str, Any], start_offset: int = 0) -> List[Chunk]:
        """Apply semantic chunking to break text into meaningful segments."""
        # Split into sentences
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return []
        
        # Group sentences semantically
        if self.semantic_model and len(sentences) > 2:
            chunks = self._semantic_grouping_with_model(sentences, base_metadata, start_offset)
        else:
            chunks = self._rule_based_semantic_grouping(sentences, base_metadata, start_offset)
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences with policy-specific rules"""
        # Handle special cases in financial policies
        text = re.sub(r'(\d+\.\d+)%', r'\1 percent', text)  # Handle percentages
        text = re.sub(r'(Rs\.|INR)\s*(\d+)', r'\1 \2', text)  # Handle currency
        text = re.sub(r'(\d+)\s*lacs?', r'\1 lacs', text)  # Handle lacs
        
        # Split on sentence boundaries
        sentences = re.split(self.sentence_endings, text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Minimum sentence length
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _rule_based_semantic_grouping(self, sentences: List[str], base_metadata: Dict[str, Any], start_offset: int) -> List[Chunk]:
        """Group sentences using rule-based semantic indicators"""
        chunks = []
        current_chunk_sentences = []
        current_length = 0
        char_offset = start_offset
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # Check if adding this sentence would exceed target size
            if (current_length + sentence_length > self.target_chunk_size and 
                current_chunk_sentences and 
                current_length > self.min_chunk_size):
                
                # Create chunk from current sentences
                content = '. '.join(current_chunk_sentences) + '.'
                chunk_metadata = {
                    **base_metadata,
                    'chunk_type': 'rule_based_semantic',
                    'sentence_count': len(current_chunk_sentences)
                }
                
                chunk_id = self._generate_chunk_id(content, chunk_metadata)
                chunk = Chunk(
                    content=content,
                    metadata=chunk_metadata,
                    chunk_id=chunk_id,
                    start_char=char_offset,
                    end_char=char_offset + len(content)
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                if self.overlap_size > 0 and current_chunk_sentences:
                    overlap_sentences = current_chunk_sentences[-1:]  # Keep last sentence
                    current_chunk_sentences = overlap_sentences + [sentence]
                    current_length = sum(len(s) for s in current_chunk_sentences)
                else:
                    current_chunk_sentences = [sentence]
                    current_length = sentence_length
                
                char_offset += len(content) - (len(overlap_sentences[0]) if overlap_sentences else 0)
            else:
                current_chunk_sentences.append(sentence)
                current_length += sentence_length
        
        # Handle remaining sentences
        if current_chunk_sentences:
            content = '. '.join(current_chunk_sentences) + '.'
            chunk_metadata = {
                **base_metadata,
                'chunk_type': 'rule_based_semantic',
                'sentence_count': len(current_chunk_sentences)
            }
            
            chunk_id = self._generate_chunk_id(content, chunk_metadata)
            chunk = Chunk(
                content=content,
                metadata=chunk_metadata,
                chunk_id=chunk_id,
                start_char=char_offset,
                end_char=char_offset + len(content)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _generate_chunk_id(self, content: str, metadata: Dict[str, Any]) -> str:
        """Generate unique chunk ID"""
        # Create a hash from content and key metadata
        id_string = f"{content[:100]}_{metadata.get('page', 0)}_{metadata.get('chunk_type', '')}"
        return hashlib.md5(id_string.encode()).hexdigest()[:12]
    
    def chunk_document(self, document) -> List[Chunk]:
        """
        Main method to chunk a document using the hybrid strategy.
        """
        text = document.page_content
        metadata = document.metadata
        
        chunks = []
        
        # Handle tables separately if present
        if metadata.get('has_tables', False):
            # Remove table content from text for regular processing (simplified)
            text = re.sub(r'\*\*Tables on Page \d+:\*\*.*?(?=\*\*Page \d+|\Z)', '', text, flags=re.DOTALL)
        
        # Apply hierarchical chunking to remaining text
        hierarchical_chunks = self.extract_hierarchical_chunks(text, metadata)
        chunks.extend(hierarchical_chunks)
        
        # Post-process chunks to ensure quality
        chunks = self._post_process_chunks(chunks)
        
        return chunks
    
    def _post_process_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Post-process chunks to ensure quality and consistency"""
        processed_chunks = []
        
        for chunk in chunks:
            # Skip empty or too-small chunks
            if len(chunk.content.strip()) < self.min_chunk_size:
                continue
            
            # Clean up content
            clean_content = self._clean_chunk_content(chunk.content)
            
            # Update chunk with cleaned content
            chunk.content = clean_content
            chunk.metadata['final_char_count'] = len(clean_content)
            chunk.metadata['final_word_count'] = len(clean_content.split())
            
            processed_chunks.append(chunk)
        
        return processed_chunks
    
    def _clean_chunk_content(self, content: str) -> str:
        """Clean chunk content for better readability"""
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove orphaned markdown markers
        content = re.sub(r'\*\*\s*\*\*', '', content)
        
        # Ensure proper sentence endings
        content = content.strip()
        if content and not content.endswith(('.', '!', '?', ':')):
            content += '.'
        
        return content

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document as LangchainDocument
from pydantic import PrivateAttr

# NEW: Enhanced Retriever with Caching (matching notebook)
class CachedEnsembleRetriever(BaseRetriever):
    """Enhanced ensemble retriever with caching and performance optimizations."""
    
    _retrievers: List = PrivateAttr()
    _retriever_names: List[str] = PrivateAttr()
    _weights: List[float] = PrivateAttr()
    _k: int = PrivateAttr()
    _retrieval_cache: Dict[str, List[LangchainDocument]] = PrivateAttr(default_factory=dict)
    _cache_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    def __init__(self, retrievers: List, retriever_names: List[str], weights: List[float], k: int = 8):
        super().__init__()
        assert len(retrievers) == len(retriever_names) == len(weights), "All lists must have same length"
        self._retrievers = retrievers
        self._retriever_names = retriever_names
        self._weights = weights
        self._k = k
        self._retrieval_cache = {}
        self._cache_lock = threading.Lock()

    @lru_cache(maxsize=100)
    def _cached_retrieve(self, query: str) -> List[LangchainDocument]:
        """Cache retrieval results for identical queries"""
        with self._cache_lock:
            if query in self._retrieval_cache:
                logger.info("Cache hit for query")
                return self._retrieval_cache[query]
            
            docs = self._get_relevant_documents(query)
            self._retrieval_cache[query] = docs
            return docs

    def _get_relevant_documents(self, query: str) -> List[LangchainDocument]:
        """Retrieve documents using ensemble approach with weighted scoring."""
        all_docs = []
        seen_contents = set()

        for retriever, name, weight in zip(self._retrievers, self._retriever_names, self._weights):
            try:
                if hasattr(retriever, 'get_relevant_documents'):
                    retrieved_docs = retriever.get_relevant_documents(query)
                elif hasattr(retriever, 'invoke'):
                    retrieved_docs = retriever.invoke(query)
                else:
                    logger.warning(f"Retriever {name} has no supported method")
                    continue
                    
                for doc in retrieved_docs:
                    if doc.page_content not in seen_contents:
                        doc.metadata = doc.metadata or {}
                        doc.metadata["source_retriever"] = name
                        doc.metadata["retriever_weight"] = weight
                        all_docs.append(doc)
                        seen_contents.add(doc.page_content)
            except Exception as e:
                logger.warning(f"{name} retrieval failed: {e}")

        return all_docs[:self._k]

class OptimizedRAGService:
    """Optimized RAG service with enhanced caching and batching."""
    
    def __init__(self):
        self.vector_store = None
        self.chain = None
        self.ensemble_retriever = None
        self._setup_complete = False
        self.embeddings = None
        self.mongo_client = None
        
        # Enhanced caching system (from notebook)
        self._retrieval_cache = {}
        self._cache_lock = threading.Lock()
        self._context_cache = {}  # For smart batching
        
    async def _setup_dependencies(self):
        """Setup dependencies only when needed."""
        if self._setup_complete:
            return
            
        # Import heavy libraries
        from langchain_voyageai import VoyageAIEmbeddings
        from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
        from langchain_openai import ChatOpenAI
        from langchain_core.documents import Document as LangchainDocument
        from langchain.prompts import PromptTemplate
        from langchain.schema.runnable import RunnablePassthrough
        from langchain_core.output_parsers import StrOutputParser
        from langchain.retrievers import BM25Retriever
        from langchain.retrievers import EnsembleRetriever  # Import EnsembleRetriever
        from pymongo import MongoClient
        
        # Store in instance
        self.VoyageAIEmbeddings = VoyageAIEmbeddings
        self.MongoDBAtlasVectorSearch = MongoDBAtlasVectorSearch
        self.ChatOpenAI = ChatOpenAI
        self.LangchainDocument = LangchainDocument
        self.PromptTemplate = PromptTemplate
        self.RunnablePassthrough = RunnablePassthrough
        self.StrOutputParser = StrOutputParser
        self.BM25Retriever = BM25Retriever
        self.EnsembleRetriever = EnsembleRetriever  # Add EnsembleRetriever
        self.MongoClient = MongoClient
        
        self._setup_complete = True
        
    async def setup_vector_store(self):
        """Setup vector store connection with optimized MongoDB client."""
        if self.vector_store is None:
            await self._setup_dependencies()
            
            DB_NAME = "Voyage_ai_RAG"
            COLLECTION_NAME = "langhcain_Voyage"
            
            # Optimized MongoDB client with connection pooling
            self.mongo_client = self.MongoClient(
                os.getenv("MONGODB_URI"),
                maxPoolSize=50,  # Increase connection pool
                minPoolSize=10,
                
                retryWrites=True,
                w="majority"
            )
            
            # Initialize embeddings with optimized settings
            self.embeddings = self.VoyageAIEmbeddings(
                model="voyage-3-large",
                batch_size=72,  # Voyage AI's max batch size
            )
            
            # Initialize vector store
            self.vector_store = self.MongoDBAtlasVectorSearch(
                collection=self.mongo_client[DB_NAME][COLLECTION_NAME],
                embedding=self.embeddings,
                index_name="vector_index",
                text_key="text",
                embedding_key="embedding"
            )

    @lru_cache(maxsize=100)
    def _cached_retrieve(self, question: str):
        """Cache retrieval results for identical questions (from notebook)"""
        with self._cache_lock:
            if question in self._retrieval_cache:
                return self._retrieval_cache[question]
            
            docs = self.ensemble_retriever.get_relevant_documents(question)
            context = "\n\n".join([doc.page_content for doc in docs])
            self._retrieval_cache[question] = context
            return context
            
    async def setup_chain(self, documents: List = None):
        """Setup the RAG chain with EnsembleRetriever (matching notebook parameters)."""
        if self.chain is None:
            await self._setup_dependencies()
            
            # Enhanced template for better insurance document understanding (same as notebook)
            template = """
Based on the document context below, answer the question factually and precisely.

CONTEXT: {context}

QUESTION: {question}

INSTRUCTIONS:
- Answer only from the context provided
- Cite exact terms, limits, sections if applicable  
- If not found, state "Not mentioned in document"
- Include key exclusions/conditions for coverage questions

ANSWER:
"""

            prompt = self.PromptTemplate.from_template(template)
            
            # Setup BM25 retriever (matching notebook parameters)
            if documents:
                bm25_retriever = self.BM25Retriever.from_documents(documents)
                bm25_retriever.k = 2  # Matching notebook
            else:
                bm25_retriever = None
            
            # Setup vector retriever with MMR (matching notebook parameters)
            vector_retriever = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"fetch_k": 25, "k": 6, "lambda_mult": 0.3}
            )
            
            # Create EnsembleRetriever with exact notebook parameters
            if bm25_retriever:
                self.ensemble_retriever = self.EnsembleRetriever(
                    retrievers=[vector_retriever, bm25_retriever],
                    weights=[0.6, 0.4]  # Exact weights from notebook
                )
            else:
                self.ensemble_retriever = vector_retriever
            
            # Use faster model (matching notebook)
            model = self.ChatOpenAI(
                model_name="gpt-3.5-turbo-1106",  # Exact model from notebook
                temperature=0,
                max_tokens=256,  # Matching notebook
                request_timeout=10,  # Add timeout
            )
            
            # Create chain with cached retrieval
            self.chain = (
                {"context": self._cached_retrieve, "question": self.RunnablePassthrough()}
                | prompt
                | model
                | self.StrOutputParser()
            )
    
    async def process_document_async(self, url: str) -> Tuple[List, List]:
        """Process document asynchronously and return both raw pages and document chunks."""
        await self._setup_dependencies()
        
        def process_sync():
            # Use the new enhanced parser from jupyter notebook
            parser = PDFParser(max_workers=8)
            parsed_documents = parser.parse_pdf_from_url(url, use_parallel=True)
            
            # Use direct approach from jupyter notebook for chunking
            chunker = FinancialPolicyChunker(
                target_chunk_size=1000,     # Ideal chunk size
                max_chunk_size=1500,        # Maximum allowed
                min_chunk_size=200,         # Minimum allowed
                overlap_size=100,           # Overlap between chunks
                use_semantic_model=False   # False = faster, True = better quality
            )
            
            all_chunks = []
            
            # Process each document
            for doc in parsed_documents:
                page_chunks = chunker.chunk_document(doc)
                all_chunks.extend(page_chunks)
            
            # Convert chunks to langchain Document objects
            documents = [
                self.LangchainDocument(
                    page_content=chunk.content,
                    metadata={**chunk.metadata}
                )
                for chunk in all_chunks
            ]
            
            return documents, parsed_documents
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, process_sync)
    
    def batch_documents(self, documents: List, batch_size: int = 72):
        """Split documents into batches for processing."""
        for i in range(0, len(documents), batch_size):
            yield documents[i:i + batch_size]
    
    async def add_documents_bulk_optimized(self, documents: List):
        """Most optimized approach using bulk operations with empty-text filtering."""
        
        DB_NAME = "Voyage_ai_RAG"
        COLLECTION_NAME = "langhcain_Voyage"
        
        # Step 1: Extract texts and filter out empties
        raw_entries = [
            (i, doc.page_content if hasattr(doc, 'page_content') else str(doc))
            for i, doc in enumerate(documents)
        ]
        
        filtered = [(i, text) for i, text in raw_entries if text and text.strip()]
        if not filtered:
            logger.info("No non-empty documents to embed.")
            return 0

        indices, texts = zip(*filtered)  # unzip valid entries
        
        logger.info(f"Computing embeddings for {len(texts)} valid documents...")
        
        # Step 2: Compute embeddings in batches
        all_embeddings = []
        for batch in self.batch_documents(texts, 72):  # Voyage AI max batch size
            batch_embeddings = self.embeddings.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)

        logger.info("Embeddings computed. Preparing bulk insert...")

        # Step 3: Prepare documents for insert
        bulk_docs = []
        for j, original_index in enumerate(indices):
            doc = documents[original_index]
            doc_dict = {
                "text": texts[j],
                "embedding": all_embeddings[j],
            }

            if hasattr(doc, 'metadata') and doc.metadata:
                doc_dict.update(doc.metadata)

            bulk_docs.append(doc_dict)

        # Step 4: Bulk insert
        collection = self.mongo_client[DB_NAME][COLLECTION_NAME]
        logger.info("Performing bulk insert...")

        try:
            result = collection.insert_many(bulk_docs, ordered=False)
            inserted_count = len(result.inserted_ids)
            logger.info(f"Successfully inserted {inserted_count} documents")
            return inserted_count
        except Exception as e:
            logger.error(f"Bulk insert failed: {e}")
            return 0

    def _safe_invoke(self, question: str) -> str:
        """Safely invoke chain with error handling (from notebook)"""
        try:
            return self.chain.invoke(question)
        except Exception as e:
            return f"Processing error: {str(e)}"

    async def answer_questions_parallel(self, questions: List[str], max_workers: int = 3) -> List[str]:
        """Answer multiple questions concurrently with parallel processing (enhanced from notebook)."""
        
        def answer_single(question: str) -> str:
            try:
                return self.chain.invoke(question)
            except Exception as e:
                logger.error(f"Error answering question '{question}': {e}")
                return f"Error processing question: {str(e)}"
        
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as thread_executor:
            # Submit all questions
            future_to_question = {
                thread_executor.submit(answer_single, q): q for q in questions
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_question):
                question = future_to_question[future]
                try:
                    answer = future.result(timeout=30)  # 30 second timeout per question
                    results.append(answer)
                except Exception as e:
                    logger.error(f"Error processing question '{question}': {e}")
                    results.append(f"Error: {str(e)}")
        
        return results

    async def answer_questions_smart_batch(self, questions: List[str]) -> List[str]:
        """Smart batching with context reuse (from notebook)"""
        # Pre-retrieve all unique contexts
        unique_contexts = {}
        question_contexts = {}
        
        for question in questions:
            context = self._cached_retrieve(question)
            context_hash = hash(context)
            unique_contexts[context_hash] = context
            question_contexts[question] = context_hash
        
        # Process questions with pre-retrieved contexts
        results = []
        for question in questions:
            try:
                context_hash = question_contexts[question]
                context = unique_contexts[context_hash]
                
                # Direct model call with pre-retrieved context
                template = """
Based on the document context below, answer the question factually and precisely.

CONTEXT: {context}

QUESTION: {question}

INSTRUCTIONS:
- Answer only from the context provided
- Cite exact terms, limits, sections if applicable  
- If not found, state "Not mentioned in document"
- Include key exclusions/conditions for coverage questions

ANSWER:
"""
                formatted_prompt = template.format(context=context, question=question)
                
                # Use the model directly
                model = self.ChatOpenAI(
                    model_name="gpt-3.5-turbo-1106",
                    temperature=0,
                    max_tokens=256,
                    request_timeout=10,
                )
                answer = model.invoke(formatted_prompt).content
                
                results.append(answer)
            except Exception as e:
                results.append(f"Error: {str(e)}")
        
        return results
    
    async def answer_questions(self, questions: List[str], method: str = "parallel") -> List[str]:
        """Answer multiple questions using specified method."""
        if method == "parallel":
            return await self.answer_questions_parallel(questions, max_workers=3)
        elif method == "smart_batch":
            return await self.answer_questions_smart_batch(questions)
        else:
            # Fallback to original method
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            def answer_single(question: str) -> str:
                try:
                    return self.chain.invoke(question)
                except Exception as e:
                    logger.error(f"Error answering question '{question}': {e}")
                    return f"Error processing question: {str(e)}"
            
            results = []
            
            with ThreadPoolExecutor(max_workers=3) as thread_executor:
                # Submit all questions
                future_to_question = {
                    thread_executor.submit(answer_single, q): q for q in questions
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_question):
                    question = future_to_question[future]
                    try:
                        answer = future.result(timeout=30)  # 30 second timeout per question
                        results.append(answer)
                    except Exception as e:
                        logger.error(f"Error processing question '{question}': {e}")
                        results.append(f"Error: {str(e)}")
            
            return results
    
    async def run(self, documents_url: str, questions: List[str], processing_method: str = "parallel") -> List[str]:
        """Main run method - optimized for speed with enhanced document processing and caching."""
        try:
            logger.info(f"Starting RAG processing for {len(questions)} questions using {processing_method} method")
            
            # Setup vector store
            await self.setup_vector_store()
            
            # Process document with enhanced parsing and chunking
            documents, parsed_pages = await self.process_document_async(documents_url)
            
            logger.info(f"Processed {len(documents)} document chunks")
            
            # Setup chain with EnsembleRetriever (matching notebook)
            await self.setup_chain(documents)
            
            # Add to vector store using optimized bulk method
            total_added = await self.add_documents_bulk_optimized(documents)
            logger.info(f"Total documents added: {total_added}")
            
            logger.info("Documents added to vector store, answering questions...")
            
            # Answer questions using specified processing method
            answers = await self.answer_questions(questions, method=processing_method)
            
            logger.info("RAG processing completed successfully")
            return answers
            
        except Exception as e:
            logger.error(f"Error in RAG service run: {e}")
            raise