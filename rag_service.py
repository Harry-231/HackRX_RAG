# enhanced_rag_service.py - Complete Two-Stage Conditional Retrieval Strategy with Enhanced Logging

import os
import asyncio
import nest_asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Create progress logger for detailed step tracking
progress_logger = logging.getLogger(f"{__name__}.progress")
progress_logger.setLevel(logging.INFO)

# Thread pool for CPU-intensive operations
executor = ThreadPoolExecutor(max_workers=8)

try:
    import fitz  # PyMuPDF
    import nltk
    from sentence_transformers import SentenceTransformer
    import faiss  # For in-memory vector store
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
                 max_workers: int = 8):
        """
        Initialize the PDF parser with table detection parameters.
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
        
        progress_logger.info("PDFParser initialized with table_strategy=%s, max_workers=%d", 
                           table_strategy, max_workers)
    
    def fetch_pdf_from_url(self, url: str) -> bytes:
        """Fetch PDF content from a URL with optimized settings."""
        progress_logger.info("📥 Starting PDF fetch from URL: %s", url)
        start_time = time.time()
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive'
            }
            
            with requests.Session() as session:
                session.headers.update(headers)
                progress_logger.info("🌐 Sending HTTP request...")
                response = session.get(url, timeout=900, stream=True)
                response.raise_for_status()
                
                progress_logger.info("📡 Receiving PDF data (Content-Length: %s)", 
                                   response.headers.get('Content-Length', 'Unknown'))
                
                content = BytesIO()
                total_size = 0
                for chunk_num, chunk in enumerate(response.iter_content(chunk_size=8192)):
                    content.write(chunk)
                    total_size += len(chunk)
                    if chunk_num % 100 == 0 and chunk_num > 0:  # Log every ~800KB
                        progress_logger.debug("Downloaded %d KB...", total_size // 1024)
                
                pdf_bytes = content.getvalue()
                
                if not pdf_bytes.startswith(b'%PDF'):
                    raise ValueError("URL does not point to a valid PDF file")
                
                fetch_time = time.time() - start_time
                progress_logger.info("✅ PDF fetched successfully: %d bytes in %.2f seconds", 
                                   len(pdf_bytes), fetch_time)
                return pdf_bytes
                
        except requests.RequestException as e:
            progress_logger.error("❌ Failed to fetch PDF from URL: %s", str(e))
            raise requests.RequestException(f"Failed to fetch PDF from URL: {e}")
    
    def clean_text(self, text: str) -> str:
        """Clean extracted text for better semantic processing."""
        if not text:
            return ""
        
        text = self._whitespace_pattern.sub(' ', text)
        text = text.strip()
        text = self._artifact_pattern.sub('', text)
        
        return text
    
    def extract_text_from_page(self, page) -> str:
        """Extract clean text from a PDF page."""
        text = page.get_text("text")
        return self.clean_text(text)
    
    def extract_tables_from_page(self, page) -> List[str]:
        """Extract tables from a PDF page and convert to markdown format."""
        tables_markdown = []
        
        try:
            table_finder = page.find_tables(
                strategy=self.table_strategy,
                min_words_vertical=self.min_words_vertical,
                min_words_horizontal=self.min_words_horizontal,
                snap_tolerance=self.snap_tolerance
            )
            
            if hasattr(table_finder, 'tables'):
                tables_count = len(table_finder.tables)
                if tables_count > 0:
                    for table_idx, table in enumerate(table_finder.tables):
                        try:
                            markdown_table = table.to_markdown()
                            
                            if markdown_table and markdown_table.strip():
                                formatted_table = f"\n**Table {table_idx + 1}** (Rows: {table.row_count}, Columns: {table.col_count})\n\n{markdown_table}\n"
                                tables_markdown.append(formatted_table)
                                
                        except Exception as e:
                            logger.debug("Failed to extract table %d: %s", table_idx + 1, str(e))
                            continue
                    
        except Exception as e:
            logger.debug("Table detection failed on page: %s", str(e))
        
        return tables_markdown
    
    def process_single_page(self, page_data: tuple) -> Document:
        """Process a single page (used for parallel processing)."""
        page_num, page, pdf_metadata, url = page_data
        
        try:
            text = self.extract_text_from_page(page)
            tables = self.extract_tables_from_page(page)
            
            doc = self.create_page_document(
                page_num=page_num,
                text=text,
                tables=tables,
                pdf_metadata=pdf_metadata,
                url=url
            )
            
            return doc
            
        except Exception as e:
            logger.error("Error processing page %d: %s", page_num + 1, str(e))
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
        """Create a Document object for a single page."""
        page_content_parts = []
        
        if text:
            page_content_parts.append(f"**Page {page_num + 1} Content:**\n\n{text}")
        
        if tables:
            page_content_parts.append(f"\n**Tables on Page {page_num + 1}:**\n")
            page_content_parts.extend(tables)
        
        page_content = "".join(page_content_parts)
        
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
        """Parse a PDF from URL and return list of Document objects."""
        progress_logger.info("🔍 Starting PDF parsing from URL")
        documents = []
        
        try:
            pdf_content = self.fetch_pdf_from_url(url)
            progress_logger.info("📖 Opening PDF document...")
            pdf_doc = fitz.open(stream=pdf_content, filetype="pdf")
            
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
            
            progress_logger.info("📊 PDF Metadata - Title: '%s', Pages: %d, Author: '%s'", 
                               pdf_metadata.get("title", "Unknown"), 
                               pdf_doc.page_count,
                               pdf_metadata.get("author", "Unknown"))
            
            processing_method = "parallel" if use_parallel and pdf_doc.page_count > 3 else "sequential"
            progress_logger.info("⚙️ Using %s processing for %d pages", processing_method, pdf_doc.page_count)
            
            if use_parallel and pdf_doc.page_count > 3:
                documents = self._process_pages_parallel(pdf_doc, pdf_metadata, url)
            else:
                documents = self._process_pages_sequential(pdf_doc, pdf_metadata, url)
            
            pdf_doc.close()
            progress_logger.info("✅ PDF parsing completed: %d documents created", len(documents))
            
        except Exception as e:
            if 'pdf_doc' in locals():
                pdf_doc.close()
            progress_logger.error("❌ PDF parsing failed: %s", str(e))
            raise Exception(f"Failed to parse PDF: {e}")
        
        return documents
    
    def _process_pages_sequential(self, pdf_doc, pdf_metadata: Dict[str, Any], url: str) -> List[Document]:
        """Process pages sequentially."""
        progress_logger.info("🔄 Processing pages sequentially...")
        documents = []
        
        for page_num in range(pdf_doc.page_count):
            try:
                page = pdf_doc[page_num]
                doc = self.process_single_page((page_num, page, pdf_metadata, url))
                documents.append(doc)
                
                if (page_num + 1) % 10 == 0 or page_num + 1 == pdf_doc.page_count:
                    progress_logger.info("📄 Processed page %d/%d (%.1f%% complete)", 
                                       page_num + 1, pdf_doc.page_count, 
                                       ((page_num + 1) / pdf_doc.page_count) * 100)
            except Exception as e:
                logger.error("Error processing page %d: %s", page_num + 1, str(e))
                continue
        
        return documents
    
    def _process_pages_parallel(self, pdf_doc, pdf_metadata: Dict[str, Any], url: str) -> List[Document]:
        """Process pages in parallel."""
        progress_logger.info("🚀 Processing pages in parallel with %d workers...", 
                           min(self.max_workers, pdf_doc.page_count))
        documents = [None] * pdf_doc.page_count
        
        page_data_list = []
        for page_num in range(pdf_doc.page_count):
            page = pdf_doc[page_num]
            page_data_list.append((page_num, page, pdf_metadata, url))
        
        with ThreadPoolExecutor(max_workers=min(self.max_workers, pdf_doc.page_count)) as executor:
            future_to_page = {
                executor.submit(self.process_single_page, page_data): page_data[0] 
                for page_data in page_data_list
            }
            
            completed_count = 0
            for future in as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    doc = future.result()
                    documents[page_num] = doc
                    completed_count += 1
                    
                    if completed_count % 10 == 0 or completed_count == pdf_doc.page_count:
                        progress_logger.info("📄 Processed %d/%d pages (%.1f%% complete)", 
                                           completed_count, pdf_doc.page_count, 
                                           (completed_count / pdf_doc.page_count) * 100)
                except Exception as e:
                    logger.error("Error processing page %d: %s", page_num + 1, str(e))
        
        valid_documents = [doc for doc in documents if doc is not None]
        progress_logger.info("✅ Parallel processing completed: %d/%d pages successful", 
                           len(valid_documents), pdf_doc.page_count)
        return valid_documents

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
        """Initialize the chunker with optimal parameters for financial documents."""
        self.target_chunk_size = target_chunk_size
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_size = overlap_size
        self.semantic_threshold = semantic_threshold
        self.use_semantic_model = use_semantic_model
        
        progress_logger.info("🔪 FinancialPolicyChunker initialized - Target: %d, Max: %d, Min: %d, Overlap: %d", 
                           target_chunk_size, max_chunk_size, min_chunk_size, overlap_size)
        
        self.semantic_model = None
        if use_semantic_model:
            try:
                progress_logger.info("🧠 Loading semantic model for advanced chunking...")
                self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
                progress_logger.info("✅ Semantic model loaded successfully")
            except Exception as e:
                progress_logger.warning("❌ Could not load semantic model, falling back to rule-based chunking: %s", str(e))
        
        self.section_patterns = [
            r'^(\d+\.?\d*)\s+(PREAMBLE|DEFINITIONS|COVERAGE|EXCLUSIONS|CONDITIONS|CLAIMS|PREMIUM)',
            r'^(\d+\.?\d*)\s+([A-Z][A-Z\s&-]+)$',
            r'^([A-Z][A-Z\s&-]+):?\s*$',
            r'^\*\*([^*]+)\*\*$',
            r'^(Table of Benefits|Benefits|Features|Plans):',
            r'^(Optional covers|Add-ons|Discounts):',
        ]
        
        self.sentence_endings = r'[.!?]\s+(?=[A-Z])'
        
        self.table_patterns = [
            r'\*\*Table \d+\*\*',
            r'\|.*\|.*\|',
            r'^\s*\|[-:]+\|',
        ]
    
    def identify_document_structure(self, text: str) -> List[Dict[str, Any]]:
        """Identify hierarchical structure in the document."""
        progress_logger.debug("🔍 Identifying document structure...")
        sections = []
        lines = text.split('\n')
        current_pos = 0
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                current_pos += len(line) + 1
                continue
            
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
        
        sections.append({
            'start_pos': len(text),
            'line_num': len(lines),
            'header': 'END',
            'type': 'end',
            'level': 0
        })
        
        progress_logger.debug("📋 Identified %d sections in document", len(sections) - 1)
        return sections
    
    def _determine_header_level(self, header: str) -> int:
        """Determine the hierarchical level of a header"""
        number_match = re.match(r'^(\d+(?:\.\d+)*)', header.strip())
        if number_match:
            return len(number_match.group(1).split('.'))
        
        if header.startswith('**') or header.isupper():
            return 1
        
        return 2
    
    def extract_hierarchical_chunks(self, text: str, page_metadata: Dict[str, Any]) -> List[Chunk]:
        """Extract chunks based on document hierarchy."""
        progress_logger.debug("🏗️ Extracting hierarchical chunks...")
        sections = self.identify_document_structure(text)
        chunks = []
        
        for i in range(len(sections) - 1):
            current_section = sections[i]
            next_section = sections[i + 1]
            
            section_start = current_section['start_pos']
            section_end = next_section['start_pos']
            section_content = text[section_start:section_end].strip()
            
            if len(section_content) < self.min_chunk_size:
                continue
            
            section_metadata = {
                **page_metadata,
                'chunk_type': 'hierarchical',
                'section_header': current_section['header'],
                'section_type': current_section['type'],
                'section_level': current_section['level'],
                'hierarchical_path': self._build_hierarchical_path(sections, i)
            }
            
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
        
        progress_logger.debug("📦 Created %d hierarchical chunks", len(chunks))
        return chunks
    
    def _build_hierarchical_path(self, sections: List[Dict], current_idx: int) -> str:
        """Build a hierarchical path for the current section"""
        path_parts = []
        current_level = sections[current_idx]['level']
        
        for i in range(current_idx, -1, -1):
            section = sections[i]
            if section['level'] < current_level:
                path_parts.insert(0, section['header'])
                current_level = section['level']
        
        path_parts.append(sections[current_idx]['header'])
        return ' > '.join(path_parts)
    
    def apply_semantic_chunking(self, text: str, base_metadata: Dict[str, Any], start_offset: int = 0) -> List[Chunk]:
        """Apply semantic chunking to break text into meaningful segments."""
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return []
        
        if self.semantic_model and len(sentences) > 2:
            chunks = self._semantic_grouping_with_model(sentences, base_metadata, start_offset)
        else:
            chunks = self._rule_based_semantic_grouping(sentences, base_metadata, start_offset)
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences with policy-specific rules"""
        text = re.sub(r'(\d+\.\d+)%', r'\1 percent', text)
        text = re.sub(r'(Rs\.|INR)\s*(\d+)', r'\1 \2', text)
        text = re.sub(r'(\d+)\s*lacs?', r'\1 lacs', text)
        
        sentences = re.split(self.sentence_endings, text)
        
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:
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
            
            if (current_length + sentence_length > self.target_chunk_size and 
                current_chunk_sentences and 
                current_length > self.min_chunk_size):
                
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
                
                if self.overlap_size > 0 and current_chunk_sentences:
                    overlap_sentences = current_chunk_sentences[-1:]
                    current_chunk_sentences = overlap_sentences + [sentence]
                    current_length = sum(len(s) for s in current_chunk_sentences)
                else:
                    current_chunk_sentences = [sentence]
                    current_length = sentence_length
                
                char_offset += len(content) - (len(overlap_sentences[0]) if overlap_sentences else 0)
            else:
                current_chunk_sentences.append(sentence)
                current_length += sentence_length
        
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
        id_string = f"{content[:100]}_{metadata.get('page', 0)}_{metadata.get('chunk_type', '')}"
        return hashlib.md5(id_string.encode()).hexdigest()[:12]
    
    def chunk_document(self, document) -> List[Chunk]:
        """Main method to chunk a document using the hybrid strategy."""
        progress_logger.debug("⚡ Starting document chunking...")
        text = document.page_content
        metadata = document.metadata
        
        chunks = []
        
        if metadata.get('has_tables', False):
            progress_logger.debug("📊 Removing table content from text for chunking")
            text = re.sub(r'\*\*Tables on Page \d+:\*\*.*?(?=\*\*Page \d+|\Z)', '', text, flags=re.DOTALL)
        
        hierarchical_chunks = self.extract_hierarchical_chunks(text, metadata)
        chunks.extend(hierarchical_chunks)
        
        chunks = self._post_process_chunks(chunks)
        
        progress_logger.debug("✅ Document chunking completed: %d final chunks", len(chunks))
        return chunks
    
    def _post_process_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Post-process chunks to ensure quality and consistency"""
        processed_chunks = []
        
        for chunk in chunks:
            if len(chunk.content.strip()) < self.min_chunk_size:
                continue
            
            clean_content = self._clean_chunk_content(chunk.content)
            
            chunk.content = clean_content
            chunk.metadata['final_char_count'] = len(clean_content)
            chunk.metadata['final_word_count'] = len(clean_content.split())
            
            processed_chunks.append(chunk)
        
        return processed_chunks
    
    def _clean_chunk_content(self, content: str) -> str:
        """Clean chunk content for better readability"""
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'\*\*\s*\*\*', '', content)
        
        content = content.strip()
        if content and not content.endswith(('.', '!', '?', ':')):
            content += '.'
        
        return content

class TwoStageRAGService:
    """Enhanced RAG service with conditional two-stage retrieval for large documents."""
    
    def __init__(self):
        self.vector_store = None
        self.chain = None
        self.ensemble_retriever = None
        self._setup_complete = False
        self.embeddings = None
        self.mongo_client = None
        
        # Enhanced caching system
        self._retrieval_cache = {}
        self._cache_lock = threading.Lock()
        self._context_cache = {}
        
        # Two-stage specific attributes
        self.faiss_store = None
        self.candidate_chunks = []
        self.page_threshold = 300  # Configurable threshold
        
        progress_logger.info("🚀 TwoStageRAGService initialized with page threshold: %d", self.page_threshold)
        
    async def _setup_dependencies(self):
        """Setup dependencies only when needed."""
        if self._setup_complete:
            return
            
        progress_logger.info("📦 Setting up dependencies...")
        
        # Import heavy libraries
        from langchain_voyageai import VoyageAIEmbeddings
        from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
        from langchain_openai import ChatOpenAI
        from langchain_core.documents import Document as LangchainDocument
        from langchain.prompts import PromptTemplate
        from langchain.schema.runnable import RunnablePassthrough
        from langchain_core.output_parsers import StrOutputParser
        from langchain.retrievers import BM25Retriever
        from langchain.retrievers import EnsembleRetriever
        from langchain_community.vectorstores import FAISS
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
        self.EnsembleRetriever = EnsembleRetriever
        self.FAISS = FAISS
        self.MongoClient = MongoClient
        
        progress_logger.info("✅ Dependencies setup completed")
        self._setup_complete = True

    async def setup_vector_store(self):
        """Setup vector store connection with optimized MongoDB client."""
        if self.vector_store is None:
            progress_logger.info("🔗 Setting up vector store connection...")
            await self._setup_dependencies()
            
            DB_NAME = "Voyage_ai_RAG"
            COLLECTION_NAME = "langhcain_Voyage"
            
            progress_logger.info("🍃 Connecting to MongoDB Atlas...")
            # Optimized MongoDB client with connection pooling
            self.mongo_client = self.MongoClient(
                os.getenv("MONGODB_URI"),
                maxPoolSize=50,
                minPoolSize=10,
                retryWrites=True,
                w="majority"
            )
            
            progress_logger.info("🚢 Initializing VoyageAI embeddings...")
            # Initialize embeddings with optimized settings
            self.embeddings = self.VoyageAIEmbeddings(
                model="voyage-3-large",
                batch_size=72,
            )
            
            progress_logger.info("🎯 Setting up MongoDB Atlas Vector Search...")
            # Initialize vector store
            self.vector_store = self.MongoDBAtlasVectorSearch(
                collection=self.mongo_client[DB_NAME][COLLECTION_NAME],
                embedding=self.embeddings,
                index_name="vector_index",
                text_key="text",
                embedding_key="embedding"
            )
            
            progress_logger.info("✅ Vector store setup completed")

    def determine_processing_strategy(self, page_count: int) -> str:
        """Determine which processing strategy to use based on page count."""
        strategy = "two_stage" if page_count > self.page_threshold else "standard"
        progress_logger.info("🎯 Strategy Selection: %s (pages: %d, threshold: %d)", 
                           strategy, page_count, self.page_threshold)
        return strategy

    async def stage_1_candidate_selection(self, documents: List, questions: List[str], k_candidates: int = 200) -> List:
        """
        Stage 1: Use BM25 for fast candidate selection from all chunks.
        
        Args:
            documents: All document chunks
            questions: List of questions to create search queries
            k_candidates: Number of candidate chunks to select
            
        Returns:
            List of candidate documents for Stage 2
        """
        progress_logger.info("🎯 STAGE 1: Starting candidate selection")
        progress_logger.info("📊 Input: %d total chunks, %d questions, target: %d candidates", 
                           len(documents), len(questions), k_candidates)
        
        start_time = time.time()
        
        # Create comprehensive search query from all questions
        combined_query = " ".join(questions)
        progress_logger.info("🔍 Combined search query length: %d characters", len(combined_query))
        
        # Build BM25 retriever on all documents
        progress_logger.info("⚙️ Building BM25 retriever on full document set...")
        bm25_retriever = self.BM25Retriever.from_documents(documents)
        bm25_retriever.k = k_candidates
        
        # Retrieve candidate chunks
        progress_logger.info("🔍 Executing BM25 search for candidates...")
        candidate_docs = bm25_retriever.get_relevant_documents(combined_query)
        
        stage1_time = time.time() - start_time
        progress_logger.info("✅ STAGE 1 COMPLETED: Selected %d/%d candidates in %.2f seconds", 
                           len(candidate_docs), len(documents), stage1_time)
        progress_logger.info("📈 Candidate selection efficiency: %.1f%% of original chunks", 
                           (len(candidate_docs) / len(documents)) * 100)
        
        return candidate_docs

    async def stage_2_in_memory_hybrid_search(self, candidate_docs: List) -> None:
        """
        Stage 2: Create in-memory FAISS store and hybrid retriever for candidates.
        
        Args:
            candidate_docs: Candidate documents from Stage 1
        """
        progress_logger.info("🎯 STAGE 2: Setting up in-memory hybrid search")
        progress_logger.info("📊 Input: %d candidate documents", len(candidate_docs))
        
        start_time = time.time()
        
        # Extract texts for embedding
        candidate_texts = [doc.page_content for doc in candidate_docs]
        avg_text_length = sum(len(text) for text in candidate_texts) / len(candidate_texts)
        progress_logger.info("📝 Candidate texts - Average length: %.0f characters", avg_text_length)
        
        # Compute embeddings in batches
        progress_logger.info("🧮 Computing embeddings for candidate chunks...")
        embedding_start = time.time()
        
        all_embeddings = []
        batch_size = 72  # Voyage AI batch size
        total_batches = (len(candidate_texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(candidate_texts), batch_size):
            batch_num = (i // batch_size) + 1
            batch = candidate_texts[i:i + batch_size]
            
            progress_logger.info("🔄 Processing embedding batch %d/%d (%d texts)", 
                               batch_num, total_batches, len(batch))
            
            batch_embeddings = self.embeddings.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)
        
        embedding_time = time.time() - embedding_start
        progress_logger.info("✅ Embeddings computed: %d vectors in %.2f seconds (%.2f vectors/sec)", 
                           len(all_embeddings), embedding_time, len(all_embeddings) / embedding_time)
        
        # Create in-memory FAISS vector store
        progress_logger.info("🏗️ Creating in-memory FAISS vector store...")
        faiss_start = time.time()
        
        self.faiss_store = self.FAISS.from_documents(
            candidate_docs,
            self.embeddings
        )
        
        faiss_time = time.time() - faiss_start
        progress_logger.info("✅ FAISS store created in %.2f seconds", faiss_time)
        
        # Create hybrid retriever with FAISS + BM25 on candidates
        progress_logger.info("🔧 Setting up hybrid retriever...")
        
        # Vector retriever from FAISS
        faiss_retriever = self.faiss_store.as_retriever(
            search_type="mmr",
            search_kwargs={"fetch_k": 15, "k": 6, "lambda_mult": 0.3}
        )
        progress_logger.info("🎯 FAISS retriever configured: MMR search, fetch_k=15, k=6, lambda=0.3")
        
        # BM25 retriever on candidate pool
        candidate_bm25 = self.BM25Retriever.from_documents(candidate_docs)
        candidate_bm25.k = 4
        progress_logger.info("📝 BM25 retriever configured: k=4")
        
        # Create ensemble retriever
        self.ensemble_retriever = self.EnsembleRetriever(
            retrievers=[faiss_retriever, candidate_bm25],
            weights=[0.7, 0.3]  # Favor semantic search slightly more
        )
        
        stage2_time = time.time() - start_time
        progress_logger.info("✅ STAGE 2 COMPLETED: Hybrid search ready in %.2f seconds", stage2_time)
        progress_logger.info("⚖️ Ensemble weights: FAISS=70%, BM25=30%")

    async def process_document_async(self, url: str) -> Tuple[List, List, int]:
        """Process document and return documents, parsed pages, and page count."""
        progress_logger.info("📄 Starting document processing pipeline")
        progress_logger.info("🔗 Document URL: %s", url)
        
        await self._setup_dependencies()
        
        def process_sync():
            pipeline_start = time.time()
            
            # Step 1: PDF Parsing
            progress_logger.info("🔄 Step 1/3: PDF Parsing")
            parser = PDFParser(max_workers=8)
            parsed_documents = parser.parse_pdf_from_url(url, use_parallel=True)
            
            # Get page count from metadata
            page_count = 0
            if parsed_documents:
                page_count = parsed_documents[0].metadata.get('total_pages', len(parsed_documents))
            
            progress_logger.info("📊 PDF Parsing Results: %d pages, %d page documents", 
                               page_count, len(parsed_documents))
            
            # Step 2: Document Chunking
            progress_logger.info("🔄 Step 2/3: Document Chunking")
            chunker = FinancialPolicyChunker(
                target_chunk_size=1000,
                max_chunk_size=1500,
                min_chunk_size=200,
                overlap_size=100,
                use_semantic_model=False
            )
            
            all_chunks = []
            for doc_idx, doc in enumerate(parsed_documents):
                if (doc_idx + 1) % 10 == 0 or doc_idx + 1 == len(parsed_documents):
                    progress_logger.info("🔪 Chunking document %d/%d", doc_idx + 1, len(parsed_documents))
                
                page_chunks = chunker.chunk_document(doc)
                all_chunks.extend(page_chunks)
            
            progress_logger.info("📦 Chunking Results: %d total chunks created", len(all_chunks))
            
            # Step 3: Document Conversion
            progress_logger.info("🔄 Step 3/3: Converting to LangChain documents")
            documents = [
                self.LangchainDocument(
                    page_content=chunk.content,
                    metadata={**chunk.metadata}
                )
                for chunk in all_chunks
            ]
            
            # Calculate statistics
            total_chars = sum(len(doc.page_content) for doc in documents)
            avg_chunk_size = total_chars / len(documents) if documents else 0
            
            pipeline_time = time.time() - pipeline_start
            progress_logger.info("✅ Document processing pipeline completed in %.2f seconds", pipeline_time)
            progress_logger.info("📈 Final statistics: %d chunks, %.0f avg chars/chunk, %.2f MB total", 
                               len(documents), avg_chunk_size, total_chars / (1024 * 1024))
            
            return documents, parsed_documents, page_count
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, process_sync)

    async def setup_chain_standard(self, documents: List = None):
        """Setup standard chain for smaller documents (< 300 pages)."""
        progress_logger.info("🔧 Setting up STANDARD RAG chain")
        progress_logger.info("📊 Input documents: %d", len(documents) if documents else 0)
        
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
        progress_logger.info("📝 Prompt template configured")
        
        # Setup retrievers
        if documents:
            progress_logger.info("🔍 Setting up BM25 retriever...")
            bm25_retriever = self.BM25Retriever.from_documents(documents)
            bm25_retriever.k = 2
            progress_logger.info("✅ BM25 retriever configured: k=2")
        else:
            bm25_retriever = None
            progress_logger.info("⚠️ No documents provided for BM25 retriever")
        
        progress_logger.info("🎯 Setting up vector retriever...")
        vector_retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"fetch_k": 25, "k": 6, "lambda_mult": 0.3}
        )
        progress_logger.info("✅ Vector retriever configured: MMR, fetch_k=25, k=6, lambda=0.3")
        
        # Create ensemble retriever
        if bm25_retriever:
            progress_logger.info("🎭 Creating ensemble retriever...")
            self.ensemble_retriever = self.EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[0.6, 0.4]
            )
            progress_logger.info("✅ Ensemble retriever created: Vector=60%, BM25=40%")
        else:
            self.ensemble_retriever = vector_retriever
            progress_logger.info("✅ Using vector retriever only")
        
        # Setup model and chain
        progress_logger.info("🤖 Setting up ChatOpenAI model...")
        model = self.ChatOpenAI(
            model_name="gpt-3.5-turbo-1106",
            temperature=0,
            max_tokens=256,
            request_timeout=10,
        )
        
        progress_logger.info("⛓️ Building RAG chain...")
        self.chain = (
            {"context": self._cached_retrieve, "question": self.RunnablePassthrough()}
            | prompt
            | model
            | self.StrOutputParser()
        )
        
        progress_logger.info("✅ STANDARD RAG chain setup completed")

    async def setup_chain_two_stage(self):
        """Setup chain for two-stage retrieval (large documents)."""
        progress_logger.info("🔧 Setting up TWO-STAGE RAG chain")
        
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
        progress_logger.info("📝 Prompt template configured")
        
        # Use the ensemble retriever created in Stage 2
        progress_logger.info("🤖 Setting up ChatOpenAI model...")
        model = self.ChatOpenAI(
            model_name="gpt-3.5-turbo-1106",
            temperature=0,
            max_tokens=256,
            request_timeout=10,
        )
        
        progress_logger.info("⛓️ Building two-stage RAG chain...")
        self.chain = (
            {"context": self._cached_retrieve_two_stage, "question": self.RunnablePassthrough()}
            | prompt
            | model
            | self.StrOutputParser()
        )
        
        progress_logger.info("✅ TWO-STAGE RAG chain setup completed")
        progress_logger.info("🎯 Chain will use in-memory FAISS + BM25 ensemble retriever")

    @lru_cache(maxsize=100)
    def _cached_retrieve(self, question: str):
        """Cache retrieval results for standard retrieval."""
        with self._cache_lock:
            cache_key = f"standard_{question}"
            if cache_key in self._retrieval_cache:
                progress_logger.debug("💾 Cache HIT for standard retrieval")
                return self._retrieval_cache[cache_key]
            
            progress_logger.debug("🔍 Cache MISS - executing standard retrieval")
            docs = self.ensemble_retriever.get_relevant_documents(question)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            self._retrieval_cache[cache_key] = context
            progress_logger.debug("💾 Cached standard retrieval result (%d chars)", len(context))
            return context

    @lru_cache(maxsize=100)
    def _cached_retrieve_two_stage(self, question: str):
        """Cache retrieval results for two-stage retrieval."""
        with self._cache_lock:
            cache_key = f"two_stage_{question}"
            if cache_key in self._retrieval_cache:
                progress_logger.debug("💾 Cache HIT for two-stage retrieval")
                return self._retrieval_cache[cache_key]
            
            progress_logger.debug("🔍 Cache MISS - executing two-stage retrieval")
            docs = self.ensemble_retriever.get_relevant_documents(question)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            self._retrieval_cache[cache_key] = context
            progress_logger.debug("💾 Cached two-stage retrieval result (%d chars)", len(context))
            return context

    def batch_documents(self, documents: List, batch_size: int = 72):
        """Split documents into batches for processing."""
        for i in range(0, len(documents), batch_size):
            yield documents[i:i + batch_size]

    async def add_documents_bulk_optimized(self, documents: List):
        """Optimized bulk document addition with enhanced logging."""
        progress_logger.info("📥 Starting bulk document addition to MongoDB")
        progress_logger.info("📊 Input: %d documents", len(documents))
        
        start_time = time.time()
        
        DB_NAME = "Voyage_ai_RAG"
        COLLECTION_NAME = "langhcain_Voyage"
        
        # Extract texts and filter out empties
        progress_logger.info("🔍 Filtering and preparing documents...")
        raw_entries = [
            (i, doc.page_content if hasattr(doc, 'page_content') else str(doc))
            for i, doc in enumerate(documents)
        ]
        
        filtered = [(i, text) for i, text in raw_entries if text and text.strip()]
        if not filtered:
            progress_logger.warning("⚠️ No non-empty documents to embed")
            return 0

        indices, texts = zip(*filtered)
        progress_logger.info("✅ Filtered documents: %d valid out of %d total", len(texts), len(documents))
        
        # Compute embeddings in batches
        progress_logger.info("🧮 Computing embeddings...")
        embedding_start = time.time()
        
        all_embeddings = []
        batch_size = 72
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for batch_num, batch in enumerate(self.batch_documents(texts, batch_size), 1):
            progress_logger.info("🔄 Processing embedding batch %d/%d (%d documents)", 
                               batch_num, total_batches, len(batch))
            batch_embeddings = self.embeddings.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)

        embedding_time = time.time() - embedding_start
        progress_logger.info("✅ Embeddings computed: %d vectors in %.2f seconds (%.2f vectors/sec)", 
                           len(all_embeddings), embedding_time, len(all_embeddings) / embedding_time)

        # Prepare documents for insert
        progress_logger.info("📦 Preparing documents for bulk insert...")
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

        # Bulk insert
        collection = self.mongo_client[DB_NAME][COLLECTION_NAME]
        progress_logger.info("💾 Executing bulk insert to MongoDB...")
        insert_start = time.time()

        try:
            result = collection.insert_many(bulk_docs, ordered=False)
            inserted_count = len(result.inserted_ids)
            
            insert_time = time.time() - insert_start
            total_time = time.time() - start_time
            
            progress_logger.info("✅ Bulk insert completed: %d documents in %.2f seconds (%.2f docs/sec)", 
                               inserted_count, insert_time, inserted_count / insert_time)
            progress_logger.info("🎯 Total bulk addition time: %.2f seconds", total_time)
            
            return inserted_count
        except Exception as e:
            progress_logger.error("❌ Bulk insert failed: %s", str(e))
            return 0

    async def answer_questions_parallel(self, questions: List[str], max_workers: int = 3) -> List[str]:
        """Answer multiple questions concurrently with enhanced logging."""
        progress_logger.info("🔄 Starting parallel question answering")
        progress_logger.info("📊 Input: %d questions, max_workers: %d", len(questions), max_workers)
        
        start_time = time.time()
        
        def answer_single(question: str) -> str:
            question_start = time.time()
            try:
                result = self.chain.invoke(question)
                question_time = time.time() - question_start
                progress_logger.debug("✅ Question answered in %.2f seconds: %s...", 
                                     question_time, question[:50])
                return result
            except Exception as e:
                question_time = time.time() - question_start
                progress_logger.error("❌ Error answering question in %.2f seconds: %s - %s", 
                                     question_time, question[:50], str(e))
                return f"Error processing question: {str(e)}"
        
        results = []
        completed_count = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as thread_executor:
            future_to_question = {
                thread_executor.submit(answer_single, q): q for q in questions
            }
            
            for future in as_completed(future_to_question):
                question = future_to_question[future]
                try:
                    answer = future.result(timeout=30)
                    results.append(answer)
                    completed_count += 1
                    
                    progress_logger.info("✅ Question %d/%d completed: %s...", 
                                       completed_count, len(questions), question[:30])
                except Exception as e:
                    progress_logger.error("❌ Error processing question '%s': %s", question[:50], str(e))
                    results.append(f"Error: {str(e)}")
                    completed_count += 1
        
        total_time = time.time() - start_time
        progress_logger.info("🎯 Parallel answering completed: %d questions in %.2f seconds (%.2f q/sec)", 
                           len(questions), total_time, len(questions) / total_time)
        
        return results

    async def answer_questions_smart_batch(self, questions: List[str]) -> List[str]:
        """Smart batching with context reuse and enhanced logging."""
        progress_logger.info("🧠 Starting smart batch question answering")
        progress_logger.info("📊 Input: %d questions", len(questions))
        
        start_time = time.time()
        
        # Pre-retrieve all unique contexts
        progress_logger.info("🔍 Phase 1: Pre-retrieving contexts...")
        retrieval_start = time.time()
        
        unique_contexts = {}
        question_contexts = {}
        
        for i, question in enumerate(questions):
            if hasattr(self, '_cached_retrieve_two_stage') and self.faiss_store:
                context = self._cached_retrieve_two_stage(question)
            else:
                context = self._cached_retrieve(question)
                
            context_hash = hash(context)
            unique_contexts[context_hash] = context
            question_contexts[question] = context_hash
            
            if (i + 1) % 10 == 0 or i + 1 == len(questions):
                progress_logger.info("🔍 Retrieved context for question %d/%d", i + 1, len(questions))
        
        retrieval_time = time.time() - retrieval_start
        progress_logger.info("✅ Context retrieval completed: %d unique contexts in %.2f seconds", 
                           len(unique_contexts), retrieval_time)
        
        # Process questions with pre-retrieved contexts
        progress_logger.info("🤖 Phase 2: Processing questions with cached contexts...")
        processing_start = time.time()
        
        results = []
        for i, question in enumerate(questions):
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
                
                if (i + 1) % 5 == 0 or i + 1 == len(questions):
                    progress_logger.info("🤖 Processed question %d/%d", i + 1, len(questions))
                    
            except Exception as e:
                progress_logger.error("❌ Error processing question %d: %s", i + 1, str(e))
                results.append(f"Error: {str(e)}")
        
        processing_time = time.time() - processing_start
        total_time = time.time() - start_time
        
        progress_logger.info("✅ Smart batch processing completed:")
        progress_logger.info("   - Retrieval: %.2f seconds", retrieval_time)
        progress_logger.info("   - Processing: %.2f seconds", processing_time)
        progress_logger.info("   - Total: %.2f seconds (%.2f q/sec)", total_time, len(questions) / total_time)
        progress_logger.info("   - Context reuse efficiency: %.1f%% (saved %d retrievals)", 
                           (1 - len(unique_contexts) / len(questions)) * 100,
                           len(questions) - len(unique_contexts))
        
        return results
    
    async def answer_questions(self, questions: List[str], method: str = "parallel") -> List[str]:
        """Answer multiple questions using specified method with enhanced logging."""
        progress_logger.info("❓ Starting question answering phase")
        progress_logger.info("📊 Questions: %d, Method: %s", len(questions), method)
        
        if method == "parallel":
            return await self.answer_questions_parallel(questions, max_workers=3)
        elif method == "smart_batch":
            return await self.answer_questions_smart_batch(questions)
        else:
            progress_logger.info("📝 Using fallback sequential method")
            # Fallback to original method
            def answer_single(question: str) -> str:
                try:
                    return self.chain.invoke(question)
                except Exception as e:
                    progress_logger.error("❌ Error answering question '%s': %s", question[:50], str(e))
                    return f"Error processing question: {str(e)}"
            
            results = []
            
            with ThreadPoolExecutor(max_workers=3) as thread_executor:
                future_to_question = {
                    thread_executor.submit(answer_single, q): q for q in questions
                }
                
                for future in as_completed(future_to_question):
                    question = future_to_question[future]
                    try:
                        answer = future.result(timeout=30)
                        results.append(answer)
                    except Exception as e:
                        progress_logger.error("❌ Error processing question '%s': %s", question[:50], str(e))
                        results.append(f"Error: {str(e)}")
            
            return results
    
    async def run(self, documents_url: str, questions: List[str], processing_method: str = "parallel") -> List[str]:
        """
        Main run method with conditional two-stage retrieval and comprehensive logging.
        
        Strategy:
        - For documents > 300 pages: Use two-stage retrieval (BM25 → In-memory FAISS + Hybrid)
        - For documents ≤ 300 pages: Use standard MongoDB + Ensemble retrieval
        """
        try:
            start_time = time.time()
            progress_logger.info("🚀 ===== STARTING RAG PROCESSING PIPELINE =====")
            progress_logger.info("📊 Configuration:")
            progress_logger.info("   - Document URL: %s", documents_url)
            progress_logger.info("   - Questions: %d", len(questions))
            progress_logger.info("   - Processing method: %s", processing_method)
            progress_logger.info("   - Page threshold: %d", self.page_threshold)
            
            # Setup vector store
            progress_logger.info("🔄 PHASE 1: Vector Store Setup")
            await self.setup_vector_store()
            
            # Process document and get page count
            progress_logger.info("🔄 PHASE 2: Document Processing")
            documents, parsed_pages, page_count = await self.process_document_async(documents_url)
            
            progress_logger.info("📈 Document Processing Results:")
            progress_logger.info("   - Pages: %d", page_count)
            progress_logger.info("   - Chunks: %d", len(documents))
            progress_logger.info("   - Avg chunk size: %.0f chars", 
                               sum(len(doc.page_content) for doc in documents) / len(documents) if documents else 0)
            
            # Determine processing strategy
            strategy = self.determine_processing_strategy(page_count)
            
            if strategy == "two_stage":
                # 🚀 PATH A: Two-Stage Retrieval for Large Documents
                progress_logger.info("🔄 PHASE 3: TWO-STAGE RETRIEVAL STRATEGY")
                progress_logger.info("🎯 Large document detected - executing advanced two-stage approach")
                
                # Stage 1: Candidate Selection with BM25
                candidate_docs = await self.stage_1_candidate_selection(
                    documents, questions, k_candidates=200
                )
                
                # Stage 2: In-Memory Hybrid Search
                await self.stage_2_in_memory_hybrid_search(candidate_docs)
                
                # Setup chain for two-stage retrieval
                progress_logger.info("🔄 PHASE 4: Two-Stage Chain Setup")
                await self.setup_chain_two_stage()
                
                # No MongoDB insertion needed for two-stage approach
                progress_logger.info("✅ Two-stage setup completed - MongoDB insertion skipped for efficiency")
                
            else:
                # 📚 PATH B: Standard Strategy for Smaller Documents
                progress_logger.info("🔄 PHASE 3: STANDARD RETRIEVAL STRATEGY")
                progress_logger.info("🎯 Standard document size - using MongoDB + Ensemble approach")
                
                # Setup standard chain
                progress_logger.info("🔄 PHASE 4: Standard Chain Setup")
                await self.setup_chain_standard(documents)
                
                # Add to MongoDB vector store
                progress_logger.info("🔄 PHASE 5: MongoDB Document Addition")
                total_added = await self.add_documents_bulk_optimized(documents)
                progress_logger.info("✅ MongoDB insertion completed: %d documents added", total_added)
            
            # Answer questions using specified processing method
            progress_logger.info("🔄 PHASE 6: Question Answering")
            progress_logger.info("🎯 Strategy: %s | Method: %s", strategy.upper(), processing_method.upper())
            
            answers = await self.answer_questions(questions, method=processing_method)
            
            # Final statistics and summary
            total_time = time.time() - start_time
            progress_logger.info("🎉 ===== RAG PROCESSING COMPLETED =====")
            progress_logger.info("📊 Final Results:")
            progress_logger.info("   - Strategy used: %s", strategy.upper())
            progress_logger.info("   - Total processing time: %.2f seconds", total_time)
            progress_logger.info("   - Questions answered: %d/%d", len(answers), len(questions))
            progress_logger.info("   - Average time per question: %.2f seconds", total_time / len(questions) if questions else 0)
            progress_logger.info("   - Processing efficiency: %.2f questions/second", len(questions) / total_time if total_time > 0 else 0)
            
            # Log answer preview
            for i, answer in enumerate(answers[:3]):  # Show first 3 answers
                preview = answer[:100] + "..." if len(answer) > 100 else answer
                progress_logger.info("   - Answer %d preview: %s", i + 1, preview)
            
            if len(answers) > 3:
                progress_logger.info("   - ... and %d more answers", len(answers) - 3)
            
            return answers
            
        except Exception as e:
            progress_logger.error("❌ ===== RAG PROCESSING FAILED =====")
            progress_logger.error("💥 Error details: %s", str(e))
            progress_logger.error("🕒 Failed after %.2f seconds", time.time() - start_time)
            logger.error("Error in RAG service run: %s", str(e))
            raise

 #Add these imports at the top of your rag_service.py file (after existing imports)
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# ========================================
# STEP 2: Replace the existing OptimizedRAGService class in your rag_service.py
# ========================================

class OptimizedRAGService(TwoStageRAGService):
    """
    Enhanced RAG service that automatically chooses between:
    1. Two-stage retrieval for large documents (>300 pages) - NOW WITH ENHANCED RETRIEVAL
    2. Standard retrieval for smaller documents (≤300 pages)
    """
    
    def __init__(self, page_threshold: int = 300):
        super().__init__()
        self.page_threshold = page_threshold
        
        # Enhanced parameters for better retrieval (NEW)
        self.stage1_candidates = 400  # Increased from 200
        self.stage2_vector_k = 12     # Increased from 6
        self.stage2_bm25_k = 8        # Increased from 4
        self.final_ensemble_k = 15    # Final number of documents to return
        
        # Multi-query expansion parameters (NEW)
        self.use_query_expansion = True
        self.max_expanded_queries = 3
        
        # Re-ranking parameters (NEW)
        self.use_mmr_reranking = True
        self.mmr_diversity_threshold = 0.7
        
        progress_logger.info("🎯 OptimizedRAGService initialized with ENHANCED retrieval")
        progress_logger.info("📊 Configuration: page_threshold=%d", page_threshold)
        progress_logger.info("🔧 Enhanced parameters:")
        progress_logger.info("   - Stage 1 candidates: %d", self.stage1_candidates)
        progress_logger.info("   - Stage 2 vector k: %d, BM25 k: %d", self.stage2_vector_k, self.stage2_bm25_k)
        progress_logger.info("   - Final ensemble k: %d", self.final_ensemble_k)
        progress_logger.info("   - Query expansion: %s", self.use_query_expansion)
        progress_logger.info("   - MMR re-ranking: %s", self.use_mmr_reranking)

    # NEW METHOD: Add this method to your OptimizedRAGService class
    def expand_query(self, question: str) -> List[str]:
        """
        Expand the original question into multiple related queries for better coverage.
        This helps capture different ways the same information might be expressed.
        """
        if not self.use_query_expansion:
            return [question]
        
        # Extract key terms and concepts
        expanded_queries = [question]  # Always include original
        
        # Create variations based on financial/insurance terminology
        financial_synonyms = {
            'premium': ['premium amount', 'policy cost', 'insurance cost'],
            'coverage': ['benefits', 'protection', 'policy benefits'],
            'claim': ['claim process', 'claim settlement', 'reimbursement'],
            'exclusion': ['not covered', 'excluded', 'limitations'],
            'deductible': ['excess', 'co-payment', 'out-of-pocket'],
            'benefit': ['coverage amount', 'sum assured', 'policy limit']
        }
        
        # Add synonym-based expansions
        question_lower = question.lower()
        for term, synonyms in financial_synonyms.items():
            if term in question_lower:
                for synonym in synonyms[:1]:  # Add top synonym
                    expanded_query = question_lower.replace(term, synonym)
                    if expanded_query != question_lower:
                        expanded_queries.append(expanded_query)
                        if len(expanded_queries) >= self.max_expanded_queries:
                            break
                break
        
        # Add question type variations
        if question.startswith(('What is', 'What are')):
            variant = question.replace('What is', 'Details about').replace('What are', 'Information on')
            expanded_queries.append(variant)
        elif question.startswith('How'):
            variant = question.replace('How', 'Process for')
            expanded_queries.append(variant)
        
        expanded_queries = expanded_queries[:self.max_expanded_queries]
        
        if len(expanded_queries) > 1:
            progress_logger.info("🔍 Query expansion: %d variants generated", len(expanded_queries))
            for i, q in enumerate(expanded_queries):
                progress_logger.debug("   %d: %s", i+1, q[:80])
        
        return expanded_queries

    # NEW METHOD: Add this method to your OptimizedRAGService class
    async def stage_1_enhanced_candidate_selection(self, documents: List, questions: List[str]) -> List:
        """
        Enhanced Stage 1: Multi-query candidate selection with better coverage.
        """
        progress_logger.info("🎯 ENHANCED STAGE 1: Starting advanced candidate selection")
        progress_logger.info("📊 Input: %d total chunks, %d questions, target: %d candidates", 
                           len(documents), len(questions), self.stage1_candidates)
        
        start_time = time.time()
        
        # Expand all questions for better coverage
        all_expanded = []
        for q in questions:
            expanded = self.expand_query(q)
            all_expanded.extend(expanded)
        
        progress_logger.info("🔍 Query expansion results: %d original → %d expanded queries", 
                           len(questions), len(all_expanded))
        
        # Create comprehensive search queries
        primary_query = " ".join(questions)  # Original approach
        expanded_query = " ".join(all_expanded)  # Expanded approach
        
        # Build BM25 retriever with optimized parameters
        progress_logger.info("⚙️ Building enhanced BM25 retriever...")
        bm25_retriever = self.BM25Retriever.from_documents(documents)
        
        # Multi-query retrieval with different k values
        candidate_docs = []
        seen_content = set()
        
        # Query 1: High-recall retrieval with expanded query
        bm25_retriever.k = int(self.stage1_candidates * 0.7)  # 70% from expanded query
        candidates_1 = bm25_retriever.get_relevant_documents(expanded_query)
        
        for doc in candidates_1:
            if doc.page_content not in seen_content:
                candidate_docs.append(doc)
                seen_content.add(doc.page_content)
        
        progress_logger.info("📈 First pass: %d candidates from expanded query", len(candidates_1))
        
        # Query 2: Focused retrieval with original questions
        remaining_k = self.stage1_candidates - len(candidate_docs)
        if remaining_k > 0:
            bm25_retriever.k = remaining_k
            candidates_2 = bm25_retriever.get_relevant_documents(primary_query)
            
            for doc in candidates_2:
                if doc.page_content not in seen_content and len(candidate_docs) < self.stage1_candidates:
                    candidate_docs.append(doc)
                    seen_content.add(doc.page_content)
        
        progress_logger.info("📈 Second pass: %d additional candidates", len(candidate_docs) - len(candidates_1))
        
        # Query 3: Individual question targeting for remaining slots
        remaining_k = self.stage1_candidates - len(candidate_docs)
        if remaining_k > 0:
            per_question_k = max(1, remaining_k // len(questions))
            bm25_retriever.k = per_question_k
            
            for question in questions:
                if len(candidate_docs) >= self.stage1_candidates:
                    break
                    
                question_candidates = bm25_retriever.get_relevant_documents(question)
                for doc in question_candidates:
                    if doc.page_content not in seen_content and len(candidate_docs) < self.stage1_candidates:
                        candidate_docs.append(doc)
                        seen_content.add(doc.page_content)
        
        stage1_time = time.time() - start_time
        progress_logger.info("✅ ENHANCED STAGE 1 COMPLETED: Selected %d/%d candidates in %.2f seconds", 
                           len(candidate_docs), len(documents), stage1_time)
        progress_logger.info("📈 Coverage efficiency: %.1f%% of original chunks", 
                           (len(candidate_docs) / len(documents)) * 100)
        
        return candidate_docs

    # NEW METHOD: Add this method to your OptimizedRAGService class
    async def stage_2_advanced_hybrid_search(self, candidate_docs: List) -> None:
        """
        Enhanced Stage 2: Advanced hybrid search with improved parameters and MMR.
        """
        progress_logger.info("🎯 ENHANCED STAGE 2: Setting up advanced hybrid search")
        progress_logger.info("📊 Input: %d candidate documents", len(candidate_docs))
        
        start_time = time.time()
        
        # Compute embeddings with progress tracking
        progress_logger.info("🧮 Computing embeddings for candidate chunks...")
        embedding_start = time.time()
        
        # Batch processing for embeddings
        all_embeddings = []
        batch_size = 72
        total_batches = (len(candidate_docs) + batch_size - 1) // batch_size
        
        for i in range(0, len(candidate_docs), batch_size):
            batch_num = (i // batch_size) + 1
            batch_docs = candidate_docs[i:i + batch_size]
            batch_texts = [doc.page_content for doc in batch_docs]
            
            progress_logger.info("🔄 Processing embedding batch %d/%d (%d texts)", 
                               batch_num, total_batches, len(batch_texts))
            
            batch_embeddings = self.embeddings.embed_documents(batch_texts)
            all_embeddings.extend(batch_embeddings)
        
        embedding_time = time.time() - embedding_start
        progress_logger.info("✅ Embeddings computed: %d vectors in %.2f seconds", 
                           len(all_embeddings), embedding_time)
        
        # Create enhanced FAISS vector store
        progress_logger.info("🏗️ Creating enhanced FAISS vector store...")
        self.faiss_store = self.FAISS.from_documents(candidate_docs, self.embeddings)
        
        # Enhanced vector retriever with MMR
        faiss_retriever = self.faiss_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "fetch_k": min(25, len(candidate_docs)),  # Adaptive fetch_k
                "k": self.stage2_vector_k,
                "lambda_mult": 0.4  # Increased diversity
            }
        )
        progress_logger.info("🎯 Enhanced FAISS retriever: MMR, fetch_k=%d, k=%d, lambda=0.4", 
                           min(25, len(candidate_docs)), self.stage2_vector_k)
        
        # Enhanced BM25 retriever on candidates
        candidate_bm25 = self.BM25Retriever.from_documents(candidate_docs)
        candidate_bm25.k = self.stage2_bm25_k
        progress_logger.info("📝 Enhanced BM25 retriever: k=%d", self.stage2_bm25_k)
        
        # Create advanced ensemble retriever with optimized weights
        self.ensemble_retriever = self.EnsembleRetriever(
            retrievers=[faiss_retriever, candidate_bm25],
            weights=[0.75, 0.25]  # Favor semantic search more heavily
        )
        
        stage2_time = time.time() - start_time
        progress_logger.info("✅ ENHANCED STAGE 2 COMPLETED: Advanced hybrid search ready in %.2f seconds", stage2_time)
        progress_logger.info("⚖️ Enhanced ensemble weights: FAISS=75%, BM25=25%")

    # NEW METHOD: Add this method to your OptimizedRAGService class
    def rerank_with_mmr(self, docs: List, query: str, diversity_threshold: float = 0.7) -> List:
        """
        Apply MMR-based re-ranking to ensure diversity in final results.
        """
        if not self.use_mmr_reranking or len(docs) <= self.final_ensemble_k:
            return docs[:self.final_ensemble_k]
        
        progress_logger.debug("🔄 Applying MMR re-ranking for diversity")
        
        try:
            # Get query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Get document embeddings
            doc_texts = [doc.page_content for doc in docs]
            doc_embeddings = self.embeddings.embed_documents(doc_texts)
            
            # Convert to numpy arrays
            query_vec = np.array(query_embedding).reshape(1, -1)
            doc_vecs = np.array(doc_embeddings)
            
            # Calculate similarities to query
            query_similarities = cosine_similarity(query_vec, doc_vecs)[0]
            
            # MMR selection
            selected_indices = []
            remaining_indices = list(range(len(docs)))
            
            # Select first document (highest query similarity)
            first_idx = np.argmax(query_similarities)
            selected_indices.append(first_idx)
            remaining_indices.remove(first_idx)
            
            # Select remaining documents using MMR
            while len(selected_indices) < self.final_ensemble_k and remaining_indices:
                mmr_scores = []
                
                for idx in remaining_indices:
                    # Relevance score
                    relevance = query_similarities[idx]
                    
                    # Diversity score (minimum similarity to already selected)
                    if len(selected_indices) == 1:
                        diversity = 1.0
                    else:
                        selected_vecs = doc_vecs[selected_indices]
                        current_vec = doc_vecs[idx].reshape(1, -1)
                        similarities = cosine_similarity(current_vec, selected_vecs)[0]
                        diversity = 1.0 - np.max(similarities)
                    
                    # MMR score
                    mmr_score = diversity_threshold * relevance + (1 - diversity_threshold) * diversity
                    mmr_scores.append((mmr_score, idx))
                
                # Select best MMR score
                best_idx = max(mmr_scores, key=lambda x: x[0])[1]
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
            
            reranked_docs = [docs[i] for i in selected_indices]
            progress_logger.debug("✅ MMR re-ranking completed: %d → %d diverse documents", 
                                 len(docs), len(reranked_docs))
            
            return reranked_docs
            
        except Exception as e:
            progress_logger.warning("⚠️ MMR re-ranking failed, using original order: %s", str(e))
            return docs[:self.final_ensemble_k]

    # REPLACE the existing _cached_retrieve_two_stage method with this enhanced version
    @lru_cache(maxsize=100)
    def _cached_retrieve_enhanced_two_stage(self, question: str):
        """
        Enhanced cached retrieval for two-stage approach with improved post-processing.
        """
        with self._cache_lock:
            cache_key = f"enhanced_two_stage_{question}"
            if cache_key in self._retrieval_cache:
                progress_logger.debug("💾 Cache HIT for enhanced two-stage retrieval")
                return self._retrieval_cache[cache_key]
            
            progress_logger.debug("🔍 Cache MISS - executing enhanced two-stage retrieval")
            
            # Multi-query retrieval for comprehensive coverage
            expanded_queries = self.expand_query(question)
            
            all_docs = []
            seen_content = set()
            
            # Retrieve for each expanded query
            for i, query in enumerate(expanded_queries):
                query_docs = self.ensemble_retriever.get_relevant_documents(query)
                
                for doc in query_docs:
                    if doc.page_content not in seen_content:
                        # Add query source information
                        doc.metadata = doc.metadata or {}
                        doc.metadata['retrieval_query_idx'] = i
                        doc.metadata['retrieval_query'] = query[:100]  # Truncated for metadata
                        
                        all_docs.append(doc)
                        seen_content.add(doc.page_content)
            
            progress_logger.debug("📊 Multi-query retrieval: %d queries → %d unique documents", 
                                 len(expanded_queries), len(all_docs))
            
            # Apply MMR re-ranking for diversity
            final_docs = self.rerank_with_mmr(all_docs, question, self.mmr_diversity_threshold)
            
            # Create context with enhanced formatting
            context_parts = []
            for i, doc in enumerate(final_docs):
                # Add document separator and metadata
                source_info = doc.metadata.get('source_retriever', 'unknown')
                page_info = doc.metadata.get('page', 'unknown')
                
                doc_header = f"\n--- Document {i+1} (Source: {source_info}, Page: {page_info}) ---\n"
                context_parts.append(doc_header)
                context_parts.append(doc.page_content)
            
            context = "".join(context_parts)
            
            self._retrieval_cache[cache_key] = context
            progress_logger.debug("💾 Cached enhanced retrieval result (%d chars, %d docs)", 
                                 len(context), len(final_docs))
            
            return context

    # REPLACE the existing setup_chain_two_stage method with this enhanced version
    async def setup_chain_enhanced_two_stage(self):
        """Setup enhanced chain for two-stage retrieval with improved prompt."""
        progress_logger.info("🔧 Setting up ENHANCED TWO-STAGE RAG chain")
        
        # Enhanced prompt template with better instructions
        template = """
You are an expert insurance policy analyst. Based on the document context below, answer the question with high precision and completeness.

CONTEXT: {context}

QUESTION: {question}

INSTRUCTIONS:
1. Answer ONLY from the context provided - do not use external knowledge
2. Be specific and cite exact terms, amounts, percentages, and section references when available
3. For coverage questions: Include limits, conditions, exclusions, and waiting periods
4. For claim questions: Include process steps, required documents, and timelines  
5. For premium questions: Include base amounts, discounts, and calculation methods
6. If information is not found, clearly state "Not mentioned in the provided documents"
7. Structure your answer with clear sections if covering multiple aspects
8. Always mention any important conditions or limitations that apply

ANSWER:
"""
        
        prompt = self.PromptTemplate.from_template(template)
        progress_logger.info("📝 Enhanced prompt template configured with detailed instructions")
        
        # Enhanced model configuration
        progress_logger.info("🤖 Setting up enhanced ChatOpenAI model...")
        model = self.ChatOpenAI(
            model_name="gpt-3.5-turbo-1106",
            temperature=0,
            max_tokens=512,  # Increased for more comprehensive answers
            request_timeout=15,  # Increased timeout
        )
        
        progress_logger.info("⛓️ Building enhanced two-stage RAG chain...")
        self.chain = (
            {"context": self._cached_retrieve_enhanced_two_stage, "question": self.RunnablePassthrough()}
            | prompt
            | model
            | self.StrOutputParser()
        )
        
        progress_logger.info("✅ ENHANCED TWO-STAGE RAG chain setup completed")
        progress_logger.info("🎯 Chain features:")
        progress_logger.info("   - Multi-query expansion")  
        progress_logger.info("   - MMR diversity re-ranking")
        progress_logger.info("   - Enhanced prompt engineering")
        progress_logger.info("   - Increased context window")

    # REPLACE the existing run method with this enhanced version
    async def run(self, documents_url: str, questions: List[str], processing_method: str = "parallel") -> List[str]:
        """Enhanced run method with automatic strategy selection and comprehensive logging."""
        try:
            start_time = time.time()
            progress_logger.info("🚀 ===== ENHANCED RAG PROCESSING PIPELINE =====")
            progress_logger.info("📊 Configuration:")
            progress_logger.info("   - Document URL: %s", documents_url)
            progress_logger.info("   - Questions: %d", len(questions))
            progress_logger.info("   - Processing method: %s", processing_method)
            progress_logger.info("   - Page threshold: %d", self.page_threshold)
            
            # Setup vector store
            progress_logger.info("🔄 PHASE 1: Vector Store Setup")
            await self.setup_vector_store()
            
            # Process document and get page count
            progress_logger.info("🔄 PHASE 2: Document Processing")
            documents, parsed_pages, page_count = await self.process_document_async(documents_url)
            
            progress_logger.info("📈 Document Processing Results:")
            progress_logger.info("   - Pages: %d", page_count)
            progress_logger.info("   - Chunks: %d", len(documents))
            progress_logger.info("   - Avg chunk size: %.0f chars", 
                               sum(len(doc.page_content) for doc in documents) / len(documents) if documents else 0)
            
            # Determine processing strategy
            strategy = self.determine_processing_strategy(page_count)
            
            if strategy == "two_stage":
                # 🚀 ENHANCED Two-Stage Retrieval for Large Documents
                progress_logger.info("🔄 PHASE 3: ENHANCED TWO-STAGE RETRIEVAL STRATEGY")
                progress_logger.info("🎯 Large document detected - executing ENHANCED two-stage approach")
                
                # Enhanced Stage 1: Better candidate selection
                candidate_docs = await self.stage_1_enhanced_candidate_selection(documents, questions)
                
                # Enhanced Stage 2: Advanced hybrid search
                await self.stage_2_advanced_hybrid_search(candidate_docs)
                
                # Setup enhanced chain
                progress_logger.info("🔄 PHASE 4: Enhanced Two-Stage Chain Setup")
                await self.setup_chain_enhanced_two_stage()
                
                progress_logger.info("✅ Enhanced two-stage setup completed - MongoDB insertion skipped for efficiency")
                
            else:
                # 📚 Standard Strategy for Smaller Documents (UNCHANGED)
                progress_logger.info("🔄 PHASE 3: STANDARD RETRIEVAL STRATEGY")
                progress_logger.info("🎯 Standard document size - using MongoDB + Ensemble approach")
                
                # Setup standard chain
                progress_logger.info("🔄 PHASE 4: Standard Chain Setup")
                await self.setup_chain_standard(documents)
                
                # Add to MongoDB vector store
                progress_logger.info("🔄 PHASE 5: MongoDB Document Addition")
                total_added = await self.add_documents_bulk_optimized(documents)
                progress_logger.info("✅ MongoDB insertion completed: %d documents added", total_added)
            
            # Answer questions using specified processing method
            progress_logger.info("🔄 PHASE 6: Question Answering")
            progress_logger.info("🎯 Strategy: %s | Method: %s", strategy.upper(), processing_method.upper())
            
            answers = await self.answer_questions(questions, method=processing_method)
            
            # Final statistics and summary
            total_time = time.time() - start_time
            progress_logger.info("🎉 ===== ENHANCED RAG PROCESSING COMPLETED =====")
            progress_logger.info("📊 Final Results:")
            progress_logger.info("   - Strategy used: %s", strategy.upper())
            progress_logger.info("   - Total processing time: %.2f seconds", total_time)
            progress_logger.info("   - Questions answered: %d/%d", len(answers), len(questions))
            progress_logger.info("   - Average time per question: %.2f seconds", total_time / len(questions) if questions else 0)
            progress_logger.info("   - Processing efficiency: %.2f questions/second", len(questions) / total_time if total_time > 0 else 0)
            
            # Log enhanced features used
            if strategy == "two_stage":
                progress_logger.info("   - Enhanced features used:")
                progress_logger.info("     * Multi-query expansion: ✅")
                progress_logger.info("     * MMR diversity re-ranking: ✅") 
                progress_logger.info("     * Enhanced prompt engineering: ✅")
                progress_logger.info("     * Increased candidate pool: ✅")
            
            # Log answer preview
            for i, answer in enumerate(answers[:3]):  # Show first 3 answers
                preview = answer[:100] + "..." if len(answer) > 100 else answer
                progress_logger.info("   - Answer %d preview: %s", i + 1, preview)
            
            if len(answers) > 3:
                progress_logger.info("   - ... and %d more answers", len(answers) - 3)
            
            return answers
            
        except Exception as e:
            progress_logger.error("❌ ===== ENHANCED RAG PROCESSING FAILED =====")
            progress_logger.error("💥 Error details: %s", str(e))
            progress_logger.error("🕒 Failed after %.2f seconds", time.time() - start_time)
            logger.error("Error in RAG service run: %s", str(e))
            raise
# Backward compatibility - keep the original interface with enhanced logging
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document as LangchainDocument
from pydantic import PrivateAttr

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
        
        progress_logger.info("🎭 CachedEnsembleRetriever initialized")
        progress_logger.info("📊 Configuration: %d retrievers, k=%d", len(retrievers), k)
        for name, weight in zip(retriever_names, weights):
            progress_logger.info("   - %s: %.1f%% weight", name, weight * 100)

    @lru_cache(maxsize=100)
    def _cached_retrieve(self, query: str) -> List[LangchainDocument]:
        """Cache retrieval results for identical queries"""
        with self._cache_lock:
            if query in self._retrieval_cache:
                progress_logger.debug("💾 CachedEnsembleRetriever cache HIT for query: %s...", query[:50])
                return self._retrieval_cache[query]
            
            progress_logger.debug("🔍 CachedEnsembleRetriever cache MISS - executing retrieval: %s...", query[:50])
            docs = self._get_relevant_documents(query)
            self._retrieval_cache[query] = docs
            progress_logger.debug("💾 Cached retrieval result: %d documents", len(docs))
            return docs

    def _get_relevant_documents(self, query: str) -> List[LangchainDocument]:
        """Retrieve documents using ensemble approach with weighted scoring."""
        all_docs = []
        seen_contents = set()
        
        progress_logger.debug("🎭 Executing ensemble retrieval with %d retrievers", len(self._retrievers))

        for retriever, name, weight in zip(self._retrievers, self._retriever_names, self._weights):
            try:
                retrieval_start = time.time()
                
                if hasattr(retriever, 'get_relevant_documents'):
                    retrieved_docs = retriever.get_relevant_documents(query)
                elif hasattr(retriever, 'invoke'):
                    retrieved_docs = retriever.invoke(query)
                else:
                    progress_logger.warning("⚠️ Retriever %s has no supported method", name)
                    continue
                
                retrieval_time = time.time() - retrieval_start
                progress_logger.debug("⚡ %s retrieval: %d docs in %.3f seconds", 
                                     name, len(retrieved_docs), retrieval_time)
                    
                for doc in retrieved_docs:
                    if doc.page_content not in seen_contents:
                        doc.metadata = doc.metadata or {}
                        doc.metadata["source_retriever"] = name
                        doc.metadata["retriever_weight"] = weight
                        all_docs.append(doc)
                        seen_contents.add(doc.page_content)
                        
            except Exception as e:
                progress_logger.warning("❌ %s retrieval failed: %s", name, str(e))

        final_docs = all_docs[:self._k]
        progress_logger.debug("✅ Ensemble retrieval completed: %d unique documents (top %d selected)", 
                             len(all_docs), len(final_docs))
        
        return final_docs

# Additional utility functions for enhanced logging and monitoring
class RAGMetrics:
    """Utility class for tracking RAG performance metrics."""
    
    def __init__(self):
        self.metrics = {
            "total_processing_time": 0,
            "document_processing_time": 0,
            "embedding_time": 0,
            "retrieval_time": 0,
            "generation_time": 0,
            "questions_processed": 0,
            "documents_processed": 0,
            "strategy_used": None,
            "cache_hits": 0,
            "cache_misses": 0
        }
        self.start_time = None
        
    def start_processing(self):
        """Mark the start of processing."""
        self.start_time = time.time()
        progress_logger.info("📊 RAG metrics tracking started")
        
    def end_processing(self):
        """Mark the end of processing and log final metrics."""
        if self.start_time:
            self.metrics["total_processing_time"] = time.time() - self.start_time
            
        progress_logger.info("📈 ===== RAG PERFORMANCE METRICS =====")
        progress_logger.info("⏱️ Total processing time: %.2f seconds", self.metrics["total_processing_time"])
        progress_logger.info("📄 Documents processed: %d", self.metrics["documents_processed"])
        progress_logger.info("❓ Questions processed: %d", self.metrics["questions_processed"])
        progress_logger.info("🎯 Strategy used: %s", self.metrics["strategy_used"])
        progress_logger.info("💾 Cache performance: %d hits, %d misses", 
                           self.metrics["cache_hits"], self.metrics["cache_misses"])
        
        if self.metrics["questions_processed"] > 0:
            avg_time_per_question = self.metrics["total_processing_time"] / self.metrics["questions_processed"]
            progress_logger.info("⚡ Average time per question: %.2f seconds", avg_time_per_question)
            
    def update_metric(self, key: str, value):
        """Update a specific metric."""
        self.metrics[key] = value
        progress_logger.debug("📊 Metric updated: %s = %s", key, value)

# Global metrics instance
rag_metrics = RAGMetrics()

def log_system_info():
    """Log system information for debugging purposes."""
    import platform
    import psutil
    
    progress_logger.info("🖥️ ===== SYSTEM INFORMATION =====")
    progress_logger.info("💻 Platform: %s %s", platform.system(), platform.release())
    progress_logger.info("🐍 Python version: %s", platform.python_version())
    progress_logger.info("💾 Available memory: %.2f GB", psutil.virtual_memory().available / (1024**3))
    progress_logger.info("⚙️ CPU cores: %d", psutil.cpu_count())
    progress_logger.info("🔧 Thread pool workers: %d", executor._max_workers)

# Log system info on module import
try:
    log_system_info()
except ImportError:
    progress_logger.info("📊 System info logging requires psutil - skipping")

progress_logger.info("✅ Enhanced RAG Service module loaded successfully")