# rag_service.py

import os
import asyncio
import nest_asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import re
import requests
import tempfile
import fitz  # PyMuPDF
import pdfplumber
import nltk
from bs4 import BeautifulSoup
from uuid import uuid4
from collections import defaultdict
from langchain_core.retrievers import BaseRetriever
from pydantic import PrivateAttr
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

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

@dataclass
class DocumentChunk:
    """Represents a semantic chunk of the document."""
    id: str
    title: str
    content: str
    section_path: str
    chunk_type: str  # 'text', 'table', 'list', 'definition', 'bullet_points'
    metadata: Dict
    semantic_score: float = 0.0  # Coherence score
    keywords: List[str] = field(default_factory=list)

class FinancialPDFParser:
    """Enhanced PDF parser optimized for financial policy documents with complex layouts and tables."""
    
    def __init__(self, 
                 table_detection_threshold: int = 3,
                 min_table_rows: int = 2,
                 min_table_cols: int = 2,
                 preserve_formatting: bool = True):
        self.table_detection_threshold = table_detection_threshold
        self.min_table_rows = min_table_rows
        self.min_table_cols = min_table_cols
        self.preserve_formatting = preserve_formatting
    
    def download_pdf(self, url: str, timeout: int = 30) -> str:
        """Download PDF from a URL to a temporary local file with better error handling."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, timeout=timeout, headers=headers, stream=True)
            response.raise_for_status()
            
            # Check if content is actually a PDF
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' not in content_type:
                # Check magic number for PDF
                first_bytes = response.content[:4]
                if not first_bytes.startswith(b'%PDF'):
                    raise ValueError("Downloaded content is not a valid PDF")
            
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tmp_file.write(response.content)
            tmp_file.flush()
            tmp_file.close()
            
            logger.info(f"Downloaded PDF: {len(response.content)} bytes")
            return tmp_file.name
            
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Failed to download PDF: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error processing PDF download: {str(e)}")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text from PDF extraction."""
        if not text:
            return ""
        
        # Remove excessive whitespace but preserve paragraph breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Fix common OCR/extraction issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between camelCase
        text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)  # Space between numbers and letters
        text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)  # Space between letters and numbers
        
        # Clean up bullet points and list formatting
        text = re.sub(r'•\s*', '• ', text)
        text = re.sub(r'^\s*[\-\*]\s*', '• ', text, flags=re.MULTILINE)
        
        return text.strip()
    
    def detect_table_indicators(self, text: str) -> Dict[str, bool]:
        """Advanced table detection using multiple heuristics for financial documents."""
        lines = text.splitlines()
        
        indicators = {
            'alignment_patterns': False,
            'numeric_columns': False,
            'financial_keywords': False,
            'separator_lines': False,
            'consistent_spacing': False
        }
        
        # Check for alignment patterns (multiple spaces or tabs)
        aligned_lines = sum(1 for line in lines if line.count('  ') >= self.table_detection_threshold or '\t' in line)
        indicators['alignment_patterns'] = aligned_lines >= 2
        
        # Check for numeric columns (financial data patterns)
        numeric_pattern = re.compile(r'[\d,]+\.?\d*%?|\$[\d,]+\.?\d*|[\d,]+\.?\d*\s*(?:million|billion|trillion|M|B|T)')
        numeric_lines = sum(1 for line in lines if len(numeric_pattern.findall(line)) >= 2)
        indicators['numeric_columns'] = numeric_lines >= 2
        
        # Check for financial keywords that often appear in tables
        financial_keywords = [
            'total', 'amount', 'balance', 'revenue', 'expense', 'profit', 'loss',
            'assets', 'liabilities', 'equity', 'cash', 'investment', 'rate',
            'percentage', 'year', 'quarter', 'month', 'date', 'period'
        ]
        keyword_pattern = re.compile(r'\b(?:' + '|'.join(financial_keywords) + r')\b', re.IGNORECASE)
        has_keywords = any(keyword_pattern.search(line) for line in lines[:10])  # Check first few lines
        indicators['financial_keywords'] = has_keywords
        
        # Check for separator lines (dashes, equals, underscores)
        separator_pattern = re.compile(r'^[\s\-=_]{10,}$')
        has_separators = any(separator_pattern.match(line) for line in lines)
        indicators['separator_lines'] = has_separators
        
        # Check for consistent spacing patterns
        if len(lines) >= 3:
            space_patterns = []
            for line in lines[:10]:  # Check first 10 lines
                spaces = [m.start() for m in re.finditer(r'\s{2,}', line)]
                if len(spaces) >= 2:
                    space_patterns.append(spaces)
            
            # If multiple lines have similar spacing patterns, likely a table
            if len(space_patterns) >= 2:
                indicators['consistent_spacing'] = True
        
        return indicators
    
    def is_likely_table_page(self, text: str) -> bool:
        """Determine if a page likely contains tables based on multiple indicators."""
        indicators = self.detect_table_indicators(text)
        
        # Weighted scoring system
        score = 0
        if indicators['alignment_patterns']:
            score += 3
        if indicators['numeric_columns']:
            score += 4  # Strong indicator for financial docs
        if indicators['financial_keywords']:
            score += 2
        if indicators['separator_lines']:
            score += 2
        if indicators['consistent_spacing']:
            score += 3
        
        return score >= 4  # Threshold for table detection
    
    def extract_tables_with_context(self, page) -> List[Dict]:
        """Extract tables with better formatting and context preservation."""
        tables_data = []
        
        try:
            tables = page.extract_tables(table_settings={
                "vertical_strategy": "lines_strict",
                "horizontal_strategy": "lines_strict",
                "snap_tolerance": 3,
                "join_tolerance": 3,
                "edge_min_length": 3,
                "min_words_vertical": 1,
                "min_words_horizontal": 1,
                "intersection_tolerance": 3
            })
            
            for i, table in enumerate(tables):
                if not table or len(table) < self.min_table_rows:
                    continue
                
                # Filter out tables with too few columns
                max_cols = max(len(row) for row in table if row)
                if max_cols < self.min_table_cols:
                    continue
                
                # Clean and format table
                cleaned_table = []
                for row in table:
                    cleaned_row = []
                    for cell in row:
                        if cell is None:
                            cleaned_row.append("")
                        else:
                            # Clean cell content
                            cell_content = str(cell).strip()
                            cell_content = re.sub(r'\s+', ' ', cell_content)
                            cleaned_row.append(cell_content)
                    cleaned_table.append(cleaned_row)
                
                # Create formatted table string
                table_str = self._format_table(cleaned_table)
                
                tables_data.append({
                    'index': i,
                    'table': table_str,
                    'rows': len(cleaned_table),
                    'cols': max_cols
                })
                
        except Exception as e:
            logger.warning(f"Error extracting tables: {str(e)}")
        
        return tables_data
    
    def _format_table(self, table: List[List[str]]) -> str:
        """Format table with proper alignment for better readability."""
        if not table:
            return ""
        
        # Calculate column widths
        col_widths = []
        max_cols = max(len(row) for row in table)
        
        for col in range(max_cols):
            max_width = 0
            for row in table:
                if col < len(row) and row[col]:
                    max_width = max(max_width, len(str(row[col])))
            col_widths.append(min(max_width, 30))  # Cap column width
        
        # Format rows
        formatted_rows = []
        for row in table:
            formatted_cells = []
            for col in range(max_cols):
                cell = row[col] if col < len(row) else ""
                formatted_cells.append(str(cell).ljust(col_widths[col]))
            formatted_rows.append(" | ".join(formatted_cells).rstrip())
        
        return "\n".join(formatted_rows)
    
    def extract_page_content(self, fitz_page, plumber_page, page_num: int) -> Dict:
        """Extract content from a single page with enhanced processing."""
        try:
            # Get text using PyMuPDF (faster)
            text = fitz_page.get_text("text").strip()
            
            # Get text with layout preservation if needed
            if self.preserve_formatting:
                layout_text = fitz_page.get_text("dict")
                # Process layout information if needed for complex documents
            
            # Clean the text
            cleaned_text = self.clean_text(text)
            
            page_info = {
                'text': cleaned_text,
                'tables': [],
                'has_tables': False,
                'metadata': {
                    'page': page_num,
                    'word_count': len(cleaned_text.split()),
                    'char_count': len(cleaned_text)
                }
            }
            
            # Check if page likely contains tables
            if self.is_likely_table_page(cleaned_text):
                logger.info(f"Detected potential tables on page {page_num}")
                tables_data = self.extract_tables_with_context(plumber_page)
                
                if tables_data:
                    page_info['tables'] = tables_data
                    page_info['has_tables'] = True
                    page_info['metadata']['table_count'] = len(tables_data)
            
            return page_info
            
        except Exception as e:
            logger.error(f"Error processing page {page_num}: {str(e)}")
            return {
                'text': f"Error processing page {page_num}: {str(e)}",
                'tables': [],
                'has_tables': False,
                'metadata': {'page': page_num, 'error': True}
            }
    
    def parse_document(self, url: str) -> List[Dict]:
        """Main parsing function with comprehensive error handling and cleanup."""
        pdf_path = None
        
        try:
            pdf_path = self.download_pdf(url)
            parsed_pages = []
            
            with fitz.open(pdf_path) as doc_fitz, pdfplumber.open(pdf_path) as doc_plumber:
                total_pages = len(doc_fitz)
                logger.info(f"Processing {total_pages} pages")
                
                for i, (fitz_page, plumber_page) in enumerate(zip(doc_fitz, doc_plumber.pages)):
                    page_num = i + 1
                    page_info = self.extract_page_content(fitz_page, plumber_page, page_num)
                    
                    # Combine text and tables
                    content_parts = [page_info['text']]
                    
                    if page_info['has_tables']:
                        content_parts.append("\n\n--- EXTRACTED TABLES ---\n")
                        for table_data in page_info['tables']:
                            content_parts.append(f"\nTable {table_data['index'] + 1}:")
                            content_parts.append(f"({table_data['rows']} rows × {table_data['cols']} columns)")
                            content_parts.append(table_data['table'])
                            content_parts.append("")
                    
                    full_content = "\n".join(content_parts).strip()
                    
                    parsed_pages.append({
                        "page_content": full_content,
                        "tables": page_info['tables'] if page_info['has_tables'] else [],
                        "metadata": {
                            'source': url,
                            'page': page_num,
                            'total_pages': total_pages,
                            'has_tables': page_info['has_tables'],
                            'word_count': page_info['metadata']['word_count'],
                            'char_count': page_info['metadata']['char_count']
                        }
                    })
            
            logger.info(f"Successfully processed {len(parsed_pages)} pages")
            return parsed_pages
            
        except Exception as e:
            logger.error(f"Error parsing document: {str(e)}")
            raise
            
        finally:
            # Clean up temporary file
            if pdf_path and os.path.exists(pdf_path):
                try:
                    os.unlink(pdf_path)
                    logger.info("Cleaned up temporary PDF file")
                except Exception as e:
                    logger.warning(f"Could not clean up temporary file: {str(e)}")

class ImprovedInsurancePDFChunker:
    """Enhanced PDF chunker with semantic awareness for insurance documents."""
    
    def __init__(self, max_chunk_size: int = 800, min_chunk_size: int = 200, chunk_overlap: int = 100):
        self.chunk_counter = 0
        self.chunks: List[DocumentChunk] = []
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Insurance-specific patterns
        self.section_patterns = [
            r'^[A-Z][A-Z\s\-:]{3,}$',  # ALL CAPS headers
            r'^\d+\.\s+[A-Z][A-Za-z\s\-:]{3,}$',  # Numbered sections
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*:?\s*$',  # Title Case
            r'^SECTION\s+\d+',  # Section markers
            r'^ARTICLE\s+\d+',  # Article markers
            r'^PART\s+[A-Z0-9]+',  # Part markers
            r'^\*{2,}.*\*{2,}$',  # Markdown-style headers
        ]
        
        self.definition_patterns = [
            r'^"([^"]+)"\s+means\s+',  # "Term" means definition
            r'^([A-Z][a-zA-Z\s]+):\s+',  # Term: definition
            r'^Definition of\s+([^:]+):',  # Definition of term:
        ]
        
        self.list_patterns = [
            r'^\s*[\-\*\+]\s+',  # Bullet points
            r'^\s*\d+[\.\)]\s+',  # Numbered lists
            r'^\s*[a-zA-Z][\.\)]\s+',  # Lettered lists
            r'^\s*[ivxlcdm]+[\.\)]\s+',  # Roman numerals
        ]

    def process_parsed_pages(self, pages: List[Dict]) -> List[DocumentChunk]:
        """Enhanced processing with better semantic awareness"""
        self.chunks = []
        current_section_hierarchy = ["General"]
        accumulated_content = []
        
        for page_index, page in enumerate(pages):
            text = page.get("page_content", "")
            tables = page.get("tables", [])
            
            # Process text with enhanced semantic chunking
            self._process_page_text(text, page_index, current_section_hierarchy, accumulated_content)
            
            # Process tables
            self._process_tables(tables, page_index, current_section_hierarchy)
        
        # Process any remaining content
        if accumulated_content:
            self._create_semantic_chunks(accumulated_content, current_section_hierarchy)
        
        # Post-process chunks for quality
        self._post_process_chunks()
        
        return self.chunks

    def _process_page_text(self, text: str, page_index: int, section_hierarchy: List[str], accumulated_content: List[str]):
        """Process page text with semantic awareness"""
        lines = text.split('\n')
        current_block = []
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_block:
                    accumulated_content.extend(current_block)
                    current_block = []
                continue
            
            # Check if this is a section header
            header_info = self._detect_section_header(line)
            if header_info:
                # Process accumulated content before starting new section
                if accumulated_content:
                    self._create_semantic_chunks(accumulated_content, section_hierarchy)
                    accumulated_content = []
                
                # Update section hierarchy
                level, title = header_info
                self._update_section_hierarchy(section_hierarchy, level, title)
                continue
            
            # Check for special content types
            content_type = self._detect_content_type(line)
            if content_type == 'definition':
                # Flush current block and process definition separately
                if current_block:
                    accumulated_content.extend(current_block)
                    current_block = []
                self._process_definition(line, section_hierarchy, page_index)
                continue
            elif content_type in ['list_item', 'bullet_point']:
                # Start or continue a list
                current_block.append(line)
                continue
            
            current_block.append(line)
        
        # Add final block
        if current_block:
            accumulated_content.extend(current_block)

    def _detect_section_header(self, line: str) -> Optional[Tuple[int, str]]:
        """Detect section headers and return (level, title)"""
        for i, pattern in enumerate(self.section_patterns):
            if re.match(pattern, line):
                # Estimate header level based on pattern type and content
                level = 1
                if re.match(r'^\d+\.\s+', line):
                    level = 1
                elif re.match(r'^\d+\.\d+\s+', line):
                    level = 2
                elif re.match(r'^[A-Z][a-z]+', line) and ':' not in line:
                    level = 1
                elif 'SECTION' in line or 'ARTICLE' in line:
                    level = 0  # Top level
                
                title = re.sub(r'^\d+\.?\s*', '', line).strip(':').strip()
                return (level, title)
        return None

    def _detect_content_type(self, line: str) -> str:
        """Detect the type of content in a line"""
        for pattern in self.definition_patterns:
            if re.match(pattern, line):
                return 'definition'
        
        for pattern in self.list_patterns:
            if re.match(pattern, line):
                return 'list_item'
        
        return 'text'

    def _update_section_hierarchy(self, hierarchy: List[str], level: int, title: str):
        """Update section hierarchy based on detected level"""
        # Ensure hierarchy has enough levels
        while len(hierarchy) <= level:
            hierarchy.append("")
        
        # Update at the detected level and clear deeper levels
        hierarchy[level] = title
        del hierarchy[level + 1:]

    def _create_semantic_chunks(self, content_lines: List[str], section_hierarchy: List[str]):
        """Create semantically coherent chunks from content lines"""
        if not content_lines:
            return
        
        text = '\n'.join(content_lines)
        
        # Use NLTK for better sentence segmentation
        sentences = nltk.sent_tokenize(text)
        
        # Group sentences into semantic units
        semantic_groups = self._group_sentences_semantically(sentences)
        
        for group in semantic_groups:
            if not group:
                continue
                
            # Create chunk from semantic group
            chunk_content = ' '.join(group)
            
            # Skip if too small unless it's a definition or special content
            if len(chunk_content) < self.min_chunk_size and not self._is_special_content(chunk_content):
                continue
            
            # Split large chunks while respecting sentence boundaries
            if len(chunk_content) > self.max_chunk_size:
                sub_chunks = self._split_large_chunk(group)
                for sub_chunk in sub_chunks:
                    self._add_semantic_chunk(sub_chunk, section_hierarchy)
            else:
                self._add_semantic_chunk(chunk_content, section_hierarchy)
        
        content_lines.clear()

    def _group_sentences_semantically(self, sentences: List[str]) -> List[List[str]]:
        """Group sentences into semantically coherent units"""
        if not sentences:
            return []
        
        groups = []
        current_group = [sentences[0]]
        current_size = len(sentences[0])
        
        for i in range(1, len(sentences)):
            sentence = sentences[i]
            
            # Check semantic coherence
            should_group = self._should_group_sentences(current_group[-1], sentence)
            
            # Check size constraints
            would_exceed_size = current_size + len(sentence) > self.max_chunk_size
            
            if should_group and not would_exceed_size:
                current_group.append(sentence)
                current_size += len(sentence)
            else:
                # Start new group
                if current_group:
                    groups.append(current_group)
                current_group = [sentence]
                current_size = len(sentence)
        
        if current_group:
            groups.append(current_group)
        
        return groups

    def _should_group_sentences(self, sent1: str, sent2: str) -> bool:
        """Determine if two sentences should be grouped together"""
        # Insurance-specific coherence indicators
        coherence_indicators = [
            # Pronouns and references
            (r'\b(this|that|these|those|it|they|such)\b', True),
            # Continuation words
            (r'\b(however|therefore|furthermore|additionally|moreover|consequently)\b', True),
            # List continuation
            (r'^\s*[\-\*\+\d+\w+[\.\)]\s+', True),
            # Definition continuation
            (r'\b(means|includes|refers to|defined as)\b', True),
            # Insurance terms
            (r'\b(policy|coverage|premium|deductible|claim|benefit)\b', True),
        ]
        
        for pattern, should_group in coherence_indicators:
            if re.search(pattern, sent2.lower()):
                return should_group
        
        return False

    def _split_large_chunk(self, sentences: List[str]) -> List[str]:
        """Split large chunks while maintaining semantic coherence"""
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            if current_size + len(sentence) > self.max_chunk_size and current_chunk:
                # Add overlap from previous chunk
                overlap_text = self._get_overlap_text(chunks[-1] if chunks else "", self.chunk_overlap)
                chunk_text = overlap_text + ' '.join(current_chunk)
                chunks.append(chunk_text.strip())
                
                current_chunk = [sentence]
                current_size = len(sentence)
            else:
                current_chunk.append(sentence)
                current_size += len(sentence)
        
        if current_chunk:
            overlap_text = self._get_overlap_text(chunks[-1] if chunks else "", self.chunk_overlap)
            chunk_text = overlap_text + ' '.join(current_chunk)
            chunks.append(chunk_text.strip())
        
        return chunks

    def _get_overlap_text(self, previous_chunk: str, overlap_size: int) -> str:
        """Get overlap text from previous chunk"""
        if not previous_chunk or overlap_size <= 0:
            return ""
        
        # Get last sentences that fit within overlap size
        sentences = nltk.sent_tokenize(previous_chunk)
        overlap_sentences = []
        current_size = 0
        
        for sentence in reversed(sentences):
            if current_size + len(sentence) <= overlap_size:
                overlap_sentences.insert(0, sentence)
                current_size += len(sentence)
            else:
                break
        
        return ' '.join(overlap_sentences) + ' ' if overlap_sentences else ""

    def _is_special_content(self, content: str) -> bool:
        """Check if content is special (definitions, lists, etc.)"""
        for pattern in self.definition_patterns:
            if re.search(pattern, content):
                return True
        return False

    def _add_semantic_chunk(self, content: str, section_hierarchy: List[str]):
        """Add a semantically processed chunk"""
        section_path = ' > '.join(filter(None, section_hierarchy))
        content_type = self._classify_chunk_type(content)
        keywords = self._extract_keywords(content)
        semantic_score = self._calculate_semantic_score(content)
        
        chunk_id = f"chunk_{self.chunk_counter}_{uuid4().hex[:6]}"
        self.chunk_counter += 1
        
        self.chunks.append(DocumentChunk(
            id=chunk_id,
            title=f"{section_hierarchy[-1] if section_hierarchy else 'General'} - {content_type.title()}",
            content=content.strip(),
            section_path=section_path,
            chunk_type=content_type,
            metadata={"word_count": len(content.split())},
            semantic_score=semantic_score,
            keywords=keywords
        ))

    def _classify_chunk_type(self, content: str) -> str:
        """Classify the type of chunk content"""
        for pattern in self.definition_patterns:
            if re.search(pattern, content):
                return 'definition'
        
        list_count = sum(1 for pattern in self.list_patterns 
                        for _ in re.finditer(pattern, content, re.MULTILINE))
        
        if list_count >= 2:
            return 'list'
        elif list_count == 1:
            return 'bullet_point'
        
        return 'text'

    def _extract_keywords(self, content: str) -> List[str]:
        """Extract key terms from content"""
        # Insurance-specific important terms
        insurance_terms = [
            'policy', 'coverage', 'premium', 'deductible', 'claim', 'benefit',
            'liability', 'damages', 'exclusion', 'endorsement', 'rider',
            'insured', 'insurer', 'policyholder', 'beneficiary'
        ]
        
        keywords = []
        content_lower = content.lower()
        
        for term in insurance_terms:
            if term in content_lower:
                keywords.append(term)
        
        # Extract capitalized terms (likely important concepts)
        capitalized_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
        keywords.extend(capitalized_terms[:5])  # Limit to top 5
        
        return list(set(keywords))

    def _calculate_semantic_score(self, content: str) -> float:
        """Calculate a semantic coherence score for the chunk"""
        score = 0.0
        
        # Length penalty/bonus
        word_count = len(content.split())
        if self.min_chunk_size <= len(content) <= self.max_chunk_size:
            score += 0.3
        
        # Sentence completeness
        sentences = nltk.sent_tokenize(content)
        if all(s.strip().endswith(('.', '!', '?', ':')) for s in sentences):
            score += 0.2
        
        # Coherence indicators
        coherence_words = ['however', 'therefore', 'furthermore', 'this', 'that', 'such']
        coherence_count = sum(1 for word in coherence_words if word in content.lower())
        score += min(coherence_count * 0.1, 0.3)
        
        # Insurance domain relevance
        domain_terms = ['policy', 'coverage', 'claim', 'premium', 'insured']
        domain_count = sum(1 for term in domain_terms if term in content.lower())
        score += min(domain_count * 0.1, 0.2)
        
        return min(score, 1.0)

    def _process_definition(self, line: str, section_hierarchy: List[str], page_index: int):
        """Process definition lines specially"""
        for pattern in self.definition_patterns:
            match = re.match(pattern, line)
            if match:
                term = match.group(1)
                self._add_semantic_chunk(
                    line, section_hierarchy
                )
                break

    def _process_tables(self, tables: List[List[List[str]]], page_index: int, section_hierarchy: List[str]):
        """Process tables with enhanced metadata"""
        for i, table in enumerate(tables):
            if not table:
                continue
                
            markdown_table = self.convert_table_to_markdown(table)
            section_path = ' > '.join(filter(None, section_hierarchy))
            
            # Extract table metadata
            headers = table[0] if table else []
            row_count = len(table) - 1 if len(table) > 1 else 0
            
            chunk_id = f"chunk_{self.chunk_counter}_{uuid4().hex[:6]}"
            self.chunk_counter += 1
            
            self.chunks.append(DocumentChunk(
                id=chunk_id,
                title=f"{section_hierarchy[-1] if section_hierarchy else 'General'} - Table {i+1}",
                content=f"**Table from Page {page_index+1}:**\n\n{markdown_table}",
                section_path=section_path,
                chunk_type="table",
                metadata={
                    "page": page_index + 1,
                    "table_index": i,
                    "headers": headers,
                    "row_count": row_count
                },
                semantic_score=0.8,  # Tables are generally well-structured
                keywords=headers[:5] if headers else []
            ))

    def _post_process_chunks(self):
        """Post-process chunks for quality improvements"""
        # Remove very small chunks that aren't definitions
        self.chunks = [chunk for chunk in self.chunks 
                      if len(chunk.content) >= self.min_chunk_size 
                      or chunk.chunk_type in ['definition', 'table']
                      or any(keyword in chunk.content.lower() for keyword in ['definition', 'means', 'refers to'])]
        
        # Sort chunks by semantic score and section order
        self.chunks.sort(key=lambda x: (x.section_path, -x.semantic_score))

    def convert_table_to_markdown(self, table: List[List[str]]) -> str:
        """Enhanced table conversion with better formatting"""
        if not table:
            return ""
        
        # Clean and normalize table data
        cleaned_table = []
        for row in table:
            cleaned_row = [cell.strip().replace('\n', ' ') for cell in row]
            cleaned_table.append(cleaned_row)
        
        max_cols = max(len(row) for row in cleaned_table) if cleaned_table else 0
        
        # Ensure all rows have the same number of columns
        for row in cleaned_table:
            while len(row) < max_cols:
                row.append("")
        
        if not cleaned_table:
            return ""
        
        header = cleaned_table[0]
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * len(header)) + " |"
        ]
        
        for row in cleaned_table[1:]:
            lines.append("| " + " | ".join(row[:len(header)]) + " |")
        
        return '\n'.join(lines)

# Replace the RerankHybridRetriever class in rag_service.py with this corrected version

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List
from pydantic import PrivateAttr
import logging

logger = logging.getLogger(__name__)

class RerankHybridRetriever(BaseRetriever):
    """Custom hybrid retriever combining vector and BM25 search with reranking."""
    
    _retrievers: List = PrivateAttr()
    _retriever_names: List[str] = PrivateAttr()
    _k: int = PrivateAttr()

    def __init__(self, retrievers: List, retriever_names: List[str], k: int = 12):
        super().__init__()
        assert len(retrievers) == len(retriever_names), "Retriever names must match retrievers"
        self._retrievers = retrievers
        self._retriever_names = retriever_names
        self._k = k

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve documents using both retrievers and combine results."""
        docs = []
        seen_contents = set()

        for retriever, name in zip(self._retrievers, self._retriever_names):
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
                        docs.append(doc)
                        seen_contents.add(doc.page_content)
            except Exception as e:
                logger.warning(f"{name} retrieval failed: {e}")

        return docs[:self._k]
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Public method for retrieving documents."""
        return self._get_relevant_documents(query)
    
    def invoke(self, query: str) -> List[Document]:
        """Alias for get_relevant_documents for compatibility."""
        return self._get_relevant_documents(query)

class OptimizedRAGService:
    """Optimized RAG service for fast document processing and question answering."""
    
    def __init__(self):
        self.vector_store = None
        self.chain = None
        self.hybrid_retriever = None
        self._setup_complete = False
        self.embeddings = None
        self.mongo_client = None
        
    async def _setup_dependencies(self):
        """Setup dependencies only when needed."""
        if self._setup_complete:
            return
            
        # Import heavy libraries
        from langchain_voyageai import VoyageAIEmbeddings
        from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
        from langchain_openai import ChatOpenAI
        from langchain_core.documents import Document
        from langchain.prompts import PromptTemplate
        from langchain.schema.runnable import RunnablePassthrough
        from langchain_core.output_parsers import StrOutputParser
        from langchain.retrievers import BM25Retriever
        from pymongo import MongoClient
        
        # Store in instance
        self.VoyageAIEmbeddings = VoyageAIEmbeddings
        self.MongoDBAtlasVectorSearch = MongoDBAtlasVectorSearch
        self.ChatOpenAI = ChatOpenAI
        self.Document = Document
        self.PromptTemplate = PromptTemplate
        self.RunnablePassthrough = RunnablePassthrough
        self.StrOutputParser = StrOutputParser
        self.BM25Retriever = BM25Retriever
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
                maxIdleTimeMS=30000,
                serverSelectionTimeoutMS=5000,
                socketTimeoutMS=20000,
                connectTimeoutMS=20000,
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
            
    async def setup_chain(self, documents: List = None):
        """Setup the RAG chain with hybrid retrieval."""
        if self.chain is None:
            await self._setup_dependencies()
            
            # Enhanced template for better insurance document understanding
            template = """
    You are a reliable assistant specialized in reading and interpreting insurance policy and technical documents.

    Using the DOCUMENT CONTEXT provided, answer the user's question **factually and precisely**, citing **exact terms, limits, exclusions, or sections** if applicable.

    ------------------
    DOCUMENT CONTEXT:
    {context}
    ------------------

    Guidelines for answering:
    - Only answer based on the context. If the answer is not found, say "The document does not mention this explicitly."
    - Prefer definitions, sections, and limits over general interpretations.
    - If the question is about coverage, always mention conditions like limits, eligibility, or hospitalization requirements.
    - If exclusions are present, highlight them clearly.
    - Avoid making assumptions or paraphrasing loosely.

    Original Question: {question}

    Answer:
    """

            prompt = self.PromptTemplate.from_template(template)
            
            # Setup vector retriever
            vector_retriever = self.vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 12})
            
            # Setup hybrid retriever
            if documents:
                # Setup BM25 retriever if documents are provided
                bm25_retriever = self.BM25Retriever.from_documents(documents)
                bm25_retriever.k = 12
                
                # Create hybrid retriever using the corrected class
                self.hybrid_retriever = RerankHybridRetriever(
                    retrievers=[vector_retriever, bm25_retriever],
                    retriever_names=["vector", "bm25"],
                    k=12
                )
            else:
                self.hybrid_retriever = vector_retriever
            
            model = self.ChatOpenAI(
                model="gpt-4o-mini", 
                temperature=0,
                max_tokens=200,  # Reasonable response length
                request_timeout=8,  # Aggressive timeout
                max_retries=1,
                streaming=False
            )
            
            self.chain = (
                {"context": self.hybrid_retriever, "question": self.RunnablePassthrough()}
                | prompt
                | model
                | self.StrOutputParser()
            )
    async def process_document_async(self, url: str) -> Tuple[List, List]:
        """Process document asynchronously and return both raw pages and document chunks."""
        await self._setup_dependencies()
        
        def process_sync():
            # Use the new enhanced parser
            parser = FinancialPDFParser(
                table_detection_threshold=3,
                min_table_rows=2,
                min_table_cols=2,
                preserve_formatting=True
            )
            
            # Parse document into pages
            parsed_pages = parser.parse_document(url)
            
            # Create semantic chunks
            chunker = ImprovedInsurancePDFChunker(
                max_chunk_size=500, 
                min_chunk_size=100, 
                chunk_overlap=100
            )
            semantic_chunks = chunker.process_parsed_pages(parsed_pages)
            
            # Convert to Document objects
            documents = [
                self.Document(
                    page_content=chunk.content,
                    metadata={
                        "id": chunk.id,
                        "title": chunk.title,
                        "section_path": chunk.section_path,
                        "chunk_type": chunk.chunk_type,
                        **chunk.metadata
                    }
                )
                for chunk in semantic_chunks
            ]
            
            return documents, parsed_pages
        
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
        
        # Step 2: Compute embeddings in large batches
        all_embeddings = []
        text_batches = [texts[i:i + 72] for i in range(0, len(texts), 72)]  # Voyage AI max batch size
        
        def compute_embeddings_sync(batch):
            return self.embeddings.embed_documents(batch)
        
        # Process batches asynchronously
        for batch in text_batches:
            batch_embeddings = await asyncio.to_thread(compute_embeddings_sync, batch)
            all_embeddings.extend(batch_embeddings)
        
        logger.info("Embeddings computed. Preparing bulk insert...")
        
        # Step 3: Prepare documents for bulk insert
        bulk_docs = []
        for j, original_index in enumerate(indices):
            doc = documents[original_index]
            doc_dict = {
                "text": texts[j],
                "embedding": all_embeddings[j],
            }
            
            # Include metadata if available
            if hasattr(doc, 'metadata') and doc.metadata:
                doc_dict.update(doc.metadata)
                
            bulk_docs.append(doc_dict)
        
        # Step 4: Bulk insert with optimized settings
        collection = self.mongo_client[DB_NAME][COLLECTION_NAME]
        
        logger.info("Performing bulk insert...")
        try:
            # Use ordered=False for better performance
            def bulk_insert_sync():
                result = collection.insert_many(bulk_docs, ordered=False)
                return len(result.inserted_ids)
            
            inserted_count = await asyncio.to_thread(bulk_insert_sync)
            logger.info(f"Successfully inserted {inserted_count} documents")
            return inserted_count
        except Exception as e:
            logger.error(f"Bulk insert failed: {e}")
            return 0
    
    async def answer_questions(self, questions: List[str]) -> List[str]:
        """Answer multiple questions concurrently."""
        async def answer_single(question: str) -> str:
            try:
                return await asyncio.to_thread(self.chain.invoke, question)
            except Exception as e:
                logger.error(f"Error answering question '{question}': {e}")
                return f"Error processing question: {str(e)}"
        
        # Process questions concurrently with semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(3)  # Limit to 3 concurrent questions
        
        async def limited_answer(question: str) -> str:
            async with semaphore:
                return await answer_single(question)
        
        tasks = [limited_answer(q) for q in questions]
        answers = await asyncio.gather(*tasks, return_exceptions=False)
        
        return answers
    
    async def run(self, documents_url: str, questions: List[str]) -> List[str]:
        """Main run method - optimized for speed with enhanced document processing."""
        try:
            logger.info(f"Starting RAG processing for {len(questions)} questions")
            
            # Setup vector store
            await self.setup_vector_store()
            
            # Process document with enhanced parsing and chunking
            documents, parsed_pages = await self.process_document_async(documents_url)
            
            logger.info(f"Processed {len(documents)} document chunks")
            
            # Setup chain with hybrid retrieval
            await self.setup_chain(documents)
            
            # Add to vector store using optimized bulk method
            total_added = await self.add_documents_bulk_optimized(documents)
            logger.info(f"Total documents added: {total_added}")
            
            logger.info("Documents added to vector store, answering questions...")
            
            # Answer questions
            answers = await self.answer_questions(questions)
            
            logger.info("RAG processing completed successfully")
            return answers
            
        except Exception as e:
            logger.error(f"Error in RAG service run: {e}")
            raise