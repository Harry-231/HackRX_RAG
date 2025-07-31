from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
import os
import asyncio
import nest_asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import TYPE_CHECKING
from rag_service import OptimizedRAGService

if TYPE_CHECKING:
    from langchain_core.documents import Document
else:
    Document = object
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["LLAMA_CLOUD_API_KEY"] = os.getenv("LLAMA_CLOUD_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["VOYAGE_API_KEY"] = os.getenv("VOYAGE_API_KEY")
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment setup
from dotenv import load_dotenv
load_dotenv()

# Async setup for notebooks/jupyter compatibility
nest_asyncio.apply()

# FastAPI app
app = FastAPI(title="HackRX RAG API", version="1.0.0")

# Security
security = HTTPBearer()

# Thread pool for CPU-intensive operations
executor = ThreadPoolExecutor(max_workers=4)

# Pydantic models
class QueryRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# API Key validation
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    expected_key = os.getenv("API_KEY", "your-secret-api-key")
    if credentials.credentials != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials



# Initialize RAG service
rag_service = OptimizedRAGService()

@app.post("/hackrx/run", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Process documents and answer questions using RAG
    """
    try:
        # Convert HttpUrl to string
        documents_url = str(request.documents)
        
        # Process with timeout
        answers = await asyncio.wait_for(
            rag_service.run(documents_url, request.questions),
            timeout=60.0  # 40 second timeout
        )
        
        return QueryResponse(answers=answers)
        
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=408,
            detail="Request timeout - processing took longer than 30 seconds"
        )
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "HackRX RAG API is running"}

