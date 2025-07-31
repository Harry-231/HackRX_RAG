# test_api.py
import requests
import json
import time
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
API_BASE_URL = "http://localhost:8000"
API_KEY = os.getenv("API_KEY", "your-secret-api-key")

def debug_api_key():
    """Debug API key configuration"""
    print("Debugging API key configuration...")
    print(f"API_KEY from .env: {API_KEY[:8]}..." if API_KEY else "No API_KEY found")
    
    # Skip debug endpoint check since it's not implemented in main.py
    print("Note: Using API key from environment variable")

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print(f"Health check status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_rag_endpoint():
    """Test the main RAG endpoint with updated timeout for optimized processing"""
    print("\nTesting RAG endpoint...")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
            "What is the waiting period for pre-existing diseases?",
            "Does this policy cover maternity expenses, and what are the limits?",
            "What is the No Claim Discount (NCD) offered in this policy?",
            "Are there any sub-limits on room rent and ICU charges for Plan A?"
        ]
    }
    
    print(f"Using API Key: {API_KEY[:8]}..." if API_KEY else "No API key")
    print(f"Sending request to {API_BASE_URL}/hackrx/run")
    print(f"Document URL: {payload['documents'][:50]}...")
    print(f"Number of questions: {len(payload['questions'])}")
    
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/hackrx/run",
            headers=headers,
            json=payload,
            timeout=120  # Increased timeout for bulk processing - first run might be slower
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\nResponse status: {response.status_code}")
        print(f"Processing time: {processing_time:.2f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\nAnswers received: {len(result['answers'])}")
            
            for i, (question, answer) in enumerate(zip(payload['questions'], result['answers']), 1):
                print(f"\n--- Question {i} ---")
                print(f"Q: {question}")
                print(f"A: {answer}")
                print("-" * 80)
                
            # Performance analysis
            avg_time_per_question = processing_time / len(payload['questions'])
            print(f"\nPerformance Analysis:")
            print(f"Total processing time: {processing_time:.2f}s")
            print(f"Average time per question: {avg_time_per_question:.2f}s")
            
            return True
        else:
            print(f"Error response: {response.text}")
            
            # Try to parse error details
            try:
                error_details = response.json()
                print(f"Error details: {json.dumps(error_details, indent=2)}")
            except:
                pass
                
            return False
            
    except requests.exceptions.Timeout:
        print(f"Request timed out (>120 seconds)")
        print("Note: First run might take longer due to document processing and embedding computation")
        return False
    except requests.exceptions.ConnectionError:
        print("Connection failed - make sure the server is running on localhost:8000")
        return False
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return False

def test_authentication():
    """Test authentication with invalid API key"""
    print("\nTesting authentication with invalid key...")
    
    headers = {
        "Authorization": "Bearer invalid-key",
        "Content-Type": "application/json"
    }
    
    payload = {
        "documents": "https://example.com/test.pdf",
        "questions": ["Test question?"]
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/hackrx/run",
            headers=headers,
            json=payload,
            timeout=10
        )
        
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        
        return response.status_code == 401
    except Exception as e:
        print(f"Auth test failed: {e}")
        return False

def test_minimal_request():
    """Test with a minimal request to verify basic functionality"""
    print("\nTesting minimal request...")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": ["What is the policy name?"]
    }
    
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/hackrx/run",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"Status: {response.status_code}")
        print(f"Processing time: {processing_time:.2f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Answer: {result['answers'][0]}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"Minimal test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 80)
    print("HackRX RAG API Test Suite - Updated for Optimized Processing")
    print("=" * 80)
    
    # Debug API key first
    debug_api_key()
    
    # Test health check
    health_ok = test_health_check()
    
    if not health_ok:
        print("❌ Health check failed. Make sure the server is running.")
        print("Try running: uvicorn main:app --reload")
        return
    
    print("✅ Health check passed")
    
    # Test authentication
    auth_ok = test_authentication()
    if auth_ok:
        print("✅ Authentication test passed")
    else:
        print("❌ Authentication test failed")
    
    # Test minimal request first
    print("\n" + "=" * 40)
    print("Running minimal test first...")
    minimal_ok = test_minimal_request()
    if minimal_ok:
        print("✅ Minimal request test passed")
    else:
        print("❌ Minimal request test failed")
        print("Skipping full test due to minimal test failure")
        return
    
    # Test main RAG functionality
    print("\n" + "=" * 40)
    print("Running full RAG test...")
    rag_ok = test_rag_endpoint()
    if rag_ok:
        print("✅ RAG endpoint test passed")
    else:
        print("❌ RAG endpoint test failed")
    
    print("\n" + "=" * 80)
    print("Test Summary:")
    print(f"Health Check: {'✅' if health_ok else '❌'}")
    print(f"Authentication: {'✅' if auth_ok else '❌'}")
    print(f"Minimal Request: {'✅' if minimal_ok else '❌'}")
    print(f"Full RAG Test: {'✅' if rag_ok else '❌'}")
    print("=" * 80)
    
    if all([health_ok, auth_ok, minimal_ok, rag_ok]):
        print("🎉 All tests passed! Your RAG API is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the logs above for details.")

if __name__ == "__main__":
    main()