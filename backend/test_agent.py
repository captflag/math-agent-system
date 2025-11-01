"""
Test script to demonstrate the Math Routing Agent
Run the backend server first: python -m uvicorn app.main:app --reload
Then run this script: python test_agent.py
"""
import requests
import json
import time
from typing import Dict, Any

# API base URL
BASE_URL = "http://localhost:8000/api/v1"

def print_separator():
    print("\n" + "="*80 + "\n")

def test_question(question: str, expected_source: str = None) -> Dict[Any, Any]:
    """
    Send a question to the agent and display the response
    """
    print(f"📝 QUESTION: {question}")
    print(f"Expected routing: {expected_source or 'Auto'}")
    print("-" * 80)
    
    try:
        # Send question
        response = requests.post(
            f"{BASE_URL}/question",
            json={"question": question},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"✅ Status: SUCCESS")
            print(f"🎯 Source: {result.get('source', 'N/A')}")
            print(f"⏱️  Processing time: {result.get('processing_time', 'N/A'):.2f}s")
            print(f"🎓 Confidence: {result.get('confidence', 'N/A'):.2%}")
            print(f"\n📊 SOLUTION:")
            print(result.get('solution', 'No solution provided'))
            
            if result.get('steps'):
                print(f"\n📋 STEPS ({len(result['steps'])} steps):")
                for i, step in enumerate(result['steps'], 1):
                    print(f"  {i}. {step}")
            
            if result.get('citations'):
                print(f"\n📚 CITATIONS:")
                for citation in result['citations']:
                    print(f"  - {citation}")
                    
            return result
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Cannot connect to the backend server!")
        print("Make sure the server is running: python -m uvicorn app.main:app --reload")
        return None
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return None

def test_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Backend server is running!")
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"⚠️  Unexpected response: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Backend server is NOT running!")
        print("Start it with: python -m uvicorn app.main:app --reload")
        return False

def main():
    print("🚀 Math Routing Agent - Test Suite")
    print_separator()
    
    # Check health
    print("Checking backend health...")
    if not test_health():
        return
    
    print_separator()
    
    # Test questions from Knowledge Base (should route to KB)
    kb_questions = [
        "Solve the quadratic equation: x² + 5x + 6 = 0",
        "Find the derivative of f(x) = x³ + 2x² - 5x + 1",
        "Calculate the area of a circle with radius 7 cm"
    ]
    
    print("🎓 TESTING KNOWLEDGE BASE QUESTIONS")
    print("These should route to the knowledge base (fast, high confidence)")
    print_separator()
    
    for i, question in enumerate(kb_questions, 1):
        print(f"\n[Test {i}/6]")
        test_question(question, expected_source="knowledge_base")
        print_separator()
        time.sleep(1)  # Small delay between requests
    
    # Test questions requiring Web Search
    web_questions = [
        "Explain the Collatz conjecture and current research status as of 2024",
        "What are the latest computational approaches to the Riemann Hypothesis?",
        "How is algebraic topology used in modern machine learning?"
    ]
    
    print("\n🌐 TESTING WEB SEARCH QUESTIONS")
    print("These should route to web search (novel/research topics)")
    print_separator()
    
    for i, question in enumerate(web_questions, 4):
        print(f"\n[Test {i}/6]")
        test_question(question, expected_source="web_search")
        print_separator()
        time.sleep(1)
    
    print("\n✨ Test suite completed!")
    print("\nNext steps:")
    print("1. Try the interactive API docs: http://localhost:8000/api/docs")
    print("2. Submit feedback to improve the agent")
    print("3. Check the routing decisions and confidence scores")

if __name__ == "__main__":
    main()
