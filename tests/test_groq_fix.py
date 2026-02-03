import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
load_dotenv()

from src.genai_assistant import query_genai_assistant

def test_groq_connection():
    context = {"status": "testing", "metric": 100}
    print("Testing Groq connection with new model...")
    response = query_genai_assistant("Hello, are you working?", context)
    print(f"Response: {response}")

if __name__ == "__main__":
    test_groq_connection()
