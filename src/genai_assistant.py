import os
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def query_genai_assistant(user_query, context_data):
    """
    Answers user queries using an LLM (Groq) + Context (RAG-lite).
    
    user_query: str
    context_data: dict (computed stats, e.g., {'high_demand_zone': 'Zone_A', 'shortage_count': 5})
    """
    
    api_key = os.getenv("GROQ_API_KEY")
    
    # Construct Prompt
    context_str = "\n".join([f"{k}: {v}" for k, v in context_data.items()])
    
    prompt = f"""
    You are an AI assistant for a Yulu-like EV fleet operations manager.
    Use the following real-time data context to answer the user's question.
    Do not hallucinate facts not present in the context.
    
    Context:
    {context_str}
    
    User Question: {user_query}
    
    Answer (be concise and professional):
    """
    
    if api_key:
        try:
            from groq import Groq
            client = Groq(api_key=api_key)
            
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="llama-3.3-70b-versatile",
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Error connecting to Groq: {str(e)}. Falling back to rule-based logic."
    else:
        # Fallback / Mock Mode for Demo w/o API Key
        return mock_response(user_query, context_data)

def mock_response(query, context):
    """
    Simple rule-based fallback if no API key is present.
    """
    query = query.lower()
    
    if "shortage" in query or "demand" in query:
        return f"Based on current data, the highest demand is in {context.get('high_demand_zone', 'N/A')}. We project a shortage of EVs in {context.get('shortage_zones', 'N/A')}."
    elif "battery" in query:
        return f"Battery status: {context.get('critical_battery_count', 0)} vehicles are Critical (<20%)."
    else:
        return f"I have analyzed the data. Key insight: High demand zone is {context.get('high_demand_zone', 'N/A')}."
