"""Business/lead capture tool using Gemini."""

import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


@tool
def business_chat(query: str) -> str:
    """Handles business/lead capture conversations using Gemini."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "Error: GOOGLE_API_KEY not found in environment variables."
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.7
    )
    
    prompt = f"""You are a helpful assistant for SocialFlow, a social-to-lead automation platform.

The user has shown business interest. Engage them in a friendly conversation to:
1. Ask for their email address
2. Ask about their company or use case
3. Ask about their preferred pricing plan (Starter, Growth, Business, or Enterprise)

Keep the conversation natural and brief. Don't be pushy.

User query: {query}

Respond with a friendly message that asks for the information above in a conversational way."""

    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Error in business chat: {str(e)}"

