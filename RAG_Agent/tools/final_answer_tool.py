"""Final answer generation tool using Gemini."""

import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


@tool
def final_answer_tool(query: str, context: str) -> str:
    """Generates final answer using Gemini with provided context."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "Error: GOOGLE_API_KEY not found in environment variables."
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.7
    )
    
    if context:
        prompt = f"""Use the following context to answer the user query.

Context:
{context}

Query:
{query}

Provide a clear, helpful answer based on the context provided."""
    else:
        prompt = f"""Answer the following user query about SocialFlow, a social-to-lead automation platform.

Query:
{query}

Provide a helpful and informative answer."""
    
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Error generating answer: {str(e)}"

