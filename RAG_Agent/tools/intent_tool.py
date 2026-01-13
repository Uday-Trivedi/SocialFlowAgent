"""Intent detection tool - keyword-based, no LLM."""

from langchain_core.tools import tool


VALID_INTENTS = [
    "overview",
    "features",
    "pricing",
    "supported_platforms",
    "limitations",
    "business"
]

INTENT_KEYWORDS = {
    "overview": ["what is", "about", "overview", "explain"],
    "features": ["features", "capabilities", "what can", "does it do"],
    "pricing": ["price", "pricing", "cost", "how much", "plans"],
    "supported_platforms": ["platform", "supports", "instagram", "facebook", "linkedin", "website"],
    "limitations": ["limitations", "limits", "cannot", "does not"],
    "business": ["buy", "demo", "trial", "contact", "email", "pricing plan"]
}


@tool
def detect_intent(query: str) -> list[str]:
    """Detects user intent from query using keyword matching."""
    query_lower = query.lower()
    detected_intents = []
    
    for intent, keywords in INTENT_KEYWORDS.items():
        for keyword in keywords:
            if keyword in query_lower:
                if intent not in detected_intents:
                    detected_intents.append(intent)
                break
    
    if not detected_intents:
        detected_intents = ["overview"]
    
    return detected_intents

