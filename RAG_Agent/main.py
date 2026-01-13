

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
from RAG_Agent.tools.intent_tool import detect_intent
from RAG_Agent.tools.context_rag_tool import retrieve_context
from RAG_Agent.tools.business_tool import business_chat
from RAG_Agent.tools.final_answer_tool import final_answer_tool
from RAG_Agent.ingestion.ingest_faiss import ingest_knowledge_base


def main():
    """Main application function."""
    st.set_page_config(
        page_title="SoAgent - Social-to-Lead AI Agent",
        page_icon="🤖",
        layout="centered"
    )
    
    st.title("🤖 SoAgent – Social-to-Lead AI Agent")
    st.markdown("Ask questions about SocialFlow, a social-to-lead automation platform.")
    
    project_root = Path(__file__).parent.parent
    faiss_db_path = project_root / "faiss_db"
    if not faiss_db_path.exists() or not any(faiss_db_path.iterdir()):
        with st.spinner("Initializing knowledge base..."):
            try:
                ingest_knowledge_base()
                st.success("Knowledge base loaded!")
            except Exception as e:
                st.error(f"Error loading knowledge base: {e}")
                return
    
    query = st.text_input(
        "User Query",
        placeholder="Ask about SocialFlow features, pricing, platforms, etc.",
        key="user_query"
    )
    
    submit_button = st.button("Submit", type="primary")
    
    if submit_button and query:
        with st.spinner("Detecting intent..."):
            intents = detect_intent.invoke({"query": query})
        
        st.subheader("Detected Intents")
        st.write(", ".join(intents))
        
        context = ""
        business_response = ""
        
        if "business" in intents and len(intents) > 1:
            with st.spinner("Handling business inquiry..."):
                business_response = business_chat.invoke({"query": query})
        
        if any(intent != "business" for intent in intents):
            with st.spinner("Retrieving context..."):
                context = retrieve_context.invoke({
                    "query": query,
                    "intents": intents
                })
        
        with st.spinner("Generating answer..."):
            final_answer = final_answer_tool.invoke({
                "query": query,
                "context": context
            })
        
        if business_response:
            st.subheader("Business Inquiry Response")
            st.write(business_response)
        
        st.subheader("Final Answer")
        st.write(final_answer)
        
        if context:
            with st.expander("Retrieved Context"):
                st.text(context[:500] + "..." if len(context) > 500 else context)


if __name__ == "__main__":
    main()

