import os
from dotenv import load_dotenv
import streamlit as st
from search import init_exa_client, perform_web_research

def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize Exa client
    try:
        exa_client = init_exa_client()
        st.success("✅ Successfully initialized Exa client")
    except Exception as e:
        st.error(f"❌ Failed to initialize Exa client: {str(e)}")
        return

    # Create a simple search interface
    st.title("Search Test")
    
    # Search parameters
    query = st.text_input("Enter your search query:")
    hours_back = st.slider("Hours to look back:", 1, 168, 24)  # 1 hour to 1 week
    num_results = st.slider("Number of results:", 1, 20, 5)
    
    search_engines = st.multiselect(
        "Select search engines:",
        ["Exa", "Tavily"],
        default=["Exa", "Tavily"]
    )

    if st.button("Search") and query:
        with st.spinner("Searching..."):
            results = perform_web_research(
                exa_client=exa_client,
                query=query,
                num_results=num_results,
                hours_back=hours_back,
                search_engines=search_engines
            )
            
            if results:
                for idx, result in enumerate(results):
                    with st.expander(f"{idx + 1}. {result['title']}"):
                        st.write(f"Source: {result['source']}")
                        st.write(f"URL: {result['url']}")
                        st.write(f"Published: {result['published_date']}")
                        st.write("Content:")
                        st.write(result['text'])
            else:
                st.warning("No results found.")

if __name__ == "__main__":
    main()
