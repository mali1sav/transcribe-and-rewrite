import os
from tavily import TavilyClient
from datetime import datetime, timedelta, timezone
import streamlit as st
from urllib.parse import urlparse
from exa_py import Exa
import httpx
from dateutil import parser
from dotenv import load_dotenv
from openai import OpenAI
import re
import html  # <--- Added for unescaping HTML entities
import unicodedata  # <--- Added for unicode normalization

# Load environment variables
load_dotenv()

def init_exa_client():
    exa_api_key = os.getenv('EXA_API_KEY')
    if not exa_api_key:
        raise ValueError("EXA_API_KEY not found in environment variables")
    return Exa(api_key=exa_api_key)

def get_domain(url):
    """Extract domain from URL"""
    try:
        return url.split('/')[2]
    except:
        return url

def format_time_ago(published_date):
    """Format a datetime string into a human-readable 'time ago' format."""
    if not published_date:
        return "Unknown time"
        
    try:
        # Try different date formats
        date_formats = [
            "%Y-%m-%dT%H:%M:%S.%fZ",  # Standard ISO format with microseconds
            "%Y-%m-%dT%H:%M:%SZ",     # ISO format without microseconds
            "%Y-%m-%d %H:%M:%S",      # Basic datetime format
            "%Y-%m-%d",               # Just date
        ]
        
        parsed_date = None
        for fmt in date_formats:
            try:
                if isinstance(published_date, str):
                    parsed_date = datetime.strptime(published_date, fmt)
                    if fmt == "%Y-%m-%d":
                        parsed_date = parsed_date.replace(hour=0, minute=0, second=0)
                    break
            except ValueError:
                continue
        
        if not parsed_date and isinstance(published_date, str):
            try:
                parsed_date = parser.parse(published_date)
            except:
                print(f"Failed to parse date: {published_date}")
                return "Unknown time"
        
        if not parsed_date:
            return "Unknown time"
            
        now = datetime.now(timezone.utc)
        if parsed_date.tzinfo is None:
            parsed_date = parsed_date.replace(tzinfo=timezone.utc)
        
        diff = now - parsed_date
        seconds = diff.total_seconds()
        if seconds < 60:
            return "just now"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        elif seconds < 86400:
            hours = int(seconds / 3600)
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        elif seconds < 604800:
            days = int(seconds / 86400)
            return f"{days} day{'s' if days != 1 else ''} ago"
        elif seconds < 2592000:
            weeks = int(seconds / 604800)
            return f"{weeks} week{'s' if weeks != 1 else ''} ago"
        else:
            return parsed_date.strftime("%Y-%m-%d")
    except Exception as e:
        print(f"Error formatting time for date {published_date}: {str(e)}")
        return "Unknown time"

def init_tavily_client():
    tavily_api_key = os.getenv('TAVILY_API_KEY')
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY not found in environment variables")
    return TavilyClient(api_key=tavily_api_key)

def init_openai_client():
    """Initialize OpenAI client with OpenRouter configuration."""
    try:
        openai_api_key = os.getenv('OPENROUTER_API_KEY')
        if not openai_api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
            
        # Initialize with OpenRouter base URL and headers
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openai_api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/your-repo", # To identify your app
                "X-Title": "Fast Transcriber" # To identify your app
            }
        )
        return client
    except Exception as e:
        st.error(f"Failed to initialize OpenAI client: {str(e)}")
        return None

def perform_exa_search(exa_client, query, num_results=5, hours_back=24, categories=None):
    """
    Perform search using Exa's search_and_contents API.
    Returns a list of dictionaries containing search results.
    """
    try:
        # Calculate the date range
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(hours=hours_back)
        
        # Format dates for Exa (ISO 8601 format)
        start_date_str = start_date.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_date_str = end_date.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        # Base search parameters
        base_params = {
            'query': query,
            'num_results': max(2, num_results // 3),  # Split results among categories
            'start_published_date': start_date_str,
            'end_published_date': end_date_str,
            'type': 'auto',  # Let Exa choose the best search type
            'use_autoprompt': True,
            'text': True  # Get full text content
        }
        
        all_results = []
        categories = categories or ['news', 'company', 'tweet']
        
        # Perform search for each category
        for category in categories:
            try:
                search_params = base_params.copy()
                search_params['category'] = category
                
                # Perform search with contents
                response = exa_client.search_and_contents(**search_params)
                
                # Get results from the response
                if response is not None:
                    category_results = []
                    if hasattr(response, 'results'):
                        category_results = response.results
                    elif isinstance(response, list):
                        category_results = response
                    
                    # Transform results to dictionary format
                    for result in category_results:
                        if hasattr(result, 'text') and result.text:
                            # Simple text cleaning - just convert to string
                            text = str(result.text)
                            title = str(result.title) if hasattr(result, 'title') and result.title else 'No Title'
                            url = str(result.url) if hasattr(result, 'url') and result.url else ''
                            
                            # Parse published date
                            published_date = None
                            if hasattr(result, 'published_date') and result.published_date:
                                try:
                                    published_date = parser.parse(result.published_date)
                                    if not published_date.tzinfo:
                                        published_date = published_date.replace(tzinfo=timezone.utc)
                                    published_date = published_date.isoformat()
                                except:
                                    pass
                            
                            transformed_result = {
                                'title': title,
                                'url': url,
                                'published_date': published_date,
                                'text': text,
                                'source': get_domain(url)
                            }
                            all_results.append(transformed_result)
            except Exception as e:
                print(f"Error searching {category} category: {str(e)}")
                continue
        
        # Sort results by date and limit to requested number
        def get_date_for_sorting(result):
            try:
                if result.get('published_date'):
                    date = parser.parse(result['published_date'])
                    return date.replace(tzinfo=timezone.utc) if not date.tzinfo else date
                return datetime.min.replace(tzinfo=timezone.utc)
            except:
                return datetime.min.replace(tzinfo=timezone.utc)
                
        all_results.sort(key=get_date_for_sorting, reverse=True)
        all_results = all_results[:num_results]
        
        if all_results:
            st.success(f"✅ Found {len(all_results)} results from Exa")
            
        return all_results
    except Exception as e:
        st.error(f"Error during Exa search: {str(e)}")
        return []

def serialize_search_results(search_results):
    """Simple serialization of search results."""
    if not search_results:
        return {"results": []}
        
    serialized_results = {
        "results": [
            {
                "title": result["title"],
                "url": result["url"],
                "text": result["text"],
                "published_date": result["published_date"],
                "source": result["source"]
            }
            for result in search_results
        ]
    }
    return serialized_results

def perform_tavily_search(tavily_client, query, num_results=5, hours_back=24):
    try:
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(hours=hours_back)
        
        start_date_str = start_date.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        response = tavily_client.search(
            query=query,
            search_depth="advanced",
            max_results=num_results,
            include_raw_content=True,
            filter_after_date=start_date_str
        )
        
        results = []
        if response and 'results' in response:
            for result in response['results']:
                cleaned_content = str(result.get('content', ''))
                
                published_date = None
                if 'published_date' in result:
                    try:
                        published_date = parser.parse(result['published_date'])
                        if not published_date.tzinfo:
                            published_date = published_date.replace(tzinfo=timezone.utc)
                    except Exception as e:
                        st.error(f"Error parsing Tavily date: {str(e)}")
                
                formatted_result = {
                    'title': result.get('title', ''),
                    'url': result.get('url', ''),
                    'text': cleaned_content,
                    'source': get_domain(result.get('url', '')),
                    'published_date': published_date.isoformat() if published_date else None
                }
                results.append(formatted_result)
        
        return results
    except Exception as e:
        st.error(f"Error performing Tavily search: {str(e)}")
        return None

def perform_web_research(exa_client, query, hours_back, search_engines):
    results = []
    try:
        exa_results = perform_exa_search(exa_client, query, num_results=3, hours_back=hours_back)
        results.extend(exa_results)
    except Exception as e:
        st.error(f"Exa search failed: {str(e)}")
    
    try:
        tavily_client = init_tavily_client()
        tavily_results = perform_tavily_search(tavily_client, query, num_results=3, hours_back=hours_back)
        results.extend(tavily_results)
    except Exception as e:
        st.error(f"Tavily search failed: {str(e)}")
    
    return results

def prepare_content_for_gpt(search_results, selected_indices):
    """Prepare content for GPT processing."""
    try:
        prepared_content = []
        for idx in selected_indices:
            result = search_results["results"][idx]
            content_item = {
                'text': result['text'].strip(),
                'source': result['source'],
                'url': result['url']
            }
            prepared_content.append(content_item)
        return prepared_content
    except Exception as e:
        st.error(f"Error preparing content: {str(e)}")
        return None

def prepare_content_for_article(selected_results):
    """Prepare selected search results for article generation."""
    try:
        prepared_content = []
        for result in selected_results:
            url = result.get('url', '').strip()
            source = result.get('source', '').strip()
            text = str(result.get('text', '').strip())
            
            content_item = {
                'url': url,
                'source': source,
                'content': text
            }
            prepared_content.append(content_item)
        return prepared_content
    except Exception as e:
        st.error(f"Error preparing content: {str(e)}")
        return None

def generate_article(client: OpenAI, transcripts, keywords=None, section_count=3):
    try:
        if not transcripts:
            return None

        keyword_list = []
        if keywords:
            keyword_list = [k.strip() for k in keywords.split('\n') if k.strip()]
        
        primary_keyword = keyword_list[0].upper() if keyword_list else ""
        
        shortcode_map = {
            "BITCOIN": '[latest_articles label="ข่าว Bitcoin (BTC) ล่าสุด" count_of_posts="6" taxonomy="category" term_id="7"]',
            "ETHEREUM": '[latest_articles label="ข่าว Ethereum ล่าสุด" count_of_posts="6" taxonomy="category" term_id="8"]',
            "SOLANA": '[latest_articles label="ข่าว Solana ล่าสุด" count_of_posts="6" taxonomy="category" term_id="501"]',
            "XRP": '[latest_articles label="ข่าว XRP ล่าสุด" count_of_posts="6" taxonomy="category" term_id="502"]',
            "DOGECOIN": '[latest_articles label="ข่าว Dogecoin ล่าสุด" count_of_posts="6" taxonomy="category" term_id="527"]'
        }

        keyword_instruction = f"""Primary Keyword Optimization:
Primary Keyword: {keyword_list[0] if keyword_list else ""}
This must appear naturally ONCE in Title, Meta Description, and H1.
Use this in H2 headings and paragraphs where they fit naturally.
Secondary Keywords: {', '.join(keyword_list[1:]) if len(keyword_list) > 1 else 'none'}
- Use these in H2 headings and paragraphs where they fit naturally
- Each secondary keyword should appear no more than 5 times in the entire content
- Only use these in Title, Meta Description, or H1 if they fit naturally.
- Skip any secondary keywords that don't fit naturally in the context"""

        angle_guidance = """Content Selection and Angle:
- Use the primary keyword and search query to determine the main angle of the article
- Only select content from sources that directly supports this angle
- Maintain focus throughout the article by excluding tangential information
- Ensure each section contributes to the main narrative
- When multiple perspectives are available, prioritize those most relevant to the chosen angle"""

        prompt = f"""write a comprehensive and in-depth Thai crypto news article for WordPress. Ensure the article is detailed and informative, providing thorough explanations and analyses for each key point discussed. Follow these instructions:

{keyword_instruction}

{angle_guidance}

First, write the following sections:

* Meta Description: Summarise the article in 160 characters in Thai.
* H1: Provide a concise title that captures the main idea of the article with a compelling hook in Thai.
* Main content: Start with a strong opening that highlights the most newsworthy aspect. Focus on picking the right angle based on user query and the primary keyword. 

* Create exactly {section_count} distinct and engaging headings (H2) for the main content, ensuring they align with and support the main angle. For each content under each H2, provide an in-depth explanation, context, and implications to Crypto investors.

* Important Instruction: When referencing a source, use this format: '<a href="[URL]">[SOURCE_NAME]</a>'

* บทสรุป: Use a H2 heading. Summarise key points and implications by emphasizing insights.

* Excerpt for WordPress: In Thai, provide 1 sentence for a brief overview.

* Image Prompt: In English, describe a scene that captures the article's essence, focus on only 1 or 2 objects. 

After writing all the above sections, analyze the key points and generate these title options:
* Title & H1 Options:
  1. News style
  2. Question style
  3. Number style

Here are the sources to base the article on:
"""
        for transcript_item in transcripts:
            prompt += f"### Content from {transcript_item['source']}\nSource URL: {transcript_item['url']}\n{transcript_item['content']}\n\n"

        completion = client.chat.completions.create(
            model="openai/gpt-4o-2024-11-20",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=5500,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            extra_headers={
                "HTTP-Referer": "https://github.com/cascade",
                "X-Title": "Cascade"
            }
        )
        
        if completion.choices and len(completion.choices) > 0:
            content = completion.choices[0].message.content
            shortcode = shortcode_map.get(primary_keyword, "")
            if shortcode:
                title_parts = content.split("Title & H1 Options:", 1)
                if len(title_parts) == 2:
                    main_content = title_parts[0]
                    title_content = "Title & H1 Options:" + title_parts[1]
                    
                    parts = main_content.split("บทสรุป:", 1)
                    if len(parts) == 2:
                        summary_parts = parts[1].split("Excerpt for WordPress:", 1)
                        if len(summary_parts) == 2:
                            content = (
                                parts[0] 
                                + "บทสรุป:" 
                                + summary_parts[0] 
                                + "\n\n" 
                                + shortcode 
                                + "\n\n----------------\n\n" 
                                + "Excerpt for WordPress:" 
                                + summary_parts[1].rstrip() 
                                + "\n\n" 
                                + title_content
                            )
                        else:
                            content = parts[0] + "บทสรุป:" + parts[1].rstrip() + "\n\n" + shortcode + "\n\n" + title_content
                    else:
                        parts = main_content.split("Excerpt for WordPress:", 1)
                        if len(parts) == 2:
                            content = (
                                parts[0].rstrip() 
                                + "\n\n" 
                                + shortcode 
                                + "\n\n----------------\n\n" 
                                + "Excerpt for WordPress:" 
                                + parts[1].rstrip() 
                                + "\n\n" 
                                + title_content
                            )
                        else:
                            content = main_content.rstrip() + "\n\n" + shortcode + "\n\n" + title_content
                else:
                    parts = content.split("Excerpt for WordPress:", 1)
                    if len(parts) == 2:
                        content = parts[0].rstrip() + "\n\n" + shortcode + "\n\n----------------\n\n" + "Excerpt for WordPress:" + parts[1]
                    else:
                        content = content.rstrip() + "\n\n" + shortcode + "\n"
            
            content = re.sub(r'\n\*\s*Meta Description:\s*', '\n### Meta Description\n', content)
            content = re.sub(r'\n\*\s*H1:\s*', '\n# ', content)
            content = re.sub(r'\n\*\s*บทสรุป:\s*', '\n## บทสรุป\n', content)
            content = re.sub(r'\n\*\s*Excerpt for WordPress:\s*', '\n### Excerpt for WordPress\n', content)
            content = re.sub(r'\n\*\s*Image Prompt:\s*', '\n### Image Prompt\n', content)
            content = re.sub(r'\n\*\s*Title & H1 Options:\s*', '\n### Title & H1 Options\n', content)
            return content
        
        return None
    except Exception as e:
        st.error(f"Error generating article: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Search and Generate Articles", layout="wide")
    st.title("Search and Generate Articles")

    if 'keywords' not in st.session_state:
        st.session_state.keywords = "Bitcoin\nBTC"
    if 'query' not in st.session_state:
        st.session_state.query = "Bitcoin Technical Price Analysis"
    if 'selected_indices' not in st.session_state:
        st.session_state.selected_indices = []
    if 'article' not in st.session_state:
        st.session_state.article = None
    if 'generating' not in st.session_state:
        st.session_state.generating = False

    try:
        exa_client = init_exa_client()
        openai_client = init_openai_client()
        if not openai_client:
            st.error("Failed to initialize OpenAI client")
            return
    except Exception as e:
        st.error(f"❌ Failed to initialize clients: {str(e)}")
        return

    with st.sidebar:
        query = st.text_input("Enter your search query:", value=st.session_state.query)
        
        st.text_area(
            "Keywords (one per line)",
            height=70,
            key="keywords",
            help="Enter one keyword per line. The first keyword will be the primary keyword for SEO optimization."
        )
        
        section_count = st.slider("Number of sections:", 2, 5, 3)
        hours_back = st.slider("Hours to look back:", 1, 168, 6)
        search_button = st.button("Search")

    if search_button and query:
        with st.spinner("Searching..."):
            results = perform_web_research(
                exa_client=exa_client,
                query=query,
                hours_back=hours_back,
                search_engines=["Exa", "Tavily"]
            )
            if results:
                st.session_state.search_results = serialize_search_results(results)
            else:
                st.warning("No results found. Try adjusting your search parameters.")
    
    if hasattr(st.session_state, 'search_results'):
        results = st.session_state.search_results
        st.subheader("Search Results")
        
        for idx, result in enumerate(results['results']):
            with st.container():
                cols = st.columns([0.05, 0.95])
                with cols[0]:
                    is_selected = idx in st.session_state.selected_indices
                    if st.checkbox("", value=is_selected, key=f"select_{idx}"):
                        if idx not in st.session_state.selected_indices:
                            st.session_state.selected_indices.append(idx)
                    else:
                        if idx in st.session_state.selected_indices:
                            st.session_state.selected_indices.remove(idx)
                
                with cols[1]:
                    title = result['title']
                    source = result['source']
                    url = result['url']
                    
                    st.markdown(f"#### [{title}]({url})")
                    source_text = f"**Source:** {source}"
                    if result.get('published_date'):
                        source_text += f" | *Published: {format_time_ago(result['published_date'])}*"
                    st.markdown(source_text)
                    
                    preview = result['text'][:300] + "..." if len(result['text']) > 300 else result['text']
                    st.markdown(preview)
                    with st.expander("Show full content"):
                        st.write(result['text'])
        
        if st.session_state.selected_indices:
            if st.button("Generate Article"):
                st.session_state.generating = True
                keywords = st.session_state.keywords.strip().split('\n')
                keywords = [k.strip() for k in keywords if k.strip()]
                
                selected_results = [results['results'][idx] for idx in st.session_state.selected_indices]
                prepared_content = prepare_content_for_article(selected_results)
                
                if prepared_content:
                    with st.spinner("Generating article..."):
                        article = generate_article(
                            client=openai_client,
                            transcripts=prepared_content,
                            keywords='\n'.join(keywords) if keywords else None,
                            section_count=section_count
                        )
                        if article:
                            st.session_state.article = article
                            st.success("Article generated successfully!")
                            if st.session_state.article:
                                st.subheader("Generated Article")
                                st.markdown(st.session_state.article, unsafe_allow_html=True)
                                
                                st.download_button(
                                    label="Download Article",
                                    data=st.session_state.article,
                                    file_name="generated_article.txt",
                                    mime="text/plain",
                                    use_container_width=True
                                )
                        else:
                            st.error("Failed to generate article. Please try again.")

if __name__ == "__main__":
    main()
