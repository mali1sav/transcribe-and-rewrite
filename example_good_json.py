import streamlit as st
from exa_py import Exa  # For web search functionality
import httpx  # For making HTTP requests
import os  # For interacting with the operating system
import json  # For JSON data handling
from datetime import datetime, timedelta, timezone  # For working with dates and times
from dotenv import load_dotenv  # For loading environment variables
from tavily import TavilyClient
from utils import initialize_exa  # Import the initialize_exa function
from dateutil import parser
from urllib.parse import urlparse
import traceback
import re

# Load environment variables from .env file
load_dotenv()

# Constants
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    st.error("OPENROUTER_API_KEY not found in environment variables.")
    st.stop()

TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
if not TAVILY_API_KEY:
    st.error("TAVILY_API_KEY not found in environment variables.")
    st.stop()

API_BASE_URL = "https://openrouter.ai/api/v1"
HEADERS = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}

# Initialize HTTP client
client = httpx.Client(base_url=API_BASE_URL, headers=HEADERS)

# Initialize Tavily client
def init_tavily_client():
    tavily_api_key = os.getenv('TAVILY_API_KEY')
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY not found in environment variables")
    return TavilyClient(api_key=tavily_api_key)

# Function to extract Title and Meta Description from the generated markdown
def extract_metadata(markdown_text):
    title = ""
    meta_description = ""
    
    lines = markdown_text.split('\n')
    for line in lines:
        line = line.strip()
        if line.lower().startswith("title:"):
            title = line.replace("Title:", "").strip().strip('"')
        elif line.lower().startswith("meta description:"):
            meta_description = line.replace("Meta Description:", "").strip().strip('"')
        
        if title and meta_description:
            break
            
    return title, meta_description

# Helper function to get domain from URL
def get_domain(url):
    """Get clean domain name from URL."""
    try:
        domain = urlparse(url).netloc.lower()
        # Remove www. and common TLDs
        domain = domain.replace('www.', '')
        for tld in ['.com', '.org', '.net', '.edu', '.gov', '.co.uk', '.io']:
            if domain.endswith(tld):
                domain = domain.replace(tld, '')
        # Convert to title case for display
        return domain.strip().title()
    except:
        return 'Unknown'

# Function to format time ago
def format_time_ago(published_date):
    """Format a datetime string into a human-readable 'time ago' format."""
    if not published_date:
        return "Unknown time"
        
    try:
        # Try different date formats
        date_formats = [
            "%Y-%m-%dT%H:%M:%S.%fZ",  # Standard ISO format with microseconds
            "%Y-%m-%dT%H:%M:%SZ",      # ISO format without microseconds
            "%Y-%m-%d %H:%M:%S",       # Basic datetime format
            "%Y-%m-%d",                # Just date
        ]
        
        parsed_date = None
        for fmt in date_formats:
            try:
                if isinstance(published_date, str):
                    parsed_date = datetime.strptime(published_date, fmt)
                    if fmt == "%Y-%m-%d":
                        # If only date is provided, set time to midnight UTC
                        parsed_date = parsed_date.replace(hour=0, minute=0, second=0)
                    break
            except ValueError:
                continue
        
        if not parsed_date and isinstance(published_date, str):
            # Try parsing ISO format with timezone info
            try:
                parsed_date = parser.parse(published_date)
            except:
                print(f"Failed to parse date: {published_date}")
                return "Unknown time"
        
        if not parsed_date:
            return "Unknown time"
            
        # Convert to UTC for consistent comparison
        now = datetime.now(timezone.utc)
        if parsed_date.tzinfo is None:
            parsed_date = parsed_date.replace(tzinfo=timezone.utc)
        
        # Calculate the time difference
        diff = now - parsed_date
        
        # Convert timedelta to human-readable format
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

# Function to convert markdown content to WordPress Gutenberg blocks
def convert_to_wordpress_blocks(content):
    """Convert the content to WordPress blocks format with proper link handling."""
    blocks = []
    
    # Split content into lines
    lines = content.split('\n')
    blocks = []
    
    # Extract and clean metadata
    metadata = {}
    clean_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Clean up metadata lines
        if line.startswith('**') and ':' in line:
            key, value = line.split(':', 1)
            key = key.strip('* ')
            value = value.strip()
            metadata[key] = value
            continue
        elif ':' in line and any(key in line.split(':')[0] for key in ['Title', 'Meta Description', 'บทคัดย่อ', 'หัวข้อ']):
            key, value = line.split(':', 1)
            key = key.strip('* ')
            value = value.strip()
            metadata[key] = value
            continue
            
        clean_lines.append(line)

    # Get title from metadata or first heading
    title = metadata.get('Title', metadata.get('หัวข้อ', ''))
    if not title and clean_lines:
        title = clean_lines[0].strip('#').strip()
        clean_lines = clean_lines[1:]

    # Add title block
    blocks.append('<!-- wp:heading {"level":1} -->')
    blocks.append(f'<h1>{title}</h1>')
    blocks.append('<!-- /wp:heading -->\n')

    # Add meta description if present
    meta_desc = metadata.get('Meta Description', '')
    if meta_desc:
        blocks.append('<!-- wp:html -->')
        blocks.append(f'<!-- Meta Description: {meta_desc} -->')
        blocks.append('<!-- /wp:html -->\n')

    # Process remaining content
    current_paragraph = []
    
    for line in clean_lines:
        line = line.strip()
        if not line:
            if current_paragraph:
                para_content = ' '.join(current_paragraph)
                if para_content.startswith('#'):  # It's a heading
                    heading_text = para_content.lstrip('#').strip()
                    blocks.append('<!-- wp:heading {"level":2} -->')
                    blocks.append(f'<h2>{heading_text}</h2>')
                    blocks.append('<!-- /wp:heading -->\n')
                else:
                    blocks.append('<!-- wp:paragraph -->')
                    blocks.append(f'<p>{para_content}</p>')
                    blocks.append('<!-- /wp:paragraph -->\n')
                current_paragraph = []
            continue
            
        # Remove any remaining markdown formatting
        line = line.replace('**', '')
        if line.startswith('#'):
            if current_paragraph:
                blocks.append('<!-- wp:paragraph -->')
                blocks.append(f'<p>{" ".join(current_paragraph)}</p>')
                blocks.append('<!-- /wp:paragraph -->\n')
                current_paragraph = []
            
            heading_text = line.lstrip('#').strip()
            blocks.append('<!-- wp:heading {"level":2} -->')
            blocks.append(f'<h2>{heading_text}</h2>')
            blocks.append('<!-- /wp:heading -->\n')
        else:
            current_paragraph.append(line)

    # Add any remaining paragraph
    if current_paragraph:
        blocks.append('<!-- wp:paragraph -->')
        blocks.append(f'<p>{" ".join(current_paragraph)}</p>')
        blocks.append('<!-- /wp:paragraph -->\n')

    # Handle image prompt if present
    image_prompt = metadata.get('Image Prompt', '')
    if image_prompt:
        blocks.append('<!-- wp:html -->')
        blocks.append(f'<!-- Image Prompt: {image_prompt.strip("*")} -->')
        blocks.append('<!-- /wp:html -->')

    return '\n'.join(blocks)

def format_source_link(source_name, url):
    """Format a source link with proper HTML anchor tag using full article URL."""
    if not url:
        return source_name
    # Clean up the URL if needed
    url = url.strip()
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    return f'<a href="{url}">{source_name}</a>'

def add_source_reference(text, source_name, url):
    """Add source reference to the end of text with proper link formatting using full article URL."""
    if not text.strip():
        return text
        
    # Remove any existing reference at the end
    if '(อ้างอิง:' in text:
        text = text.split('(อ้างอิง:')[0].strip()
    
    # Ensure we have a valid URL
    if url and not url.strip().startswith(('http://', 'https://')):
        url = 'https://' + url.strip()
        
    if not text.strip().endswith(')'):  # Only add if not already ending with a reference
        source_link = format_source_link(source_name, url)
        text = f"{text} (อ้างอิง: {source_link})"
    return text

def process_search_results(results):
    """Process and format search results for article generation."""
    processed_content = []
    
    if results and "results" in results:
        for item in results["results"]:
            try:
                if item and isinstance(item, dict):
                    text = item['text'].strip()
                    source = item.get('source', 'Unknown')
                    url = item.get('url', '').strip()  # Get the full article URL
                    
                    # Get domain for display name but keep full URL for link
                    if source:
                        display_name = source.title()  # Use source name for display
                    else:
                        display_name = get_domain(url).title()  # Fallback to domain name
                    
                    # Add source reference with proper link using full URL
                    text = add_source_reference(text, display_name, url)
                    processed_content.append(text)
            except Exception as e:
                print(f"Error processing result: {str(e)}")
                continue
    
    return "\n\n".join(processed_content) if processed_content else ""

# Function to perform web research using Exa and Tavily APIs
def perform_web_research(exa_client, query, num_results=5, hours_back=24, search_engines=None):
    try:
        combined_results = []
        
        # Use selected search engines or default to both
        search_engines = search_engines or ["Exa", "Tavily"]
        
        # Perform Exa search if selected
        if "Exa" in search_engines:
            exa_results = perform_exa_search(exa_client, query, num_results, hours_back)
            combined_results.extend(exa_results)
            
        # Perform Tavily search if selected
        if "Tavily" in search_engines:
            tavily_client = init_tavily_client()
            tavily_results = perform_tavily_search(tavily_client, query, num_results, hours_back)
            combined_results.extend(tavily_results)
        
        # Sort results by date (newest first) and remove duplicates
        if combined_results:
            # Convert dates to datetime objects for sorting
            for result in combined_results:
                if isinstance(result['published_date'], str):
                    try:
                        result['datetime'] = parser.parse(result['published_date'])
                    except:
                        result['datetime'] = datetime.now(timezone.utc)
            
            # Sort by date
            combined_results.sort(key=lambda x: x['datetime'], reverse=True)
            
            # Remove the temporary datetime field
            for result in combined_results:
                del result['datetime']
            
            return combined_results
        else:
            st.warning("No results found from the selected search engines.")
            return None
            
    except Exception as e:
        st.error(f"Error during web research: {str(e)}")
        return None

# Function to perform web research using Exa API
def perform_exa_search(exa_client, query, num_results=5, hours_back=24, categories=None):
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
        categories = ['news', 'company', 'tweet']
        
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
                            transformed_result = {
                                'title': str(result.title) if hasattr(result, 'title') and result.title else 'No Title',
                                'url': str(result.url) if hasattr(result, 'url') and result.url else '',
                                'published_date': result.published_date if hasattr(result, 'published_date') and result.published_date else None,
                                'text': str(result.text),
                                'source': get_domain(str(result.url) if hasattr(result, 'url') and result.url else '')
                            }
                            all_results.append(transformed_result)
            except Exception as e:
                print(f"Error searching {category} category: {str(e)}")
                continue
        
        # Sort results by date and limit to requested number
        all_results.sort(
            key=lambda x: parser.parse(x['published_date']) if x.get('published_date') else datetime.min,
            reverse=True
        )
        all_results = all_results[:num_results]
        
        if all_results:
            st.success(f"✅ Found {len(all_results)} results from Exa")
            
        return all_results
    except Exception as e:
        st.error(f"Error during Exa search: {str(e)}")
        return []

# Perform Tavily search
def perform_tavily_search(tavily_client, query, num_results=5, hours_back=24):
    try:
        # Set up search parameters specifically for Tavily
        search_params = {
            'search_depth': "advanced",
            'max_results': num_results,
            'include_raw_content': True,  # Get full content for better filtering
            'include_images': False,
            'include_answer': False,
            'topic': 'general'  # Use general instead of tech to avoid limitations
        }
        
        # Add time parameters for Tavily
        if hours_back <= 24:
            search_params['days'] = 1
        else:
            search_params['days'] = max(1, int(hours_back/24))
        
        # Perform search
        search_response = tavily_client.search(query, **search_params)
        
        # Transform Tavily results
        transformed_results = []
        if isinstance(search_response, dict) and 'results' in search_response:
            for result in search_response['results']:
                # Skip if the content seems unrelated
                title = result.get('title', '')
                content = result.get('content', '')
                query_keywords = set(query.lower().split())
                result_text = (title + ' ' + content).lower()
                
                if not any(keyword in result_text for keyword in query_keywords):
                    continue
                
                # Get the published date
                published_date = result.get('published_date')
                if not published_date:
                    published_date = datetime.now(timezone.utc).isoformat()
                    
                # Extract actual source from URL
                url = result.get('url', '')
                source = get_domain(url)
                    
                transformed_results.append({
                    'title': title or 'No Title',
                    'url': url,
                    'published_date': published_date,
                    'text': content,
                    'source': source  # Use actual source instead of 'Tavily'
                })
        
        if transformed_results:
            st.success(f"✅ Found {len(transformed_results)} results from Tavily")
            
        return transformed_results
    except Exception as e:
        st.error(f"Error performing Tavily search: {str(e)}")
        return []

# Function to serialize search results
def serialize_search_results(search_results):
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

# Function to prepare content for GPT
def prepare_content_for_gpt(search_results, selected_indices):
    """Prepare content for GPT processing."""
    try:
        prepared_content = []
        
        for idx in selected_indices:
            result = search_results["results"][idx]
            
            # Extract core information
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

# Function to generate article using OpenRouter's API
def generate_article(content, query):
    """Generate an article from the selected content."""
    processed_content = process_search_results({"results": content})
    
    # Check if OpenRouter client is initialized
    if not hasattr(generate_article, 'client'):
        openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        if not openrouter_api_key:
            st.error("OpenRouter API key not found in environment variables")
            return None, None, None
        
        generate_article.client = httpx.Client(
            base_url="https://openrouter.ai/api/v1",
            headers={
                "Authorization": f"Bearer {openrouter_api_key}",
                "HTTP-Referer": "https://github.com/cascade", 
                "X-Title": "Cascade"
            }
        )
    
    # First, generate three different titles
    try:
        titles_prompt = {
            "model": "openai/gpt-4o-2024-11-20",
            "messages": [
                {"role": "system", "content": f"""
You are a Thai technology journalist. Generate THREE different title options for an article about {query}.
Each title should be unique and focus on a different aspect or angle of the topic.

Requirements for each title:
- Must be in Thai
- Maximum 60 characters
- News-like headline style, is engaging and click-worthy.
- Must directly relate to {query}
- Keep entity terms (company names, product names, etc.) in English

Return ONLY the three titles in this format:
Title 1: [title]
Title 2: [title]
Title 3: [title]
"""},
                {"role": "user", "content": processed_content}
            ],
            "temperature": 0.8,
            "max_tokens": 300
        }
        
        titles_response = generate_article.client.post("/chat/completions", json=titles_prompt)
        titles_response.raise_for_status()
        titles_content = titles_response.json()['choices'][0]['message']['content']
        
        # Store titles in session state for later selection
        titles = []
        for line in titles_content.split('\n'):
            if line.startswith('Title '):
                title = line.split(': ', 1)[1].strip()
                titles.append(title)
        
        if len(titles) != 3:
            st.error("Failed to generate three distinct titles")
            return None, None, None
        
        # Let user select a title
        st.write("Please select a title:")
        selected_title = st.radio("Choose a title:", titles, key="title_selection")
        
    except Exception as e:
        st.error(f"Error generating titles: {str(e)}")
        return None, None, None
    
    # Generate article using OpenRouter's API with the selected title
    payload = {
        "model": "openai/gpt-4o-2024-11-20",
        "messages": [
            {"role": "system", "content": f"""
You are an expert Thai technology journalist with extensive knowledge across various aspects of '{query}' topic. 
Your task is to create a focused, well-researched article about {query}. The article should be tailored for a Thai audience interested in {query}.

Use this exact title: {selected_title}

**CRITICAL RULES FOR TERMS AND NAMES:**
1. ALWAYS keep the entity terms in English, NEVER translate company, organization, technical terms, product names, people's names, country names, or any other names that are not in Thai. Keep entity terms in English throughout the article, even when the surrounding text is in Thai.
2. When referencing a source, naturally integrate the Brand Name of the source into the sentence as a clickable hyperlink. Encode the URL within the Brand Name. Ensure the Brand Name is a clickable link to the source.
**CONTENT FOCUS AND RELEVANCE:**
1. Content Requirements:
   - Every paragraph must directly support or explain the main topic in the title
   - Do not include tangential or loosely related topics
   - If mentioning other topics, they must have a clear, direct connection to the main topic
   - Remove any information that doesn't strengthen the main narrative

**STRICT FORMAT REQUIREMENTS:**
1. **First lines must be exactly:**
   - Title: {selected_title}
   - Meta Description: [Expand on title's main topic, 160 chars max]
   - H1: {selected_title}
(อ้างอิง: <a href="https://cryptopotato.com/full-article-url">CryptoPotato</a>)
2. **Structure:**
   - Introduction: Set up the main topic and its significance
   - Body (3-4 unique sections): Each unique section must directly support or explain the title's topic
   - Each section should use an H2 heading. Keep each section directly focused on supporting or explaining the title.
   - In each H2 section, use 3-5 paragraphs or list items, whichever best improves readability. Provide in-depth explanations, context, and implications for Crypto investors. Includе direct quotes or specific data points from the transcript where relevant to add credibility.
   - Remove any sections that drift from the main topic
   - Avoid duplicate heading and content. If there is a duplicate, remove it.

3. **Guidelines:**
   - Write in Thai, use semi-professional language.
   - Present complex technology concepts in a way that Thai readers with basic knowledge of the topic can easily understand.
   - Ensure smooth transitions between sections.

4. **Required Additional Sections:**
   - สรุป: Summarise key points about the main topic in Thai.
   - Excerpt for WordPress: [1 Thai short sentence for overview that reinforces the H1 and summarizes the main topic.]
   - Suggest a Wordpress slug format in English in 3-6 words separated by '-'
   - Image Prompt: [Provide a single, simple English sentence in 8-10 words for an AI image generator that captures the essence of the article's main points. The sentence should focus on only one or two key objects or concepts. Avoid complex descriptions, multiple elements, and detailed instructions about image composition.]
   - Image ALT text recommendation: [Provide a single, simple Thai sentence in 4-7 words describing the image which aligns with the article's main topic.]
"""}, 
            {"role": "user", "content": processed_content}
        ],
        "temperature": 0.7,
        "max_tokens": 8000  # Increased to allow for longer content
    }
    
    try:
        response = generate_article.client.post("/chat/completions", json=payload)
        response.raise_for_status()
        
        article_content = response.json()['choices'][0]['message']['content']
        
        # Extract metadata
        title, meta_description = extract_metadata(article_content)
        
        if not meta_description:
            try:
                # Generate missing metadata in Thai
                metadata_prompt = f"Based on this Thai article content, generate a Meta description in Thai (160 chars max)\n\nArticle:\n{article_content}"
                metadata_response = generate_article.client.post("/chat/completions", json={
                    "model": "openai/gpt-4o-2024-11-20",
                    "messages": [
                        {"role": "system", "content": "You are a Thai SEO expert. Return only the Meta Description in Thai, prefixed with 'Meta Description: '."},
                        {"role": "user", "content": metadata_prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 300  # Sufficient for metadata
                })
                
                metadata = metadata_response.json()['choices'][0]['message']['content']
                _, meta_description = extract_metadata(metadata)
            except Exception as e:
                st.error(f"Error generating metadata: {str(e)}")
                return None, None, None
            
            article_content = f"Title: {selected_title}\nMeta Description: {meta_description}\n# {selected_title}\n{article_content}"
        
        return article_content, selected_title, meta_description
        
    except httpx.HTTPStatusError as e:
        st.error(f"HTTP error from OpenRouter: {e.response.status_code} - {e.response.text}")
        return None, None, None
    except Exception as e:
        st.error(f"Error during OpenRouter API call: {str(e)}")
        return None, None, None

# Streamlit App Layout
def main():
    # Initialize session state for search results if not exists
    if 'search_performed' not in st.session_state:
        st.session_state.search_performed = False
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
    if 'query' not in st.session_state:
        st.session_state.query = "SUI and Franklin Templeton Partnership"
    if 'error_message' not in st.session_state:
        st.session_state.error_message = None

    st.set_page_config(page_title="Article Generator", layout="wide")
    st.title("Article Generator")
    st.markdown("""
    ### Generate SEO-Optimized Articles from Web Research
    Enter your research query below, specify the number of results and the time frame, and generate an article with Title and Meta Description.
    """)

    # Search input and button in main area
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        query = st.text_input("Enter your web research query:", st.session_state.query, key="query_input")
        perform_research_button = st.button("Perform Research")

    # Sidebar for search parameters
    st.sidebar.title("Search Configuration")
    
    # Search engine selection with descriptions
    st.sidebar.subheader("🔎 Search Engines")
    st.sidebar.markdown("""
    Choose one or both search engines:
    - 🔵 **Exa**: Specialized neural search
    - 🟣 **Tavily**: General web search with good coverage
    """)
    search_engines = st.sidebar.multiselect(
        "Select search engines:",
        ["Exa", "Tavily"],
        default=["Exa", "Tavily"]
    )
    
    if not search_engines:
        st.sidebar.warning("⚠️ Please select at least one search engine")
        st.stop()
    
    # Time frame selection
    st.sidebar.subheader("⏰ Time Frame")
    
    # Define time frame options with more granular recent options
    look_back_options = {
        "30 minutes": 0.5,
        "1 hour": 1,
        "2 hours": 2,
        "4 hours": 4,
        "6 hours": 6,
        "12 hours": 12,
        "24 hours": 24,
        "48 hours": 48,
        "72 hours": 72
    }
    
    look_back_label = st.sidebar.selectbox(
        "Look back period:",
        options=list(look_back_options.keys()),
        index=2  # Default to "2 hours"
    )
    hours_back = look_back_options[look_back_label]
    
    # Number of results per search engine
    num_results = st.sidebar.number_input(
        "Number of results per engine:", 
        min_value=1, 
        max_value=25, 
        value=5
    )

    if perform_research_button:
        st.session_state.query = query
        if not query.strip():
            st.error("Please enter a search query.")
            st.stop()
            
        with st.spinner("Performing web research..."):
            try:
                exa_client = initialize_exa()
                search_result = perform_web_research(
                    exa_client, 
                    query, 
                    num_results, 
                    hours_back,
                    search_engines=search_engines
                )
                
                if search_result is None:
                    st.stop()
                
                st.session_state.search_results = serialize_search_results(search_result)
                st.session_state.search_performed = True
                
                if not st.session_state.search_results["results"]:
                    st.warning("""
                    No results found. Try:
                    1. Using different search terms
                    2. Extending the time window
                    3. Selecting more categories
                    4. Making the query more general
                    """)
                    st.stop()
                
            except Exception as e:
                st.error(f"""
                Error during research: {str(e)}
                
                Possible solutions:
                1. Check your internet connection
                2. Verify your API key
                3. Try again in a few moments
                4. Reduce the number of results requested
                """)
                st.session_state.error_message = str(e)
                st.stop()

    # Display results if search has been performed
    if st.session_state.search_performed and st.session_state.search_results:
        if not st.session_state.search_results["results"]:
            st.warning("No results found for the given query and time frame.")
            st.stop()

        # Display search results with checkbox selection
        st.subheader("Search Results")
        st.write("Select the sources you want to use for article generation:")
        
        selected_indices = []
        for i, result in enumerate(st.session_state.search_results["results"]):
            domain = get_domain(result['url'])
            source_label = f"{domain}"
            
            col1, col2 = st.columns([0.1, 0.9])
            with col1:
                if st.checkbox("", key=f"result_{i}"):
                    selected_indices.append(i)
            with col2:
                st.markdown(f"**[{result['title']}]({result['url']})**")
                st.markdown(f"*Source: {source_label} | Published: {format_time_ago(result['published_date'])}*")
                
                # Get and clean text preview
                text = result.get('text', '').strip()
                if text:
                    # Find the first substantial paragraph (more than 100 characters)
                    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
                    preview = next((p for p in paragraphs if len(p) > 100), paragraphs[0] if paragraphs else text)
                    
                    # Limit preview length but try to end at a sentence
                    if len(preview) > 300:
                        # Try to find the last sentence end within first 300 chars
                        end_pos = -1
                        for end_char in ['. ', '! ', '? ']:
                            pos = preview[:300].rfind(end_char)
                            if pos > end_pos:
                                end_pos = pos
                        
                        if end_pos > 0:
                            preview = preview[:end_pos + 1]
                        else:
                            preview = preview[:300] + "..."
                    
                    st.write(preview)
                st.markdown("---")
        
        if not selected_indices:
            st.error("Please select at least one source to generate the article.")
            st.stop()

        # Add a "Generate Article" button after source selection
        generate_article_button = st.button("Generate Article")

        if generate_article_button:
            # Prepare content for GPT
            content = prepare_content_for_gpt(st.session_state.search_results, selected_indices)

            if not content:
                st.error("No content available to generate the article.")
                st.stop()

            # Generate the article
            with st.spinner("Generating article..."):
                try:
                    article_markdown, title, meta_description = generate_article(content, st.session_state.query)

                    if article_markdown and title and meta_description:
                        st.session_state.generated_article = article_markdown
                        st.session_state.error_message = None
                        st.success("Article generated successfully!")
                        
                        # **Separate Metadata from Article Content**
                        lines = article_markdown.split('\n')
                        metadata_end_index = 0
                        for i, line in enumerate(lines):
                            if line.startswith('# '):
                                metadata_end_index = i + 1
                                break
                        
                        # Extract the article body excluding metadata
                        article_body = '\n'.join(lines[metadata_end_index:]).strip()

                        # **Display the Clean Article**
                        st.subheader("Generated Article")
                        st.markdown(article_body, unsafe_allow_html=True)

                        # **Generate Gutenberg Blocks from Full Article (Including Metadata)**
                        wordpress_blocks = convert_to_wordpress_blocks(article_markdown)

                        # **Provide Download Button for Gutenberg Blocks**
                        st.download_button(
                            label="Download WordPress Gutenberg Blocks",
                            data=wordpress_blocks,
                            file_name=f"wordpress_blocks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain",
                        )
                    else:
                        error_msg = "Failed to generate the article. Check the logs below for details."
                        st.error(error_msg)
                        st.session_state.error_message = error_msg
                except Exception as e:
                    error_msg = f"Error generating article: {str(e)}"
                    st.error(error_msg)
                    st.session_state.error_message = error_msg
                    traceback.print_exc()

    # Display error message if any
    if 'error_message' in st.session_state and st.session_state.error_message:
        st.error(f"Error Details: {st.session_state.error_message}")
        
    # Display debug information
    if st.checkbox("Show Debug Information"):
        st.subheader("Debug Information")
        st.write("Query:", st.session_state.query)
        st.write("Content Items:", len(st.session_state.search_results["results"]) if st.session_state.search_results else 0)
        if st.session_state.search_results:
            st.json(st.session_state.search_results)

if __name__ == "__main__":
    main()