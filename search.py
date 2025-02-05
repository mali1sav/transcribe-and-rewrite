import os
import pytz
from datetime import datetime, timedelta, timezone
import streamlit as st
from urllib.parse import urlparse
from exa_py import Exa
import httpx
from dateutil import parser
from dotenv import load_dotenv
import re
import html
import unicodedata
import json
import base64
from together import Together
import google.generativeai as genai
import time
import tenacity
from pydantic import BaseModel
from firecrawl import FirecrawlApp

load_dotenv()

def init_exa_client():
    exa_api_key = os.getenv('EXA_API_KEY')
    if not exa_api_key:
        raise ValueError("EXA_API_KEY not found in environment variables")
    return Exa(api_key=exa_api_key)

def get_domain(url):
    try:
        return url.split('/')[2]
    except:
        return url

def init_gemini_client():
    """Initialize Google Gemini client."""
    try:
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            st.error("Gemini API key not found. Please set GEMINI_API_KEY in your environment variables.")
            return None
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        return {
            'model': model,
            'name': 'gemini-2.0-flash-exp'
        }
    except Exception as e:
        st.error(f"Failed to initialize Gemini client: {str(e)}")
        return None

@tenacity.retry(
    stop=tenacity.stop_after_attempt(5),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=20),
    retry=tenacity.retry_if_exception_type((Exception)),
    retry_error_callback=lambda retry_state: None
)
def make_gemini_request(client, prompt):
    """Make Gemini API request with retries and proper error handling"""
    try:
        # Add retry logic for Gemini API
        for attempt in range(3):
            try:
                response = client['model'].generate_content(prompt)
                if response and response.text:
                    return response.text
            except Exception as e:
                if attempt == 2:  # Last attempt
                    raise
                st.warning(f"Retrying Gemini request (attempt {attempt + 2}/3)...")
                time.sleep(2 ** attempt)  # Exponential backoff
                
        raise Exception("Failed to get valid response from Gemini API after all retries")
            
    except Exception as e:
        st.error(f"Error making Gemini request: {str(e)}")
        raise

class ArticleContent(BaseModel):
    """Schema for article content extraction"""
    title: str
    author: str = None
    published_date: str = None
    content: str

def extract_with_firecrawl(url):
    try:
        api_key = os.getenv('FIRECRAWL_API_KEY')
        if not api_key:
            st.error("Firecrawl API key not found. Please set FIRECRAWL_API_KEY in your environment")
            return None
            
        app = FirecrawlApp(api_key=api_key)
        
        # Define extraction schema
        class NestedModel1(BaseModel):
            title: str
            author: str = None
            published_date: str = None
            content: str

        class ExtractSchema(BaseModel):
            article: NestedModel1
            
        data = app.extract([url], {
            'prompt': 'Extract the article title, author, published date, and content. Ensure the title and content are always included.',
            'schema': ExtractSchema.model_json_schema(),
        })
        
        if data and data.get('success') and data.get('data'):
            article = data['data'].get('article')
            if article:
                return {
                    'title': article.get('title'),
                    'author': article.get('author'),
                    'published_date': article.get('published_date'),
                    'text': article.get('content')
                }
        
        st.warning("Firecrawl could not extract content")
        return None
            
    except Exception as e:
        st.error(f"Firecrawl extraction failed: {str(e)}")
        return None

def extract_url_content(gemini_client, url, messages_placeholder):
    """Extract article content using Firecrawl first, then fallback to Gemini if needed."""
    try:
        # First try with Firecrawl
        with messages_placeholder:
            st.info(f"Attempting to extract content from {url} using Firecrawl...")
        fc_content = extract_with_firecrawl(url)
        if fc_content:
            return {
                'title': fc_content.get('title', 'Extracted Content'),
                'url': url,
                'text': fc_content.get('text', ''),
                'published_date': None,
                'source': url,
                'author': fc_content.get('author')
            }
        
        # If Firecrawl fails, try Gemini
        with messages_placeholder:
            st.info(f"Firecrawl failed, trying Gemini for {url}...")
        prompt = f"""Extract the main article content from this URL: {url}
        Return ONLY the article text content, no additional formatting or commentary.
        If you cannot access the content, respond with 'EXTRACTION_FAILED'."""
        
        response = make_gemini_request(gemini_client, prompt)
        
        if response and 'EXTRACTION_FAILED' not in response:
            return {
                'title': 'Extracted Content',
                'url': url,
                'text': response,
                'published_date': None,
                'source': url,
                'author': None
            }
        
        return None
    except Exception as e:
        print(f"Error extracting content from {url}: {str(e)}")
        return None

def perform_exa_search(exa_client, query, num_results=10, hours_back=12):
    try:
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(hours=hours_back)
        
        start_date_str = start_date.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_date_str = end_date.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        response = exa_client.search_and_contents(
            query=query,
            num_results=num_results,
            start_published_date=start_date_str,
            end_published_date=end_date_str,
            type='auto',
            category='news',
            text=True
        )
        
        if not response:
            return []
            
        results = getattr(response, 'results', None) or response
        all_results = []
        
        excluded_domains = ['twitter.com', 'youtube.com', 'youtu.be']
        
        for result in results:
            # Get URL and check against excluded domains
            url = str(getattr(result, 'url', '') or '')
            domain = get_domain(url)
            if any(excluded in domain for excluded in excluded_domains):
                continue
                
            title = str(getattr(result, 'title', 'No Title') or 'No Title')
            text = str(getattr(result, 'text', '') or '')
            
            # Parse date if available
            published_date = None
            if hasattr(result, 'published_date') and result.published_date:
                try:
                    parsed = parser.parse(result.published_date)
                    if not parsed.tzinfo:
                        parsed = parsed.replace(tzinfo=timezone.utc)
                    published_date = parsed.isoformat()
                except:
                    pass
            
            transformed_result = {
                'title': title,
                'url': url,
                'published_date': published_date,
                'text': text,
                'source': domain
            }
            all_results.append(transformed_result)
            
        if all_results:
            st.success(f"✅ Found {len(all_results)} results from Exa")
            
        return all_results
        
    except Exception as e:
        st.error(f"Error during Exa search: {str(e)}")
        return []

def perform_web_research(exa_client, query, hours_back, search_engines=None):
    """
    Perform web research using available search engines.
    Currently using Exa for comprehensive search results.
    """
    results = []
    
    try:
        # Get results from Exa
        exa_results = perform_exa_search(
            exa_client=exa_client,
            query=query,
            hours_back=hours_back
        )
        if exa_results:
            results.extend(exa_results)
            
        return results
    except Exception as e:
        st.error(f"Error during web research: {str(e)}")
        return []

def serialize_search_results(search_results):
    if not search_results:
        return {"results": []}
    serialized_results = {
        "results": [
            {
                "title": r["title"],
                "url": r["url"],
                "text": r["text"],
                "published_date": r["published_date"],
                "source": r["source"]
            }
            for r in search_results
        ]
    }
    return serialized_results

def prepare_content_for_article(selected_results):
    """Takes a list of results, each with 'url','source','text', and returns a 'transcript' list for GPT."""
    try:
        prepared_content = []
        for result in selected_results:
            url = result.get('url', '').strip()
            source = result.get('source', '').strip()
            # Handle both 'text' and 'content' fields for compatibility
            content = result.get('content', result.get('text', '')).strip()
            
            content_item = {
                'url': url,
                'source': source,
                'content': content  # Changed from 'text' to match generate_article's expected format
            }
            prepared_content.append(content_item)
        return prepared_content
    except Exception as e:
        st.error(f"Error preparing content: {str(e)}")
        return []

def escape_special_chars(text):
    """Escape special characters that might interfere with Markdown formatting."""
    # Escape dollar signs that aren't already part of a LaTeX equation
    text = re.sub(r'(?<!\$)\$(?!\$)', r'\$', text)
    # Escape other special Markdown characters
    chars_to_escape = ['*', '_', '`', '#', '~', '|', '<', '>', '[', ']']
    for char in chars_to_escape:
        text = text.replace(char, '\\' + char)
    return text

def generate_article(client, transcripts, keywords=None, news_angle=None, section_count=3, promotional_text=None):
    try:
        if not transcripts:
            return None

        # Keywords should already be a list from the text area split
        keyword_list = keywords if keywords else []
        primary_keyword = keyword_list[0].upper() if keyword_list else ""
        secondary_keywords = ", ".join(keyword_list[1:]) if len(keyword_list) > 1 else ""
        
        shortcode_map = {
            "BITCOIN": '[latest_articles label="ข่าว Bitcoin (BTC) ล่าสุด" count_of_posts="6" taxonomy="category" term_id="7"]',
            "ETHEREUM": '[latest_articles label="ข่าว Ethereum ล่าสุด" count_of_posts="6" taxonomy="category" term_id="8"]',
            "SOLANA": '[latest_articles label="ข่าว Solana ล่าสุด" count_of_posts="6" taxonomy="category" term_id="501"]',
            "XRP": '[latest_articles label="ข่าว XRP ล่าสุด" count_of_posts="6" taxonomy="category" term_id="502"]',
            "DOGECOIN": '[latest_articles label="ข่าว Dogecoin ล่าสุด" count_of_posts="6" taxonomy="category" term_id="527"]'
        }
        
        prompt = f"""
Write a comprehensive and in-depth news article in Thai (Title, Main Content, บทสรุป, Excerpt for WordPress, Title & H1 Options, and Meta Description Options all in Thai).
When creating section headings (H2) and subheadings (H3), use one of the relevant guidelines below:
1. Use power words that often in appear in Thai crypto news headline. Choose words that best fit the specific news context.
2. Include specific numbers/stats when relevant (examples: "10 เท่า!", "5 เหรียญ Meme ที่อาจพุ่ง 1000%")
3. Create curiosity gaps (examples:"เบื้องหลังการพุ่งทะยานของราคา...", "จับตา! สัญญาณที่บ่งชี้ว่า...")
4. Make bold, specific statements (examples:"เหรียญคริปโตที่ดีที่สุด", "ทำไมวาฬถึงทุ่มเงินหมื่นล้านใส่...")

Primary Keyword: {primary_keyword or ""}
Secondary Keywords: {keywords or ""}
News Angle: {news_angle or ""}

# Content Focus Instructions:
* The article MUST be written from the perspective of the specified News Angle above
* Only include information from sources that is relevant to and supports this news angle
* Analyze each source and extract only the content that aligns with or provides context for the news angle
* If a source contains information not relevant to the news angle, exclude it
* Ensure each section contributes to developing the specified news angle
* Maintain focus throughout the article - avoid tangents or unrelated information

# Source Citation Rules:
* CRITICAL: Each source must be cited EXACTLY ONCE at the beginning of each section
* EVERY section MUST start with "ตามรายงานจาก" followed by the source citation
* Citation format is STRICTLY required as: "ตามรายงานจาก [PUBLICATION_NAME](SOURCE_URL) ..." where:
  - PUBLICATION_NAME is the actual source name (e.g. Forbes, Bloomberg, CoinDesk)
  - SOURCE_URL is the full URL to the source
  - The source name must be a clickable markdown hyperlink
* Example of correct citation:
  "ตามรายงานจาก [Forbes](https://forbes.com/article) Jerome Powell ประธาน Federal Reserve..."
* Example of incorrect citations:
  - "Forbes รายงานว่า..." (Missing ตามรายงานจาก and hyperlink)
  - "[Forbes] Jerome Powell..." (Missing ตามรายงานจาก)
  - "ตามรายงานจาก Forbes..." (Missing hyperlink)
* NEVER repeat a source citation in the same section
* NEVER use placeholder text - always use the actual publication name

# Main Content Guidelines:
* Keep the following terms in English, rest in Thai:
  - Technical terms
  - Entity names
  - People names
  - Place names (including cities, states, countries)
  - Organizations
  - Company names
  - Cryptocurrency names (use proper capitalization: "Bitcoin", not "BITCOIN" or "bitcoin")
  - Platform names
  - Government entities (e.g., "Illinois State", not "รัฐอิลลินอยส์")
* Use proper capitalization for all terms:
  - Cryptocurrencies: "Bitcoin", "Ethereum", "Solana"
  - Companies: "Binance", "Coinbase"
  - Organizations: "Federal Reserve", "Securities and Exchange Commission"
  - Never use ALL CAPS unless it's a widely recognized acronym (e.g., "FBI", "SEC")

* Ensure to avoid unintended Markdown or LaTeX formatting.
* Create exactly {section_count} heading level 2 in Thai for the main content (keep technical terms and entity names in English).
* For each section, ensure a thorough and detailed exploration of the topic, with each section comprising at least 2-4 paragraphs of comprehensive analysis. Strive to simplify complex ideas, making them accessible and easy to grasp. Where applicable, incorporate relevant data, examples, or case studies to substantiate your analysis and provide clarity.
* Use heading level 2 for each section heading. Use sub-headings if necessary. For each sub-heading, provide real examples, references, or numeric details from the sources, with more extensive context.
* If the content contains numbers that represent monetary values, remove $ signs before numbers and add "ดอลลาร์" after the number, ensuring a single space before and after the numeric value.

# Promotional Content Guidelines
{f'''
**Promotional Integration Guidelines**
----------------------------------------
{promotional_text}

* Requirements:
  1. Create a seamless Heading Level 2 that is semantically aligned with the {news_angle} in Thai (keep technical terms and entity names in English but the rest of sentence in Thai).
  2. The content must be in Thai (keep technical terms and entity names in English but the rest in Thai).
  3. Transit seamlessly from the main content into the promotional text.
  4. Find a way to mention {primary_keyword} in a seamless way.
  5. Limit promotional text to 120 words, placed at the end of the article.
''' if promotional_text else ''}

# Article Structure:
## Title
[Create an engaging news-style title that includes {primary_keyword} once, maintaining its original form]

## Main Content
[First paragraph must include {primary_keyword} once in its original form]

[Create H2 sections below, with at least 2 containing {primary_keyword}. Each H2 should align with the news angle]

[Write supporting paragraphs using {primary_keyword} and {secondary_keywords} (if exists) where they fit naturally]

[Conclude with a summary emphasizing the news angle's significance include{primary_keyword} if they fit naturally]

## SEO Elements
1. Title Options (include {primary_keyword} once, maintain original form):
   - [Option 1: News-focused title]
   - [Option 2: Number-focused title]
   - [Option 3: Question-based title]

2. Meta Description Options (include {primary_keyword} once):
   - [Option 1: News angle + key benefit]
   - [Option 2: Number-focused]
   - [Option 3: Question to stimulate curiosity]

3. H1 Options (aligned with Title and Meta Description including {primary_keyword}):
   - [Option 1: Direct news statement]
   - [Option 2: Number-focused statement]
   - [Option 3: Engaging question]

## Additional Elements
Slug URL in English (must include {primary_keyword}; translate Thai keywords to English)
- Image ALT Text in Thai including {primary_keyword} (keep technical terms and entity names in English, rest in Thai)
- Excerpt for WordPress: One sentence in Thai that briefly describes the article

## Image Prompt
[Create a photorealistic scene that fits the main news article, focusing on 1-2 main objects. Keep it simple and clear. Don't include anything from promotional content. Avoid charts, graphs, or technical diagrams as they don't work well with image generation.]

# Source Usage Instructions:
* Use provided sources as primary reference for facts and data
* Your knowledge can supplement for context and understanding, but never override source information
* When source information exists on a topic, it takes precedence over general knowledge

Here are the sources to base the article on:
"""
        # Append transcripts with escaped special characters
        for t in transcripts:
            prompt += f"### Content from {t['source']}\nSource URL: {t['url']}\n{escape_special_chars(t['content'])}\n\n"

        # Make API request with retries and error handling
        content = make_gemini_request(client, prompt)
        if not content:
            return None
            
        shortcode = shortcode_map.get(primary_keyword, "")
        
        # 1) Insert the relevant shortcode in the final text
        if shortcode:
            excerpt_split = "Excerpt for WordPress:"
            if excerpt_split in content:
                parts = content.split(excerpt_split, 1)
                content = (
                    parts[0].rstrip() + "\n\n" +
                    shortcode + "\n\n----------------\n\n" +
                    excerpt_split + parts[1]
                )
            else:
                content += f"\n\n{shortcode}\n\n----------------\n"
        
        # 2) Post-processing to remove "$" and replace with " ดอลลาร์"
        #    A simple regex that targets something like $100,000 → 100,000 ดอลลาร์
        content = re.sub(r"\$(\d[\d,\.]*)", r"\1 ดอลลาร์", content)
        
        return content
    except Exception as e:
        st.error(f"Error generating article: {str(e)}")
        return None

def load_promotional_content():
    """Load promotional content from files in the pr directory."""
    pr_dir = os.path.join(os.path.dirname(__file__), 'pr')
    content_map = {}
    
    if os.path.exists(pr_dir):
        for filename in os.listdir(pr_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(pr_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        # Use filename without extension as key
                        name = os.path.splitext(filename)[0]
                        content_map[name] = content
                except Exception as e:
                    print(f"Error loading {filename}: {str(e)}")
    
    return content_map

def extract_image_prompt(article_text):
    pattern = r"(?i)Image Prompt.*?\n(.*)"
    match = re.search(pattern, article_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def extract_alt_text(article_text):
    pattern = r"(?i)Image ALT Text.*?\n(.*?)(?:\n\n|\Z)"
    match = re.search(pattern, article_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def main():
    st.set_page_config(page_title="Search and Generate Articles", layout="wide")
    
    # Initialize session state
    if 'keywords' not in st.session_state:
        st.session_state.keywords = "Bitcoin"
    if 'query' not in st.session_state:
        st.session_state.query = "Strategic Bitcoin Reserve - Latest news"
    if 'selected_indices' not in st.session_state:
        st.session_state.selected_indices = []
    if 'article' not in st.session_state:
        st.session_state.article = None
    if 'generating' not in st.session_state:
        st.session_state.generating = False
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
    if 'search_results_json' not in st.session_state:
        st.session_state.search_results_json = None
    if 'process_urls' not in st.session_state:
        st.session_state.process_urls = False
    if 'urls_to_process' not in st.session_state:
        st.session_state.urls_to_process = None

    try:
        exa_client = init_exa_client()
        gemini_client = init_gemini_client()
        if not gemini_client:
            st.error("Failed to initialize Gemini client")
            return
    except Exception as e:
        st.error(f"❌ Failed to initialize clients: {str(e)}")
        return

    # Create a placeholder for messages in the main area
    messages_placeholder = st.empty()

    # Sidebar for inputs and controls
    with st.sidebar:
        content_source = st.radio(
            "How would you like to generate your article?",
            ["Search and Generate", "Generate from URLs"],
            key="content_source"
        )
        
        # Common elements that always appear
        st.text_area(
            "Keywords (one per line)",
            height=68,
            key="keywords",
            help="Enter one keyword per line. The first keyword will be the primary keyword for SEO optimization."
        )
        
        news_angle = st.text_input(
            "News Angle",
            value="",
            help="""This can be in Thai or English. Having a clear news angle is essential, especially when sources may lack focus which is bad for SEO. A well-defined angle helps create a coherent narrative around your chosen perspective.
            Tips: You can use one of the English headlines from your selected news sources as your news angle."""
        )

        section_count = st.slider("Number of sections:", 2, 8, 3, key="section_count")
        
        # Promotional content selection
        promotional_content = load_promotional_content()
        st.write("Promotional Content")
        selected_promotions = []
        for name in sorted(promotional_content.keys()):
            if st.checkbox(name, key=f"promo_{name}"):
                selected_promotions.append(promotional_content[name])
        promotional_text = "\n\n".join(selected_promotions) if selected_promotions else None
        
        # Source-specific inputs
        if content_source == "Search and Generate":
            query = st.text_input("Enter your search query:", value=st.session_state.query)
            hours_back = st.slider("Hours to look back:", 1, 168, 12)
            
            if st.button("Search Content", type="primary"):
                st.session_state.process_urls = False
                results = perform_web_research(
                    exa_client=exa_client,
                    query=query,
                    hours_back=hours_back
                )
                if results:
                    st.session_state.search_results = serialize_search_results(results)
                    st.session_state.search_results_json = json.dumps(
                        st.session_state.search_results, 
                        ensure_ascii=False, 
                        indent=4
                    )
                else:
                    with messages_placeholder:
                        st.error("No results found. Try adjusting your search parameters.")

        else:  # Generate from URLs
            user_main_text = st.text_area(
                "URLs to Extract",
                height=100,
                help="Enter URLs (one per line) to automatically extract content from news articles. Each URL will be processed to extract its content using Firecrawl or Gemini."
            )
            
            if st.button("Generate Article from URLs", type="primary"):
                if not user_main_text.strip():
                    with messages_placeholder:
                        st.error("Please enter at least one URL in the URLs to Extract box")
                else:
                    st.session_state.generating = True
                    st.session_state.selected_indices = []  # Clear any previous selections
                    st.session_state.process_urls = True
                    st.session_state.urls_to_process = [url.strip() for url in user_main_text.splitlines() if url.strip()]
                    
    # Main area for displaying status and content
    
    # Show search results first if available
    if st.session_state.search_results:
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
                    published_date = result['published_date'] or "Unknown time"
                        
                    def format_local_date(iso_date):
                        if not iso_date:
                            return "Unknown time"
                        try:
                            utc_time = datetime.fromisoformat(iso_date.replace('Z', '+00:00'))
                            local_tz = pytz.timezone('Asia/Bangkok')  # Default to Bangkok time
                            local_time = utc_time.astimezone(local_tz)
                            return local_time.strftime('%Y-%m-%d %H:%M:%S')
                        except:
                            return "Unknown time"
                        
                    formatted_date = format_local_date(published_date)
                    st.markdown(f'<div class="search-result"><a href="{url}" target="_blank">{title}</a><br><span class="search-result-source">Source: {source} | Published: {formatted_date}</span></div>', unsafe_allow_html=True)
                        
                    preview = result['text'][:300] + "..." if len(result['text']) > 300 else result['text']
                    st.markdown(preview)
                    with st.expander("Show full content"):
                        st.write(result['text'])
        
        if st.session_state.selected_indices:
            if st.button("Generate Article from all sources", type="primary"):
                st.session_state.generating = True
                
                # Prepare content from selected search results
                prepared_content = []
                for idx in st.session_state.selected_indices:
                    result = results['results'][idx]
                    prepared_content.append({
                        "url": result['url'],
                        "source": result['source'],
                        "content": result['text']
                    })
                
                # Add additional content if provided
                if content_source == "Generate from URLs":
                    additional_content = []
                    for line in user_main_text.strip().split('\n'):
                        line = line.strip()
                        if line.startswith('http://') or line.startswith('https://'):
                            st.session_state.status_message = f"Extracting content from {line}..."
                            extracted = extract_url_content(gemini_client, line, messages_placeholder)
                            if extracted:
                                additional_content.append(extracted)
                        elif line:  # If it's not a URL but has content
                            additional_content.append({
                                'title': 'User Provided Content',
                                'url': '',
                                'text': line,
                                'published_date': None,
                                'source': 'User Input',
                                'author': None
                            })
                    
                    for item in additional_content:
                        prepared_content.append({
                            "url": item.get('url', ''),
                            "source": item.get('source', 'User Input'),
                            "content": item.get('text', '')
                        })
                
                if prepared_content:
                    try:
                        with messages_placeholder:
                            st.info("Generating article...")
                        # Convert keywords string to list if it exists
                        keywords = st.session_state.keywords.split('\n') if st.session_state.keywords else []
                        
                        # Always include promotional text if provided
                        promo_text = promotional_text.strip() if promotional_text else None
                        
                        article = generate_article(
                            client=gemini_client,
                            transcripts=prepared_content,
                            keywords=keywords,
                            news_angle=news_angle if news_angle.strip() else None,
                            section_count=section_count,
                            promotional_text=promo_text
                        )
                        if article:
                            st.session_state.article = article
                            with messages_placeholder:
                                st.success("Article generated successfully!")
                        else:
                            with messages_placeholder:
                                st.error("Failed to generate article. Please try again.")
                    except Exception as e:
                        with messages_placeholder:
                            st.error(f"Error generating article: {str(e)}")
                else:
                    with messages_placeholder:
                        st.error("No content available to generate article from")
                    st.session_state.generating = False

    # Process URLs section
    if st.session_state.process_urls and st.session_state.urls_to_process:
        # Process URLs
        additional_content = []
        for line in st.session_state.urls_to_process:
            try:
                extracted = extract_url_content(gemini_client, line, messages_placeholder)
                if extracted:
                    additional_content.append(extracted)
                    with messages_placeholder:
                        st.success(f"Successfully extracted content from {line}")
                else:
                    with messages_placeholder:
                        st.error(f"Both Firecrawl and Gemini failed to extract content from {line}")
            except Exception as e:
                with messages_placeholder:
                    st.error(f"Error extracting content from {line}: {str(e)}")

        if additional_content:
            prepared_content = []
            for item in additional_content:
                prepared_content.append({
                    "url": item.get('url', ''),
                    "source": item.get('source', 'User Input'),
                    "content": item.get('text', '')
                })
                
            try:
                with messages_placeholder:
                    st.info("Generating article...")
                # Get keywords as a list directly from the text area
                keywords = [k.strip() for k in st.session_state.keywords.split('\n') if k.strip()]
                # Ensure promotional text is properly handled
                promo_text = promotional_text.strip() if promotional_text else None
                article = generate_article(
                    client=gemini_client,
                    transcripts=prepared_content,
                    keywords=keywords,
                    news_angle=news_angle if news_angle.strip() else None,
                    section_count=section_count,
                    promotional_text=promo_text
                )
                if article:
                    st.session_state.article = article
                    with messages_placeholder:
                        st.success("Article generated successfully!")
                else:
                    with messages_placeholder:
                        st.error("Failed to generate article. Please try again.")
            except Exception as e:
                with messages_placeholder:
                    st.error(f"Error generating article: {str(e)}")
        else:
            with messages_placeholder:
                st.error("No content could be extracted from the provided URLs")
        
        # Reset URL processing flags
        st.session_state.process_urls = False
        st.session_state.urls_to_process = None
        st.session_state.generating = False

    # Show generated article at the bottom
    if st.session_state.article:
        st.markdown("---")  # Add a visual separator
        st.subheader("Generated Article")
        cleaned_article = st.session_state.article.replace("**Title:**", "## Title")
        cleaned_article = cleaned_article.replace("**Main Content:**", "## Main Content")
        cleaned_article = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned_article)
        st.markdown(cleaned_article)
            
        st.download_button(
            label="Download Article",
            data=st.session_state.article,
            file_name="generated_article.md",
            mime="text/markdown",
            use_container_width=True
        )

        # Handle image generation
        image_prompt = extract_image_prompt(st.session_state.article)
        alt_text = extract_alt_text(st.session_state.article)
            
        if image_prompt:
            st.session_state.status_message = "Generating image from Together AI..."
            together_client = Together()
            try:
                response = together_client.images.generate(
                    prompt=image_prompt,
                    model="black-forest-labs/FLUX.1-schnell-free",
                    width=1200,
                    height=800,
                    steps=4,
                    n=1,
                    response_format="b64_json"
                )
                    
                if response and response.data and len(response.data) > 0:
                    b64_data = response.data[0].b64_json
                    st.image(
                        "data:image/png;base64," + b64_data,
                        caption=alt_text or "Generated image"
                    )
                else:
                    st.session_state.error_message = "Failed to generate image from Together AI."
            except Exception as e:
                st.session_state.error_message = f"Error generating image: {str(e)}"

def run_app():
    main()

if __name__ == "__main__":
    run_app()
