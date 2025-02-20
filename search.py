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
import markdown

load_dotenv()

def init_exa_client():
    exa_api_key = os.getenv('EXA_API_KEY')
    if not exa_api_key:
        raise ValueError("EXA_API_KEY not found in environment variables")
    return Exa(api_key=exa_api_key)

def get_domain(url):
    try:
        return url.split('/')[2]
    except Exception:
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
        
        # Use only supported parameters for v1 API
        response = app.scrape_url(url=url, params={
            'formats': ['markdown']
        })
        
        # Debug output in expander
        with st.expander("Debug: Firecrawl Response", expanded=False):
            st.write("Raw response:", response)
        
        if response and isinstance(response, dict):
            # Get metadata
            metadata = response.get('metadata', {})
            
            # Get content, trying different possible fields
            content = response.get('markdown', '')
            if not content:
                content = response.get('content', '')
            if not content:
                content = response.get('text', '')
            
            if content:
                # Clean up the content by removing navigation-like patterns
                lines = [line for line in content.split('\n') 
                        if not any(pattern in line.lower() 
                                 for pattern in ['menu', 'navigation', '- [', 'search', 'login', 'sign up'])]
                cleaned_content = '\n'.join(lines).strip()
                
                if cleaned_content:
                    # Get title from metadata
                    title = (metadata.get('ogTitle') or 
                            metadata.get('title') or 
                            metadata.get('parsely-title') or
                            'Extracted Content')
                    
                    # Clean up the title if it contains the site name
                    if ' | ' in title:
                        title = title.split(' | ')[0]
                    elif ' - ' in title:
                        title = title.split(' - ')[0]
                    
                    return {
                        'title': title,
                        'author': (metadata.get('parsely-author') or 
                                 metadata.get('author')),
                        'published_date': (metadata.get('publishedTime') or 
                                         metadata.get('parsely-pub-date')),
                        'text': cleaned_content
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

# ---------------------------------------------------------------------------
# Helper function to sanitize text by removing control characters.
# ---------------------------------------------------------------------------
def sanitize_text(text):
    # Remove control characters (except newline and tab)
    return ''.join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\t ")

# ---------------------------------------------------------------------------
# Use a full day filter for Exa's search.
# ---------------------------------------------------------------------------
def perform_exa_search(exa_client, query, num_results=10, specific_date=None):
    try:
        if specific_date:
            start_date = datetime(specific_date.year, specific_date.month, specific_date.day, 0, 0, 0, tzinfo=timezone.utc)
            end_date = datetime(specific_date.year, specific_date.month, specific_date.day, 23, 59, 59, 999000, tzinfo=timezone.utc)
        else:
            now = datetime.now(timezone.utc)
            start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = now.replace(hour=23, minute=59, second=59, microsecond=999000)
        
        start_date_str = start_date.strftime('%Y-%m-%dT%H:%M:%S.000Z')
        end_date_str = end_date.strftime('%Y-%m-%dT%H:%M:%S.999Z')
        
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
            url = str(getattr(result, 'url', '') or '')
            domain = get_domain(url)
            if any(excluded in domain for excluded in excluded_domains):
                continue
                
            title = str(getattr(result, 'title', 'No Title') or 'No Title')
            text = str(getattr(result, 'text', '') or '')
            text = sanitize_text(text)
            
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
        exa_results = perform_exa_search(
            exa_client=exa_client,
            query=query,
            num_results=10
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
    """Takes a list of results and returns a list for GPT article generation."""
    try:
        prepared_content = []
        for result in selected_results:
            url = result.get('url', '').strip()
            source = result.get('source', '').strip()
            content = result.get('content', result.get('text', '')).strip()
            
            content_item = {
                'url': url,
                'source': source,
                'content': content
            }
            prepared_content.append(content_item)
        return prepared_content
    except Exception as e:
        st.error(f"Error preparing content: {str(e)}")
        return []

def escape_special_chars(text):
    """Escape special characters that might interfere with Markdown formatting."""
    text = re.sub(r'(?<!\$)\$(?!\$)', r'\$', text)
    chars_to_escape = ['*', '_', '`', '#', '~', '|', '<', '>', '[', ']']
    for char in chars_to_escape:
        text = text.replace(char, '\\' + char)
    return text

def generate_article(client, transcripts, keywords=None, news_angle=None, section_count=3, promotional_text=None):
    try:
        if not transcripts:
            return None

        keyword_list = keywords if keywords else []
        primary_keyword = keyword_list[0] if keyword_list else ""
        secondary_keywords = ", ".join(keyword_list[1:]) if len(keyword_list) > 1 else ""
        
        shortcode_map = {
            "BITCOIN": '[latest_articles label="ข่าว Bitcoin (BTC) ล่าสุด" count_of_posts="6" taxonomy="category" term_id="7"]',
            "ETHEREUM": '[latest_articles label="ข่าว Ethereum ล่าสุด" count_of_posts="6" taxonomy="category" term_id="8"]',
            "SOLANA": '[latest_articles label="ข่าว Solana ล่าสุด" count_of_posts="6" taxonomy="category" term_id="501"]',
            "XRP": '[latest_articles label="ข่าว XRP ล่าสุด" count_of_posts="6" taxonomy="category" term_id="502"]',
            "DOGECOIN": '[latest_articles label="ข่าว Dogecoin ล่าสุด" count_of_posts="6" taxonomy="category" term_id="527"]'
        }
        
        # --- Revised prompt with explicit, concise citation rules ---
        prompt = f"""
Write a comprehensive and in-depth news article in Thai (including Title, Main Content, บทสรุป, Excerpt for WordPress, Title & H1 Options, and Meta Description Options all in Thai).
When creating section headings (H2) and subheadings (H3), use engaging language, power words, and when needed, specific numbers/stats that show analytical value.
Keep technical terms and entity names in English but the rest should be in Thai.

Primary Keyword: {primary_keyword}
Secondary Keywords: {keywords or ""}
News Angle: {news_angle or ""}

**IMPORTANT CITATION RULES:**
- In the **first paragraph only** of the Main Content, include a single, natural citation for each source using this format: [Source Domain Name](url).
- Group all facts from a source under that single citation.
- Do **not** include any source citations in any other part of the article.
- If you refer to the same source later, simply state the facts without re-citing.

# Content Focus Instructions:
* The article MUST be written from the perspective of the specified News Angle.
* Only include information from sources that is relevant to and supports this news angle.
* Analyze each source and extract only the content that aligns with or provides context for the news angle.
* Ensure each section contributes to developing the specified news angle.
* Maintain focus throughout the article.

SEO Guidelines:
* Title: Include {primary_keyword} exactly once. Must be under 60 characters. Must be engaging, and click-worthy, news-like headline.
* Meta Description: Include {primary_keyword} exactly once. Must be under 160 characters. Must be engaging, and click-worthy, encourage curiosity and interest.
* H1: Include {primary_keyword} exactly once. Must align with the title. Must be engaging, and click-worthy, news-like headline.

# Main Content Guidelines:
* Include exactly {section_count} H2 sections in Thai (with technical terms and entity names in English).
* Each section should have 2-4 detailed paragraphs.
* Remove any $ signs from monetary values and replace them with " ดอลลาร์" (with proper spacing).

# Promotional Content Guidelines
{f'''
**Promotional Integration Guidelines**
----------------------------------------
{promotional_text}

* Create a seamless H2 section aligned with the {news_angle} in Thai.
* Integrate the promotional text (max 120 words) at the end of the article.
* Mention {primary_keyword} naturally.
''' if promotional_text else ''}

# Article Structure:
## Title
[Create an engaging title that includes {primary_keyword} exactly once]

## Main Content
[The first paragraph must include one natural citation per source as specified above]

[Include H2 sections and supporting paragraphs with natural use of {primary_keyword} and {secondary_keywords}]

## SEO Elements
1. Title Options:
   - [Option 1]
   - [Option 2]
   - [Option 3]

2. Meta Description Options:
   - [Option 1]
   - [Option 2]
   - [Option 3]

3. H1 Options:
   - [Option 1]
   - [Option 2]
   - [Option 3]

## Additional Elements
Slug URL in English (must include {primary_keyword})
- Image ALT Text in Thai including {primary_keyword}
- Excerpt for WordPress: One sentence in Thai that briefly describes the article

## Image Prompt
[Create a photorealistic scene based on the article content. Avoid promotional elements.]

# Source Usage Instructions:
* Use the provided sources as the basis for all facts and data.
* Do not repeat a source citation beyond the first paragraph.
"""
        # Append source content. Note: Only include the citation header once per source.
        seen_sources = set()
        for t in transcripts:
            source = t['source']
            if source not in seen_sources:
                seen_sources.add(source)
                prompt += f"### Content from {source}\nSource URL: {t['url']}\n{escape_special_chars(t['content'])}\n\n"
            else:
                prompt += f"{escape_special_chars(t['content'])}\n\n"

        content = make_gemini_request(client, prompt)
        if not content:
            return None
            
        shortcode = shortcode_map.get(primary_keyword.upper(), "")
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

    messages_placeholder = st.empty()

    # -------------------------------
    # SIDEBAR: Re-ordered UI Elements
    # -------------------------------
    with st.sidebar:
        content_source = st.radio(
            "How would you like to generate your article?",
            ["Generate from URLs", "Search and Generate"],
            key="content_source"
        )
        
        if content_source == "Generate from URLs":
            st.text_area(
                "URLs to Extract",
                height=100,
                key="user_main_text",
                help="Enter URLs (one per line) to extract content from news articles."
            )
        elif content_source == "Search and Generate":
            st.text_input("Enter your search query:", value=st.session_state.query, key="query")
            st.slider("Hours to look back:", 1, 744, 12, key="hours_back")
        
        st.text_area(
            "Keywords (one per line)",
            height=68,
            key="keywords",
            help="Enter one keyword per line. The first keyword is the primary keyword for SEO."
        )
        news_angle = st.text_input(
            "News Angle",
            value="",
            key="news_angle",
            help="A clear news angle is essential to create a coherent narrative."
        )
        section_count = st.slider("Number of sections:", 2, 8, 3, key="section_count")
        
        promotional_content = load_promotional_content()
        st.write("Promotional Content")
        selected_promotions = []
        for name in sorted(promotional_content.keys()):
            if st.checkbox(name, key=f"promo_{name}"):
                selected_promotions.append(promotional_content[name])
        promotional_text = "\n\n".join(selected_promotions) if selected_promotions else None

        if content_source == "Generate from URLs":
            action_clicked = st.button("Generate Article from URLs", type="primary")
        elif content_source == "Search and Generate":
            action_clicked = st.button("Search Content", type="primary")

    # --------------------------------
    # MAIN AREA: Process Action Buttons
    # --------------------------------
    if action_clicked:
        if content_source == "Generate from URLs":
            user_main_text = st.session_state.get("user_main_text", "")
            if not user_main_text.strip():
                with messages_placeholder:
                    st.error("Please enter at least one URL in the URLs to Extract box")
            else:
                st.session_state.generating = True
                st.session_state.selected_indices = []
                st.session_state.process_urls = True
                st.session_state.urls_to_process = [url.strip() for url in user_main_text.splitlines() if url.strip()]
        elif content_source == "Search and Generate":
            query = st.session_state.get("query", st.session_state.query)
            hours_back = st.session_state.get("hours_back", 12)
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

    # --------------------------------
    # MAIN AREA: Display Search Results and Process URLs
    # --------------------------------
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
                            local_tz = pytz.timezone('Asia/Bangkok')
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
                prepared_content = []
                for idx in st.session_state.selected_indices:
                    result = results['results'][idx]
                    prepared_content.append({
                        "url": result['url'],
                        "source": result['source'],
                        "content": result['text']
                    })
                
                if content_source == "Generate from URLs":
                    additional_content = []
                    for line in st.session_state.get("user_main_text", "").strip().split('\n'):
                        line = line.strip()
                        if line.startswith('http://') or line.startswith('https://'):
                            st.session_state.status_message = f"Extracting content from {line}..."
                            extracted = extract_url_content(gemini_client, line, messages_placeholder)
                            if extracted:
                                additional_content.append(extracted)
                        elif line:
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
                        keywords = st.session_state.keywords.split('\n') if st.session_state.keywords else []
                        promo_text = promotional_text.strip() if promotional_text else None
                        
                        article = generate_article(
                            client=gemini_client,
                            transcripts=prepared_content,
                            keywords=keywords,
                            news_angle=st.session_state.news_angle if st.session_state.news_angle.strip() else None,
                            section_count=st.session_state.section_count,
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

    if st.session_state.process_urls and st.session_state.urls_to_process:
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
                keywords = [k.strip() for k in st.session_state.keywords.split('\n') if k.strip()]
                promo_text = promotional_text.strip() if promotional_text else None
                article = generate_article(
                    client=gemini_client,
                    transcripts=prepared_content,
                    keywords=keywords,
                    news_angle=st.session_state.news_angle if st.session_state.news_angle.strip() else None,
                    section_count=st.session_state.section_count,
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
        
        st.session_state.process_urls = False
        st.session_state.urls_to_process = None
        st.session_state.generating = False

    if st.session_state.article:
        st.markdown("---")
        st.subheader("Generated Article (Rendered Markdown)")
        
        # Show the rendered Markdown as usual
        cleaned_article = st.session_state.article.replace("**Title:**", "## Title")
        cleaned_article = cleaned_article.replace("**Main Content:**", "## Main Content")
        cleaned_article = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned_article)
        
        # This displays it as nicely formatted Markdown in Streamlit
        st.markdown(cleaned_article)
        
        # Optionally, provide a download button for the raw Markdown
        st.download_button(
            label="Download Article (Markdown)",
            data=st.session_state.article,
            file_name="generated_article.md",
            mime="text/markdown",
            use_container_width=True
        )

        # --- New Part: Convert Markdown to HTML and show in a text area ---
        html_output = markdown.markdown(st.session_state.article)

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
