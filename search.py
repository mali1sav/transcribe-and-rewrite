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
        model = genai.GenerativeModel('gemini-2.0-flash')
        return {'model': model, 'name': 'gemini-2.0-flash'}
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
        for attempt in range(3):
            try:
                response = client['model'].generate_content(prompt)
                if response and response.text:
                    return response.text
            except Exception as e:
                if attempt == 2:
                    raise
                st.warning(f"Retrying Gemini request (attempt {attempt + 2}/3)...")
                time.sleep(2 ** attempt)
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
            url = str(getattr(result, 'url', '') or '')
            domain = get_domain(url)
            if any(excluded in domain for excluded in excluded_domains):
                continue
            title = str(getattr(result, 'title', 'No Title') or 'No Title')
            text = str(getattr(result, 'text', '') or '')
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
        exa_results = perform_exa_search(exa_client, query, hours_back=hours_back)
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

def deduplicate_prompt(prompt):
    """
    Remove duplicate non-empty lines from the prompt while preserving the original order.
    """
    seen = set()
    deduped_lines = []
    for line in prompt.splitlines():
        stripped = line.strip()
        if stripped and stripped not in seen:
            seen.add(stripped)
            deduped_lines.append(line)
        elif not stripped:
            deduped_lines.append(line)
    return "\n".join(deduped_lines)

def generate_article(client, transcripts, keywords=None, news_angle=None, section_count=3, promotional_text=None):
    try:
        if not transcripts:
            return None

        # Process keywords without altering their original case for display
        keyword_list = [k.strip() for k in keywords if k.strip()] if keywords else []
        # Use the keyword exactly as provided for display (e.g., "Dogecoin")
        primary_keyword_display = keyword_list[0] if keyword_list else ""
        secondary_keywords = ", ".join(keyword_list[1:]) if len(keyword_list) > 1 else ""

        # For shortcode mapping we still need uppercase keys, but we don't output them in the article
        primary_keyword_for_shortcode = primary_keyword_display.upper()

        shortcode_map = {
            "BITCOIN": '[latest_articles label="ข่าว Bitcoin (BTC) ล่าสุด" count_of_posts="6" taxonomy="category" term_id="7"]',
            "ETHEREUM": '[latest_articles label="ข่าว Ethereum ล่าสุด" count_of_posts="6" taxonomy="category" term_id="8"]',
            "SOLANA": '[latest_articles label="ข่าว Solana ล่าสุด" count_of_posts="6" taxonomy="category" term_id="501"]',
            "XRP": '[latest_articles label="ข่าว XRP ล่าสุด" count_of_posts="6" taxonomy="category" term_id="502"]',
            "DOGECOIN": '[latest_articles label="ข่าว Dogecoin ล่าสุด" count_of_posts="6" taxonomy="category" term_id="527"]'
        }
        
        # Build the prompt header with instructions.
        # Note: We add an extra instruction to use the primary keyword exactly as provided.
        prompt = f"""
Write a comprehensive and in-depth news article in Thai (Title, Main Content, บทสรุป, Excerpt for WordPress, Title & H1 Options, and Meta Description Options all in Thai).
When creating section headings (H2) and subheadings (H3), follow these guidelines:
1. Use power words that often appear in Thai crypto news headlines.
2. Include specific numbers/stats when relevant.
3. Create curiosity gaps.
4. Make bold, specific statements.
The above guidelines are examples. You must choose your own words or modify them as needed.

Primary Keyword: {primary_keyword_display or ""}
Secondary Keywords: {keywords or ""}
News Angle: {news_angle or ""}

**Important:** Use the primary keyword exactly as provided (e.g., "Dogecoin") without converting it to uppercase in the article text.

# Content Focus Instructions:
* Write the article from the perspective of the specified News Angle.
* Include only information that supports this news angle.
* Analyze each source and use only the content that provides context for the news angle.
* Exclude information not relevant to the news angle.
* Ensure each section develops the news angle thoroughly.
* Avoid tangents or unrelated information.

# Section Instructions:
Create exactly {section_count} heading level 2 sections in Thai for the main content.
For each section, do not merely summarize—provide a detailed discussion that thoroughly explores the topic. The discussion should be as detailed and lengthy as the original source text, matching its depth and nuance. Use as many tokens as needed.
Do not number each heading.

# Currency Formatting Instruction:
* **IMPORTANT:** Write all monetary values in USD without the '$' sign. After each numeric value (which may include commas or multipliers like K/M), append a space followed by "ดอลลาร์". For example, "$96,000K" should be written as "96,000K ดอลลาร์".

# Source Citation Rules:
* CRITICAL: Include concise attributions to each source only ONCE in the entire article. Embed each citation naturally in the introduction sentence a contextual manner using markdown hyperlinks [source name](url). Ensure that each source is cited only once—do not repeat the citation in multiple sections.

# Main Content Guidelines:
* [Additional instructions...]

# Promotional Content Guidelines
{f'''
**Promotional Integration Guidelines**
----------------------------------------
{promotional_text}
* Requirements:
  1. Create a seamless Heading Level 2 aligned with the {news_angle} in Thai.
  2. The content must be in Thai.
  3. Transit smoothly from the main content into the promotional text.
  4. Mention {primary_keyword_display} seamlessly.
  5. Limit promotional text to 120 words, placed at the end of the article.
''' if promotional_text else ''}

# Article Structure:
[Create an engaging news-style title that includes {primary_keyword_display} once, exactly as provided]
[The first paragraph must include {primary_keyword_display} exactly as provided]
[Create H2 sections below, with at least 2 sections containing {primary_keyword_display} exactly as provided, aligned with the news angle]
[Write supporting paragraphs using {primary_keyword_display} and {secondary_keywords} where they fit naturally]
[Conclude with a summary emphasizing the news angle's significance, including {primary_keyword_display} if it fits naturally]

## SEO Elements
[SEO Instructions...]

## Additional Elements
Slug URL in English (must include {primary_keyword_display}; translate Thai keywords to English)
- Image ALT Text in Thai including {primary_keyword_display}
- Excerpt for WordPress: One sentence in Thai that briefly describes the article

## Image Prompt
[Create a photorealistic scene that fits the main news article in English. Keep it simple and focus on 1 or 2 objects.]

# Source Usage Instructions:
* Use provided sources as the primary reference for facts and data.
* Supplement with your knowledge only if it does not override source information.
* When source information exists on a topic, it takes precedence over general knowledge.

Here are the sources to base the article on:
"""
        # --- Consolidate sources to avoid repeated citations ---
        seen_sources = {}
        for t in transcripts:
            source = t['source']
            if source not in seen_sources:
                seen_sources[source] = t
        for source, t in seen_sources.items():
            # Ensure the source is formatted with the first letter in uppercase.
            formatted_source = source[0].upper() + source[1:] if source else source
            prompt += f"### Content from {formatted_source}\nSource URL: {t['url']}\n{escape_special_chars(t['content'])}\n\n"
        # --- End of citation consolidation ---

        # Deduplicate repeated lines in the prompt to save tokens.
        prompt = deduplicate_prompt(prompt)

        content = make_gemini_request(client, prompt)
        if not content:
            return None

        # Use the uppercase version for matching the shortcode_map key (for shortcode lookup only).
        shortcode = shortcode_map.get(primary_keyword_for_shortcode, "")
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
    """
    Extract the image prompt by looking for a section starting with "## Image Prompt" on its own line.
    If not found, return a default prompt.
    """
    pattern = r"(?is)^##\s*Image Prompt\s*\n(.*?)\n(?=##|\Z)"
    match = re.search(pattern, article_text, re.MULTILINE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return "Create a photorealistic scene that fits the main news article in English. Keep it simple and focus on 1 or 2 objects."

def extract_alt_text(article_text):
    pattern = r"(?i)Image ALT Text.*?\n(.*?)(?:\n\n|\Z)"
    match = re.search(pattern, article_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def main():
    st.set_page_config(page_title="Search and Generate Articles", layout="wide")
    
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

    with st.sidebar:
        content_source = st.radio(
            "How would you like to generate your article?",
            ["Generate from URLs", "Search and Generate"],
            key="content_source",
            index=0
        )
        if content_source == "Generate from URLs":
            st.text_area(
                "URLs to Extract",
                height=100,
                key="user_main_text",
                help="Enter URLs (one per line) to automatically extract content from news articles. Each URL will be processed using Firecrawl or Gemini."
            )
        elif content_source == "Search and Generate":
            st.text_input("Enter your search query:", value=st.session_state.query, key="query")
            st.slider("Hours to look back:", 1, 168, 12, key="hours_back")
        st.text_area(
            "Keywords (one per line)",
            height=68,
            key="keywords",
            help="Enter one keyword per line. The first keyword will be the primary keyword for SEO optimization."
        )
        news_angle = st.text_input(
            "News Angle",
            value="",
            key="news_angle",
            help="""This can be in Thai or English. Having a clear news angle is essential, especially when sources may lack focus which is bad for SEO. A well-defined angle helps create a coherent narrative around your chosen perspective.
            Tips: You can use one of the English headlines from your selected news sources as your news angle."""
        )
        section_count = st.slider("Number of sections:", 3, 8, 4, key="section_count")
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
            results = perform_web_research(exa_client, query, hours_back)
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
        # -------------------------------
        # Image Generation and Optional Editing Workflow
        # -------------------------------
        image_prompt = extract_image_prompt(st.session_state.article)
        alt_text = extract_alt_text(st.session_state.article)
        # If no image prompt was found, use a default prompt
        if not image_prompt:
            image_prompt = "Create a photorealistic scene that fits the main news article."
        st.write("Image Prompt:", image_prompt)  # Debug: Show current prompt
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
                st.image("data:image/png;base64," + b64_data, caption=alt_text or "Generated image")
            else:
                st.session_state.error_message = "Failed to generate image from Together AI."
        except Exception as e:
            st.session_state.error_message = f"Error generating image: {str(e)}"
        
        # Optional image editing workflow: always allow editing
        if st.checkbox("Edit image prompt", key="edit_image_checkbox"):
            edited_image_prompt = st.text_area("Edit Image Prompt", value=image_prompt, key="edited_image_prompt")
            if st.button("Submit Edited Image Prompt", key="submit_edited_image"):
                st.session_state.status_message = "Generating edited image from Together AI..."
                try:
                    edited_response = together_client.images.generate(
                        prompt=edited_image_prompt,
                        model="black-forest-labs/FLUX.1-schnell-free",
                        width=1200,
                        height=800,
                        steps=4,
                        n=1,
                        response_format="b64_json"
                    )
                    if edited_response and edited_response.data and len(edited_response.data) > 0:
                        b64_data = edited_response.data[0].b64_json
                        st.image("data:image/png;base64," + b64_data, caption=alt_text or "Edited generated image")
                    else:
                        st.error("Failed to generate image from Together AI using the edited prompt.")
                except Exception as e:
                    st.error(f"Error generating image with edited prompt: {str(e)}")

def run_app():
    main()

if __name__ == "__main__":
    run_app()
