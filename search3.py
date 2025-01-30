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
    stop=tenacity.stop_after_attempt(5),  # Increase max retries to 5
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=20),  # Longer max wait time
    retry=tenacity.retry_if_exception_type((Exception)),  # Retry on any exception
    retry_error_callback=lambda retry_state: None
)
def make_gemini_request(client, prompt):
    """Make Gemini API request with retries and proper error handling"""
    try:
        # Split long prompts if needed (Gemini has lower context window)
        if len(prompt) > 30000:  # Approximate limit
            st.warning("⚠️ Input text is too long, truncating to fit Gemini's context window")
            prompt = prompt[:30000]
            
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

def generate_article(client: dict, transcripts, keywords=None, news_angle=None, section_count=3, promotional_text=None):
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
        
        prompt = f"""
Write a comprehensive and in-depth news article in Thai (Title, Main Content, บทสรุป, Excerpt for WordPress, Title & H1 Options, and Meta Description Options all in Thai).
When creating section headings (H2) and subheadings (H3), use one of the relevant guidelines below:
1. Use power words that evoke emotion in Thai (examples: "ทะลุเป้า!" "ทุปสถิติใหม่!", "เปิดโผ!"), ensure correct grammar and punctuation according to Thai sentnence structure, particularly news-like headlines
2. Include specific numbers/stats when relevant (examples: "10 เท่า!", "5 เหรียญ Meme ที่อาจพุ่ง 1000%")
3. Create curiosity gaps (examples:"เบื้องหลังการพุ่งทะยานของราคา...", "จับตา! สัญญาณที่บ่งชี้ว่า...")
4. Make bold, specific statements (examples:"เหรียญคริปโตที่ดีที่สุด", "ทำไมวาฬถึงทุ่มเงินหมื่นล้านใส่...")

Primary Keyword: {primary_keyword or ""}
Secondary Keywords: {keywords or ""}

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
* Citation format:
  - When referencing a source, use this exact format in Thai sentence structure:
    - End the sentence with "(อ้างอิง: [Source Name](url))"
    - Example: "ราคา Bitcoin พุ่งขึ้นแตะ 50,000 ดอลลาร์ (อ้างอิง: [Bloomberg](https://www.bloomberg.com))"
    - Always capitalize source names properly
    - Place the citation at the end of the relevant statement
    - Keep the entire citation on the same line as the sentence
* Ensure that any special characters and avoid unintended Markdown or LaTeX formatting.
* Open with the most newsworthy aspect.
* Create exactly {section_count} heading level 2 in Thai for the main content (keep technical terms and entity names in English).
* For each section, ensure a thorough and detailed exploration of the topic, with each section comprising at least 2-4 paragraphs of comprehensive analysis. Strive to simplify complex ideas, making them accessible and easy to grasp. Where applicable, incorporate relevant data, examples, or case studies to substantiate your analysis and provide clarity.
* Use heading level 2 for each section heading. Use sub-headings if necessary. For each sub-heading, provide real examples, references, or numeric details from the sources, with more extensive context.
* If the content contains numbers that represent monetary values, remove $ signs before numbers and add "ดอลลาร์" after the number, ensuring a single space before and after the numeric value.
* When referencing a source, naturally integrate the Brand Name into the sentence as a clickable markdown hyperlink to the source webpage like this: [brand name](url).
* Ensure that any special characters and avoid unintended Markdown or LaTeX formatting.

# Article Structure:
## Title
[Your engaging title here]

## Main Content
[Your main content with H2 sections]

## บทสรุป
[Summary of key points]

## Additional Elements
- Slug URL in English (must include {primary_keyword})
- Image ALT Text in Thai including {primary_keyword} (keep technical terms and entity names in English, rest in Thai)
- Excerpt for WordPress: One sentence in Thai that briefly describes the article

## SEO Elements (3 options each)
1. Title (SEO best practices)
2. Meta Description (SEO best practices)
3. H1 (aligned with Title and Meta Description)
- Integrate {primary_keyword} in its original form naturally in all elements
- All elements must be engaging and news-style

## Image Prompt
[English description of a scene that fits the article, focusing on 1-2 objects]

Here are the sources to base the article on:
"""
        # Append transcripts (which may include search results + user-provided main text)
        for t in transcripts:
            prompt += f"### Content from {t['source']}\nSource URL: {t['url']}\n{t['content']}\n\n"

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
    
    st.markdown("""
    <style>
    .block-container {
        padding: 2rem;
        margin: 2rem;
    }
    .element-container{
        font-weight: bold;
    }
    .stTitle, h1 {
        font-size: 1.5rem !important;
        margin-bottom: 0.5rem !important;
        color: #333 !important;
    }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.5rem;
        padding-top: 0.5rem;
    }
    div[data-testid="stSidebarUserContent"] > div:nth-child(1) {
        padding-top: 0.5rem;
    }
    .st-emotion-cache-16txtl3 {
        padding-top: 0.5rem;
    }
    /* Make checkboxes larger and easier to click */
    [data-testid="stCheckbox"] {
        scale: 1.5;
        padding: 0;
        margin: 0 10px;
    }
    /* Style all buttons to be blue by default */
    .stButton button, .secondary {
        background-color: #0066FF !important;
        color: white !important;
        width:70%;
        font-size: 1.8em;
        line-height: 1.8em;
        padding: 5px 30px 5px 30px;
        width: 40%;
    }
    .stButton button:hover {
        background-color: #0052CC !important;
        color: white !important;
    }
    [data-testid="stTextArea"][aria-label="Keywords (one per line)"] textarea {
        min-height: 45px !important;
        height: 45px !important;
    }
    /* Style search results */
    .search-result {
        font-size: 1rem !important;
        color: #333;
        margin: 0.25rem 0;
    }
    .search-result-source {
        color: green;
        font-size: 0.8rem;
        font-weight: normal;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("Search and Generate Articles")

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

    try:
        exa_client = init_exa_client()
        gemini_client = init_gemini_client()
        if not gemini_client:
            st.error("Failed to initialize Gemini client")
            return
    except Exception as e:
        st.error(f"❌ Failed to initialize clients: {str(e)}")
        return

    with st.sidebar:
        query = st.text_input("Enter your search query:", value=st.session_state.query)
        
        hours_back = st.slider("Hours to look back:", 1, 168, 12)
        
        st.text_area(
            "Keywords (one per line)",
            height=68,
            key="keywords",
            help="Enter one keyword per line. The first keyword will be the primary keyword for SEO optimization."
        )
        
        # Larger text area for user-provided main content:
        user_main_text = st.text_area(
            "Additional Main News Text (Optional)",
            height=150,
            help="Paste any additional text you want included in the main portion of the article. This is not a promotional block."
        )
        
        promotional_text = st.text_area(
            "Promotional Text (Optional)",
            height=100,
            help="Paste any promotional content or CTA you want appended (about 10% weighting)."
        )

        search_button = st.button("Search")

    if search_button and query:
        with st.spinner("Searching..."):
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
                st.warning("No results found. Try adjusting your search parameters.")
    
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
        
        if st.session_state.selected_indices or user_main_text.strip():
            news_angle = st.text_input(
                "News Angle",
                value="",
                help="""This can be in Thai or English. Having a clear news angle is essential, especially when sources may lack focus which is bad for SEO. A well-defined angle helps create a coherent narrative around your chosen perspective.
                Tips: You can use one of the English headlines from your selected news sources as your news angle."""
            )
            
            cols = st.columns([0.4, 0.6])
            with cols[0]:
                section_count = st.slider("Number of sections:", 2, 6, 3, key="section_count")
            with cols[1]:
                st.markdown(
                    """
                    <style>
                    div.stButton > button {
                        background-color: #0066FF;
                        color: white;
                        font-size: 1.8em;
                        line-height: 1.8em;
                        padding: 5px 30px 5px 30px;
                        width: 40%;
                    }
                    div.stButton > button:hover {
                        background-color: #0052CC;
                        color: white;
                    }
                    </style>
                    """, 
                    unsafe_allow_html=True
                )
                generate_btn = st.button("Generate Article")

            if generate_btn:
                st.session_state.generating = True
                keywords = st.session_state.keywords.strip().split('\n')
                keywords = [k.strip() for k in keywords if k.strip()]
                
                selected_results = [results['results'][idx] for idx in st.session_state.selected_indices]
                prepared_content = prepare_content_for_article(selected_results)
                
                # If the user provides "Additional Main News Text", append it as a source
                if user_main_text.strip():
                    prepared_content.append({
                        "url": "UserProvided",
                        "source": "User Main Text",
                        "content": user_main_text.strip()
                    })
                
                if prepared_content:
                    with st.spinner("Generating article..."):
                        article = generate_article(
                            client=gemini_client,
                            transcripts=prepared_content,
                            keywords='\n'.join(keywords) if keywords else None,
                            news_angle=news_angle,
                            section_count=section_count,
                            promotional_text=promotional_text
                        )
                        if article:
                            st.session_state.article = article
                            st.success("Article generated successfully!")
                            
                            st.subheader("Generated Article")
                            cleaned_article = article.replace("**Title:**", "## Title")
                            cleaned_article = cleaned_article.replace("**Main Content:**", "## Main Content")
                            cleaned_article = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned_article)  # Remove other ** markers
                            st.markdown(cleaned_article)
                            st.download_button(
                                label="Download Article",
                                data=article,
                                file_name="generated_article.md",
                                mime="text/markdown",
                                use_container_width=True
                            )

                            image_prompt = extract_image_prompt(st.session_state.article)
                            alt_text = extract_alt_text(st.session_state.article)
                            
                            # Only proceed with image generation if we have an English prompt
                            if image_prompt:
                                with st.spinner("Generating image from Together AI..."):
                                    together_client = Together()
                                    try:
                                        response = together_client.images.generate(
                                            prompt=image_prompt,  # Only use English prompt for generation
                                            model="black-forest-labs/FLUX.1-schnell-Free",
                                            width=1200,
                                            height=800,
                                            steps=4,
                                            n=1,
                                            response_format="b64_json"
                                        )
                                        if response and response.data and len(response.data) > 0:
                                            b64_data = response.data[0].b64_json
                                            # Display image with Thai ALT text as caption
                                            st.image(
                                                "data:image/png;base64," + b64_data,
                                                caption=alt_text or "Generated image"  # Use Thai ALT text or fallback to simple English caption
                                            )
                                        else:
                                            st.error("Failed to generate image from Together AI.")
                                    except Exception as e:
                                        st.error(f"Error generating image: {str(e)}")
                        else:
                            st.error("Failed to generate article. Please try again.")

def run_app():
    main()

if __name__ == "__main__":
    run_app()
