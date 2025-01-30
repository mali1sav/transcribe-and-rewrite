import os
import pytz
from tavily import TavilyClient
from datetime import datetime, timedelta, timezone
import streamlit as st
from urllib.parse import urlparse
from exa_py import Exa
from dateutil import parser
from dotenv import load_dotenv
from openai import OpenAI
import re
import json
import base64
from together import Together

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

def serialize_search_results(search_results):
    if not search_results:
        return {"results": []}
        
    def format_date(date):
        if isinstance(date, datetime):
            return date.isoformat()
        return str(date) if date else ""
        
    serialized_results = {
        "results": [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "text": r.get("text", r.get("content", "")),  # Support both text and content fields
                "published_date": format_date(r.get("published_date")),
                "source": r.get("source", "")
            }
            for r in search_results
        ]
    }
    return serialized_results

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
                
        raise ValueError("Failed to get valid response from Gemini API after retries")
            
    except Exception as e:
        st.error(f"Gemini API error: {str(e)}")
        st.error("Retrying request...")
        raise

def perform_exa_search(exa_client, query, num_results=12, hours_back=24):
    """Simple Exa search function that returns news articles from the past hours_back hours"""
    try:
        # Calculate time range
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(hours=hours_back)
        
        # Make the API request
        response = exa_client.search_and_contents(
            query=query,
            num_results=num_results,
            start_published_date=start_date.strftime('%Y-%m-%dT%H:%M:%SZ'),
            end_published_date=end_date.strftime('%Y-%m-%dT%H:%M:%SZ'),
            type='auto',
            use_autoprompt=True,
            text=True
        )
        
        # Process results
        all_results = []
        if response and hasattr(response, 'results'):
            for result in response.results:
                # Skip if no text content
                if not hasattr(result, 'text') or not result.text:
                    continue
                    
                # Get basic fields
                url = getattr(result, 'url', '')
                source = get_domain(url)
                
                # Skip Twitter results
                if 'twitter.com' in source.lower():
                    continue
                    
                # Parse date
                published_date = None
                if hasattr(result, 'published_date') and result.published_date:
                    try:
                        parsed = parser.parse(result.published_date)
                        published_date = parsed.replace(tzinfo=timezone.utc).isoformat() if not parsed.tzinfo else parsed.isoformat()
                    except:
                        pass
                
                # Format result
                all_results.append({
                    'title': getattr(result, 'title', 'No Title'),
                    'url': url,
                    'text': str(result.text),
                    'source': source,
                    'published_date': published_date
                })
        
        # Sort by date and limit results
        all_results.sort(
            key=lambda x: parser.parse(x['published_date']) if x['published_date'] else datetime.min.replace(tzinfo=timezone.utc),
            reverse=True
        )
        all_results = all_results[:num_results]
        
        if all_results:
            st.success(f"✅ Found {len(all_results)} results from Exa")
        
        return all_results
        
    except Exception as e:
        st.error(f"Error during Exa search: {str(e)}")
        return []

def perform_web_research(exa_client, query, hours_back, search_engines):
    results = []
    search_status = []
    
    try:
        if "Exa" in search_engines:
            exa_results = perform_exa_search(exa_client, query, num_results=10, hours_back=hours_back)
            if exa_results:
                results.extend(exa_results)
                search_status.append(f"✅ Found {len(exa_results)} results from Exa")
            else:
                search_status.append("❌ No results from Exa")
    except Exception as e:
        st.error(f"❌ Exa search failed: {str(e)}")
    
    # Display consolidated status
    for status in search_status:
        st.write(status)
    
    return results

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

def clean_markdown(text):
    """Clean up any raw markdown or HTML from text"""
    import re
    
    # Remove HTML-style links
    text = re.sub(r'<a href="[^"]*">[^<]*</a>', lambda m: m.group(0).split('>')[1].split('<')[0], text)
    
    # Remove markdown links while keeping the text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    return text.strip()

def generate_article(client, transcripts, keywords=None, news_angle=None, section_count=3, promotional_text=None):
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
เขียนบทความภาษาไทยจากข้อมูลที่ให้มา โดยมีเงื่อนไขดังนี้:
1. ห้ามใส่ลิงก์หรือ markdown ใดๆ ในเนื้อหา
2. เขียนในรูปแบบบทความข่าว
3. ใช้ภาษาที่เป็นทางการ
4. สรุปประเด็นสำคัญจากแหล่งข้อมูลทั้งหมด
5. ความยาวประมาณ 3-4 ย่อหน้า

ข้อมูล:
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
        
        # Clean any markdown that might have slipped through
        cleaned_response = clean_markdown(content)
        return cleaned_response
            
    except Exception as e:
        st.error(f"Error generating article: {str(e)}")
        return None

def extract_image_prompt(article_text):
    pattern = r"(?i)Image Prompt.*?\n(.*)"
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
    
    st.title("🔍 Search and Generate Articles")

    if 'keywords' not in st.session_state:
        st.session_state.keywords = "Bitcoin"
    if 'query' not in st.session_state:
        st.session_state.query = "Strategic Bitcoin Reserve News"
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
        
        hours_back = st.slider("Hours to look back:", 1, 168, 6)
        
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
                hours_back=hours_back,
                search_engines=["Exa"]
            )
            if results:
                st.session_state.search_results = serialize_search_results(results)
                st.session_state.search_results_json = json.dumps(
                    st.session_state.search_results, 
                    ensure_ascii=False, 
                    indent=4,
                    cls=DateTimeEncoder
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
                            st.markdown(st.session_state.article, unsafe_allow_html=True)
                            st.download_button(
                                label="Download Article",
                                data=st.session_state.article,
                                file_name="generated_article.txt",
                                mime="text/plain",
                                use_container_width=True
                            )

                            image_prompt = extract_image_prompt(st.session_state.article)
                            if image_prompt:
                                with st.spinner("Generating image from Together AI..."):
                                    together_client = Together()
                                    response = together_client.images.generate(
                                        prompt=image_prompt,
                                        model="black-forest-labs/FLUX.1-schnell-Free",
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
                                            caption=image_prompt
                                        )
                                    else:
                                        st.error("Failed to generate image from Together AI.")
                                    pass
                        else:
                            st.error("Failed to generate article. Please try again.")

def run_app():
    main()

if __name__ == "__main__":
    run_app()
