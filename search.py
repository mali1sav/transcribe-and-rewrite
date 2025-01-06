import os
import pytz
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
import html
import unicodedata
import json
import base64
from together import Together

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

def init_tavily_client():
    tavily_api_key = os.getenv('TAVILY_API_KEY')
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY not found in environment variables")
    return TavilyClient(api_key=tavily_api_key)

def init_openai_client():
    try:
        openai_api_key = os.getenv('OPENROUTER_API_KEY')
        if not openai_api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openai_api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/your-repo",
                "X-Title": "Fast Transcriber"
            }
        )
        return client
    except Exception as e:
        st.error(f"Failed to initialize OpenAI client: {str(e)}")
        return None

def perform_exa_search(exa_client, query, num_results=5, hours_back=24, categories=None):
    try:
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(hours=hours_back)
        
        start_date_str = start_date.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_date_str = end_date.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        base_params = {
            'query': query,
            'num_results': 5,  # Always request 5 results
            'start_published_date': start_date_str,
            'end_published_date': end_date_str,
            'type': 'auto',
            'use_autoprompt': True,
            'text': True
        }
        
        all_results = []
        categories = categories or ['news']
        
        # Try each category until we have 5 results
        for category in categories:
            if len(all_results) >= 5:
                break
                
            try:
                search_params = base_params.copy()
                search_params['category'] = category
                response = exa_client.search_and_contents(**search_params)
                if response is not None:
                    category_results = getattr(response, 'results', None) or response
                    for result in category_results:
                        if hasattr(result, 'text') and result.text:
                            # Handle text content with proper encoding
                            try:
                                if isinstance(result.text, bytes):
                                    # Try UTF-8 first, then fallback to latin-1
                                    try:
                                        text = result.text.decode('utf-8')
                                    except UnicodeDecodeError:
                                        text = result.text.decode('latin-1')
                                else:
                                    text = str(result.text)
                                
                                # Validate text content
                                if not text.strip() or any(ord(char) > 127 and not unicodedata.category(char).startswith('P') for char in text[:100]):
                                    continue
                            except Exception:
                                continue
                            
                            # Handle title with proper encoding
                            try:
                                if hasattr(result, 'title') and result.title:
                                    if isinstance(result.title, bytes):
                                        try:
                                            title = result.title.decode('utf-8')
                                        except UnicodeDecodeError:
                                            title = result.title.decode('latin-1')
                                    else:
                                        title = str(result.title)
                                else:
                                    title = 'No Title'
                            except Exception:
                                title = 'No Title'
                                
                            # Handle URL
                            try:
                                if hasattr(result, 'url') and result.url:
                                    if isinstance(result.url, bytes):
                                        try:
                                            url = result.url.decode('utf-8')
                                        except UnicodeDecodeError:
                                            url = result.url.decode('latin-1')
                                    else:
                                        url = str(result.url)
                                else:
                                    url = ''
                            except Exception:
                                url = ''
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
                                'source': get_domain(url)
                            }
                            all_results.append(transformed_result)
                            
                            # Stop if we have 5 results
                            if len(all_results) >= 5:
                                break
            except Exception as e:
                continue
        
        def get_date_for_sorting(item):
            try:
                if item.get('published_date'):
                    date = parser.parse(item['published_date'])
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

def perform_tavily_search(tavily_client, query, num_results=5, hours_back=24):
    try:
        days = max(1, round(hours_back / 24))
        response = tavily_client.search(
            query=query,
            search_depth="advanced",
            topic="news", 
            days=days,
            max_results=num_results,
            include_raw_content=True
        )
        
        results = []
        if response and 'results' in response:
            for result in response['results']:
                cleaned_content = str(result.get('content', ''))
                published_date_str = result.get('published_date')
                parsed_published_date = None
                if published_date_str:
                    try:
                        parsed = parser.parse(published_date_str)
                        if not parsed.tzinfo:
                            parsed = parsed.replace(tzinfo=timezone.utc)
                        parsed_published_date = parsed.isoformat()
                    except:
                        pass
                
                formatted_result = {
                    'title': result.get('title', ''),
                    'url': result.get('url', ''),
                    'text': cleaned_content,
                    'source': get_domain(result.get('url', '')),
                    'published_date': parsed_published_date
                }
                results.append(formatted_result)

        def get_date_for_sorting(item):
            try:
                if item.get('published_date'):
                    date = parser.parse(item['published_date'])
                    return date.replace(tzinfo=timezone.utc) if not date.tzinfo else date
                return datetime.min.replace(tzinfo=timezone.utc)
            except:
                return datetime.min.replace(tzinfo=timezone.utc)
                
        results.sort(key=get_date_for_sorting, reverse=True)
        results = results[:num_results]
        
        if results:
            st.success(f"✅ Found {len(results)} results from Tavily")
        
        return results
    except Exception as e:
        st.error(f"Error performing Tavily search: {str(e)}")
        return None

def perform_web_research(exa_client, query, hours_back, search_engines):
    results = []
    try:
        exa_results = perform_exa_search(exa_client, query, num_results=5, hours_back=hours_back)
        results.extend(exa_results)
    except Exception as e:
        st.error(f"Exa search failed: {str(e)}")
    
    try:
        tavily_client = init_tavily_client()
        tavily_results = perform_tavily_search(tavily_client, query, num_results=5, hours_back=hours_back)
        if tavily_results:
            results.extend(tavily_results)
    except Exception as e:
        st.error(f"Tavily search failed: {str(e)}")
    
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

def generate_article(client: OpenAI, transcripts, keywords=None, news_angle=None, section_count=3, promotional_text=None):
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
Write a comprehensive and in-depth Thai crypto news article (Title, Main Content, บทสรุป, Excerpt for WordPress, Title & H1 Options, and Meta Description Options all in Thai).
Then provide an Image Prompt in English describing the scene in one or two sentences.

If there is promotional content provided below, seamlessly blend it into roughly 10% of the final article. The main news content (including any user-pasted main text) should remain the priority (~90% focus), but do a smooth transition into promotional text.

Focus the article's perspective on the following news angle (optional), prioritise info and insights that are most relevant to this perspective and structure the article to build a coherent narrative around this angle:
{news_angle or ""}

Promotional Text (optional):
{promotional_text or ""}

Follow these keyword instructions:

Primary Keyword Optimization:
Primary Keyword: {keyword_list[0] if keyword_list else ""}
- Use the primary keyword in its original form (Thai or English)
- Integrate the keyword naturally while maintaining grammatical correctness
- The keyword must appear naturally ONCE in Title, Meta Description, and H1
- Use the keyword in section headings and paragraphs where it fits naturally

Secondary Keywords: {', '.join(keyword_list[1:]) if len(keyword_list) > 1 else 'none'}
* Use these in H2 headings and paragraphs where they fit naturally
* Each secondary keyword should appear no more than 5 times in the entire content
* Skip any secondary keywords that don't fit naturally in the context

In the main content:
* Provide a concise but news-like Title, ensure it's engaging.
* Open with the most newsworthy aspect.
* Create exactly {section_count} sub-headings in Thai for the main content.
* For each section, give in-depth context and the implications for crypto investors. Make complex concepts simple and easy to understand.
* Use heading level 2 for each section heading.
* If the content contains numbers that represent monetary values, remove $ signs before numbers and add "ดอลลาร์" after the number, ensuring a single space before and after the numeric value.
* Whenever referencing a source, use the source's Brand Name as a clickable hyperlink in Thai text, e.g., [Brand Name](URL).

Use a H2 heading for บทสรุป: Summarize key points.

Excerpt for WordPress: In Thai, one sentence that briefly describes the article.

Title & H1 Options (in Thai) under 60 characters. Meta Description Options (in Thai) under 160 characters, both integrating the primary keyword in its original English form naturally, both are engaging, with news-like sentences, and are click-worthy.

Finally, provide an Image Prompt in English describing a scene that fits the article in 1-2 sentences.

Here are the sources to base the article on:
"""
        # Append transcripts (which may include search results + user-provided main text)
        for t in transcripts:
            prompt += f"### Content from {t['source']}\nSource URL: {t['url']}\n{t['content']}\n\n"

        completion = client.chat.completions.create(
            model="openai/gpt-4o-2024-11-20",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=7000,
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
                # Example insertion point
                excerpt_split = "Excerpt for WordPress:"
                if excerpt_split in content:
                    parts = content.split(excerpt_split, 1)
                    content = (
                        parts[0].rstrip() + "\n\n" +
                        shortcode + "\n\n----------------\n\n" +
                        excerpt_split + parts[1]
                    )
            
            return content
        else:
            st.error("No valid response content received")
            return None

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
    
    st.title("Search and Generate Articles")

    if 'keywords' not in st.session_state:
        st.session_state.keywords = "Bitcoin"
    if 'query' not in st.session_state:
        st.session_state.query = "Bitcoin Situation Analysis"
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
        openai_client = init_openai_client()
        if not openai_client:
            st.error("Failed to initialize OpenAI client")
            return
    except Exception as e:
        st.error(f"❌ Failed to initialize clients: {str(e)}")
        return

    with st.sidebar:
        query = st.text_input("Enter your search query:", value=st.session_state.query)
        
        hours_back = st.slider("Hours to look back:", 1, 168, 6)
        
        st.text_area(
            "Keywords (one per line)",
            height=68,  # Reduced height to show approximately 2 rows
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
                search_engines=["Exa", "Tavily"]
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
                help="""This can be in Thai or English. Having a clear news angle is essential, especially when sources may lack focus which is bad for SEO. A well-defined angle helps create a coherent narrative around your chosen perspective
                Tips: You can use one of the English headlines from your selected news sources as your news angle."""
            )
            
            # Move number of sections slider to the right
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
                            client=openai_client,
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
                        else:
                            st.error("Failed to generate article. Please try again.")

def run_app():
    main()

if __name__ == "__main__":
    run_app()
