import os
import pytz
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

def perform_exa_search(exa_client, query, num_results=10, max_age_days=None):
    """
    Perform a more flexible Exa search with improved result handling
    """
    try:
        base_params = {
            'query': query,
            'num_results': num_results,  # Increased default to 10
            'type': 'keyword',  # Changed from 'auto' to 'keyword' for broader results
            'use_autoprompt': True,
            'text': True,
            'exclude_domains': ['twitter.com', 'reddit.com']  # Exclude social media
        }
        
        # If max_age_days is provided, we compute a start date, else no date restriction
        if max_age_days is not None:
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=max_age_days)
            start_date_str = start_date.strftime('%Y-%m-%dT%H:%M:%SZ')
            end_date_str = end_date.strftime('%Y-%m-%dT%H:%M:%SZ')
            base_params['start_published_date'] = start_date_str
            base_params['end_published_date'] = end_date_str
        
        all_results = []
        response = exa_client.search_and_contents(**base_params)
        if response is not None:
            exa_results = getattr(response, 'results', None) or response
            for result in exa_results:
                # Simplified text handling
                try:
                    text = str(result.text) if hasattr(result, 'text') and result.text else ''
                    title = str(result.title) if hasattr(result, 'title') and result.title else 'No Title'
                    url = str(result.url) if hasattr(result, 'url') and result.url else ''
                except Exception as e:
                    st.error(f"Error processing search result: {str(e)}")
                    continue
                
                # Get published date if available
                published_date = None
                if hasattr(result, 'published_date') and result.published_date:
                    try:
                        parsed_dt = parser.parse(result.published_date)
                        if not parsed_dt.tzinfo:
                            parsed_dt = parsed_dt.replace(tzinfo=timezone.utc)
                        published_date = parsed_dt.isoformat()
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
                
                if len(all_results) >= num_results:
                    break
        
        # Optional date-based sorting
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

def perform_web_research(exa_client, query, max_age_days):
    """
    Perform web research using only Exa search
    """
    try:
        results = perform_exa_search(
            exa_client=exa_client,
            query=query,
            num_results=5,
            max_age_days=max_age_days
        )
        return results
    except Exception as e:
        st.error(f"Search failed: {str(e)}")
        return []

def prepare_content_for_article(selected_results):
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

def generate_article(client: OpenAI, transcripts, keywords=None, evergreen_focus=None, section_count=3, promotional_text=None):
    """
    Prompt focusing on evergreen content instead of news.
    """
    try:
        if not transcripts:
            return None

        keyword_list = []
        if keywords:
            keyword_list = [k.strip() for k in keywords.split('\n') if k.strip()]
        
        primary_keyword = keyword_list[0].upper() if keyword_list else ""
        
        # Evergreen prompt
        prompt = f"""
Write a comprehensive, evergreen-style article in Thai with these components: 
(Title, Main Content, บทสรุป, Excerpt for WordPress, Title & H1 Options, and Meta Description Options all in Thai).
Then provide an Image Prompt in English describing the scene in 1-2 sentences.

Focus the article's perspective on this evergreen angle or topic: {evergreen_focus or ""}.

In the main content:
* Create {section_count} unique section headings in Thai for the main content
* For each section, provide a thorough and detailed exploration of the topic. Use at least four detailed paragraphs per section, each containing specific examples, data points, and case studies. If it hasn't reached four paragraphs, do not conclude the section. Try to add more paragraphs and word countas you go.
* If the content contains numbers that represent monetary values, remove $ signs and add "ดอลลาร์" after each number with a single space
* When referencing sources, naturally integrate the Brand Name as a clickable hyperlink to the source webpage

Excerpt for WordPress: 1 sentence in Thai giving a quick overview of the article.

Title & H1 Options in Thai under 60 characters. Meta Description Options in Thai under 160 characters with the primary keyword. Make them engaging and evergreen.

Finally, provide an Image Prompt in English describing a scene that fits this evergreen content.

(If promotional content is provided below, weave it in at ~10% of the final article.)

Promotional Text (optional):
{promotional_text or ""}

Follow these keyword instructions:

Primary Keyword Optimization:
Primary Keyword: {keyword_list[0] if keyword_list else ""}
- Use it in Title, Meta Description, and H1, as is (original form).
- Insert it naturally in the content.

Secondary Keywords: {', '.join(keyword_list[1:]) if len(keyword_list) > 1 else 'none'}
* Use these in H2 headings/paragraphs if relevant.

Sources:
"""
        for t in transcripts:
            prompt += f"### Content from {t['source']}\nSource URL: {t['url']}\n{t['content']}\n\n"

        completion = client.chat.completions.create(
            model="openai/gpt-4o-2024-11-20",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=10000,
            top_p=0.9,
            frequency_penalty=0.2,
            presence_penalty=0.2,
            extra_headers={
                "HTTP-Referer": "https://github.com/cascade",
                "X-Title": "Cascade"
            }
        )
        
        if completion.choices and len(completion.choices) > 0:
            content = completion.choices[0].message.content
            
            # Replace "$" with " ดอลลาร์"
            content = re.sub(r"\$(\d[\d,\.]*)", r"\1 ดอลลาร์", content)
            
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
    st.set_page_config(page_title="Search and Generate Evergreen Articles", layout="wide")
    
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
    [data-testid="stCheckbox"] {
        scale: 1.5;
        padding: 0;
        margin: 0 10px;
    }
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
    
    st.title("Search and Generate Evergreen Articles")

    if 'keywords' not in st.session_state:
        st.session_state.keywords = "AI Agent"
    if 'query' not in st.session_state:
        st.session_state.query = "AI Agent Economy in Crypto industry"
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
        query = st.text_input("Enter your evergreen topic or query:", value=st.session_state.query)
        
        # Instead of hours, let users specify a day range or no limit:
        max_age_days = st.number_input("Max Age of Content in Days (30 recommended, 0 for none):", min_value=0, value=30)
        max_age_days = None if max_age_days == 0 else max_age_days
        
        st.text_area(
            "Keywords (one per line)",
            height=68,
            key="keywords",
            help="Enter one keyword per line. The first keyword will be the primary keyword for SEO optimization."
        )
        
        user_main_text = st.text_area(
            "Additional Evergreen Text (Optional)",
            height=150,
            help="Paste any reference text you want included (not promotional)."
        )
        
        promotional_text = st.text_area(
            "Promotional Text (Optional)",
            height=100,
            help="Paste any promotional content or CTA you want appended (~10%)."
        )

        search_button = st.button("Search")

    if search_button and query:
        with st.spinner("Searching..."):
            results = perform_web_research(
                exa_client=exa_client,
                query=query,
                max_age_days=max_age_days
            )
            if results:
                st.session_state.search_results = serialize_search_results(results)
                st.session_state.search_results_json = json.dumps(
                    st.session_state.search_results, 
                    ensure_ascii=False, 
                    indent=4
                )
            else:
                st.warning("No results found. Try a different query or broaden your search parameters.")
    
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
                    st.markdown(
                        f'<div class="search-result"><a href="{url}" target="_blank">{title}</a>'
                        f'<br><span class="search-result-source">Source: {source} | Published: {formatted_date}</span></div>',
                        unsafe_allow_html=True
                    )
                    
                    preview = result['text'][:300] + "..." if len(result['text']) > 300 else result['text']
                    st.markdown(preview)
                    with st.expander("Show full content"):
                        st.write(result['text'])
        
        if st.session_state.selected_indices or user_main_text.strip():
            evergreen_angle = st.text_input(
                "Evergreen Topic/Angle",
                value="",
                help="""You can specify the core angle or focus of your evergreen article. 
                For instance, 'Long-term benefits of blockchain technology.'"""
            )
            
            cols = st.columns([0.4, 0.6])
            with cols[0]:
                section_count = st.slider("Number of sections:", 2, 8, 5, key="section_count")
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
                generate_btn = st.button("Generate Evergreen Article")

            if generate_btn:
                st.session_state.generating = True
                keywords = st.session_state.keywords.strip().split('\n')
                keywords = [k.strip() for k in keywords if k.strip()]
                
                selected_results = [results['results'][idx] for idx in st.session_state.selected_indices]
                prepared_content = prepare_content_for_article(selected_results)
                
                # Append user-provided text as additional source if present
                if user_main_text.strip():
                    prepared_content.append({
                        "url": "UserProvided",
                        "source": "User Main Text",
                        "content": user_main_text.strip()
                    })
                
                if prepared_content:
                    with st.spinner("Generating evergreen article..."):
                        article = generate_article(
                            client=openai_client,
                            transcripts=prepared_content,
                            keywords='\n'.join(keywords) if keywords else None,
                            evergreen_focus=evergreen_angle,
                            section_count=section_count,
                            promotional_text=promotional_text
                        )
                        if article:
                            st.session_state.article = article
                            st.success("Evergreen article generated successfully!")
                            
                            st.subheader("Generated Evergreen Article")
                            st.markdown(st.session_state.article, unsafe_allow_html=True)
                            st.download_button(
                                label="Download Article",
                                data=st.session_state.article,
                                file_name="evergreen_article.txt",
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
