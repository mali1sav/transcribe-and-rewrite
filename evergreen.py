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

@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
    retry_error_callback=lambda retry_state: None
)
def make_openai_request(client, prompt, model="deepseek/deepseek-r1-distill-llama-70b", temp=0.6):
    """Make OpenAI API request with retries and proper error handling"""
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
            max_tokens=1000000,
            top_p=0.9,
            frequency_penalty=0.2,
            presence_penalty=0.2,
            extra_headers={
                "HTTP-Referer": "https://github.com/cascade",
                "X-Title": "Cascade"
            }
        )
        
        if not completion or not completion.choices:
            raise ValueError("Empty response received from API")
            
        if not completion.choices[0].message or not completion.choices[0].message.content:
            raise ValueError("No content in API response")
            
        return completion.choices[0].message.content
            
    except httpx.TimeoutException:
        st.error("Request timed out. Please try again.")
        raise
    except httpx.RequestError as e:
        st.error(f"Network error occurred: {str(e)}")
        raise
    except Exception as e:
        st.error(f"API request failed: {str(e)}")
        raise

def generate_article(client: OpenAI, transcripts, keywords=None, evergreen_focus=None, section_count=3, promotional_text=None):
    try:
        if not transcripts:
            return None

        keyword_list = []
        if keywords:
            keyword_list = [k.strip() for k in keywords.split('\n') if k.strip()]
        
        primary_keyword = keyword_list[0].upper() if keyword_list else ""
        
        # Determine if the only source is Additional Evergreen Text
        is_additional_text_only = all(t['source'] == "User Main Text" for t in transcripts)
        
        # Check if the pasted content is longer than 2000 words
        pasted_content_length = len(transcripts[0]['content'].split()) if is_additional_text_only else 0
        needs_condensing = pasted_content_length > 2000

        # Evergreen prompt
        if is_additional_text_only:
            if needs_condensing:
                prompt = f"""
Adapt and translate the following content into a comprehensive article in Thai (keep technical terms and entities in English but the rest should be in Thai). 
Focus on improving readability and structure while preserving the original key points. 
Integrate the provided keywords naturally and ensure to keep the original length while prioritizing key aspects of the pasted content.
Reference sources naturally e.g. use the Brand Name as a clickable markdown hyperlink to the source webpage like this: ([brand name](url)).

Primary Keyword: {primary_keyword}
Secondary Keywords: {', '.join(keyword_list[1:]) if len(keyword_list) > 1 else 'none'}

Content to adapt:
{transcripts[0]['content']}

Instructions:
1. Translate the content into Thai (keep technical terms and entities in English but the rest should be in Thai).
2. Adjust the structure for better readability.
3. Integrate keywords naturally.
4. Prioritize key aspects of the pasted content to condense it to approximately 2000 words.
5. Provide the following components in Thai:
   - Title
   - Main Content (with {section_count} sections), ensure each section heading is engaging
   - บทสรุป
   - Excerpt for WordPress
   - Title & H1 Options
   - Meta Description Options
6. Provide an Image Prompt in English describing the scene in 1-2 sentences that fits this article.

Promotional Text (optional):
{promotional_text or ""}

Note: The main content should remain the priority (~90% focus). If promotional content is provided, blend it in naturally at ~10% of the final article.
"""
            else:
                prompt = f"""
Adapt and translate the following content into a comprehensive, evergreen-style article in Thai. 
Focus on improving readability by structuring content while preserving the original key points. 
Integrate the provided keywords naturally and ensure to keep the original length while prioritizing key aspects of the pasted content.

Primary Keyword: {primary_keyword}
Secondary Keywords: {', '.join(keyword_list[1:]) if len(keyword_list) > 1 else 'none'}

Content to adapt:
{transcripts[0]['content']}

Instructions:
1. Translate the content into Thai.
2. For better readability, Adjust the structure by choosing appropriate headings, paragraphs, lists or table.
3. Integrate keywords naturally.
4. Keep the original length.
5. Provide the following components in Thai (keep technical terms and entities in English but the rest should be in Thai):
   - Title - ensure it's engaging and click-worthy
   - Main Content (with {section_count} sections). Ensure each section heading is engaging for the crypto news reader
   - บทสรุป
   - Excerpt for WordPress
   - Title & H1 Options
   - Meta Description Options
6. Provide an Image Prompt in English describing the scene that fits this article in 1-2 sentences.

Promotional Text (optional):
{promotional_text or ""}

Note: The main content should remain the priority (~90% focus). If promotional content is provided, blend it in naturally at ~10% of the final article.
"""
        else:
            prompt = f"""
Write a comprehensive, evergreen-style article in Thai with these components (keep technical terms and entities in English but the rest should be in Thai): 
(Title, Main Content, บทสรุป, Excerpt for WordPress, Title & H1 Options, and Meta Description Options all in Thai).
Then provide an Image Prompt in English describing the scene in 1-2 sentences.

Focus the article's perspective on this evergreen angle or topic: {evergreen_focus or ""}.

In the main content:
* Create {section_count} unique, engaging, and semantically relevant H2 section headings in Thai for the main content. 
* For each section, provide a thorough and detailed exploration of the topic. Use at least 3 detailed paragraphs per section, each containing specific examples, data points, and case studies. If it hasn't reached four paragraphs, do not conclude the section. Try to add more paragraphs and word count as you go.
* Ensure the article reaches approximately 1800 words. If it doesn't, continue expanding on the content without concluding.
* If the content contains numbers that represent monetary values, remove $ signs and add "ดอลลาร์" after each number with a single space
* When referencing sources, naturally integrate the web Brand Name as a clickable markdown hyperlink to the source webpage like this: ([brand name](url)).

Excerpt for WordPress: 1 sentence in Thai giving a quick overview of the article.

Title & H1 Options in Thai under 60 characters. Meta Description Options in Thai under 160 characters with the primary keyword. Make them engaging and cliick-worthy.

Finally, provide an Image Prompt in English describing a scene that fits this article.

If there is promotional content provided below, seamlessly blend it into roughly 10% of the final article. The main news content (including any user-pasted main text) should remain the priority (~90% focus), but do a smooth transition into promotional text, ensure it's engaging and semantically relevant.

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

        # Make initial API request
        content = make_openai_request(client, prompt)
        if not content:
            return None
            
        # Replace "$" with " ดอลลาร์"
        content = re.sub(r"\$(\d[\d,\.]*)", r"\1 ดอลลาร์", content)
        
        # Ensure the article reaches approximately 2000 words
        word_count = len(content.split())
        if word_count < 2000 and not needs_condensing:
            additional_prompt = f"""
The article is currently {word_count} words long. Please continue expanding on the content to reach 2000 words. 
Focus on adding more detailed paragraphs, examples, and case studies without concluding the article.
"""
            try:
                additional_content = make_openai_request(
                    client, 
                    additional_prompt, 
                    model="openai/gpt-4o-2024-11-20",
                    temp=0.6
                )
                if additional_content:
                    content += "\n\n" + additional_content
            except Exception as e:
                st.warning(f"Failed to expand article content: {str(e)}")
                # Continue with original content if expansion fails
        
        elif word_count > 2000 and needs_condensing:
            condensing_prompt = f"""
The article is currently {word_count} words long. Please condense it to approximately 2000 words by prioritizing key aspects of the pasted content.
"""
            try:
                content = make_openai_request(
                    client, 
                    condensing_prompt, 
                    model="openai/gpt-4o-2024-11-20",
                    temp=0.6
                )
            except Exception as e:
                st.warning(f"Failed to condense article content: {str(e)}")
                # Continue with original content if condensing fails
        
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
        font-size: 1.2em;
        line-height: 1.5em;
        padding: 10px 20px;
        width: 100%;
        border-radius: 5px;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .stButton button:hover {
        background-color: #0052CC !important;
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
        st.session_state.keywords = "AI Agent คือ\nลงทุน AI Agent"
    if 'query' not in st.session_state:
        st.session_state.query = "2025 will be the year of AI agents"
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
        
        # Instead of hours, let users specify a day range or no limit:
        max_age_days = st.number_input("Max Age of Content in Days (30 recommended):", min_value=0, value=30)
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

        # Move the Generate Evergreen Article button to the sidebar
        evergreen_angle = st.text_input(
            "Evergreen Topic/Angle",
            value="",
            help="""You can specify the core angle or focus of your evergreen article. 
            For instance, 'Long-term benefits of blockchain technology.'"""
        )
        
        section_count = st.slider("Number of sections:", 2, 8, 5, key="section_count")
        
        # Renamed button: Generate Article from Pasted Text
        generate_btn_pasted_text = st.button("Generate Article from Pasted Text")
        st.markdown("<div style='text-align: center; margin: 10px; padding: 5px'>Or</div>", unsafe_allow_html=True)
        # Move the Search button below the Generate Article from Pasted Text button
        search_button = st.button("Search Content")

    # Initialize generate_btn_all_sources to False
    generate_btn_all_sources = False

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

        # Renamed button: Generate Article from All Sources
        generate_btn_all_sources = st.button("Generate Article from All Sources")

    # Article generation logic (works with or without search results)
    if (generate_btn_pasted_text or generate_btn_all_sources) and (st.session_state.search_results or user_main_text.strip()):
        st.session_state.generating = True
        keywords = st.session_state.keywords.strip().split('\n')
        keywords = [k.strip() for k in keywords if k.strip()]
        
        selected_results = []
        if st.session_state.search_results:
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