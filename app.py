import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional
import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    """Configuration class to manage environment variables and paths"""
    OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
    BASE_DIR = Path(__file__).parent
    TEMP_DIR = BASE_DIR / "temp"
    TEMP_DIR.mkdir(exist_ok=True)

# Initialize OpenAI client with OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=Config.OPENROUTER_API_KEY
)

class TranscriptFetcher:
    def get_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL."""
        try:
            video_id = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
            return video_id.group(1) if video_id else None
        except Exception as e:
            logger.error(f"Error extracting video ID: {str(e)}")
            return None

    def get_transcript(self, url: str, language: str = "Auto-detect") -> Optional[Dict]:
        """Get transcript from YouTube."""
        try:
            video_id = self.get_video_id(url)
            if not video_id:
                logger.error(f"Could not extract video ID from URL: {url}")
                return None

            try:
                if language == "Thai":
                    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['th'])
                elif language == "English":
                    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
                else:
                    transcript = YouTubeTranscriptApi.get_transcript(video_id)
                
                full_text = " ".join(segment['text'] for segment in transcript)
                logger.info(f"Successfully got YouTube transcript for video {video_id}")
                return {
                    'text': full_text,
                    'segments': transcript,
                    'language': language
                }
            except (TranscriptsDisabled, NoTranscriptFound) as e:
                logger.error(f"No transcript available for video {video_id}: {str(e)}")
                return None

        except Exception as e:
            logger.error(f"Error getting YouTube transcript: {str(e)}")
            return None

def process_sources(fetcher: TranscriptFetcher, yt_url1, channel1, yt_url2, channel2, yt_url3, channel3, text_content, language):
    """Process all sources synchronously with retry logic"""
    MAX_RETRIES = 3
    transcripts = []
    sources_to_process = []
    
    # Prepare sources to process
    if yt_url1 and channel1:
        sources_to_process.append({"url": yt_url1, "channel": channel1})
    if yt_url2 and channel2:
        sources_to_process.append({"url": yt_url2, "channel": channel2})
    if yt_url3 and channel3:
        sources_to_process.append({"url": yt_url3, "channel": channel3})

    # Track processed sources
    processed_sources = set()
    retry_count = 0

    while sources_to_process and retry_count < MAX_RETRIES:
        retry_count += 1
        failed_sources = []

        for source in sources_to_process:
            if source["url"] in processed_sources:
                continue

            with st.spinner(f"Processing {source['channel']} (Attempt {retry_count}/{MAX_RETRIES})..."):
                try:
                    transcript = fetcher.get_transcript(source["url"], language)
                    if transcript:
                        transcripts.append({
                            'source': source['channel'],
                            'url': source['url'],
                            'content': transcript['text']
                        })
                        processed_sources.add(source["url"])
                        st.success(f"✅ Completed {source['channel']}")
                    else:
                        failed_sources.append(source)
                        st.warning(f"⚠️ Failed to get transcript for {source['channel']} - Will retry")
                except Exception as e:
                    logger.error(f"Error processing {source['channel']}: {str(e)}")
                    failed_sources.append(source)
                    st.warning(f"⚠️ Error processing {source['channel']} - Will retry")

        sources_to_process = failed_sources
        if failed_sources:
            if retry_count < MAX_RETRIES:
                st.info(f"Retrying failed transcriptions... ({retry_count}/{MAX_RETRIES})")
            else:
                for source in failed_sources:
                    st.error(f"❌ Failed to get transcript for {source['channel']} after {MAX_RETRIES} attempts")

    # Process direct text input - now without requiring text_source
    if text_content:
        transcripts.append({
            'source': 'Article',  # Default source name for pasted content
            'content': text_content
        })
        st.success("✅ Added pasted content")

    return transcripts

def generate_article(client: OpenAI, transcripts, keywords=None, language=None, angle=None, section_count=3):
    """Generate article from transcripts using OpenRouter"""
    try:
        if not transcripts:
            return None

        # Process keywords properly
        keyword_list = []
        if keywords:
            keyword_list = [k.strip() for k in keywords.split('\n') if k.strip()]
        
        # Get the primary keyword (first keyword)
        primary_keyword = keyword_list[0].upper() if keyword_list else ""
        
        # Define shortcode mapping
        shortcode_map = {
            "BITCOIN": '[latest_articles label="ข่าว Bitcoin (BTC) ล่าสุด" count_of_posts="6" taxonomy="category" term_id="7"]',
            "ETHEREUM": '[latest_articles label="ข่าว Ethereum ล่าสุด" count_of_posts="6" taxonomy="category" term_id="8"]',
            "SOLANA": '[latest_articles label="ข่าว Solana ล่าสุด" count_of_posts="6" taxonomy="category" term_id="501"]',
            "XRP": '[latest_articles label="ข่าว XRP ล่าสุด" count_of_posts="6" taxonomy="category" term_id="502"]',
            "DOGECOIN": '[latest_articles label="ข่าว Dogecoin ล่าสุด" count_of_posts="6" taxonomy="category" term_id="527"]'
        }

        # Format keywords for the prompt
        keyword_instruction = """Primary Keyword Optimization:
Primary Keyword: {}
This must appear naturally ONCE in Title, Meta Description, and H1.
Use this in H2 headings and paragraphs where they fit naturally.
Secondary Keywords: {}
- Use these in H2 headings and paragraphs where they fit naturally
- Each secondary keyword should appear no more than 5 times in the entire content
- Only use these in Title, Meta Description, or H1 if they fit naturally.
- Skip any secondary keywords that don't fit naturally in the context""".format(
    keyword_list[0] if keyword_list else "",
    ', '.join(keyword_list[1:]) if len(keyword_list) > 1 else 'none'
)
        
        # Add angle instruction
        angle_instruction = ""
        if angle:
            angle_instruction = f"""
Main Article Angle: {angle}
- Focus the article's perspective and analysis primarily through this lens
- Ensure all sections contribute to or relate back to this main angle
- Prioritize information and insights that are most relevant to this perspective
- Structure the article to build a coherent narrative around this angle"""

        # Prepare the attribution format for multiple sources
        attribution_instruction = "Include concise attributions to all source videos. Recognise the speaker names if possible. "
        if any('url' in t for t in transcripts):
            attribution_instruction += "Use these formats for attribution: "
            sources_with_urls = [t for t in transcripts if 'url' in t]
            attribution_instruction += ", ".join(
                f"<a href=\"{t['url']}\">{t['source']}</a>" 
                for t in sources_with_urls
            )
        else:
            attribution_instruction += "The content includes citations and references within the text."

        prompt = """write a comprehensive and in-depth Thai crypto news article for WordPress. """ + angle_instruction + """ Ensure the Thai article is detailed and informative, providing thorough explanations and analyses for each key point discussed. Incorporate information from all provided sources, balancing content from each channel appropriately. Make complex concepts easy to understand for Thai readers. Follow these keyword instructions:

""" + keyword_instruction + """

First, write the following sections:

* Meta Description: Summarise the article in 160 characters in Thai.
* Provide a concise and engaging news headline that captures the main idea of the article.
* Main content: Start with a strong opening that highlights the most newsworthy aspect, specifically focusing on the chosen angle. 
  """ + attribution_instruction + """

* Create exactly """ + str(section_count) + """ distinct and engaging headings for the main content, ensuring they align with and support the main angle. For each content section, pick the right format like sub-headings, paragraphs or list items for improve readability. Write content continuously without any <hr> separator line between sections.
* For each content section, provide an in-depth explanation, context, and implications to Crypto investors, maintaining focus on the chosen angle. If relevant, include direct quotes or specific data points from the transcript to add credibility.
* If the content contains numbers that represent monetary values, remove $ signs before numbers and add "ดอลลาร์" after the number, ensure proper spacing with one space before and after numbers, and maintain consistent formatting throughout the article.
* Important Instruction: When referencing a source, naturally integrate the Brand Name into the sentence as a clickable hyperlink.
* บทสรุป: Use a H2 heading. Summarise key points and implications by emphasizing insights related to the main angle.

* Excerpt for WordPress: In Thai, provide 1 sentence for a brief overview.

* Image Prompt: In English, describe a scene that captures the article's essence and chosen angle, focus on only 1 or 2 objects. 

After writing all the above sections, analyze the key points and generate these title options:
* Title & H1 Options:
  1. News style: State the main news with a compelling hook (integrate primary keyword and angle naturally)
  2. Question style: Ask an engaging question that addresses the main concern (integrate primary keyword and angle naturally)
  3. Number style: Start with a number or statistic that captures attention (integrate primary keyword and angle naturally)

Here are the transcripts to base the article on:
"""
        for transcript_item in transcripts:
            prompt += """### Transcript from """ + transcript_item['source'] + "\n" + transcript_item['content'] + "\n\n"

        completion = client.chat.completions.create(
            model="openai/gpt-4o-2024-11-20",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=5500,  # Increased to ensure we get all content including titles
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        
        # Access the response content using the new API format
        if completion.choices and len(completion.choices) > 0:
            content = completion.choices[0].message.content
            
            # Add the appropriate shortcode based on primary keyword before the Excerpt section
            shortcode = shortcode_map.get(primary_keyword, "")
            if shortcode:
                # First split at "Title & H1 Options:" to preserve it at the end
                title_parts = content.split("Title & H1 Options:", 1)
                if len(title_parts) == 2:
                    main_content = title_parts[0]
                    title_content = "Title & H1 Options:" + title_parts[1]
                    
                    # Then split main content at "Excerpt for WordPress:"
                    parts = main_content.split("Excerpt for WordPress:", 1)
                    if len(parts) == 2:
                        # Reconstruct with shortcode and titles
                        content = (parts[0].rstrip() + "\n\n" + shortcode + 
                                 "\n\n----------------\n\n" + "Excerpt for WordPress:" + 
                                 parts[1].rstrip() + "\n\n" + title_content)
                    else:
                        content = main_content.rstrip() + "\n\n" + shortcode + "\n\n" + title_content
                else:
                    # If no title section found, just handle excerpt
                    parts = content.split("Excerpt for WordPress:", 1)
                    if len(parts) == 2:
                        content = parts[0].rstrip() + "\n\n" + shortcode + "\n\n----------------\n\n" + "Excerpt for WordPress:" + parts[1]
                    else:
                        content = content.rstrip() + "\n\n" + shortcode + "\n"
            
            return content
        else:
            logger.error("No content in completion response")
            st.error("No valid response content received")
            return None

    except Exception as e:
        logger.error(f"Error generating article: {str(e)}")
        st.error(f"Error generating article: {str(e)}")
        return None

def main():
    # Initialize session state for input fields if not exists
    if 'keywords' not in st.session_state:
        st.session_state.keywords = "Bitcoin\nBTC"
    if 'yt_url1' not in st.session_state:
        st.session_state.yt_url1 = "https://www.youtube.com/watch?v=dW2KZDucujw"
    if 'channel1' not in st.session_state:
        st.session_state.channel1 = "InvestAnswers"
    if 'yt_url2' not in st.session_state:
        st.session_state.yt_url2 = ""
    if 'channel2' not in st.session_state:
        st.session_state.channel2 = ""
    if 'yt_url3' not in st.session_state:
        st.session_state.yt_url3 = ""
    if 'channel3' not in st.session_state:
        st.session_state.channel3 = ""
    if 'text_content' not in st.session_state:
        st.session_state.text_content = ""
    if 'angle' not in st.session_state:
        st.session_state.angle = "Bitcoin Situation Analysis"
    if 'transcripts' not in st.session_state:
        st.session_state.transcripts = []
    if 'article' not in st.session_state:
        st.session_state.article = None
    if 'processing_done' not in st.session_state:
        st.session_state.processing_done = False
    if 'generating' not in st.session_state:
        st.session_state.generating = False

    # X button callbacks
    def X_field(field_name):
        st.session_state[field_name] = ""

    def process_clicked():
        st.session_state.processing_done = True

    def generate_clicked():
        st.session_state.generating = True

    # Set page to full screen and wide mode
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
    
    # Custom CSS for full screen and compact layout
    st.markdown("""
        <style>
        .main {
            max-width: 100% !important;
            padding: 0rem;
        }

        .stButton button {
            width: 100%;
        }
        div[data-testid="stVerticalBlock"] {
            gap: 0.3rem !important;
        }
        header {
            visibility: hidden;
        }
        #MainMenu {
            visibility: hidden;
        }
        footer {
            visibility: hidden;
        }
        div[data-testid="stAppViewContainer"] > section[data-testid="stSidebar"] {
            display: none;
        }
        div[data-testid="stAppViewContainer"] > section:first-child {
            padding-top: 0rem;
        }
        div[data-testid="stToolbar"] {
            display: none;
        }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
        }
        h1 {
            margin-top: -1rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Streamlit UI - moved title inside container for better spacing
    with st.container():
        st.title("Transcribe and Rewrite")
        # Layout optimization - use columns for better space usage
        input_col, output_col = st.columns([1, 1])

        with input_col:
            # Compact input section
            with st.expander("Input Sources", expanded=True):
                # Language selection in a row with keywords
                lang_col, key_col = st.columns([1, 3])
                with lang_col:
                    language = st.selectbox(
                        "Language",
                        options=["Thai", "English"],
                        index=1,
                        key="language"
                    )
                with key_col:
                    keywords = st.text_area("Keywords (one per line, first keyword is primary)", 
                          help="Enter keywords, one per line. The first keyword will be used as the primary keyword.", 
                          height=70, 
                          key="keywords")
                
                # Add angle input
                angle = st.text_input("Article Angle", 
                         help="If video lack focus or has multiple topics, enter the main angle you want the article to focus on (e.g., 'BTC situation analysis', 'ETH Technical analysis', 'Regulatory implications')", 
                         key="angle")

                # Add section count input
                section_count = st.number_input("Number of Content Sections", 
                                              min_value=1, 
                                              max_value=10, 
                                              value=4, 
                                              help="Specify how many H2 sections you want in the article. If the video length is 12-15 mins, consider 2-3 sections. If it's longer than 20-30 mins, consider 4-5 sections.",
                                              key="section_count")

                # YouTube inputs in more compact rows
                yt1_col1, yt1_col2, yt1_col3 = st.columns([5, 2, 0.5])
                with yt1_col1:
                    yt_url1 = st.text_input("YouTube URL 1", value=st.session_state.yt_url1, key="yt_url1", label_visibility="collapsed", placeholder="YouTube URL 1")
                with yt1_col2:
                    channel1 = st.text_input("Channel 1", value=st.session_state.channel1, key="channel1", label_visibility="collapsed", placeholder="Channel 1")
                with yt1_col3:
                    st.button("X", key="X_yt1", on_click=lambda: [X_field("yt_url1"), X_field("channel1")])
                
                yt2_col1, yt2_col2, yt2_col3 = st.columns([5, 2, 0.5])
                with yt2_col1:
                    yt_url2 = st.text_input("YouTube URL 2", value=st.session_state.yt_url2, key="yt_url2", label_visibility="collapsed", placeholder="YouTube URL 2 (Optional)")
                with yt2_col2:
                    channel2 = st.text_input("Channel 2", value=st.session_state.channel2, key="channel2", label_visibility="collapsed", placeholder="Channel 2")
                with yt2_col3:
                    st.button("X", key="X_yt2", on_click=lambda: [X_field("yt_url2"), X_field("channel2")])
                
                yt3_col1, yt3_col2, yt3_col3 = st.columns([5, 2, 0.5])
                with yt3_col1:
                    yt_url3 = st.text_input("YouTube URL 3", value=st.session_state.yt_url3, key="yt_url3", label_visibility="collapsed", placeholder="YouTube URL 3 (Optional)")
                with yt3_col2:
                    channel3 = st.text_input("Channel 3", value=st.session_state.channel3, key="channel3", label_visibility="collapsed", placeholder="Channel 3")
                with yt3_col3:
                    st.button("X", key="X_yt3", on_click=lambda: [X_field("yt_url3"), X_field("channel3")])
                
                # Text input in a more compact layout
                text_col1, text_col3 = st.columns([6, 0.5])
                with text_col1:
                    text_content = st.text_area("Text Content", value=st.session_state.text_content, height=200, key="text_content", label_visibility="collapsed", placeholder="Or paste text content (supports Markdown with references)")
                with text_col3:
                    st.button("X", key="X_text", on_click=lambda: X_field("text_content"))

                # Process button at the bottom of input section
                if st.button("Process Sources", type="primary", key="process_btn", use_container_width=True):
                    process_clicked()

        with output_col:
            # Process sources when the processing_done flag is set
            if st.session_state.processing_done:
                try:
                    fetcher = TranscriptFetcher()
                    st.session_state.transcripts = process_sources(
                        fetcher, yt_url1, channel1, yt_url2, channel2, yt_url3, channel3, text_content, language
                    )
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    logger.error(f"Error processing sources: {str(e)}")
                st.session_state.processing_done = False

            # Show Generate Article and Download buttons if we have transcripts
            if st.session_state.transcripts:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Generate Article", type="primary", key="generate_btn", use_container_width=True):
                        generate_clicked()
                with col2:
                    combined_text = "\n\n".join([f"=== {t['source']} ===\n{t['content']}" for t in st.session_state.transcripts])
                    st.download_button(
                        label="Download Transcripts",
                        data=combined_text,
                        file_name="transcripts.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
            
            # Generate article when button is clicked
            if getattr(st.session_state, 'generating', False):
                with st.spinner("Generating article..."):
                    article = generate_article(client, st.session_state.transcripts, keywords, language, angle, section_count)
                    if article:
                        st.session_state.article = article
                    else:
                        st.error("Failed to generate article. Please try again.")
                st.session_state.generating = False
            
            # Show article and download button if generated
            if st.session_state.article:
                st.subheader("Meta Description")
                st.markdown(st.session_state.article, unsafe_allow_html=True)
                
                st.download_button(
                    label="Download Article",
                    data=st.session_state.article,
                    file_name="generated_article.txt",
                    mime="text/plain",
                    use_container_width=True
                )

if __name__ == "__main__":
    main()
