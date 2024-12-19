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

def process_sources(fetcher: TranscriptFetcher, yt_url1, channel1, yt_url2, channel2, text_content, text_source, language):
    """Process all sources synchronously"""
    try:
        transcripts = []

        # Process YouTube URL 1
        if yt_url1 and channel1:
            with st.spinner(f"Processing {channel1}..."):
                transcript = fetcher.get_transcript(yt_url1, language)
                if transcript:
                    transcripts.append({
                        'source': channel1,
                        'url': yt_url1,
                        'content': transcript['text']
                    })
                    st.success(f"✅ Completed {channel1}")
                else:
                    st.error(f"Failed to get transcript for {channel1}")

        # Process YouTube URL 2
        if yt_url2 and channel2:
            with st.spinner(f"Processing {channel2}..."):
                transcript = fetcher.get_transcript(yt_url2, language)
                if transcript:
                    transcripts.append({
                        'source': channel2,
                        'url': yt_url2,
                        'content': transcript['text']
                    })
                    st.success(f"✅ Completed {channel2}")
                else:
                    st.error(f"Failed to get transcript for {channel2}")

        # Process direct text input
        if text_content and text_source:
            transcripts.append({
                'source': text_source,
                'content': text_content
            })
            st.success(f"✅ Added text from {text_source}")

        return transcripts

    except Exception as e:
        logger.error(f"Error processing sources: {str(e)}")
        return []

def generate_article(client: OpenAI, transcripts, keywords=None, language=None, angle=None, section_count=3):
    """Generate article from transcripts using OpenRouter"""
    if not Config.OPENROUTER_API_KEY:
        st.error("OpenRouter API key not found in environment variables")
        return None

    try:
        # Process keywords
        primary_keyword = ""
        secondary_keywords = []
        if keywords:
            keyword_list = [k.strip() for k in keywords.split('\n') if k.strip()]
            if keyword_list:
                primary_keyword = keyword_list[0]
                secondary_keywords = keyword_list[1:] if len(keyword_list) > 1 else []

        keyword_instruction = f"""Primary Keyword: {primary_keyword}
This must appear naturally ONCE in Title, Meta Description, and H1. Do not force it if it doesn't fit naturally.
Use primary keyword in the rest of the article 2-3 more times.
Secondary Keywords: {', '.join(secondary_keywords) if secondary_keywords else 'none'}
- Only use these in H2 headings and paragraphs where they fit naturally
- Each secondary keyword should appear 3-5 times in the entire content
- Only use these in Title, Meta Description, or H1 if they can fit in naturally.
- Skip any secondary keywords that don't fit naturally in the context"""

        # Add angle instruction
        angle_instruction = ""
        if angle:
            angle_instruction = f"""
Main Article Angle: {angle}
- Focus the article's perspective and analysis primarily through this lens
- Ensure all sections contribute to or relate back to this main angle
- Prioritize information and insights that are most relevant to this perspective
- Structure the article to build a coherent narrative around this angle"""

        # Add attribution based on source type
        attribution = ""
        if transcripts and transcripts[0]['url']:
            attribution = f"Use this format for attribution: '<a href=\"{transcripts[0]['url']}\">{transcripts[0]['source']}</a>'"
        elif transcripts:
            attribution = f"Source: {transcripts[0]['source']}"

        prompt = f"""write a comprehensive and in-depth Thai crypto news article for WordPress. {angle_instruction} Ensure the article is detailed and informative, providing thorough explanations and analyses for each key point discussed. Follow these keyword instructions:

{keyword_instruction}

First, write the following sections:

* Meta Description: Summarise the article in 160 characters in Thai.
* H1: Provide a concise title that captures the main idea of the article with a compelling hook in Thai.
* Main content: Start with a strong opening that highlights the most newsworthy aspect, specifically focusing on the chosen angle. {attribution if attribution else ""}

* Create exactly {section_count} distinct and engaging headings (H2) for the main content, ensuring they align with and support the main angle. For each content section, pick the right format like sub-headings, paragraphs or list items for improve readability. Write content continuously without line separators between sections.
* For each content under each H2, provide an in-depth explanation, context, and implications to Crypto investors, maintaining focus on the chosen angle. If relevant, include direct quotes or specific data points from the transcript to add credibility.
* Important Instruction: When referencing a source, naturally integrate the Brand Name into the sentence as a clickable hyperlink.
* บทสรุป: Summarise key points and implications without a heading, emphasizing insights related to the main angle.

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
            prompt += f"### Transcript from {transcript_item['source']}\n{transcript_item['content']}\n\n"

        # Using the new OpenAI API format via OpenRouter
        completion = client.chat.completions.create(
            model="openai/gpt-4o-2024-11-20",
            messages=[
                {"role": "system", "content": "You are an expert Thai crypto journalist. You know everything about crypto and cryptotrading. You know how to write and structure articles for crypto news sites."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=5000
        )
        
        # Access the response content using the new API format
        if completion.choices and len(completion.choices) > 0:
            return completion.choices[0].message.content
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
        st.session_state.keywords = "Bitcoin\nราคา BTC"
    if 'yt_url1' not in st.session_state:
        st.session_state.yt_url1 = "https://www.youtube.com/watch?v=QfgiMhP-PlU"
    if 'channel1' not in st.session_state:
        st.session_state.channel1 = "ช่อง Youtube"
    if 'yt_url2' not in st.session_state:
        st.session_state.yt_url2 = ""
    if 'channel2' not in st.session_state:
        st.session_state.channel2 = ""
    if 'text_content' not in st.session_state:
        st.session_state.text_content = ""
    if 'text_source' not in st.session_state:
        st.session_state.text_source = ""
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
        st.title("Video Transcriber")
        # Layout optimization - use columns for better space usage
        input_col, output_col = st.columns([1, 1])

        with input_col:
            # Compact input section
            with st.expander("Input Sources", expanded=True):
                # Language selection in a row with keywords
                lang_col, key_col = st.columns([1, 3])
                with lang_col:
                    language = st.selectbox(
                        "Input Language",
                        options=["English", "Thai"],
                        index=0,
                        key="language"
                    )
                with key_col:
                    keywords = st.text_area("Keywords (one per line, first keyword is primary)", 
                          help="Enter keywords, one per line. The first keyword will be used as the primary keyword.", value=st.session_state.keywords, height=70, key="keywords")
                
                # Add angle input
                angle = st.text_input("Article Angle", 
                         help="Enter the main perspective or angle you want the article to focus on (e.g., 'Impact on retail investors', 'Technical analysis perspective', 'Regulatory implications')", key="angle")

                # Add section count input
                section_count = st.number_input("Number of Content Sections", 
                                              min_value=1, 
                                              max_value=10, 
                                              value=3, 
                                              help="Specify how many main content sections (H2 headings) you want in the article",
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
                
                # Text input in a more compact layout
                text_col1, text_col2, text_col3 = st.columns([5, 2, 0.5])
                with text_col1:
                    text_content = st.text_area("Text Content", value=st.session_state.text_content, height=200, key="text_content", label_visibility="collapsed", placeholder="Or paste text content")
                with text_col2:
                    text_source = st.text_input("Source", value=st.session_state.text_source, key="text_source", label_visibility="collapsed", placeholder="Text Source")
                with text_col3:
                    st.button("X", key="X_text", on_click=lambda: [X_field("text_content"), X_field("text_source")])

                # Process button at the bottom of input section
                if st.button("Process Sources", type="primary", key="process_btn", use_container_width=True):
                    process_clicked()

        with output_col:
            # Process sources when the processing_done flag is set
            if st.session_state.processing_done:
                try:
                    fetcher = TranscriptFetcher()
                    st.session_state.transcripts = process_sources(
                        fetcher, yt_url1, channel1, yt_url2, channel2, text_content, text_source, language
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
                        st.session_state.generating = True

                        # Process text content if provided
                        if text_content.strip():
                            text_transcript = {
                                'content': text_content,
                                'source': text_source if text_source else "User Input",
                                'url': ""  # Empty URL for pasted text
                            }
                            st.session_state.transcripts = [text_transcript]
                        
                        # Process YouTube URLs if no text content
                        elif yt_url1 or yt_url2:
                            transcripts = []
                            
                            if yt_url1:
                                transcript1 = process_sources(TranscriptFetcher(), yt_url1, channel1 if channel1 else "YouTube", "", "", "", "", language)
                                if transcript1:
                                    transcripts.extend(transcript1)
                            
                            if yt_url2:
                                transcript2 = process_sources(TranscriptFetcher(), "", "", yt_url2, channel2 if channel2 else "YouTube", "", "", language)
                                if transcript2:
                                    transcripts.extend(transcript2)
                            
                            if transcripts:
                                st.session_state.transcripts = transcripts
                            else:
                                st.error("No valid transcripts found from the provided URLs")
                                st.session_state.generating = False
                                return
                        else:
                            st.error("Please provide either text content or YouTube URLs")
                            st.session_state.generating = False
                            return

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
