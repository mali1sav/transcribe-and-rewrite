import streamlit as st
st.set_page_config(page_title="Fast Transcriber", layout="wide")

import os
import tempfile
import asyncio
from openai import AsyncOpenAI
from fast_transcripts import extract_and_split_audio, transcribe_all_chunks, combine_transcripts
import yt_dlp
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled
import re

# Load environment variables
load_dotenv()

# Add custom CSS for better formatting
st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 0rem;
            margin-top: -4rem;
        }
        .stTextArea textarea {
            font-size: 0.8rem;
            line-height: 1.2;
        }
        .stTextInput input {
            font-size: 0.8rem;
        }
        .stMarkdown {
            font-size: 0.9rem;
        }
        .main > div {
            padding: 0rem 1rem;
        }
        .stButton > button {
            margin-top: 1rem;
            margin-bottom: 1rem;
        }
        div[data-testid="stToolbar"] {
            display: none;
        }
        .stProgress > div > div > div {
            height: 0.5rem;
        }
        .stProgress {
            margin-bottom: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# Check for API keys at startup
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENAI_API_KEY:
    st.error("Please ensure OPENAI_API_KEY is set in your .env file")
    st.stop()

if not OPENROUTER_API_KEY:
    st.error("Please ensure OPENROUTER_API_KEY is set in your .env file")
    st.stop()

# Initialize session state
if 'transcript' not in st.session_state:
    st.session_state.transcript = None
if 'article' not in st.session_state:
    st.session_state.article = None
if 'openrouter_client' not in st.session_state:
    try:
        st.session_state.openrouter_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
            default_headers={
                "HTTP-Referer": "https://github.com/cascade",
                "X-Title": "Fast Transcriber"
            }
        )
    except Exception as e:
        st.error(f"Error initializing OpenRouter client: {str(e)}")
        st.session_state.openrouter_client = None

async def generate_article(transcript, keywords=None):
    """Generate article from transcript using OpenRouter"""
    if not st.session_state.openrouter_client:
        st.error("OpenRouter client is not properly initialized")
        return None
    
    # Process keywords
    primary_keyword = ""
    secondary_keywords = []
    if keywords:
        keyword_list = [k.strip() for k in keywords.split('\n') if k.strip()]
        if keyword_list:
            primary_keyword = keyword_list[0]  # First keyword is primary
            secondary_keywords = keyword_list[1:] if len(keyword_list) > 1 else []  # Rest are secondary
    
    keyword_instruction = f"""Primary Keyword: {primary_keyword}
This must appear naturally ONCE in Title, Meta Description, and H1. Do not force it if it doesn't fit naturally.

Secondary Keywords: {', '.join(secondary_keywords) if secondary_keywords else 'none'}
- Only use these in H2 headings and paragraphs where they fit naturally
- Each secondary keyword should appear no more than 2 times in the entire content
- Do NOT use these in Title, Meta Description, or H1
- Skip any secondary keywords that don't fit naturally in the context"""

    prompt = f"""Based on these transcripts, write a Thai crypto news article for WordPress. Follow these keyword instructions:

{keyword_instruction}

First, write the following sections:

* Meta Description: Summarise the article in 160 characters in Thai.

* Main content: Start with a brief introduction that includes a concise attribution to the source videos. Use this format for attribution: '<a href="{transcript[0]['url']}">{transcript[0]['source']}</a>'. Then introduce the topic and its significance for Thai crypto audiences. 

* Use 3-7 distinct headings (H2) for the main content, with 2-3 paragraphs (or list items if more appropriate) under each heading.
* Important Instruction: When referencing a source, naturally integrate the Brand Name into the sentence as a clickable hyperlink.
* บทสรุป: Summarise key points and implications without a heading.

* Excerpt for WordPress: In Thai, provide 1 sentence for a brief overview.

* Image Prompt: In English, describe a scene that captures the article's essence, focus on only 1 or 2 objects. 

After writing all the above sections, analyze the key points and generate these title options:
* Title Options:
  1. News style: State the main news with a compelling hook (integrate primary keyword naturally)
  2. Question style: Ask an engaging question that addresses the main concern (integrate primary keyword naturally)
  3. Number style: Start with a number or statistic that captures attention (integrate primary keyword naturally)

Here are the transcripts to base the article on:
"""
    for transcript_item in transcript:
        prompt += f"### Transcript from {transcript_item['source']}\n{transcript_item['content']}\n\n"

    try:
        response = await st.session_state.openrouter_client.chat.completions.create(
            model="openai/gpt-4o-2024-11-20",
            messages=[
                {"role": "system", "content": "You are an expert Thai technology journalist."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=4000
        )
        
        # Debug information
        with st.expander("Debug Information", expanded=False):
            st.write("Debug - Response:", response)
        
        # Safely access response content
        if response and response.choices:
            return response.choices[0].message.content
        else:
            st.error("No valid response content received")
            return None
            
    except Exception as e:
        st.error(f"Error generating article: {str(e)}")
        if hasattr(e, 'response'):
            st.error(f"API Response: {e.response}")
        return None

async def transcribe_chunk(client, audio_chunk_path):
    """Transcribe a single audio chunk using OpenAI's Whisper API"""
    try:
        with open(audio_chunk_path, 'rb') as audio_file:
            response = await client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
            return response
    except Exception as e:
        st.error(f"Error transcribing chunk: {str(e)}")
        return ""

async def download_youtube_audio(url):
    """Download audio from YouTube video"""
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            output_path = temp_file.name

        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'outtmpl': output_path,
            'progress_hooks': [update_progress],
            'quiet': True,
            'no_warnings': True
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            
        return output_path
    except Exception as e:
        st.error(f"Error downloading audio: {str(e)}")
        return None

async def process_youtube_url(url, source_name, progress_placeholder):
    """Process a YouTube URL to get transcription"""
    try:
        # Extract video ID from URL
        video_id = None
        if 'youtu.be' in url:
            video_id = url.split('/')[-1].split('?')[0]
        elif 'youtube.com' in url:
            video_id = re.search(r'v=([^&]+)', url).group(1)
        
        if not video_id:
            progress_placeholder.error(f"Could not extract video ID from URL: {url}")
            return None
            
        # Update progress
        progress_placeholder.progress(0.2, text=f"Processing {source_name}: Checking for YouTube transcript...")
            
        # First try to get transcript using YouTube API
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
            # Combine all transcript pieces into one text
            transcript_text = ' '.join([entry['text'] for entry in transcript_list])
            progress_placeholder.progress(1.0, text=f"✅ Completed {source_name} using YouTube transcript")
            return {"type": "youtube", "url": url, "source": source_name, "content": transcript_text}
            
        except (NoTranscriptFound, TranscriptsDisabled) as e:
            progress_placeholder.warning(f"No YouTube transcript available for {source_name}. Falling back to audio download...")
            progress_placeholder.progress(0.3, text=f"Processing {source_name}: Downloading audio...")
            
            # Fall back to audio download and transcription
            audio_path = await download_youtube_audio(url)
            if not audio_path:
                progress_placeholder.error(f"Failed to download audio from {source_name}")
                return None
                
            progress_placeholder.progress(0.6, text=f"Processing {source_name}: Transcribing audio...")
            transcript = await transcribe_audio(audio_path)
            if not transcript:
                progress_placeholder.error(f"Failed to transcribe audio from {source_name}")
                return None
                
            progress_placeholder.progress(1.0, text=f"✅ Completed {source_name}")
            return {"type": "youtube", "url": url, "source": source_name, "content": transcript}
                
    except Exception as e:
        progress_placeholder.error(f"Error processing {source_name}: {str(e)}")
        return None

def progress_update(d, progress_bar, progress_text):
    if d['status'] == 'downloading':
        p = d.get('downloaded_bytes', 0) / d.get('total_bytes', 100)
        progress_bar.progress(min(p, 1.0))
        progress_text.text(f"Downloading: {p*100:.1f}%")
    elif d['status'] == 'finished':
        progress_bar.progress(1.0)
        progress_text.text("Download finished, extracting audio...")

def update_progress(d):
    """Update download progress in session state"""
    if d['status'] == 'downloading':
        p = d.get('downloaded_bytes', 0) / d.get('total_bytes', 100)
        st.session_state.download_progress = min(p, 1.0)
        st.session_state.download_message = f"Downloading: {p*100:.1f}%"
    elif d['status'] == 'finished':
        st.session_state.download_progress = 1.0
        st.session_state.download_message = "Download finished, extracting audio..."

def main():
    # Initialize session state
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'article_generated' not in st.session_state:
        st.session_state.article_generated = False
    if 'sources' not in st.session_state:
        st.session_state.sources = []
    
    # Initialize OpenRouter client
    if 'openrouter_client' not in st.session_state:
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
            return
        st.session_state.openrouter_client = AsyncOpenAI(
            api_key=openai_api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        
    st.markdown("## Fast Transcriber", help="Generate articles from YouTube videos and web content")
    
    # Layout optimization - use columns for better space usage
    input_col, output_col = st.columns([1, 1])
    
    with input_col:
        # Compact input section
        st.markdown("### Input Sources")
        keywords = st.text_area("Keywords (one per line)", value="Bitcoin\nราคา BTC/USD", height=80, key="keywords")
        
        # YouTube Source 1 - more compact layout
        yt1_col1, yt1_col2 = st.columns([2, 1])
        with yt1_col1:
            yt_url1 = st.text_input("YouTube URL 1", value="https://www.youtube.com/watch?v=6I6-7pMhuRM", key="yt_url1")
        with yt1_col2:
            channel1 = st.text_input("Channel 1", value="InvestAnswers", key="channel1")
        
        # YouTube Source 2 - more compact layout
        yt2_col1, yt2_col2 = st.columns([2, 1])
        with yt2_col1:
            yt_url2 = st.text_input("YouTube URL 2 (Optional)", key="yt_url2")
        with yt2_col2:
            channel2 = st.text_input("Channel 2", key="channel2")
        
        # Webpage Source - more compact layout
        web_col1, web_col2 = st.columns([2, 1])
        with web_col1:
            webpage_url = st.text_input("Webpage URL (Optional)", key="webpage_url")
        with web_col2:
            webpage_source = st.text_input("Source", key="webpage_source")
        webpage_content = st.text_area("Or paste webpage content", height=80, key="webpage_content")
        
        # Collect sources
        sources = []
        if yt_url1 and channel1:
            sources.append({"type": "youtube", "url": yt_url1, "source": channel1})
        if yt_url2 and channel2:
            sources.append({"type": "youtube", "url": yt_url2, "source": channel2})
        if webpage_url and webpage_source:
            sources.append({"type": "webpage", "url": webpage_url, "source": webpage_source})
        elif webpage_content and webpage_source:
            sources.append({"type": "text", "content": webpage_content, "source": webpage_source})
        
        st.session_state.sources = sources
        
        if st.button("Process Sources", type="primary", use_container_width=True):
            st.session_state.processed = False
            st.session_state.article_generated = False
            
            if not sources:
                st.error("Please add at least one source")
            else:
                try:
                    all_transcripts = []
                    total_sources = len(sources)
                    
                    # Progress tracking section - moved up
                    progress_section = st.container()
                    with progress_section:
                        # Overall progress
                        overall_progress = st.progress(0.0, text="Starting processing...")
                        
                        # Individual progress bars
                        progress_containers = []
                        for source in sources:
                            container = st.empty()
                            progress_containers.append(container)
                        
                        for idx, (source, progress_container) in enumerate(zip(sources, progress_containers)):
                            with progress_container:
                                progress_bar = st.empty()
                                
                                if source["type"] == "youtube":
                                    transcript = asyncio.run(process_youtube_url(
                                        source["url"], 
                                        source["source"],
                                        progress_bar
                                    ))
                                    if transcript:
                                        all_transcripts.append(transcript)
                                        
                                elif source["type"] in ["webpage", "text"]:
                                    progress_bar.progress(1.0, text=f"✅ Added content from {source['source']}")
                                    all_transcripts.append({
                                        "content": source.get("content", ""),
                                        "source": source["source"],
                                        "url": source.get("url", "")
                                    })
                            
                            overall_progress.progress((idx + 1) / total_sources, 
                                                   text=f"Processed {idx + 1} of {total_sources} sources")
                        
                        if all_transcripts:
                            st.session_state.all_transcripts = all_transcripts
                            st.session_state.processed = True
                            overall_progress.progress(1.0, text="✅ All sources processed successfully!")
                        else:
                            st.error("No transcripts were generated. Please check the errors above.")
                            
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
    
    with output_col:
        if st.session_state.processed:
            if st.button("Generate Article", type="primary", use_container_width=True):
                with st.spinner("Generating article..."):
                    async def generate():
                        return await generate_article(st.session_state.all_transcripts, keywords=keywords)
                    
                    article = asyncio.run(generate())
                    if article:
                        st.session_state.article = article
                        st.session_state.article_generated = True
                    else:
                        st.error("Failed to generate article. Please try again.")
                    
        if st.session_state.article_generated:
            # Extract and display title options first
            article_text = st.session_state.article
            title_section = ""
            content_section = ""
            
            # Split article into title options and content
            if "* Title Options:" in article_text:
                parts = article_text.split("* Title Options:")
                if len(parts) > 1:
                    content_section = parts[0]
                    title_section = "* Title Options:" + parts[1].split("* Meta Description:")[0]
                    remaining_content = "* Meta Description:" + parts[1].split("* Meta Description:")[1]
                    
                    # Display title options first
                    st.markdown("### Title Options")
                    st.markdown(title_section)
                    
                    # Then display the rest of the content
                    st.markdown("### Article Content")
                    st.markdown(remaining_content)
            else:
                # Fallback if structure is different
                st.markdown(article_text, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
