import os
import re
import logging
from pathlib import Path
from typing import Dict, Optional
import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from openai import OpenAI
from together import Together  # For image generation
import markdown  # Added to convert markdown to HTML
import requests
from contextlib import suppress

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
    # Optional YouTube cookies file (Netscape format) and proxies
    YT_COOKIES_PATH = os.getenv('YT_COOKIES_PATH')  # e.g., /path/to/cookies.txt
    HTTP_PROXY = os.getenv('HTTP_PROXY')
    HTTPS_PROXY = os.getenv('HTTPS_PROXY')
    # Optional Firecrawl API key
    FIRECRAWL_API_KEY = os.getenv('FIRECRAWL_API_KEY')

# Initialize OpenAI client with OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=Config.OPENROUTER_API_KEY
)

class TranscriptFetcher:
    def get_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL."""
        try:
            # Support common formats: full, short, shorts, embed
            patterns = [
                r"(?:v=)([0-9A-Za-z_-]{11})",                   # https://www.youtube.com/watch?v=VIDEOID
                r"youtu\.be\/([0-9A-Za-z_-]{11})",            # https://youtu.be/VIDEOID
                r"youtube\.com\/shorts\/([0-9A-Za-z_-]{11})",# https://www.youtube.com/shorts/VIDEOID
                r"youtube\.com\/embed\/([0-9A-Za-z_-]{11})"  # https://www.youtube.com/embed/VIDEOID
            ]
            for pat in patterns:
                m = re.search(pat, url)
                if m:
                    return m.group(1)
            # Fallback: last 11-char token
            m = re.search(r"([0-9A-Za-z_-]{11})(?:[&?#].*)?$", url)
            return m.group(1) if m else None
        except Exception as e:
            logger.error(f"Error extracting video ID: {str(e)}")
            return None

    def get_transcript(self, url: str, language: str = "Auto-detect") -> Optional[Dict]:
        """Get transcript from YouTube with enhanced error handling and retries."""
        import time
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api._errors import (
            TranscriptsDisabled,
            NoTranscriptFound,
            VideoUnavailable,
            TooManyRequests,
        )

        video_id = self.get_video_id(url)
        if not video_id:
            logger.error(f"Could not extract video ID from URL: {url}")
            return None

        # Preferred language codes
        preferred = {
            "Thai": ["th"],
            "English": ["en"],
        }.get(language, None)

        for attempt in range(3):
            try:
                if attempt > 0:
                    # Exponential backoff on retry
                    time.sleep(2 ** attempt)

                # Use transcript listing API for fine-grained control
                # Pass proxies/cookies if available
                proxies = None
                if Config.HTTP_PROXY or Config.HTTPS_PROXY:
                    proxies = {}
                    if Config.HTTP_PROXY:
                        proxies['http'] = Config.HTTP_PROXY
                    if Config.HTTPS_PROXY:
                        proxies['https'] = Config.HTTPS_PROXY

                # list_transcripts accepts proxies and cookies path in recent versions
                with suppress(TypeError):
                    tlist = YouTubeTranscriptApi.list_transcripts(video_id, proxies=proxies, cookies=Config.YT_COOKIES_PATH)
                if 'tlist' not in locals():
                    # Fallback without extras if running older library version
                    tlist = YouTubeTranscriptApi.list_transcripts(video_id)

                def fetch_segments(t):
                    try:
                        return t.fetch()
                    except Exception as fe:
                        logger.warning(f"Failed to fetch transcript object: {fe}")
                        return None

                segments = None

                # 1) Try preferred language directly (manual first, then generated)
                if preferred:
                    try:
                        t = tlist.find_manually_created_transcript(preferred)
                        segments = fetch_segments(t)
                    except Exception:
                        pass
                    if segments is None:
                        try:
                            t = tlist.find_generated_transcript(preferred)
                            segments = fetch_segments(t)
                        except Exception:
                            pass

                # 2) Try translate from any available to preferred
                if segments is None and preferred:
                    for t in tlist:
                        try:
                            tt = t.translate(preferred[0])
                            segments = fetch_segments(tt)
                            if segments:
                                break
                        except Exception:
                            continue

                # 3) Fallback: fetch any available transcript (first successful)
                if segments is None:
                    for t in tlist:
                        segments = fetch_segments(t)
                        if segments:
                            break

                if segments:
                    full_text = " ".join(s.get('text', '').strip() for s in segments if s.get('text'))
                    logger.info(f"Successfully got YouTube transcript for video {video_id}")
                    return {
                        'text': full_text,
                        'segments': segments,
                        'language': language,
                        'method': 'yt-api',
                    }

                # If we reach here, no segments could be fetched
                logger.warning(f"No transcript segments available for video {video_id} (attempt {attempt+1}/3)")

            except TranscriptsDisabled:
                logger.error(f"Transcripts are disabled for video {video_id}")
                # Try yt-dlp fallback
                return self.get_transcript_via_ytdlp(video_id, language)
            except NoTranscriptFound:
                logger.error(f"No transcript found for video {video_id}")
                # Try yt-dlp fallback
                return self.get_transcript_via_ytdlp(video_id, language)
            except VideoUnavailable as e:
                logger.error(f"Video {video_id} is unavailable: {str(e)}")
                return None
            except TooManyRequests:
                logger.warning(f"Rate limited when fetching transcript for video {video_id} (attempt {attempt+1}/3)")
                continue
            except Exception as e:
                logger.error(f"Unexpected error getting transcript for video {video_id}: {str(e)}")
                continue

        logger.error(f"Failed to get transcript for video {video_id} after all attempts")
        # Try Firecrawl (if configured), then yt-dlp
        result = self.get_transcript_via_firecrawl(video_id, language)
        if result:
            return result
        return self.get_transcript_via_ytdlp(video_id, language)

    def get_transcript_via_firecrawl(self, video_id: str, language: str) -> Optional[Dict]:
        """Fallback using Firecrawl API to scrape youtubetranscript.com."""
        if not Config.FIRECRAWL_API_KEY:
            return None
        try:
            target_url = f"https://youtubetranscript.com/?server_vid2={video_id}"
            headers = {
                'Authorization': f'Bearer {Config.FIRECRAWL_API_KEY}',
                'Content-Type': 'application/json'
            }
            payload = {
                'url': target_url,
                'formats': ['html'],
                'onlyMainContent': True,
                'waitFor': 2000
            }
            resp = requests.post('https://api.firecrawl.dev/v1/scrape', headers=headers, json=payload, timeout=20)
            if resp.status_code != 200:
                logger.warning(f"Firecrawl scrape failed ({resp.status_code}): {resp.text[:200]}")
                return None
            data = resp.json()
            html = ''
            if isinstance(data, dict):
                html = data.get('data', {}).get('html') or data.get('html') or ''
            if not html:
                logger.warning("Firecrawl returned no HTML content")
                return None
            # crude text extraction from transcript html
            # youtubetranscript.com returns a simple HTML with transcript lines
            text = re.sub(r'<[^>]+>', ' ', html)
            text = re.sub(r'\s+', ' ', text).strip()
            if not text:
                return None
            logger.info(f"Successfully retrieved transcript via Firecrawl for video {video_id}")
            return {
                'text': text,
                'segments': [],
                'language': language,
                'method': 'firecrawl',
            }
        except Exception as e:
            logger.warning(f"Firecrawl fallback error: {e}")
            return None

    def get_transcript_via_ytdlp(self, video_id: str, language: str) -> Optional[Dict]:
        """Fallback: use yt-dlp to fetch subtitles/auto-captions and parse to text."""
        try:
            import yt_dlp
        except Exception as e:
            logger.error(f"yt-dlp not available for fallback: {e}")
            return None

        preferred = {
            "Thai": ["th"],
            "English": ["en"],
        }.get(language, ["th", "en"])  # try both when auto

        ydl_opts = {
            'quiet': True,
            'skip_download': True,
            # Try to fetch both manually provided subtitles and auto-captions
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': preferred + ['en', 'th'],
        }
        # Proxy support
        if Config.HTTPS_PROXY or Config.HTTP_PROXY:
            # yt-dlp uses 'proxy' param
            ydl_opts['proxy'] = Config.HTTPS_PROXY or Config.HTTP_PROXY
        # Cookies support
        if Config.YT_COOKIES_PATH and Path(Config.YT_COOKIES_PATH).exists():
            ydl_opts['cookiefile'] = Config.YT_COOKIES_PATH

        url = f"https://www.youtube.com/watch?v={video_id}"
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
        except Exception as e:
            logger.error(f"yt-dlp extract failed: {e}")
            return None

        # Collect caption entries from subtitles or automatic_captions
        caption_sources = []
        for key in ('subtitles', 'automatic_captions'):
            caps = info.get(key) or {}
            for lang, entries in caps.items():
                caption_sources.append((lang, entries))

        # Rank selection by preferred languages and format preference
        def score(lang: str) -> int:
            if not preferred:
                return 100
            try:
                idx = preferred.index(lang.split('-')[0])
                return idx
            except ValueError:
                return 100

        # Try best-matching language first
        caption_sources.sort(key=lambda x: score(x[0]))

        session = requests.Session()
        proxies = {}
        if Config.HTTP_PROXY:
            proxies['http'] = Config.HTTP_PROXY
        if Config.HTTPS_PROXY:
            proxies['https'] = Config.HTTPS_PROXY
        if proxies:
            session.proxies.update(proxies)

        for lang, entries in caption_sources:
            # Prefer .vtt or .srv3 if available
            sorted_entries = sorted(entries, key=lambda e: (e.get('ext') != 'vtt', e.get('ext') != 'srv3'))
            for entry in sorted_entries:
                url = entry.get('url')
                if not url:
                    continue
                try:
                    r = session.get(url, timeout=15)
                    if r.status_code != 200 or not r.text.strip():
                        continue
                    raw = r.text
                    text = self._parse_captions(raw, entry.get('ext'))
                    if text:
                        logger.info(f"Successfully retrieved captions via yt-dlp for video {video_id} (lang={lang})")
                        return {
                            'text': text,
                            'segments': [],
                            'language': language,
                            'method': 'yt-dlp',
                        }
                except Exception as e:
                    logger.warning(f"Failed to download/parse caption file ({entry.get('ext')}): {e}")
                    continue

        logger.error(f"No usable captions found via yt-dlp for video {video_id}")
        return None

    def _parse_captions(self, raw: str, ext: Optional[str]) -> str:
        """Basic parser for VTT/SBV/SRV caption formats -> plain text."""
        try:
            if ext == 'vtt' or (raw.startswith('WEBVTT')):
                # Strip WEBVTT headers and timestamps
                lines = []
                for line in raw.splitlines():
                    if not line.strip():
                        continue
                    # skip headers, timestamps, and sequence numbers
                    if line.startswith('WEBVTT') or '-->' in line or line.strip().isdigit():
                        continue
                    # remove HTML tags
                    clean = re.sub(r'<[^>]+>', '', line)
                    lines.append(clean)
                return ' '.join(lines).strip()
            else:
                # Fallback: remove timestamps and tags heuristically
                no_tags = re.sub(r'<[^>]+>', ' ', raw)
                no_ts = re.sub(r"\d{1,2}:\d{2}:\d{2}\.\d{3} --> .*", ' ', no_tags)
                no_nums = re.sub(r"^\s*\d+\s*$", ' ', no_ts, flags=re.MULTILINE)
                collapsed = re.sub(r"\s+", ' ', no_nums)
                return collapsed.strip()
        except Exception as e:
            logger.warning(f"Caption parse error: {e}")
            return ''

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
                        text = (transcript.get('text') or '').strip()
                        # Heuristic: skip known block/placeholder pages from third-party scrapers
                        block_phrases = [
                            "we're sorry, youtube is currently blocking us",
                            "preventing us from generating a summary",
                        ]
                        if any(p in text.lower() for p in block_phrases):
                            st.warning(f"{source['channel']}: Transcript appears blocked/invalid and was skipped.")
                        elif len(text) < 40:  # too short to be useful
                            st.warning(f"{source['channel']}: Transcript is too short and was skipped.")
                        else:
                            transcripts.append({
                                'source': source['channel'],
                                'url': source['url'],
                                'content': text,
                                'method': transcript.get('method', 'unknown'),
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

    # Process direct text input
    if text_content:
        transcripts.append({
            'source': 'Article',
            'content': text_content
        })
        st.success("✅ Added pasted content")

    return transcripts

def generate_article(client: OpenAI, transcripts, keywords=None, language=None, angle=None, section_count=3):
    """
    Generate an article from transcripts using OpenRouter.
    The article must be written entirely in Thai (except technical terms and proper names remain in English).
    The image prompt must be in English only. 
    All source attributions must be included as markdown hyperlinks (e.g. ([Source Name](URL))) only once in the introduction.
    """
    try:
        if not transcripts:
            return None

        # Process keywords
        keyword_list = []
        if keywords:
            keyword_list = [k.strip() for k in keywords.split('\n') if k.strip()]
        primary_keyword = keyword_list[0].upper() if keyword_list else ""
        
        # Shortcode mapping (for promotional integration if applicable)
        shortcode_map = {
            "BITCOIN": '[latest_articles label="ข่าว Bitcoin (BTC) ล่าสุด" count_of_posts="6" taxonomy="category" term_id="7"]',
            "ETHEREUM": '[latest_articles label="ข่าว Ethereum ล่าสุด" count_of_posts="6" taxonomy="category" term_id="8"]',
            "SOLANA": '[latest_articles label="ข่าว Solana ล่าสุด" count_of_posts="6" taxonomy="category" term_id="501"]',
            "XRP": '[latest_articles label="ข่าว XRP ล่าสุด" count_of_posts="6" taxonomy="category" term_id="502"]',
            "DOGECOIN": '[latest_articles label="ข่าว Dogecoin ล่าสุด" count_of_posts="6" taxonomy="category" term_id="527"]'
        }

        # Keyword instructions (example)
        keyword_instruction = f"""Primary Keyword: {keyword_list[0] if keyword_list else ""}
Secondary Keywords: {', '.join(keyword_list[1:]) if len(keyword_list) > 1 else 'none'}
- The primary keyword must appear naturally ONCE in the Title, Meta Description, and H1.
- Secondary keywords should be used sparingly and naturally within the content."""
        
        # Angle instruction
        angle_instruction = ""
        if angle:
            angle_instruction = f"""Main Article Angle: {angle}
- Structure the analysis and narrative around this perspective.
- Ensure every section supports this angle."""
        
        # Attribution instructions
        attribution_instruction = ("Include concise source attributions using markdown hyperlinks "
                                   "(for example, ([CryptoQuant](https://www.cryptoquant.com))) only once in the article’s introduction.")
        
        # Build the prompt
        prompt = f"""
Write a comprehensive and in-depth Thai crypto news article for WordPress. Align to {angle_instruction} if provided.
Ensure the entire article is written in Thai, except that all technical terms and proper names (including names of people, places, organizations, platforms, coin names, etc.) must remain in English.
Make the narrative natural and engaging. All explanations, analysis, and commentary should be in Thai.
Follow these guidelines:
- Use the keyword instructions below:
{keyword_instruction}
- When creating section headings (H2) and subheadings (H3):
  1. Use power words and include specific numbers or statistics where relevant.
  2. Create curiosity gaps and bold statements.
- Include concise source attributions only once in the introduction. {attribution_instruction}
- For monetary values, remove any "$" symbols and append "ดอลลาร์" with proper spacing.
- Create exactly {section_count} distinct and engagingH2 sections; for each section provide 3-4 paragraphs of in-depth explanation.
- At the end, include an Excerpt for WordPress (1 sentence in Thai) and an Image Prompt.
- The Image Prompt must be in English only. Create a photorealistic scene that fits the main news article, focusing on 1-2 main objects. Keep it simple and clear. Avoid charts, graphs, or technical diagrams as they don't work well with image generation.
- Always escapes special characters to avoid Streamlit rendering issues. 

 CRITICAL SOURCE POLICY:
 - Use ONLY the transcripts provided below as source material. Do NOT invent facts or speculate beyond what the transcripts state.
 - If the transcripts are missing, blocked, or insufficient to write the article, respond with exactly: INS U F F I C I E N T _ S O U R C E _ C O N T E N T (without spaces) and nothing else.

Now, generate markdown article with the following structure:
[Title includes {primary_keyword}, ensure it's engaging and is new-like headline.]
[Meta Description includes {primary_keyword}, ensure it's click-worthy and stimulates curiosity]
[Main Content in {section_count} sections]
[Slug URL in English includes {primary_keyword}, translate Thai keywords to English]
[Image ALT Text in Thai includes {primary_keyword}. Keep technical terms and entity names in English, rest in Thai]
[Image Prompt]
Here are the transcripts to base the article on:
"""
        for transcript_item in transcripts:
            prompt += f"### Transcript from {transcript_item['source']}\n{transcript_item['content']}\n\n"

        # Expose prompt for debugging in the UI
        try:
            st.session_state.last_llm_prompt = prompt
        except Exception:
            pass

        completion = client.chat.completions.create(
            model="google/gemini-2.0-flash-001",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        
        if completion.choices and len(completion.choices) > 0:
            content = completion.choices[0].message.content
            # If model indicates insufficient sources, stop
            if content and 'INSUFFICIENT_SOURCE_CONTENT' in content.replace(" ", "").upper():
                st.error("Generation halted: insufficient/blocked source content. Please provide a valid transcript or pasted text.")
                return None
            # Insert shortcode if applicable based on primary keyword
            shortcode = shortcode_map.get(primary_keyword, "")
            if shortcode:
                parts = content.split("Excerpt for WordPress:", 1)
                if len(parts) == 2:
                    content = parts[0].rstrip() + "\n\n" + shortcode + "\n\n----------------\n\nExcerpt for WordPress:" + parts[1]
                else:
                    content += f"\n\n{shortcode}\n"
            # Post-processing: Convert monetary values ($amount) to "amount ดอลลาร์"
            content = re.sub(r"\$(\d[\d,\.]*)", r"\1 ดอลลาร์", content)
            return content
        else:
            logger.error("No content in completion response")
            st.error("No valid response content received")
            return None

    except Exception as e:
        logger.error(f"Error generating article: {str(e)}")
        st.error(f"Error generating article: {str(e)}")
        return None

def extract_image_prompt(article_text):
    """Extract the Image Prompt from the generated article using a more robust pattern."""
    # Try multiple possible patterns
    patterns = [
        r"(?i)^Image Prompt\s*[:\-]\s*(.+)$",  # Standard format
        r"(?i)\[Image Prompt\]\s*(.+)(?:\n|$)",  # Markdown format
        r"(?i)Image Prompt[:\-]\s*(.+)(?:\n|$)"  # Flexible format
    ]
    
    for pattern in patterns:
        match = re.search(pattern, article_text, re.MULTILINE)
        if match:
            prompt = match.group(1).strip()
            if prompt:
                logging.info(f"Found image prompt: {prompt}")
                return prompt
            
    # If no prompt is found, log the issue
    logging.warning("No image prompt found in article text. Using fallback prompt.")
    logging.debug(f"Article text snippet: {article_text[:500]}...")
    return "A futuristic digital scene depicting cryptocurrency market trends with vibrant colors and dynamic motion."

def extract_alt_text(article_text):
    """Extract the Image ALT text from the generated article using the marker 'Image ALT Text:'"""
    pattern = r"(?i)Image ALT Text.*?\n(.*?)(?:\n\n|\Z)"
    match = re.search(pattern, article_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def X_field(field_name):
    st.session_state[field_name] = ""

def main():
    # Initialize session state for input fields if not exists
    if 'keywords' not in st.session_state:
        st.session_state.keywords = ""
    if 'yt_url1' not in st.session_state:
        st.session_state.yt_url1 = ""
    if 'channel1' not in st.session_state:
        st.session_state.channel1 = ""
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
        st.session_state.angle = ""
    if 'transcripts' not in st.session_state:
        st.session_state.transcripts = []
    if 'article' not in st.session_state:
        st.session_state.article = None
    if 'processing_done' not in st.session_state:
        st.session_state.processing_done = False
    if 'generating' not in st.session_state:
        st.session_state.generating = False
    if 'last_llm_prompt' not in st.session_state:
        st.session_state.last_llm_prompt = ""

    def process_clicked():
        st.session_state.processing_done = True

    def generate_clicked():
        st.session_state.generating = True

    st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
    st.markdown("""
        <style>
        .main { max-width: 100% !important; padding: 0rem; }
        .stButton button { width: 100%; }
        div[data-testid="stVerticalBlock"] { gap: 0.3rem !important; }
        header, #MainMenu, footer, div[data-testid="stToolbar"] { visibility: hidden; }
        .block-container { padding-top: 1rem; padding-bottom: 0rem; }
        h1 { margin-top: -1rem; }
        </style>
    """, unsafe_allow_html=True)

    with st.container():
        st.title("Transcribe and Rewrite")
        input_col, output_col = st.columns([1, 1])

        with input_col:
            with st.expander("Input Sources", expanded=True):
                # Language selector at the very top
                language = st.selectbox("Language", options=["Thai", "English"], index=1, key="language")

                # Single YouTube URL field at the top
                yt_url1 = st.text_input("YouTube URL", value=st.session_state.yt_url1, key="yt_url1", placeholder="https://www.youtube.com/watch?v=...")

                # Keywords and Article Angle below URL
                keywords = st.text_area(
                    "Keywords (one per line; first is primary)",
                    help="Enter keywords; the first keyword is primary.",
                    height=70,
                    key="keywords",
                )
                angle = st.text_input(
                    "Article Angle",
                    help="Enter the main angle for the article.",
                    key="angle",
                )

                # Section count
                section_count = st.number_input(
                    "Number of Content Sections",
                    min_value=1,
                    max_value=10,
                    value=4,
                    help="Number of H2 sections in the article.",
                    key="section_count",
                )

                # Optional pasted text content
                text_col1, text_col3 = st.columns([6, 0.5])
                with text_col1:
                    text_content = st.text_area(
                        "Text Content (optional)",
                        value=st.session_state.text_content,
                        height=200,
                        key="text_content",
                        placeholder="Or paste text content (supports Markdown with references)",
                    )
                with text_col3:
                    st.button("X", key="X_text", on_click=lambda: X_field("text_content"))

                if st.button("Process Sources", type="primary", key="process_btn", use_container_width=True):
                    process_clicked()

        with output_col:
            if st.session_state.processing_done:
                try:
                    fetcher = TranscriptFetcher()
                    # Use a single URL; use a default channel label
                    channel1 = "YouTube"
                    yt_url2 = ""
                    yt_url3 = ""
                    channel2 = ""
                    channel3 = ""
                    st.session_state.transcripts = process_sources(
                        fetcher, yt_url1, channel1, yt_url2, channel2, yt_url3, channel3, text_content, language
                    )
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    logger.error(f"Error processing sources: {str(e)}")
                st.session_state.processing_done = False

            if st.session_state.transcripts:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Generate Article", type="primary", key="generate_btn", use_container_width=True):
                        generate_clicked()
                with col2:
                    combined_text = "\n\n".join([f"=== {t['source']} ===\n{t['content']}" for t in st.session_state.transcripts])
                    st.download_button(label="Download Transcripts", data=combined_text, file_name="transcripts.txt", mime="text/plain", use_container_width=True)

                # Show extraction method info clearly
                methods = {t.get('method', 'unknown') for t in st.session_state.transcripts}
                method_label = ", ".join(sorted(methods))
                if 'firecrawl' in methods:
                    st.info(f"Transcript extraction method: {method_label} (Firecrawl used)")
                else:
                    st.success(f"Transcript extraction method: {method_label}")

                # Collapsible box to display the LLM prompt that will be/was sent
                with st.expander("Debug: LLM Prompt Sent", expanded=False):
                    if st.session_state.last_llm_prompt:
                        st.text_area("Prompt", st.session_state.last_llm_prompt, height=300)
                    else:
                        st.caption("Prompt will appear here after you click 'Generate Article'.")

            if st.session_state.generating:
                with st.spinner("Generating article..."):
                    article = generate_article(client, st.session_state.transcripts, keywords, language, angle, section_count)
                    if article:
                        st.session_state.article = article
                    else:
                        st.error("Failed to generate article. Please try again.")
                st.session_state.generating = False

            if st.session_state.article:
                # Convert the generated markdown article to HTML for fully rendered output
                html_article = markdown.markdown(st.session_state.article)
                
                # Add custom CSS for styling the rendered HTML
                st.markdown("""
                    <style>
                    .markdown-content {
                        padding: 1rem;
                        background: white;
                        border-radius: 5px;
                    }
                    .markdown-content h1 { 
                        font-size: 2em;
                        margin: 1.2em 0 0.6em 0;
                        color: #1e1e1e;
                    }
                    .markdown-content h2 { 
                        font-size: 1.5em;
                        margin: 1em 0 0.5em 0;
                        color: #2c2c2c;
                        border-bottom: 1px solid #eee;
                        padding-bottom: 0.3em;
                    }
                    .markdown-content h3 { 
                        font-size: 1.2em;
                        margin: 0.8em 0 0.4em 0;
                        color: #3c3c3c;
                    }
                    .markdown-content p { 
                        margin: 0 0 1em 0;
                        line-height: 1.6;
                    }
                    .markdown-content ul { 
                        margin: 0 0 1em 0;
                        padding-left: 2em;
                    }
                    .markdown-content li { 
                        margin: 0.5em 0;
                        line-height: 1.4;
                    }
                    .markdown-content a {
                        color: #0366d6;
                        text-decoration: none;
                    }
                    .markdown-content a:hover {
                        text-decoration: underline;
                    }
                    </style>
                """, unsafe_allow_html=True)
                
                # Create a container with the custom CSS class and render the HTML article
                with st.container():
                    st.markdown('<div class="markdown-content">' + html_article + '</div>', unsafe_allow_html=True)
                
                # Download button now provides the fully rendered HTML version
                st.download_button(label="Download Article (HTML)", data=html_article, file_name="generated_article.html", mime="text/html", use_container_width=True)
                
                # Together AI Image Generation
                image_prompt = extract_image_prompt(st.session_state.article)
                alt_text = extract_alt_text(st.session_state.article)
                st.write("Extracted Image Prompt:", image_prompt)  # Debug output to verify prompt extraction
                if image_prompt:
                    st.info("Generating image from Together AI...")
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
                        st.write("Together API Response:", response)  # Debug output to inspect API response
                        if response and response.data and len(response.data) > 0:
                            b64_data = response.data[0].b64_json
                            st.image("data:image/png;base64," + b64_data, caption=alt_text or "Generated image")
                        else:
                            st.error("Failed to generate image from Together AI.")
                    except Exception as e:
                        st.error(f"Error generating image: {str(e)}")

if __name__ == "__main__":
    main()
