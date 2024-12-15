import streamlit as st
import yt_dlp
import openai
from pathlib import Path
import logging
import os
from dotenv import load_dotenv
import shutil
import asyncio
from typing import Dict, Optional

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    BASE_DIR = Path(__file__).parent
    TEMP_DIR = BASE_DIR / "temp"
    TEMP_DIR.mkdir(exist_ok=True)

class VideoDownloader:
    def __init__(self, temp_dir: Path):
        self.temp_dir = temp_dir
        self.ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            'outtmpl': str(temp_dir / '%(title)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True
        }

    async def download(self, url: str) -> Optional[Dict]:
        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                return {
                    'title': info['title'],
                    'path': self.temp_dir / f"{info['title']}.wav",
                    'duration': info.get('duration'),
                    'url': url
                }
        except Exception as e:
            logger.error(f"Error downloading video: {str(e)}")
            raise

class Transcriber:
    def __init__(self, api_key: str):
        self.client = openai.Client(api_key=api_key)

    async def transcribe(self, audio_path: Path) -> Dict:
        try:
            with open(audio_path, 'rb') as audio_file:
                response = await self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json"
                )
                return {
                    'text': response.text,
                    'segments': response.segments,
                    'language': response.language
                }
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            raise

def cleanup_temp_files():
    try:
        shutil.rmtree(Config.TEMP_DIR)
        Config.TEMP_DIR.mkdir(exist_ok=True)
    except Exception as e:
        logger.error(f"Error cleaning up temp files: {str(e)}")

# Initialize components
if not Config.OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

downloader = VideoDownloader(Config.TEMP_DIR)
transcriber = Transcriber(Config.OPENAI_API_KEY)

# Streamlit UI
st.title("Fast Video Transcriber")

url = st.text_input("Enter YouTube URL:")
if st.button("Transcribe"):
    try:
        with st.spinner("Downloading video..."):
            video_info = asyncio.run(downloader.download(url))
            
        with st.spinner("Transcribing..."):
            transcription = asyncio.run(transcriber.transcribe(video_info['path']))
            
        st.success("Transcription completed!")
        st.write(transcription['text'])
        
        # Cleanup
        cleanup_temp_files()
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.error(f"Error processing video: {str(e)}")