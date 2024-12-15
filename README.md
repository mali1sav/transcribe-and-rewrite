# Fast Video Transcriber

A Streamlit application that downloads YouTube videos and transcribes them using OpenAI's Whisper API.

## Features

- Download audio from YouTube videos using yt-dlp
- Transcribe audio using OpenAI's Whisper API
- Clean and simple Streamlit interface
- Automatic cleanup of temporary files

## Requirements

- Python 3.8+
- FFmpeg (system requirement)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mali1sav/transcribe-and-rewrite.git
cd transcribe-and-rewrite
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with your API keys:
```bash
cp .env.example .env
```
Then edit `.env` with your actual API keys.

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Enter a YouTube URL in the input field
3. Click "Transcribe" to start the process
4. The transcription will appear once complete

## Environment Variables

Required environment variables in `.env`:
- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENROUTER_API_KEY`: Your OpenRouter API key (optional)

## License

MIT License
