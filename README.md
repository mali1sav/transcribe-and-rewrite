# Transcribe and Rewrite

A Streamlit application that transcribes YouTube videos and generates article summaries using AI. The app supports both English and Thai languages.

## Features

- Fetch transcripts from YouTube videos
- Support for multiple input sources (YouTube URLs and text)
- Generate AI-powered article summaries
- Download transcripts and generated articles
- Supports English and Thai languages

## Setup

1. Clone the repository:
```bash
git clone https://github.com/mali1sav/transcribe-and-rewrite.git
cd transcribe-and-rewrite
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file based on `.env.example`:
```bash
cp .env.example .env
```

4. Add your OpenRouter API key to `.env`:
```
OPENROUTER_API_KEY=your_api_key_here
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Enter YouTube URLs or paste text content
3. Add keywords to guide the article generation
4. Select your preferred language (English or Thai)
5. Click "Process Sources" to fetch transcripts
6. Use "Generate Article" to create an AI-powered summary
7. Download transcripts or generated articles using the provided buttons

## API Keys

This application uses the OpenRouter API for article generation. You'll need to:
1. Sign up at [OpenRouter](https://openrouter.ai/)
2. Get your API key from the dashboard
3. Add it to your `.env` file

## License

MIT License
