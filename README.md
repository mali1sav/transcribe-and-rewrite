# Fast Transcriber

A Streamlit application that transcribes YouTube videos and web content, then generates well-structured articles in Thai. The app supports multiple sources including YouTube videos and webpage content.

## Features

- **Multiple Source Support**:
  - YouTube videos (with automatic transcript detection)
  - Webpage URLs
  - Direct text input

- **Smart Transcription**:
  - Automatically uses YouTube's transcript when available
  - Falls back to audio download and OpenAI transcription if needed
  - Supports multiple languages

- **Article Generation**:
  - Generates structured Thai articles
  - Creates engaging titles in multiple styles
  - Includes meta descriptions and excerpts
  - Properly attributes sources with hyperlinks

## Prerequisites

- Python 3.8+
- FFmpeg installed on your system
- OpenAI API key
- Streamlit account (for deployment)

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

3. Install FFmpeg:
   - **Mac**: `brew install ffmpeg`
   - **Linux**: `sudo apt-get install ffmpeg`
   - **Windows**: Download from [FFmpeg website](https://ffmpeg.org/download.html)

4. Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Enter your sources:
   - Add YouTube URLs and channel names
   - Add webpage URLs or paste content directly
   - Enter relevant keywords

3. Click "Process Sources" to start transcription
4. Click "Generate Article" once processing is complete
5. View and copy your generated article

## Configuration

- Adjust text area sizes in `app.py`
- Modify article generation prompts in the `generate_article` function
- Customize the UI layout using Streamlit's column system

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Streamlit](https://streamlit.io/) for the web framework
- [OpenAI](https://openai.com/) for transcription and text generation
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) for YouTube video processing
- [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api) for YouTube transcript fetching
