import os
import re
import json
import time
import base64
import logging
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
import requests
import streamlit as st
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth
from firecrawl import FirecrawlApp
import markdown
import tenacity
from datetime import datetime, timezone
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
import google.generativeai as genai
from together import Together
from slugify import slugify

load_dotenv()

# Define site-specific edit URLs
SITE_EDIT_URLS = {
    "BITCOINIST": "https://bitcoinist.com/wp-admin/post.php?post={post_id}&action=edit&classic-editor",
    "NEWSBTC": "https://www.newsbtc.com/wp-admin/post.php?post={post_id}&action=edit&classic-editor",
    "ICOBENCH": "https://icobench.com/th/wp-admin/post.php?post={post_id}&action=edit&classic-editor"
}

# ------------------------------
# Utility Functions
# ------------------------------

def escape_special_chars(text):
    """Escape special characters that might interfere with Markdown formatting."""
    text = re.sub(r'(?<!\$)\$(?!\$)', r'\$', text)
    chars_to_escape = ['*', '_', '`', '#', '~', '|', '<', '>', '[', ']']
    for char in chars_to_escape:
        text = text.replace(char, '\\' + char)
    return text

def generate_slug_custom(text):
    """Generate a sanitized slug using python-slugify."""
    # Convert to lowercase first
    text = text.lower()
    # Use basic slugify without extra parameters
    slug = slugify(text)
    # Trim to max length if needed
    return slug[:200] if len(slug) > 200 else slug

def construct_endpoint(wp_url, endpoint_path):
    """Construct the WordPress endpoint."""
    wp_url = wp_url.rstrip('/')
    # Skip adding /th for Bitcoinist and NewsBTC
    if not any(domain in wp_url for domain in ["bitcoinist.com", "newsbtc.com"]) and "/th" not in wp_url:
        wp_url += "/th"
    return f"{wp_url}{endpoint_path}"

# ------------------------------
# YouTube Transcript Fetcher
# ------------------------------

class TranscriptFetcher:
    def get_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL."""
        try:
            video_id = re.search(r'(?:v=|\/|youtu\.be\/)([0-9A-Za-z_-]{11}).*', url)
            return video_id.group(1) if video_id else None
        except Exception as e:
            logging.error(f"Error extracting video ID: {str(e)}")
            return None

    def get_transcript(self, url: str) -> Optional[Dict]:
        """Get English transcript from YouTube."""
        try:
            video_id = self.get_video_id(url)
            if not video_id:
                logging.error(f"Could not extract video ID from URL: {url}")
                return None

            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
                full_text = " ".join(segment['text'] for segment in transcript)
                logging.info(f"Successfully got YouTube transcript for video {video_id}")
                return {
                    'text': full_text,
                    'segments': transcript
                }
            except (TranscriptsDisabled, NoTranscriptFound) as e:
                logging.error(f"No transcript available for video {video_id}: {str(e)}")
                return None

        except Exception as e:
            logging.error(f"Error getting YouTube transcript: {str(e)}")
            return None

# ------------------------------
# Data Models
# ------------------------------

class ArticleSection(BaseModel):
    heading: str = Field(..., description="H2 heading with power words in Thai")
    paragraphs: List[str] = Field(..., min_items=2, max_items=4, description="2-4 detailed paragraphs")

class ArticleContent(BaseModel):
    intro: str = Field(..., description="First paragraph with primary keyword")
    sections: List[ArticleSection]
    conclusion: str = Field(..., description="Summary emphasizing news angle")

class Source(BaseModel):
    domain: str
    url: str

class ArticleSEO(BaseModel):
    slug: str = Field(..., description="English URL-friendly with primary keyword")
    metaTitle: str = Field(..., description="Thai SEO title with primary keyword")
    metaDescription: str = Field(..., description="Thai meta desc with primary keyword")
    excerpt: str = Field(..., description="One Thai sentence summary")
    imagePrompt: str = Field(..., description="English photo description")
    altText: str = Field(..., description="Thai ALT text with English terms")

class Article(BaseModel):
    title: str = Field(..., description="Thai news-style title with primary keyword")
    content: ArticleContent
    sources: List[Source]
    seo: ArticleSEO

# ------------------------------
# SEO Parsing Functions
# ------------------------------

def parse_article(article_json):
    """
    Parses the generated article JSON into structured elements.
    Returns a dict with keys needed for WordPress upload.
    """
    try:
        article = json.loads(article_json) if isinstance(article_json, str) else article_json
        content_parts = []
        
        # Handle intro (either string or dict with parts)
        intro = article['content']['intro']
        if isinstance(intro, dict):
            content_parts.append(intro.get('Part 1', ''))
            content_parts.append(intro.get('Part 2', ''))
        else:
            content_parts.append(intro)
        
        # Handle sections with different formats
        for section in article['content']['sections']:
            content_parts.append(f"## {section['heading']}")
            format_type = section.get('format', 'paragraph')
            
            if format_type == 'list':
                # Format list items with bullets
                content_parts.extend([f"* {item}" for item in section['paragraphs']])
            elif format_type == 'table':
                # Format table data
                if section['paragraphs']:
                    # Handle if the data is already in markdown table format
                    if isinstance(section['paragraphs'][0], str) and section['paragraphs'][0].startswith('|'):
                        content_parts.extend(section['paragraphs'])
                    # Handle if the data is in array format
                    elif isinstance(section['paragraphs'][0], (list, tuple)):
                        headers = section['paragraphs'][0]
                        rows = section['paragraphs'][1:]
                        # Create markdown table
                        content_parts.append('| ' + ' | '.join(str(h) for h in headers) + ' |')
                        content_parts.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')
                        for row in rows:
                            content_parts.append('| ' + ' | '.join(str(cell) for cell in row) + ' |')
                    # Handle if each row is a dict
                    elif isinstance(section['paragraphs'][0], dict):
                        headers = section['paragraphs'][0].keys()
                        # Create markdown table
                        content_parts.append('| ' + ' | '.join(str(h) for h in headers) + ' |')
                        content_parts.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')
                        for row in section['paragraphs']:
                            content_parts.append('| ' + ' | '.join(str(row.get(h, '')) for h in headers) + ' |')
            else:
                # Regular paragraphs
                content_parts.extend(section['paragraphs'])
            content_parts.append("")
            
        content_parts.append(article['content']['conclusion'])
        return {
            "main_title": article['title'],
            "main_content": "\n\n".join(content_parts),
            "yoast_title": article['seo']['metaTitle'],
            "yoast_metadesc": article['seo']['metaDescription'] if len(article['seo']['metaDescription']) <= 160 \
                              else article['seo']['metaDescription'][:157] + "...",
            "seo_slug": article['seo']['slug'],
            "excerpt": article['seo']['excerpt'],
            "image_prompt": article['seo']['imagePrompt'],
            "image_alt": article['seo']['altText']
        }
    except (json.JSONDecodeError, KeyError) as e:
        st.error(f"Failed to parse article JSON: {str(e)}")
        return {}

# ------------------------------
# WordPress Uploader Functions
# ------------------------------

def upload_image_to_wordpress(b64_data, wp_url, username, wp_app_password, filename="generated_image.png", alt_text="Generated Image"):
    """
    Uploads an image (base64 string) to WordPress via the REST API.
    Returns a dict with 'media_id' and 'source_url'.
    """
    try:
        image_bytes = base64.b64decode(b64_data)
    except Exception as e:
        st.error(f"[Upload] Decoding error: {e}")
        return None

    media_endpoint = construct_endpoint(wp_url, "/wp-json/wp/v2/media")
    st.write(f"[Upload] Uploading image to {media_endpoint} with alt text: {alt_text}")
    try:
        files = {'file': (filename, image_bytes, 'image/png')}
        data = {'alt_text': alt_text, 'title': alt_text}
        response = requests.post(media_endpoint, files=files, data=data, auth=HTTPBasicAuth(username, wp_app_password))
        st.write(f"[Upload] Response status: {response.status_code}")
        if response.status_code in (200, 201):
            media_data = response.json()
            media_id = media_data.get('id')
            source_url = media_data.get('source_url', '')
            st.write(f"[Upload] Received Media ID: {media_id}")
            # Update alt text via PATCH (or PUT)
            update_endpoint = f"{media_endpoint}/{media_id}"
            update_data = {'alt_text': alt_text, 'title': alt_text}
            update_response = requests.patch(update_endpoint, json=update_data, auth=HTTPBasicAuth(username, wp_app_password))
            st.write(f"[Upload] Update response status: {update_response.status_code}")
            if update_response.status_code in (200, 201):
                st.success(f"[Upload] Image uploaded and alt text updated. Media ID: {media_id}")
                return {"media_id": media_id, "source_url": source_url}
            else:
                st.error(f"[Upload] Alt text update failed. Status: {update_response.status_code}, Response: {update_response.text}")
                return {"media_id": media_id, "source_url": source_url}
        else:
            st.error(f"[Upload] Image upload failed. Status: {response.status_code}, Response: {response.text}")
            return None
    except Exception as e:
        st.error(f"[Upload] Exception during image upload: {e}")
        return None

def submit_article_to_wordpress(article, wp_url, username, wp_app_password, primary_keyword="", site_name=None):
    """
    Submits the article to WordPress using the WP REST API.
    Sets Yoast SEO meta fields and auto-selects the featured image.
    """
    # Deep normalization for article data
    while isinstance(article, list) and len(article) > 0:
        article = article[0]  # Unpack nested lists
    if not isinstance(article, dict):
        article = {}
    
    endpoint = construct_endpoint(wp_url, "/wp-json/wp/v2/posts")
    st.write("Submitting article with Yoast SEO fields...")
    st.write("Yoast Title:", article.get("yoast_title"))
    st.write("Yoast Meta Description:", article.get("yoast_metadesc"))
    
    data = {
        "title": article.get("main_title", ""),
        "content": article.get("main_content", ""),
        "slug": article.get("seo_slug", ""),
        "status": "draft",
        "meta_input": {
            "_yoast_wpseo_title": article.get("yoast_title", ""),
            "_yoast_wpseo_metadesc": article.get("yoast_metadesc", ""),
            "_yoast_wpseo_focuskw": primary_keyword
        }
    }

    if "image" in article:
        # Normalize image data if it's a list
        image_data = article["image"]
        if isinstance(image_data, list) and len(image_data) > 0:
            image_data = image_data[0]
        
        if isinstance(image_data, dict) and image_data.get("media_id"):
            data["featured_media"] = image_data["media_id"]

    # Example category/tag logic:
    keyword_to_cat_tag = {"Dogecoin": 527, "Bitcoin": 7}
    if primary_keyword in keyword_to_cat_tag:
        cat_tag_id = keyword_to_cat_tag[primary_keyword]
        data["categories"] = [cat_tag_id]
        data["tags"] = [cat_tag_id]

    try:
        response = requests.post(endpoint, json=data, auth=HTTPBasicAuth(username, wp_app_password))
        if response.status_code in (200, 201):
            post = response.json()
            post_id = post.get('id')
            st.success(f"Article '{data['title']}' submitted successfully! Post ID: {post_id}")
            
            # If we have a site-specific edit URL, create a clickable link
            if site_name in SITE_EDIT_URLS:
                edit_url = SITE_EDIT_URLS[site_name].format(post_id=post_id)
                st.markdown(f"📝 [Click here to edit your draft article]({edit_url})")
                # Open the URL in a new browser tab
                import webbrowser
                webbrowser.open(edit_url)
            return post
        else:
            st.error(f"Failed to submit article. Status: {response.status_code}")
            st.error(f"Response: {response.text}")
            return None
    except Exception as e:
        st.error(f"Exception during article submission: {e}")
        return None

# ------------------------------
# Jina-Based Fallback Extraction (Using Jina API)
# ------------------------------

def jina_extract_via_r(url: str) -> dict:
    """
    Uses the Jina REST API endpoint (https://r.jina.ai/) to extract LLM-ready text.
    This appends the URL to the Jina base URL and returns the extracted markdown content along with minimal SEO fields.
    """
    JINA_BASE_URL = "https://r.jina.ai/"
    full_url = JINA_BASE_URL + url
    try:
        r = requests.get(full_url)
    except Exception as e:
        st.error(f"Jina request error: {e}")
        return {
            "title": "Extracted Content", 
            "content": {"intro": "", "sections": [], "conclusion": ""},
            "seo": {
                "slug": "", "metaTitle": "", "metaDescription": "", "excerpt": "", "imagePrompt": "", "altText": ""
            }
        }
    if r.status_code == 200:
        text = r.text
        # Assume the API returns LLM-ready text with a clear "Markdown Content:" section.
        md_index = text.find("Markdown Content:")
        md_content = text[md_index + len("Markdown Content:"):].strip() if md_index != -1 else text.strip()
        title_match = re.search(r"Title:\s*(.*)", text)
        title = title_match.group(1).strip() if title_match else "Extracted Content"
        fallback_json = {
            "title": title,
            "content": {"intro": md_content, "sections": [], "conclusion": ""},
            "seo": {
                "slug": generate_slug_custom(title),
                "metaTitle": title,
                "metaDescription": title,
                "excerpt": title,
                "imagePrompt": "",
                "altText": ""
            }
        }
        return fallback_json
    else:
        st.error(f"Jina extraction failed with status code {r.status_code}")
        return {
            "title": "Extracted Content", 
            "content": {"intro": "", "sections": [], "conclusion": ""},
            "seo": {
                "slug": "", "metaTitle": "", "metaDescription": "", "excerpt": "", "imagePrompt": "", "altText": ""
            }
        }

def extract_url_content(gemini_client, url, messages_placeholder):
    """
    Extracts article content from a URL. Handles both YouTube URLs (using transcript)
    and regular web pages (using Jina REST API).
    """
    # Check if it's a YouTube URL
    fetcher = TranscriptFetcher()
    video_id = fetcher.get_video_id(url)
    
    if video_id:
        with messages_placeholder:
            st.info(f"Extracting YouTube transcript from {url}...")
        transcript_data = fetcher.get_transcript(url)
        
        if transcript_data:
            with st.expander("Debug: YouTube Transcript", expanded=False):
                st.write("Transcript data:", transcript_data)
            return {
                'title': "YouTube Video Transcript",
                'url': url,
                'content': transcript_data['text'],
                'published_date': None,
                'source': f"YouTube Video ({video_id})",
                'author': None
            }
        else:
            st.warning(f"Could not extract transcript from YouTube video. Falling back to Jina extraction...")
    
    # Fallback to Jina for non-YouTube URLs or if transcript extraction fails
    with messages_placeholder:
        st.info(f"Extracting content from {url} using Jina REST API...")
    fallback_data = jina_extract_via_r(url)
    with st.expander("Debug: Jina Extraction", expanded=False):
        st.write("Jina extracted JSON:", fallback_data)
    content_text = fallback_data.get("content", {}).get("intro", "")
    with st.expander("Debug: Jina Extracted Content", expanded=False):
        st.write("Extracted content:", content_text)
    return {
        'title': fallback_data.get("title", "Extracted Content"),
        'url': url,
        'content': content_text,
        'published_date': None,
        'source': url,
        'author': None
    }

# ------------------------------
# Gemini Article Generation Functions
# ------------------------------

def init_gemini_client():
    """Initialize Google Gemini client."""
    try:
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            st.error("Gemini API key not found. Please set GEMINI_API_KEY in your environment variables.")
            return None
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        return {'model': model, 'name': 'gemini-2.0-flash-exp'}
    except Exception as e:
        st.error(f"Failed to initialize Gemini client: {str(e)}")
        return None

@tenacity.retry(
    stop=tenacity.stop_after_attempt(5),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=20),
    retry=tenacity.retry_if_exception_type((Exception)),
    retry_error_callback=lambda retry_state: None
)
def clean_gemini_response(text):
    """Clean Gemini response to extract valid JSON."""
    # Extract JSON from code blocks if present
    json_match = re.search(r'```(?:json)?\s*({[\s\S]*?})\s*```', text)
    if json_match:
        return json_match.group(1).strip()
    
    # If no code blocks, try to find a JSON object
    json_match = re.search(r'({[\s\S]*?})', text)
    if json_match:
        return json_match.group(1).strip()
    
    # If no JSON object found, clean the text
    text = re.sub(r'```(?:json)?\s*|```', '', text)
    text = text.strip()
    
    # Ensure it's wrapped in curly braces
    if not text.startswith('{'):
        text = '{' + text
    if not text.endswith('}'):
        text = text + '}'
    
    return text

def validate_article_json(json_str):
    """Validate article JSON against schema and return cleaned data."""
    try:
        # Parse JSON and ensure we have a dictionary
        data = json.loads(json_str)
        if isinstance(data, list):
            data = next((item for item in data if isinstance(item, dict)), {})
        elif not isinstance(data, dict):
            data = {}
            
        # Skip validation if we have an empty dictionary
        if not data:
            return {}
            
        if not data.get('title'):
            raise ValueError("Missing required field: title")
        if not data.get('content'):
            raise ValueError("Missing required field: content")
        # Handle case-insensitive 'seo' or 'SEO' field
        seo_field = next((k for k in data.keys() if k.lower() == 'seo'), None)
        if not seo_field:
            raise ValueError("Missing required field: seo")
        content = data['content']
        if not content.get('intro'):
            raise ValueError("Missing required field: content.intro")
        if not content.get('sections'):
            raise ValueError("Missing required field: content.sections")
        if not content.get('conclusion'):
            raise ValueError("Missing required field: content.conclusion")
        seo = data[seo_field]
        required_seo = ['slug', 'metaTitle', 'metaDescription', 'excerpt', 'imagePrompt', 'altText']
        for field in required_seo:
            if not seo.get(field):
                raise ValueError(f"Missing required SEO field: {field}")
        return data
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON: {str(e)}")
        st.code(json_str, language="json")
        return {}
    except ValueError as e:
        st.error(f"Validation error: {str(e)}")
        return {}

def make_gemini_request(client, prompt):
    """Make Gemini API request with retries and proper error handling.
    If Gemini’s response isn’t valid JSON, return a fallback JSON structure using the raw text.
    """
    try:
        for attempt in range(3):
            try:
                response = client['model'].generate_content(prompt)
                if response and response.text:
                    cleaned_text = clean_gemini_response(response.text)
                    if not cleaned_text.strip().startswith("{"):
                        st.warning("Gemini response is not valid JSON. Using raw text fallback.")
                        lines = cleaned_text.splitlines()
                        title = lines[0].strip() if lines else "Untitled"
                        fallback_json = {
                            "title": title,
                            "content": {"intro": cleaned_text, "sections": [], "conclusion": ""},
                            "seo": {
                                "slug": generate_slug_custom(title),
                                "metaTitle": title,
                                "metaDescription": title,
                                "excerpt": title,
                                "imagePrompt": "",
                                "altText": ""
                            }
                        }
                        return fallback_json
                    try:
                        return validate_article_json(cleaned_text)
                    except (json.JSONDecodeError, ValueError) as e:
                        if attempt == 2:
                            raise e
                        st.warning("Invalid JSON format, retrying...")
                        continue
            except Exception as e:
                if attempt == 2:
                    raise e
                st.warning(f"Retrying Gemini request (attempt {attempt+2}/3)...")
                time.sleep(2**attempt)
        raise Exception("Failed to get valid response from Gemini API after all retries")
    except Exception as e:
        st.error(f"Error making Gemini request: {str(e)}")
        raise

def load_promotional_content():
    """Loads promotional content from text files in the 'pr' folder."""
    import random
    pr_folder = os.path.join(os.path.dirname(__file__), "pr")
    if not os.path.isdir(pr_folder):
        return ""
    promo_files = [f for f in os.listdir(pr_folder) if f.endswith(".txt")]
    if not promo_files:
        return ""
    promo_file = random.choice(promo_files)
    promo_path = os.path.join(pr_folder, promo_file)
    with open(promo_path, "r", encoding="utf-8") as f:
        return f.read().strip()

def clean_source_content(content):
    """Clean source content by handling special characters and escape sequences"""
    # Replace problematic escape sequences
    content = content.replace('!\[', '![')  # Fix image markdown escapes
    content = content.replace('\\[', '[')   # Fix link markdown escapes
    content = content.replace('\\]', ']')   # Fix link markdown escapes
    content = content.replace('\\(', '(')   # Fix parentheses escapes
    content = content.replace('\\)', ')')   # Fix parentheses escapes
    # Handle other special characters if needed
    return content

def generate_article(client, transcripts, keywords=None, news_angle=None, section_count=3, promotional_text=None, selected_site=None):
    """
    Generates a comprehensive news article in Thai using the Gemini API.
    Uses the extracted source content (via Jina) in the prompt.
    Returns a JSON-structured article following the Article schema.
    Always returns a dictionary, even if input is a list.
    """
    try:
        if not transcripts:
            return None
        keyword_list = keywords if keywords else []
        primary_keyword = keyword_list[0] if keyword_list else ""
        secondary_keywords = ", ".join(keyword_list[1:]) if len(keyword_list) > 1 else ""
        
        # Concatenate source content from transcripts with clear delimiters.
        source_texts = ""
        seen_sources = set()
        for t in transcripts:
            content = clean_source_content(t.get('content') or "")
            source = t.get('source', 'Unknown')
            if source not in seen_sources:
                seen_sources.add(source)
                source_texts += f"\n---\nSource: {source}\nURL: {t.get('url', '')}\n\n{content}\n---\n"
            else:
                source_texts += f"\n{content}\n"
        
        prompt = f"""
You are an expert Thai crypto journalist and SEO specialist. Using ONLY the exact source content provided below, craft an SEO-optimized article in Thai. DO NOT invent or modify any factual details. Your article must faithfully reflect the provided source text.

Keep technical terms and entity names in English but the rest should be in Thai.

Primary Keyword: {primary_keyword}
Secondary Keywords: {secondary_keywords}
News Angle: {news_angle}

SEO Guidelines:
1. Primary Keyword ({primary_keyword}) Distribution:
   - Title: Include naturally in the first half (1x)
   - First Paragraph: Include naturally (1x)
   - H2 Headings: Include in at least 2 headings
   - Body content: Include naturally 3 times where relevant
   - Meta Description: Include naturally (1x)
   - Maintain original form (Thai/English) consistently

2. Secondary Keywords ({secondary_keywords}) Usage:
   - Include in 2 H2 headings where relevant
   - Use naturally in supporting paragraphs
   - Maximum density: 3% (3 mentions per 100 words)

Structure your output as valid JSON with the following keys:
- title: An engaging and click-worthy title (max 60 characters) with {primary_keyword} in first half.
- content: An object with:
   - intro: Two-part introduction:
     - Part 1 (meta description): Compelling 160-character that will also represent as Meta Description with {primary_keyword} included. This part must be click-worthy and stimulate curiosity
     - Part 2: Detailed paragraph expanding on Part 1
   - sections: An array of exactly {section_count} objects, each with:
     - heading: H2 heading using power words (include {primary_keyword} and {secondary_keywords} where natural)
     - format: Choose the most appropriate format for this section:
       - 'paragraph': For explanatory content (default). Each section MUST have 2-3 detailed paragraphs (at least 3-4 sentences each) with in-depth analysis, data points, examples, and thorough explanations. Don't just summarize - dive deep into each point with supporting evidence and context.
       - 'list': For steps, features, or benefits. Each list item MUST be comprehensive with 2-3 sentences of explanation, not just a short phrase. Include relevant examples, data points, or use cases for each item.
       - 'table': For comparisons or data. Each cell MUST contain detailed explanations (2-3 sentences) with context and examples, not just single words. Include at least 3 rows of comprehensive comparisons.
     - paragraphs: Array of 2-4 elements based on the chosen format:
       - For 'paragraph': Regular paragraphs
       - For 'list': Bullet points or numbered items
       - For 'table': Array of rows with headers
       Include {primary_keyword} naturally where relevant.
         {f'One section MUST seamlessly integrate this promotional content while maintaining natural flow and mentioning the {primary_keyword}: {promotional_text}' if promotional_text else ''}
   - conclusion: A concluding paragraph summarizing the key points of the article while mentioning the {primary_keyword} naturally.
- sources: An array of objects with keys "domain" and "url" for each source (each included only once).
- seo: An object with keys:
   - slug: English URL-friendly slug that MUST include {primary_keyword}{' and end with "-thailand"' if selected_site in ['BITCOINIST', 'NEWSBTC'] else ''}
   - metaTitle: Thai title with {primary_keyword}
   - metaDescription: Use the same text as the Part 1 of the intro.
   - excerpt: One Thai sentence summary
   - imagePrompt: The Image Prompt must be in English only. Create a photorealistic scene that fits the main news article, focusing on 1-2 main objects. Keep it simple and clear. Avoid charts, graphs, or technical diagrams as they don't work well with image generation.
   - altText: Thai ALT text with {primary_keyword} while keeping technical terms and entities in English
   Ensure {primary_keyword} appears exactly once in title, metaTitle, and metaDescription.

IMPORTANT NOTES:
1. Content Balance Guidelines:
   - Main Content (70%):
     * Focus on news, analysis, and key information from source URLs
     * Ensure comprehensive coverage of the main topic
     * Maintain journalistic integrity and depth
   - Promotional Content (30%):
     * Select only the most relevant points from promotional text
     * Limit to 2-3 paragraphs maximum
     * Focus on points that naturally connect with the main topic

2. Promotional Content Integration:
   - Choose the most contextually relevant position (preferably middle or end of article)
   - Create a dedicated section with a natural Thai heading that bridges the main topic and promotional content. You must create a sentence or two to transit seamlessly between them.
   - Weave in {primary_keyword} naturally within this section
   - Extract and use only key points that enhance the article's value
   - Trim lengthy promotional content while preserving core message
   - Ensure the promotional section reads like a natural part of the article's narrative

3. General Guidelines:
   - Maintain consistent tone and style throughout
   - DO NOT modify or invent any factual details
   - Preserve any image markdown exactly as provided
   - Ensure promotional content supports rather than overshadows the main story

IMPORTANT: DO NOT modify or invent any new factual details. Preserve any image markdown found within the analysis paragraphs exactly as provided.

Below is the source content (in markdown) extracted from the articles:
{source_texts}

Return ONLY valid JSON, no additional commentary.
"""
        st.write("Prompt being sent to Gemini API:")
        st.code(prompt, language='python')
        response = make_gemini_request(client, prompt)
        if not response:
            return {}
        
        # If response is already a dict, convert it to JSON string first
        if isinstance(response, dict):
            response = json.dumps(response)
        
        try:
            # Try to parse the JSON response
            content = json.loads(response)
            
            # Force dict output format
            if isinstance(content, list):
                content = content[0] if content else {}
            
            # Validate and ensure required fields exist
            if not isinstance(content, dict):
                st.error("Response is not a valid dictionary")
                return {}
                
            # Initialize required fields if missing
            if 'title' not in content:
                content['title'] = f"Latest News about {primary_keyword}"
                
            if 'content' not in content:
                content['content'] = {}
                
            if 'sections' not in content['content']:
                content['content']['sections'] = []
                
            # Ensure each section has required fields
            for section in content['content'].get('sections', []):
                if 'format' not in section:
                    section['format'] = 'paragraph'
                if 'paragraphs' not in section:
                    section['paragraphs'] = []
                    
            if 'conclusion' not in content['content']:
                content['content']['conclusion'] = 'บทความนี้นำเสนอข้อมูลเกี่ยวกับ ' + primary_keyword + ' ซึ่งเป็นประเด็นสำคัญในตลาดคริปโตที่ควรติดตาม'
                
            # Initialize SEO fields if missing
            seo_field = next((k for k in content.keys() if k.lower() == 'seo'), 'seo')
            if seo_field not in content:
                content[seo_field] = {
                    'slug': generate_slug_custom(primary_keyword),
                    'metaTitle': f"Latest Updates on {primary_keyword}",
                    'metaDescription': content.get('content', {}).get('intro', {}).get('Part 1', f"Latest news and updates about {primary_keyword}"),
                    'excerpt': f"Stay updated with the latest {primary_keyword} developments",
                    'imagePrompt': f"A photorealistic scene showing {primary_keyword} in a professional setting",
                    'altText': f"{primary_keyword} latest updates and developments"
                }
            
            return content
            
        except json.JSONDecodeError as e:
            st.error(f"JSON parsing error: {str(e)}")
            st.error("Raw response:")
            st.code(response)
            return {}
    except Exception as e:
        st.error(f"Error generating article: {str(e)}")
        return {}

# ------------------------------
# Main App Function
# ------------------------------

def main():
    st.set_page_config(page_title="Generate and Upload Article", layout="wide")
    
    # Default values:
    default_url = ""
    default_keyword = "Bitcoin"
    default_news_angle = ""
    
    gemini_client = init_gemini_client()
    if not gemini_client:
        st.error("Failed to initialize Gemini client")
        return
    
    st.sidebar.header("Article Generation")
    urls_input = st.sidebar.text_area("Enter URLs (one per line) to extract content from:", value=default_url)
    keywords_input = st.sidebar.text_area("Keywords (one per line):", value=default_keyword, height=100)
    news_angle = st.sidebar.text_input("News Angle:", value=default_news_angle)
    section_count = st.sidebar.slider("Number of sections:", 2, 8, 3)
    
    # Additional content text area
    additional_content = st.sidebar.text_area(
        "Additional Content",
        placeholder="Paste any extra content here. It will be treated as an additional source.",
        height=150
    )
    
    # Promotional content selection
    st.sidebar.header("Promotional Content")
    pr_folder = os.path.join(os.path.dirname(__file__), "pr")
    if os.path.isdir(pr_folder):
        promo_files = [f for f in os.listdir(pr_folder) if f.endswith(".txt")]
        if promo_files:
            promo_files = ["None"] + promo_files
            selected_promo = st.sidebar.selectbox("Select promotional content:", promo_files)
            if selected_promo != "None":
                promo_path = os.path.join(pr_folder, selected_promo)
                with open(promo_path, "r", encoding="utf-8") as f:
                    promotional_text = f.read().strip()
            else:
                promotional_text = None
        else:
            st.sidebar.warning("No promotional content files found in 'pr' folder")
            promotional_text = None
    else:
        st.sidebar.warning("'pr' folder not found")
        promotional_text = None
    
    # ----------------------------
    # Replaced WordPress Credentials Section
    # ----------------------------
    st.sidebar.header("Select WordPress Site to Upload")
    sites = {
        "ICOBENCH": {
            "url": os.getenv("ICOBENCH_WP_URL"),
            "username": os.getenv("ICOBENCH_WP_USERNAME"),
            "password": os.getenv("ICOBENCH_WP_APP_PASSWORD")
        },
        "CRYPTONEWS": {
            "url": os.getenv("CRYPTONEWS_WP_URL"),
            "username": os.getenv("CRYPTONEWS_WP_USERNAME"),
            "password": os.getenv("CRYPTONEWS_WP_APP_PASSWORD")
        },
        "BITCOINIST": {
            "url": os.getenv("BITCOINIST_WP_URL"),
            "username": os.getenv("BITCOINIST_WP_USERNAME"),
            "password": os.getenv("BITCOINIST_WP_APP_PASSWORD")
        },
        "NEWSBTC": {
            "url": os.getenv("NEWSBTC_WP_URL"),
            "username": os.getenv("NEWSBTC_WP_USERNAME"),
            "password": os.getenv("NEWSBTC_WP_APP_PASSWORD")
        },
    }
    site_options = list(sites.keys())
    selected_site = st.sidebar.selectbox("Choose a site:", site_options)
    
    wp_url = sites[selected_site]["url"]
    wp_username = sites[selected_site]["username"]
    wp_app_password = sites[selected_site]["password"]
    # ----------------------------
    
    messages_placeholder = st.empty()
    
    if st.sidebar.button("Generate Article"):
        transcripts = []
        
        # Process URLs if provided
        if urls_input.strip():
            urls = [line.strip() for line in urls_input.splitlines() if line.strip()]
            for url in urls:
                extracted = extract_url_content(gemini_client, url, messages_placeholder)
                if extracted:
                    transcripts.append(extracted)
        
        # Add additional content if provided
        if additional_content.strip():
            transcripts.append({
                'content': additional_content.strip(),
                'source': 'Additional Content',
                'url': ''
            })
        
        # Check if we have any content to process
        if not transcripts:
            st.error("Please provide either URLs or Additional Content to generate an article")
            return
            
        # Process keywords
        keywords = [k.strip() for k in keywords_input.splitlines() if k.strip()]
            
        if transcripts:
            # Get selected site
            selected_site = st.session_state.get('selected_site')
            
            article_content = generate_article(
                gemini_client,
                transcripts,
                keywords=keywords,
                news_angle=news_angle,
                section_count=section_count,
                promotional_text=promotional_text,
                selected_site=selected_site
            )
            if article_content:
                st.session_state.article = json.dumps(article_content, ensure_ascii=False, indent=2)
                st.success("Article generated successfully!")
            else:
                st.error("Failed to generate article.")
        else:
            st.error("No content extracted from provided URLs.")
    
    if "article" in st.session_state and st.session_state.article:
        st.subheader("Generated Article (JSON)")
        try:
            article_json = json.loads(st.session_state.article)
            st.json(article_json)
            st.subheader("Article Content Preview")
            st.write(f"**{article_json['title']}**")
            # Display intro parts
            if isinstance(article_json['content']['intro'], dict):
                st.write(article_json['content']['intro'].get('Part 1', ''))
                st.write(article_json['content']['intro'].get('Part 2', ''))
            else:
                st.write(article_json['content']['intro'])
            
            # Display sections with different formats
            for section in article_json['content']['sections']:
                st.write(f"### {section['heading']}")
                format_type = section.get('format', 'paragraph')
                
                if format_type == 'list':
                    for item in section['paragraphs']:
                        st.write(f"* {item}")
                elif format_type == 'table':
                    # If the data is already in markdown table format
                    if isinstance(section['paragraphs'][0], str) and section['paragraphs'][0].startswith('|'):
                        for row in section['paragraphs']:
                            st.write(row)
                    # If the data is in array or dict format
                    elif section['paragraphs']:
                        import pandas as pd
                        if isinstance(section['paragraphs'][0], (list, tuple)):
                            df = pd.DataFrame(section['paragraphs'][1:], columns=section['paragraphs'][0])
                        elif isinstance(section['paragraphs'][0], dict):
                            df = pd.DataFrame(section['paragraphs'])
                        st.table(df)
                else:  # default paragraph format
                    for para in section['paragraphs']:
                        st.write(para)
                st.write("")
            
            st.write("**บทสรุป:**")
            st.write(article_json['content']['conclusion'])
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON format: {str(e)}")
            st.text_area("Raw Content", value=st.session_state.article, height=300)
        
        # Process images in the article content (if any markdown images exist)
        if "article_data" not in st.session_state:
            st.session_state.article_data = {}
        if "processed_article" not in st.session_state.article_data:
            parsed = parse_article(st.session_state.article)
            st.session_state.article_data["processed_article"] = parsed
        
        # Supplementary image generation via Together AI
        if "image" not in st.session_state.article_data:
            parsed_for_image = parse_article(st.session_state.article)
            image_prompt = parsed_for_image.get("image_prompt")
            alt_text = parsed_for_image.get("image_alt")
            if image_prompt:
                # Show original prompt
                st.info(f"Original image prompt: '{image_prompt}'")
                
                # Remove Thai characters to pass a simpler English prompt to the image generator
                image_prompt_english = re.sub(r'[\u0E00-\u0E7F]+', '', image_prompt).strip()
                if not image_prompt_english:
                    image_prompt_english = "A photo-realistic scene of cryptocurrencies floating in the air, depicting the Crypto news"
                    st.warning("Using fallback image prompt since no English text was found")
                
                # Show cleaned prompt
                st.info(f"Cleaned English prompt for Together AI: '{image_prompt_english}'")
            else:
                image_prompt_english = None
            
            if image_prompt_english:
                together_client = Together()
                try:
                    response = together_client.images.generate(
                        prompt=image_prompt_english,
                        model="black-forest-labs/FLUX.1-schnell-free",
                        width=1200,
                        height=800,
                        steps=4,
                        n=1,
                        response_format="b64_json"
                    )
                    if response and response.data and len(response.data) > 0:
                        b64_data = response.data[0].b64_json
                        if not alt_text:
                            primary_kw = keywords_input.splitlines()[0] if keywords_input.splitlines() else "Crypto"
                            alt_text = f"Featured image related to {primary_kw}"
                        st.image("data:image/png;base64," + b64_data, caption=alt_text)
                        st.session_state.article_data["image"] = {"b64_data": b64_data, "alt_text": alt_text, "media_id": None}
                    else:
                        st.error("Failed to generate supplementary image from Together AI.")
                except Exception as e:
                    st.error(f"Error generating supplementary image: {str(e)}")
            else:
                st.info("No valid image prompt found in generated article for supplementary image.")
        else:
            st.info("Using previously generated image.")
        
        if st.button("Upload Article to WordPress"):
            if not all([wp_url, wp_username, wp_app_password]):
                st.error("Please ensure your .env file has valid credentials for the selected site.")
            else:
                try:
                    parsed = parse_article(st.session_state.article)
                    media_info = None
                    if "image" in st.session_state.article_data:
                        image_data = st.session_state.article_data["image"]
                        if image_data.get("b64_data"):
                            article_json = json.loads(st.session_state.article)
                            alt_text = article_json['seo']['altText']
                            media_info = upload_image_to_wordpress(
                                image_data["b64_data"],
                                wp_url,
                                wp_username,
                                wp_app_password,
                                filename="generated_image.png",
                                alt_text=alt_text
                            )
                    article_data = {
                        "main_title": parsed.get("main_title", "No Title"),
                        "main_content": markdown.markdown(parsed.get("main_content", "")),
                        "seo_slug": parsed.get("seo_slug", ""),
                        "excerpt": parsed.get("excerpt", ""),
                        "yoast_title": parsed.get("yoast_title", ""),
                        "yoast_metadesc": parsed.get("yoast_metadesc", ""),
                        "image": st.session_state.article_data.get("image") if "image" in st.session_state.article_data else {}
                    }
                    if media_info and "media_id" in media_info:
                        article_data["image"]["media_id"] = media_info["media_id"]
                    primary_keyword_upload = keywords_input.splitlines()[0] if keywords_input.strip() else ""
                    submit_article_to_wordpress(
                        article_data, 
                        wp_url, 
                        wp_username, 
                        wp_app_password, 
                        primary_keyword=primary_keyword_upload,
                        site_name=st.session_state.get('selected_site')
                    )
                except Exception as e:
                    st.error(f"Error during upload process: {str(e)}")

if __name__ == "__main__":
    main()
