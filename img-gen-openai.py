
import os
import re
import json
import time
import base64
import logging
import html  # <-- Needed for escaping alt text
# YouTube functionality removed - focusing on non-YouTube content only
import requests
import streamlit as st
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth
import markdown
import tenacity
from datetime import datetime, timezone
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
# import google.generativeai as genai
# from together import Together
import openai
# OpenRouter uses OpenAI-compatible API, so we use openai package


# Custom slugify implementation is used instead of external library

load_dotenv()

# Define site-specific edit URLs (ensure every site gets an edit link)
SITE_EDIT_URLS = {
    "BITCOINIST": "https://bitcoinist.com/wp-admin/post.php?post={post_id}&action=edit&classic-editor",
    "NEWSBTC": "https://www.newsbtc.com/wp-admin/post.php?post={post_id}&action=edit&classic-editor",
    "ICOBENCH": "https://icobench.com/th/wp-admin/post.php?post={post_id}&action=edit&classic-editor",
    "CRYPTONEWS": "https://cryptonews.com/th/wp-admin/post.php?post={post_id}&action=edit&classic-editor",
    "INSIDEBITCOINS": "https://insidebitcoins.com/th/wp-admin/post.php?post={post_id}&action=edit&classic-editor"
}

# Promotional Images Data Structure
PROMOTIONAL_IMAGES = {
    "Solaxy": {
        "url": "https://icobench.com/th/wp-content/uploads/sites/17/2025/02/solaxy-1.png",
        "alt": "Solaxy Thailand",
        "width": "600",
        "height": "520",
    },
    "BTC Bull Token": {
        "url": "https://icobench.com/th/wp-content/uploads/sites/17/2025/02/btc-bull-token.png",
        "alt": "BTC Bull Token",
        "width": "600",
        "height": "408",
    },
    "Mind of Pepe": {
        "url": "https://icobench.com/th/wp-content/uploads/sites/17/2025/02/mind-of-pepe-e1740672348698.png",
        "alt": "Mind of Pepe",
        "width": "600",
        "height": "490",
    },
    "Meme Index": {
        "url": "https://icobench.com/th/wp-content/uploads/sites/17/2025/02/memeindex.png",
        "alt": "Meme Index",
        "width": "600",
        "height": "468",
    },

    "SUBBD": {
        "url": "https://icobench.com/th/wp-content/uploads/sites/17/2025/04/Subbd-token.png",
        "alt": "SUBBD Token",
        "width": "600",
        "height": "514",
    }
}

# Affiliate Links Data Structure
AFFILIATE_LINKS = {
    "ICOBENCH": {
        "Best Wallet": "https://cryptoquad.care/box_c3ff67e3750f4482393b82c4f7f6799e",
        "Solaxy": "https://icobench.com/th/visit/solaxy",
        "BTC Bull Token": "https://icobench.com/th/visit/bitcoin-bull",
        "Mind of Pepe": "https://icobench.com/th/visit/mindofpepe",
        "Meme Index": "https://icobench.com/th/visit/memeindex",
        "SUBBD Token": "https://cryptoquad.care/box_d036ffde8379690af8a250c50dbdd00a"
    },
    "BITCOINIST": {
        "Best Wallet": "https://bs_332b25fb.bitcoinist.care/",
        "Solaxy": "https://bs_ddfb0f8c.bitcoinist.care/",
        "BTC Bull Token": "https://bs_919798f4.bitcoinist.care/",
        "Mind of Pepe": "https://bs_1f5417eb.bitcoinist.care/",
        "Meme Index": "https://bs_89e992a3.bitcoinist.care",
        "SUBBD Token": "https://bitcoinist.care/box_0a57d9080feda10ad1d596bf6b65b364"
    },
    "CRYPTONEWS": {
        "Best Wallet": "https://bestwallet.com/th",
        "Solaxy": "https://solaxy.io/th/?tid=156",
        "BTC Bull Token": "https://btcbulltoken.com/th?tid=156",
        "Mind of Pepe": "https://mindofpepe.com/th?tid=156",
        "Meme Index": "https://memeindex.com/?tid=156",
        "SUBBD Token": "https://nsjbxj.care/box_a4cb299d00dac3c226d1f66424dbce92"
    },
    "INSIDEBITCOINS": {
        "Best Wallet": "https://solaxy.io/th/?tid=156",
        "Solaxy": "https://insidebitcoins.com/th/visit/solaxy",
        "BTC Bull Token": "https://insidebitcoins.com/th/visit/bitcoin-bull",
        "Mind of Pepe": "https://insidebitcoins.com/th/visit/mindofpepe",
        "Meme Index": "https://insidebitcoins.com/th/visit/memeindex",
        "SUBBD Token": "https://cryptocodes.care/box_6064e1aad8581d077fd585c28176ab57"
    }
}

def generate_slug_custom(text: str) -> str:
    """
    Generate a very simple slug - just lowercase with spaces replaced by hyphens.
    This avoids any dependency on external libraries.
    """
    try:
        # Empty check with fallback
        if not text:
            return f"article-{int(time.time())}"
            
        # Simple conversion: lowercase and replace spaces with hyphens
        simple_slug = text.lower().replace(' ', '-')
        
        # Remove any characters that aren't alphanumeric, hyphens, or underscores
        simple_slug = ''.join(c for c in simple_slug if c.isalnum() or c == '-' or c == '_')
        
        # Remove multiple consecutive hyphens and trim
        while '--' in simple_slug:
            simple_slug = simple_slug.replace('--', '-')
        simple_slug = simple_slug.strip('-')
        
        # Add fallback if empty after processing
        if not simple_slug:
            return f"article-{int(time.time())}"
            
        return simple_slug[:200]  # Limit length
    except Exception as e:
        # Safe fallback with timestamp
        return f"article-{int(time.time())}"

def process_affiliate_links(content, website, project_name=None):
    """
    Scan content for anchor texts (outside of HTML attributes) and replace with appropriate affiliate links.
    If project_name is provided, also add the corresponding promotional image in a dedicated section or fallback.
    """
    if website not in AFFILIATE_LINKS:
        return content
    
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        # If bs4 not available, return as-is (fallback).
        return content

    soup = BeautifulSoup(content, 'html.parser')

    # Replace anchor text only in text nodes
    for anchor_text, link_url in AFFILIATE_LINKS[website].items():
        text_nodes = soup.find_all(text=True)
        link_count = 0
        for text_node in text_nodes:
            if not text_node.parent or text_node.parent.name in ['style', 'script', 'head', 'title', 'meta', 'img', 'figure', 'a']:
                continue
            pattern = r'\b' + re.escape(anchor_text) + r'\b'
            def replace_limited(match):
                nonlocal link_count
                if link_count < 3:
                    link_count += 1
                    return f'<a href="{link_url}" target="_blank" rel="sponsored noopener nofollow">{anchor_text}</a>'
                else:
                    return anchor_text
            replaced = re.sub(pattern, replace_limited, text_node, flags=re.IGNORECASE)
            if replaced != text_node:
                text_node.replace_with(BeautifulSoup(replaced, 'html.parser'))

    # Update all affiliate links to add nofollow to rel (in case some are already present)
    for a in soup.find_all('a', href=True):
        rel = a.get('rel', [])
        rel_set = set(rel if isinstance(rel, list) else rel.split())
        rel_set.update(['sponsored', 'noopener', 'nofollow'])
        a['rel'] = ' '.join(sorted(rel_set))


    # Insert promotional image if project_name is provided
    if project_name and project_name in PROMOTIONAL_IMAGES:
        img_data = PROMOTIONAL_IMAGES[project_name]
        # Escape alt text to avoid HTML breakage
        escaped_alt = html.escape(img_data["alt"])  
        img_html = (
            f'<img src="{img_data["url"]}" '
            f'alt="{escaped_alt}" '
            f'width="{img_data["width"]}" '
            f'height="{img_data["height"]}" />\n\n'
        )

        # Try to place the promotional image after an h2 containing the project name
        found_heading = None
        h2_tags = soup.find_all('h2')
        for h2 in h2_tags:
            if project_name.lower() in h2.get_text(strip=True).lower():
                found_heading = h2
                break

        if found_heading:
            found_heading.insert_after(BeautifulSoup(img_html, 'html.parser'))
        else:
            # If not found, fallback to after the 3rd heading or near conclusion
            if len(h2_tags) >= 3:
                h2_tags[2].insert_after(BeautifulSoup(img_html, 'html.parser'))
            else:
                conclusion_tags = soup.find_all(string=re.compile(r'บทความนี้นำเสนอ|บทสรุป|สรุป'))
                if conclusion_tags:
                    conclusion_node = conclusion_tags[0]
                    conclusion_node.insert_before(BeautifulSoup(img_html, 'html.parser'))
                else:
                    soup.append(BeautifulSoup(img_html, 'html.parser'))

    return str(soup)

def convert_to_gutenberg_format(content):
    """
    Convert standard HTML content to Gutenberg blocks format.
    """
    if '<!-- wp:' in content:
        return content
    
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(content, 'html.parser')
        
        gutenberg_content = []
        # Process each recognized element in order
        for element in soup.find_all(['h1','h2','h3','h4','h5','h6','p','ul','ol','img','figure']):
            if element.name in ['h1','h2','h3','h4','h5','h6']:
                # Convert headings to H2
                gutenberg_content.append('<!-- wp:heading -->')
                gutenberg_content.append(f'<h2>{element.get_text()}</h2>')
                gutenberg_content.append('<!-- /wp:heading -->')
            
            elif element.name == 'p':
                gutenberg_content.append('<!-- wp:paragraph -->')
                gutenberg_content.append(f'<p>{element.decode_contents()}</p>')
                gutenberg_content.append('<!-- /wp:paragraph -->')
            
            elif element.name == 'ul':
                gutenberg_content.append('<!-- wp:list -->')
                gutenberg_content.append(str(element))
                gutenberg_content.append('<!-- /wp:list -->')
            
            elif element.name == 'ol':
                gutenberg_content.append('<!-- wp:list {"ordered":true} -->')
                gutenberg_content.append(str(element))
                gutenberg_content.append('<!-- /wp:list -->')
            
            elif element.name == 'img':
                align = ''
                attrs = {}
                if align:
                    attrs['align'] = align
                
                # Escape alt text to avoid breakage
                alt_escaped = html.escape(element.get("alt",""))
                
                gutenberg_content.append(f'<!-- wp:image {json.dumps(attrs)} -->')
                gutenberg_content.append(
                    f'<figure class="wp-block-image{" align" + align if align else ""}">'
                    f'<img src="{element.get("src", "")}" alt="{alt_escaped}" '
                    f'width="{element.get("width", "")}" height="{element.get("height", "")}"/>'
                    f'</figure>'
                )
                gutenberg_content.append('<!-- /wp:image -->')
            
            elif element.name == 'figure':
                # If there's an <img> inside, handle similarly
                img = element.find('img')
                if img:
                    align = ''
                    attrs = {}
                    if align:
                        attrs['align'] = align
                    
                    alt_escaped = html.escape(img.get("alt",""))
                    
                    gutenberg_content.append(f'<!-- wp:image {json.dumps(attrs)} -->')
                    # Replace raw alt text with escaped alt text inside the figure
                    figure_html = str(element).replace(img.get("alt",""), alt_escaped)
                    gutenberg_content.append(figure_html)
                    gutenberg_content.append('<!-- /wp:image -->')
        
        if not gutenberg_content:
            # fallback
            gutenberg_content.append('<!-- wp:paragraph -->')
            gutenberg_content.append(f'<p>{content}</p>')
            gutenberg_content.append('<!-- /wp:paragraph -->')
        
        return '\n'.join(gutenberg_content)
    except Exception as e:
        st.error(f"Error converting to Gutenberg format: {str(e)}")
        return f'<!-- wp:paragraph -->\n<p>{content}</p>\n<!-- /wp:paragraph -->'

def escape_special_chars(text):
    text = re.sub(r'(?<!\$)\$(?!\$)', r'\$', text)
    chars_to_escape = ['*', '_', '`', '#', '~', '|', '<', '>', '[', ']']
    for char in chars_to_escape:
        text = text.replace(char, '\\' + char)
    return text

def construct_endpoint(wp_url, endpoint_path):
    wp_url = wp_url.rstrip('/')
    if not any(domain in wp_url for domain in ["bitcoinist.com", "newsbtc.com"]) and "/th" not in wp_url:
        wp_url += "/th"
    return f"{wp_url}{endpoint_path}"

# YouTube functionality (TranscriptFetcher class) removed - focusing on non-YouTube content only

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

def parse_article(article_json, add_affiliate_note=False):
    """
    Convert the model's JSON into a final structure for WP.
    Forces the slug to ASCII only. Optionally appends an affiliate note at the end.
    """
    try:
        article = json.loads(article_json) if isinstance(article_json, str) else article_json
        content_parts = []
        
        # Intro
        intro = article['content']['intro']
        if isinstance(intro, dict):
            content_parts.append(intro.get('Part 1', ''))
            content_parts.append(intro.get('Part 2', ''))
        else:
            content_parts.append(intro)
        
        # Sections
        for section in article['content']['sections']:
            content_parts.append(f"## {section['heading']}")
            format_type = section.get('format', 'paragraph')
            
            if format_type == 'list':
                content_parts.extend([f"* {item}" for item in section['paragraphs']])
            elif format_type == 'table':
                if section['paragraphs']:
                    if isinstance(section['paragraphs'][0], str) and section['paragraphs'][0].startswith('|'):
                        content_parts.extend(section['paragraphs'])
                    elif isinstance(section['paragraphs'][0], (list, tuple)):
                        headers = section['paragraphs'][0]
                        rows = section['paragraphs'][1:]
                        content_parts.append('| ' + ' | '.join(str(h) for h in headers) + ' |')
                        content_parts.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')
                        for row in rows:
                            content_parts.append('| ' + ' | '.join(str(cell) for cell in row) + ' |')
                    elif isinstance(section['paragraphs'][0], dict):
                        headers = section['paragraphs'][0].keys()
                        content_parts.append('| ' + ' | '.join(str(h) for h in headers) + ' |')
                        content_parts.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')
                        for row in section['paragraphs']:
                            content_parts.append('| ' + ' | '.join(str(row.get(h, '')) for h in headers) + ' |')
            else:
                content_parts.extend(section['paragraphs'])
            content_parts.append("")
        
        # Conclusion
        content_parts.append(article['content']['conclusion'])

        # Optional affiliate note
        if add_affiliate_note:
            content_parts.append("[su_note note_color=\"#FEFEEE\"] เรามุ่งมั่นในการสร้างความโปร่งใสอย่างเต็มที่กับผู้อ่านของเรา บางเนื้อหาในเว็บไซต์อาจมีลิงก์พันธมิตร ซึ่งเราอาจได้รับค่าคอมมิชชั่นจากความร่วมมือเหล่านี้ [/su_note]")

        # Simple slug processing - lowercase, replace spaces with hyphens, remove special chars
        try:
            slug = article['seo'].get('slug', '')
            if not slug:
                slug = article['title']
                
            # Very simple processing: lowercase, replace spaces
            slug = slug.lower().replace(' ', '-')
            
            # Keep only alphanumeric and hyphens
            slug = ''.join(c for c in slug if c.isalnum() or c == '-')
            
            # Remove multiple consecutive hyphens
            while '--' in slug:
                slug = slug.replace('--', '-')
                
            # Ensure not empty
            if not slug:
                slug = f"article-{int(time.time())}"
                
            # Limit length
            article['seo']['slug'] = slug[:100]  # shorter is better for URLs
        except Exception as e:
            # Fallback with timestamp if any errors
            article['seo']['slug'] = f"article-{int(time.time())}"

        # --- META DESCRIPTION POST-PROCESSING ---
        meta_desc = article['seo']['metaDescription']
        title = article['title']
        # Remove hashtags
        import re
        meta_desc = re.sub(r'#\w+', '', meta_desc)
        meta_desc = meta_desc.replace('  ', ' ').strip()
        # Ensure not identical to title
        if meta_desc.strip() == title.strip():
            meta_desc = f"{meta_desc} - อ่านรายละเอียดเพิ่มเติม!"
        # Trim to 155 chars
        if len(meta_desc) > 155:
            meta_desc = meta_desc[:152] + '...'

        return {
            "main_title": title,
            "main_content": "\n\n".join(content_parts),
            "yoast_title": article['seo']['metaTitle'],
            "yoast_metadesc": meta_desc,
            "seo_slug": article['seo']['slug'],
            "excerpt": article['seo']['excerpt'],
            "image_prompt": article['seo']['imagePrompt'],
            "image_alt": article['seo']['altText']
        }
    except (json.JSONDecodeError, KeyError) as e:
        st.error(f"Failed to parse article JSON: {str(e)}")
        return {}

def upload_image_to_wordpress(b64_data, wp_url, username, wp_app_password, filename="generated_image.png", alt_text="Generated Image"):
    """
    Uploads an image (base64 string) to WordPress via the REST API.
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
            # Update alt text via PATCH
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

def submit_article_to_wordpress(article, wp_url, username, wp_app_password, primary_keyword="", site_name=None, content_type="post"):
    """
    Submits the article to WordPress using the WP REST API.
    """
    if not isinstance(article, dict):
        article = {}

    if content_type.lower() == "page":
        endpoint = construct_endpoint(wp_url, "/wp-json/wp/v2/pages")
    else:
        endpoint = construct_endpoint(wp_url, "/wp-json/wp/v2/posts")
        
    st.write("Submitting article with Yoast SEO fields...")
    st.write("Yoast Title:", article.get("yoast_title"))
    st.write("Yoast Meta Description:", article.get("yoast_metadesc"))
    st.write("Selected site:", site_name)
    
    # Insert affiliate links
    content = article.get("main_content", "")
    if site_name and content:
        content = process_affiliate_links(content, site_name)
    
    # Convert to Gutenberg blocks if post
    if content_type.lower() == "post":
        content = convert_to_gutenberg_format(content)
    
    data = {
        "title": article.get("main_title", ""),
        "content": content,
        "slug": article.get("seo_slug", ""),
        "status": "draft",
        "meta_input": {
            "_yoast_wpseo_title": article.get("yoast_title", ""),
            "_yoast_wpseo_metadesc": article.get("yoast_metadesc", ""),
            "_yoast_wpseo_focuskw": primary_keyword
        }
    }

    # Featured image if available
    if "image" in article:
        image_data = article["image"]
        if isinstance(image_data, list) and len(image_data) > 0:
            image_data = image_data[0]
        if isinstance(image_data, dict) and image_data.get("media_id"):
            data["featured_media"] = image_data["media_id"]

    # Example: category/tag assignment by primary_keyword
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
            st.success(f"Article '{data['title']}' submitted successfully! ID: {post_id}")
            
            edit_url = SITE_EDIT_URLS.get(
                site_name,
                f"{wp_url.rstrip('/')}/wp-admin/post.php?post={{post_id}}&action=edit&classic-editor"
            ).format(post_id=post_id)
            st.markdown(f"[Click here to edit your draft article]({edit_url})")
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

def jina_extract_via_r(url: str) -> dict:
    """
    Uses the Jina REST API endpoint (https://r.jina.ai/) to extract text.
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
        md_index = text.find("Markdown Content:")
        md_content = text[md_index + len("Markdown Content:"):].strip() if md_index != -1 else text.strip()
        title_match = re.search(r"Title:\s*(.*)", text)
        title = title_match.group(1).strip() if title_match else "Extracted Content"
        fallback_json = {
            "title": title,
            "content": {"intro": md_content, "sections": [], "conclusion": ""},
            "seo": {
                "slug": title.lower().replace(' ', '-')[:100],  # Simple direct conversion without external functions
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
    Extracts article content from a URL using Jina REST API.
    If extraction fails, recommends using the Additional Content area.
    """
    try:
        # Validate input URL
        if not url or not url.startswith(('http://', 'https://')):
            with messages_placeholder:
                st.error(f"Invalid URL format: {url}. Please provide a valid URL starting with http:// or https://")
                st.info("Please use the 'Additional Content' section in the sidebar to manually input content.")
            return None
            
        # Extract content using Jina
        with messages_placeholder:
            st.info(f"Extracting content from {url} using Jina REST API...")
            
        # Get content with error handling
        try:
            extraction_data = jina_extract_via_r(url)
            
            # Log extraction results for debugging
            with st.expander("Debug: Content Extraction", expanded=False):
                st.write("Extraction data:", extraction_data)
                
            # Check for CloudFlare protection or empty content
            content_text = extraction_data.get("content", {}).get("intro", "")
            title = extraction_data.get("title", "")
            
            # Handle CloudFlare "Just a moment..." challenge pages
            if title == "Just a moment..." or not content_text.strip():
                with messages_placeholder:
                    st.warning(f"No meaningful content could be extracted from {url}")
                    st.error("The website may be protected by CloudFlare or other anti-scraping measures.")
                    st.info("Please use the 'Additional Content' section in the sidebar to manually input content.")
                logging.warning(f"CloudFlare protection detected or empty content from {url}")
                return None
            
            # Return structured content if extraction was successful
            return {
                'title': title or "Extracted Content",
                'url': url,
                'content': content_text,
                'published_date': None,
                'source': url,
                'author': None
            }
            
        except Exception as e:
            with messages_placeholder:
                st.error(f"Error during content extraction: {str(e)}")
                st.info("Please use the 'Additional Content' section in the sidebar to manually input content.")
            logging.error(f"Content extraction error for {url}: {str(e)}")
            return None
            
    except Exception as e:
        # Catch-all for any unexpected errors
        with messages_placeholder:
            st.error(f"Unexpected error processing URL {url}: {str(e)}")
            st.info("Please use the 'Additional Content' section in the sidebar to manually input content.")
        logging.error(f"URL processing error: {str(e)}")
        return None

def init_openrouter_client():
    """
    Initialize OpenRouter client for Gemini-2.5 via OpenAI-compatible API (>=1.0.0).
    """
    try:
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            st.error("OpenRouter API key not found. Please set OPENROUTER_API_KEY in your environment variables.")
            return None
        # Use openai>=1.0.0 client
        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        model = "google/gemini-2.5-flash-preview-05-20"
        return {'client': client, 'model': model, 'name': 'gemini-2.5-flash-preview-05-20'}
    except Exception as e:
        st.error(f"Failed to initialize OpenRouter client: {str(e)}")
        return None

def clean_gemini_response(text):
    """
    Clean Gemini response to extract valid JSON.
    """
    json_match = re.search(r'```(?:json)?\s*({[\s\S]*?})\s*```', text)
    if json_match:
        return json_match.group(1).strip()
    json_match = re.search(r'({[\s\S]*?})', text)
    if json_match:
        return json_match.group(1).strip()
    text = re.sub(r'```(?:json)?\s*|```', '', text)
    text = text.strip()
    if not text.startswith('{'):
        text = '{' + text
    if not text.endswith('}'):
        text = text + '}'
    return text

def validate_article_json(json_str):
    """
    Validate article JSON against schema.
    """
    try:
        data = json.loads(json_str)
        if isinstance(data, list):
            data = next((item for item in data if isinstance(item, dict)), {})
        elif not isinstance(data, dict):
            data = {}
        if not data:
            return {}
        if not data.get('title'):
            raise ValueError("Missing required field: title")
        if not data.get('content'):
            raise ValueError("Missing required field: content")
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

def make_openrouter_request(client_dict, prompt):
    """
    Make OpenRouter API request (Gemini-2.5) with retries. Return validated JSON or fallback. Compatible with openai>=1.0.0 interface.
    """
    try:
        for attempt in range(3):
            try:
                response = client_dict['client'].chat.completions.create(
                    model=client_dict['model'],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=8192,
                )
                text = response.choices[0].message.content
                if text:
                    cleaned_text = clean_gemini_response(text)
                    if not cleaned_text.strip().startswith("{"):
                        st.warning("Model response is not valid JSON. Using raw text fallback.")
                        lines = cleaned_text.splitlines()
                        title = lines[0].strip() if lines else "Untitled"
                        fallback_json = {
                            "title": title,
                            "content": {"intro": cleaned_text, "sections": [], "conclusion": ""},
                            "seo": {
                                "slug": title.lower().replace(' ', '-')[:100],
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
                st.warning(f"Retrying OpenRouter request (attempt {attempt+2}/3)...")
                time.sleep(2**attempt)
        raise Exception("Failed to get valid response from OpenRouter API after all retries")
    except Exception as e:
        st.error(f"Error making OpenRouter request: {str(e)}")
        raise

def load_promotional_content(promo_name=None):
    """
    Loads promotional content from the 'pr' folder.
    If promo_name is provided, loads that specific file.
    Otherwise returns empty string.
    """
    try:
        if not promo_name or promo_name == "None":
            logging.info("No promotional content requested")
            return ""
            
        # Validate input and construct path
        pr_folder = os.path.join(os.path.dirname(__file__), "pr")
        if not os.path.isdir(pr_folder):
            logging.warning("'pr' folder not found")
            return ""
            
        # Look for exact file or with .txt extension
        promo_file = os.path.join(pr_folder, promo_name)
        if not os.path.exists(promo_file):
            promo_file = os.path.join(pr_folder, f"{promo_name}.txt")
            
        # Try loading from files with error handling
        if os.path.exists(promo_file) and os.path.isfile(promo_file):
            with open(promo_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                logging.info(f"Loaded promotional content: {promo_name} ({len(content)} chars)")
                return content
        else:
            # If no file exists, use the name as the content (for backward compatibility)
            logging.info(f"Using promotional name as content: {promo_name}")
            return promo_name
            
    except Exception as e:
        logging.error(f"Error loading promotional content: {str(e)}")
        return ""  # Return empty string on error

def clean_source_content(content):
    """
    Clean source content by handling special characters and escape sequences
    """
    content = content.replace('!\\[', '!')
    content = content.replace('\\[', '[')
    content = content.replace('\\]', ']')
    content = content.replace('\\(', '(')
    content = content.replace('\\)', ')')
    return content

def generate_article(client, transcripts, keywords=None, news_angle=None, section_count=3, promotional_text=None, selected_site=None):
    """    
    Generates a comprehensive news article in Thai using Gemini.
    If promotional_text is provided, it will be included in the prompt to Gemini.
    """
    """
    Generates a comprehensive news article in Thai using Gemini.
    """
    try:
        if not transcripts:
            return None
        if promotional_text is None:
            promotional_text = ""
        keyword_list = keywords if keywords else []
        primary_keyword = keyword_list[0] if keyword_list else ""
        secondary_keywords = ", ".join(keyword_list[1:]) if len(keyword_list) > 1 else ""
        
        source_texts = ""
        seen_sources = set()
        for t in transcripts:
            c = clean_source_content(t.get('content') or "")
            s = t.get('source', 'Unknown')
            # Only add each source once to avoid duplication:
            if s not in seen_sources:
                seen_sources.add(s)
                source_texts += f"\n---\nSource: {s}\nURL: {t.get('url', '')}\n\n{c}\n---\n"

        # If promotional_text is provided, instruct Gemini to add a final promotional section in Thai
        promo_instructions = ""
        if promotional_text:
            # Extract project name from promotional_text for use in instructions
            project_name = ""
            if "BTC Bull Token" in promotional_text:
                project_name = "BTC Bull Token"
            elif "BTC Bull" in promotional_text or "Bitcoin Bull" in promotional_text:
                project_name = "BTC Bull"
            elif "Solaxy" in promotional_text:
                project_name = "Solaxy"
            elif "Best Wallet" in promotional_text:
                project_name = "Best Wallet"
            elif "Mind of Pepe" in promotional_text:
                project_name = "Mind of Pepe"
            elif "Meme Index" in promotional_text:
                project_name = "Meme Index"

            
            promo_instructions = f"""IMPORTANT: In addition to the {section_count} main content sections, add ONE FINAL promotional section (before the conclusion). This promotional section is EXTRA and not counted in the {section_count} main sections. Here's the source promotional text:

{promotional_text}

Using the source promotional text, create the promotional section in Thai as follows:
- Create a natural-looking H2 heading that seamlessly blends with previous headings. DO NOT use "โปรโมชัน:" prefix.
- The heading should incorporate both {primary_keyword} and {project_name} naturally, making it look like a regular content section.
- Make the heading semantically relevant to both the main content and the promotional content.
- LIMIT TO EXACTLY 3 PARAGRAPHS TOTAL. Choose the most relevant parts of the promotional text to create semantically relevant paragraphs.
- Ensure the section flows naturally from previous sections and maintains the article's tone and style.
- Make sure to mention the project name (e.g., "{project_name}") multiple times in the promotional section text."""

        prompt = f"""
You are an expert Thai crypto journalist and SEO specialist. Using ONLY the exact source content provided below, craft a COMPREHENSIVE SEO-optimized article in Thai. DO NOT invent or modify any factual details. 
Your article must reflect the provided source text.
Produce the article with an active and direct tone using a subject-verb-object structure. For instance, instead of writing 'ราคา Etheruem จะถูกดันโดยการอัพเกรด?', write 'การอัพเกรด Pectra จะช่วยดันราคา Ethereum หรือไม่?'
Keep crypto names and entity names in English (e.g., 'Donald Trump', not 'โดนัลด์ ทรัมป์'; 'Bitcoin', not 'บิตคอยน์'). The rest of the article should be in Thai.

Secondary Keywords: {secondary_keywords}
News Angle: {news_angle}

IMPORTANT TITLE INSTRUCTIONS:
- RULE #1 - DO NOT PREFIX: ABSOLUTELY DO NOT start the title with the keyword followed by a colon or any direct prefixing. For example, if the keyword is 'Trump Coin', AVOID starting the title with 'Trump Coin: ...'. This is an incorrect and unnatural format.
- RULE #2 - NATURAL INTEGRATION: The user-provided keyword (e.g., 'Trump Coin') MUST be woven NATURALLY into the title. It should appear as part of a compelling, contextually relevant sentence or phrase, as demonstrated in ALL provided good examples. The goal is human-like, seamless integration, not forced placement.
- RULE #3 - ALL COMPONENTS PRESENT & KEYWORD IS KING: The title MUST include the entity, the user-provided {primary_keyword} (integrated naturally as per Rule #2), and an interesting point or fact. The {primary_keyword} is the most critical component and ABSOLUTELY MUST be present. All three components must be present.
- RULE #4 - FLEXIBLE ARRANGEMENT: Arrange all components (entity, keyword, fact) in an order that is most natural and contextually relevant for the story. Do not adhere to a fixed left-to-right order.
- RULE #5 - LENGTH: Title length MUST be between 50 and 60 characters. If the title is too short or too long, rewrite it to fit this range.
- RULE #6 - ENGAGEMENT & STYLE: Make the title highly emotionally engaging, news-style, and natural in Thai (keep entities in English). Achieve this by using techniques seen in the effective examples, such as: strong action verbs (e.g., พุ่งชน, ฟันธง, กลับลำ, เทขาย), specific numbers that create impact (e.g., price points like $107,000, percentages, large monetary values), highlighting conflict/drama/urgency (e.g., โดนฟ้อง, สงครามเดือด, พลาดกำไร), referencing authoritative figures or entities, or posing intriguing questions/stakes (e.g., ใครช้าอาจไม่ทันรวย).
- Example flexible arrangements:
  - "มหาเศรษฐีจีนร่วมงานดินเนอร์คริปโตของ Trump จะดัน Trump Coin หรือไม่"
  - "ดินเนอร์คริปโตสุดหรู: จะส่งอิทธิพลต่อ Trump Coin อย่างไร"
  - "จะพุ่งหรือจะร่วง จับตา Trump Coin หลัง Trump ประกาศงานดินเนอร์คริปโต"
- Examples of highly effective titles (analyze these for patterns):
  - Example 1:
    - Headline: "Arthur Hayes ฟันธง! Altcoin Season จะมาก็ต่อเมื่อราคา Bitcoin พุ่งทะลุ $110,000"
    - Entity: Arthur Hayes
    - Keyword: Altcoin Season
    - Urgency/Interesting Point: พุ่งทะลุ $110,000
    - Note: ✔ ใช้ทั้ง Authority (Entity) + Keyword + Future Promise
  - Example 2:
    - Headline: "Bitcoin พุ่งทะลุ $106,000 หลังรองประธาน Fed ส่งสัญญาณ “พร้อมอัดฉีดสภาพคล่อง”"
    - Entity: Bitcoin / Fed
    - Keyword: Bitcoin
    - Urgency/Interesting Point: พร้อมอัดฉีดสภาพคล่อง
    - Note: ✔ ใช้ตัวเลขแรง + Entity + เหตุการณ์
  - Example 3:
    - Headline: "Strategy ซื้อ Bitcoin เพิ่มอีก 765 ล้านดอลลาร์ แต่โดนฟ้องเรื่องแกล้งผู้ถือ!"
    - Entity: Strategy
    - Keyword: ซื้อ Bitcoin
    - Urgency/Interesting Point: ฟ้องผู้ถือ / 765 ล้านดอลลาร์
    - Note: ✔ มีตัวเลข + ความดราม่า = คลิกได้
  - Example 4:
    - Headline: "JPMorgan กลับลำ? ยอมเปิดให้ลูกค้าซื้อขาย Bitcoin ได้ แต่ยังไม่รับ Custody เต็มรูป"
    - Entity: JPMorgan
    - Keyword: Bitcoin
    - Urgency/Interesting Point: กลับลำ / ยังไม่รับ custody
    - Note: ✔ ชัดเจน + มีความขัดแย้ง = กระตุ้น curiosity
- Further examples of highly effective headlines seen in Google Discover (analyze these for patterns of strong verbs, numbers, conflict, and intrigue):
  - "ราคา Bitcoin พุ่งชน $107,000 ห่าง ATH แค่ 1.9% นักเทรดแห่เปิดสัญญาฟิวเจอร์สลุ้นเบรคสถิติใหม่"
  - "บิทคอยน์ส่งสัญญาณพุ่งทะลุสถิติสร้างนิวไฮรอบใหม่"
  - "JPMorgan เปิดให้ลูกค้าซื้อ Bitcoin ครั้งแรก แม้ซีอีโอ ยืนกรานว่า ไม่เชื่อ และมองว่าไร้ค่า"
  - "4 หุ้นคริปโตวิ่ง! รับ “เจพีมอร์แกน” ไฟเขียวลูกค้าซื้อบิตคอยน์ ดันราคาทะลุ 106,000 เหรียญ"
  - "ผลสำรวจใหม่เผย ! ชาวอเมริกันเริ่มทิ้งทองคำและหันไปเก็บ Bitcoin แทน"
  - "รัฐบาลเยอรมันพลาดกำไรกว่า 2.3 พันล้านดอลลาร์! เหตุเทขาย Bitcoin ที่ $57,000"
  - "Michael Saylor ชี้ชัด! ศึกแย่งซื้อ Bitcoin จะถึงจุดจบในปี 2035 ใครช้าอาจไม่ทันรวย"
  - "Saylor เผยการคาดการณ์ราคา Bitcoin ใหม่ – ไม่ใช่ 13 ล้านดอลลาร์"
  - "Strategy งานเข้า! โดนฟ้องร้องแบบกลุ่ม ฐานให้ข้อมูลเท็จเกี่ยวกับกลยุทธ์ลงทุนใน Bitcoin"
  - "Metaplanet ทุบสถิติ! เก็บ BTC เพิ่ม 1,004 เหรียญดันยอดถือครองทะลุ 7,800 เหรียญ"
  - "Strategy แห่งอินโดนีเซียมาแล้ว! DigiAsia เตรียมระดมทุน $100 ล้าน ซื้อ Bitcoin ดันหุ้นพุ่ง 90%"
  - "เจ้ามือกระเป๋าหนัก ! มั่นใจ “ขาขึ้น” ยังอีกยาว เปิด Long Bitcoin 40X มูลค่ากว่า 1 หมื่นล้านบาท"
  - "Jim Cramer เตือน ! อย่ากลัวเกินเหตุ หลัง Moody’s ลดอันดับสหรัฐฯ ชี้ Bitcoin-ทองคำ คือทางออก"
  - "ค่าธรรมเนียมธุรกรรม Bitcoin พุ่งทำสถิติสูงสุดในปี 2025 หลังราคาเหรียญเฉียด $106,000"
  - "สงครามเดือด! “Sats” ปะทะ “Bits” ชื่อไหนควรจะถูกตั้งเป็นหน่วยย่อยของ Bitcoin"
- Heading level 1 (H1) instruction:
- Must align with the title but be more engaging.
- MUST include the Entity, the primary Keyword (e.g., 'Trump Coin' if central to the news), and the most newsworthy fact in a highly engaging way. Ensure the keyword is integrated naturally and not forced.
- Should be in Thai but keep entity names in English.
- Can be slighly longer than Title.

IMPORTANT HEADING INSTRUCTIONS:
- Each section heading (H2) should prioritize the entity, keyword, and interesting fact, but arrange them in the most natural and contextually relevant way for the section. Do NOT force or awkwardly insert the keyword at the start of every heading, and avoid repetition. Only use the keyword where it adds value or clarity. 
- Always keep entity names in English, not Thai transliteration.
- Headings should provide clear context for the section—aim for 12–20 words or 50–120 characters.
- Avoid short, generic, or awkward headings. Headings should read as natural, and engaging.
- Example flexible arrangements:
  - "ผลกระทบของงานดินเนอร์คริปโตต่อ Trump Coin และตลาด"
  - "บทบาทของมหาเศรษฐีจีนในงานดินเนอร์คริปโตต่อ Trump Coin"

CONTENT COMPREHENSIVENESS INSTRUCTIONS:
- IMPORTANT: Your Thai article must be AS COMPLETE AND DETAILED as the original source material.
- Include ALL important facts, figures, quotes, and context from the original content.
- Make sure to cover all aspects mentioned in the original article, including:
  * All specific details about the ETF filing (dates, companies involved, processes)
  * Market analysis and expert opinions
  * Historical context and background information
  * Related initiatives or partnerships mentioned
  * Statistical data and market positions
  * Future prospects and predictions
- Write detailed paragraphs with multiple sentences rather than brief summaries.
- Each section should thoroughly explore its topic with substantial content (3-4 paragraphs per section).
- The intro should be comprehensive, setting up the complete story.
- The conclusion should summarize all key points and implications.

Generate exactly {section_count} main sections. Each section must be detailed and comprehensive.

{promo_instructions}

Structure the JSON as:
{{
 "title": "...",
 "content": {{
    "intro": "...",
    "sections": [
       {{
         "heading": "H2 heading in Thai that NATURALLY includes the {primary_keyword}, the main entity, and an interesting fact/angle relevant to this section. Prioritize natural inclusion of {primary_keyword}.",
         "paragraphs": [...]
       }},
       ...
    ],
    "conclusion": "..."
 }},
 "sources": [...],
 "seo": {{
     "slug": "Concise english slug (6-10 words max) with {primary_keyword} that reflects the news content. Use lowercase-hyphenated format, letters/numbers/hyphens only",
     "metaTitle": "Thai SEO title with {primary_keyword}",
     "metaDescription": "Create a click-worthy Thai description (up to 155 characters) that MUST create curiosity and make readers want to know more. The primary keyword must appear within the first 4–5 words. Do NOT use hashtags. Do NOT repeat or copy the title; the meta description should complement, not mirror, the title. Use a semi-formal, curiosity-building tone. Keep entities (people, companies, cities, coin names, platform or projects etc.) in English, rest is Thai.",
     "excerpt": "...",
    "imagePrompt": "English prompt only",
    "altText": "Thai alt text with {primary_keyword}"
 }}
}}

Below is the source content (in markdown):
{source_texts}

Return ONLY valid JSON, no commentary.
"""
        with st.expander("Prompt being sent to Gemini API", expanded=False):
            st.code(prompt, language='python')
        response = make_openrouter_request(client, prompt)
        if not response:
            return {}
        if isinstance(response, dict):
            response = json.dumps(response)
        try:
            content_json = json.loads(response)
            if isinstance(content_json, list):
                content_json = content_json[0] if content_json else {}
                
            # Separate main vs. promotional sections
            if promotional_text and promotional_text.strip() != "None" and 'content' in content_json and 'sections' in content_json['content']:
                main_sections = []
                promo_sections = []
                
                # Extract project name from promotional_text
                project_name = ""
                if "BTC Bull Token" in promotional_text:
                    project_name = "BTC Bull Token"
                elif "BTC Bull" in promotional_text or "Bitcoin Bull" in promotional_text:
                    project_name = "BTC Bull"
                elif "Solaxy" in promotional_text:
                    project_name = "Solaxy"
                elif "Best Wallet" in promotional_text:
                    project_name = "Best Wallet"
                elif "Mind of Pepe" in promotional_text:
                    project_name = "Mind of Pepe"
                elif "Meme Index" in promotional_text:
                    project_name = "Meme Index"

                
                for sec in content_json['content'].get('sections', []):
                    heading = sec.get('heading', "")
                    paragraphs = sec.get('paragraphs', [])
                    
                    # Get the text of all paragraphs to check content
                    paragraph_text = ' '.join(paragraphs) if paragraphs else ''
                    
                    # Check for promotional section markers in the heading or content
                    is_promo = False
                    
                    # Check if the heading contains promotional markers
                    promo_markers = ["โปรโมชัน", "โปรโมท", "สนับสนุน", "แนะนำ", "ส่งเสริม"]
                    if any(marker in heading.lower() for marker in promo_markers):
                        is_promo = True
                    # Check if the heading contains the project name
                    elif project_name and project_name.lower() in heading.lower():
                        is_promo = True
                    # Check if the project name appears multiple times in the paragraphs
                    elif project_name and project_name.lower() in paragraph_text.lower() and paragraph_text.lower().count(project_name.lower()) >= 2:
                        is_promo = True
                    # Also look for keywords specific to BTC Bull Token in the content
                    elif "btc bull" in paragraph_text.lower() or "btcbull" in paragraph_text.lower():
                        is_promo = True
                    # Last section is likely the promotional section if we have promotional text
                    elif len(content_json['content'].get('sections', [])) > section_count and sec == content_json['content'].get('sections', [])[-1]:
                        is_promo = True
                    
                    # Log detection for debugging
                    if is_promo:
                        st.info(f"Identified promotional section: {heading}")
                    
                    if is_promo:
                        promo_sections.append(sec)
                    else:
                        main_sections.append(sec)
                
                # Force exactly 'section_count' main sections
                if len(main_sections) > section_count:
                    main_sections = main_sections[:section_count]
                
                # Keep only the first promotional section if multiple
                if len(promo_sections) > 1:
                    promo_sections = [promo_sections[0]]
                elif len(promo_sections) == 0 and promotional_text:
                    # If no promotional section was detected but promotional text was provided,
                    # create a simple promotional section
                    st.warning("No promotional section detected in the generated content. Creating a default one.")
                    promo_section = {
                        "heading": f"ทำความรู้จักกับ {project_name} โอกาสใหม่ในวงการ {primary_keyword}",
                        "paragraphs": [
                            f"หนึ่งในโครงการที่น่าสนใจในตลาด {primary_keyword} คือ {project_name} ซึ่งกำลังได้รับความนิยมอย่างมาก",
                            f"{project_name} นำเสนอนวัตกรรมใหม่ที่แตกต่างจากโครงการอื่นๆ ในตลาด โดยมุ่งเน้นการสร้างประโยชน์ให้กับผู้ถือครอง",
                            f"หากคุณกำลังมองหาโอกาสการลงทุนใหม่ๆ ในวงการ {primary_keyword}, {project_name} อาจเป็นตัวเลือกที่น่าสนใจสำหรับการศึกษาเพิ่มเติม"
                        ]
                    }
                    promo_sections.append(promo_section)
                
                # Recombine them in the correct order
                content_json['content']['sections'] = main_sections + promo_sections
            
            return content_json
        except json.JSONDecodeError as e:
            st.error(f"JSON parsing error: {str(e)}")
            st.error("Raw response:")
            st.code(response)
            return {}
    except Exception as e:
        st.error(f"Error generating article: {str(e)}")
        return {}
    finally:
        # Add promotional image to the returned content if we have promotional text
        if 'content_json' in locals() and promotional_text and promotional_text.strip() and promotional_text != "None":
            try:
                # Get the appropriate promotional image data
                project_name = ""
                if "BTC Bull Token" in promotional_text:
                    project_name = "BTC Bull Token"
                elif "BTC Bull" in promotional_text or "Bitcoin Bull" in promotional_text:
                    project_name = "BTC Bull"
                elif "Solaxy" in promotional_text:
                    project_name = "Solaxy"
                elif "Best Wallet" in promotional_text:
                    project_name = "Best Wallet"
                elif "Mind of Pepe" in promotional_text:
                    project_name = "Mind of Pepe"
                elif "Meme Index" in promotional_text:
                    project_name = "Meme Index"

                
                if project_name and project_name in PROMOTIONAL_IMAGES:
                    img_data = PROMOTIONAL_IMAGES[project_name]
                    if 'content_json' in locals() and isinstance(content_json, dict):
                        if 'media' not in content_json:
                            content_json['media'] = {}
                        if 'images' not in content_json['media']:
                            content_json['media']['images'] = []
                        content_json['media']['images'].append({
                            'url': img_data.get("url", ""),
                            'alt': img_data.get("alt", "Promotional Image"),
                            'width': img_data.get("width", "600"),
                            'height': img_data.get("height", "558")
                        })
            except Exception as ex:
                st.error(f"Error adding promotional image: {str(ex)}")

def main():
    st.set_page_config(page_title="Generate and Upload Article", layout="wide")
    
    default_url = ""
    default_keyword = "Bitcoin"
    default_news_angle = ""
    
    gemini_client = init_openrouter_client()
    if not gemini_client:
        st.error("Failed to initialize Gemini client")
        return
    
    urls_input = st.sidebar.text_area("Enter URLs (one per line) to extract from:", value=default_url)
    keywords_input = st.sidebar.text_area("Keywords (one per line):", value=default_keyword, height=100)
    news_angle = st.sidebar.text_input("News Angle:", value=default_news_angle)
    section_count = st.sidebar.slider("Number of sections:", 2, 8, 3)
    
    additional_content = st.sidebar.text_area(
        "Additional Content",
        placeholder="Paste any extra content here. It will be treated as an additional source.",
        height=150
    )
    
    st.sidebar.header("Promotional Content")
    pr_folder = os.path.join(os.path.dirname(__file__), "pr")
    try:
        # Load available promotional options
        available_promos = []
        
        if os.path.isdir(pr_folder):
            # Get files from pr folder
            promo_files = [f.replace('.txt', '') for f in os.listdir(pr_folder) 
                          if f.endswith(".txt") and os.path.isfile(os.path.join(pr_folder, f))]
            if promo_files:
                available_promos.extend(promo_files)
                
            # Add promotional images options if available
            available_promos.extend([k for k in PROMOTIONAL_IMAGES.keys() 
                                  if k not in available_promos])
                
            # Deduplicate and normalize names to prevent duplicates like 'Best Wallet' and 'BestWallet'
            normalized_promos = {}
            for promo in available_promos:
                # Normalize by removing spaces and converting to lowercase
                norm_key = promo.replace(' ', '').lower()
                # Keep the prettier formatted version when duplicates are found
                if norm_key not in normalized_promos or len(promo) < len(normalized_promos[norm_key]):
                    normalized_promos[norm_key] = promo
            
            # Convert back to list and sort
            available_promos = sorted(normalized_promos.values())
            
            # Include None option
            display_options = ["None"] + available_promos
            
            # Display selectbox with all options
            selected_promo = st.sidebar.selectbox("Select promotional content:", display_options)
            st.sidebar.caption("Promotional content will be integrated into the article")
            
            # Load the actual content if a promo is selected
            if selected_promo != "None":
                # Load content from file with proper error handling
                promotional_text = load_promotional_content(selected_promo)
                if promotional_text:
                    st.sidebar.success(f"Loaded promotional content: {selected_promo}")
                    # Log that content will be included in the Gemini prompt
                    st.sidebar.info("This content will be included in the prompt sent to Gemini")
                else:
                    st.sidebar.warning(f"Selected promotional content '{selected_promo}' could not be loaded")
                    promotional_text = selected_promo  # Fallback to using the name directly
            else:
                promotional_text = None
        else:
            st.sidebar.warning("'pr' folder not found")
            promotional_text = None
    except Exception as e:
        logging.error(f"Error loading promotional content options: {str(e)}")
        st.sidebar.error(f"Error loading promotional options: {str(e)}")
        promotional_text = None
    
    st.sidebar.header("Select WordPress Site to Upload")
    sites = {
        "ICOBENCH": {
            "url": os.getenv("ICOBENCH_WP_URL"),
            "username": os.getenv("ICOBENCH_WP_USERNAME"),
            "password": os.getenv("ICOBENCH_WP_APP_PASSWORD")
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
        "CRYPTONEWS": {
            "url": os.getenv("CRYPTONEWS_WP_URL"),
            "username": os.getenv("CRYPTONEWS_WP_USERNAME"),
            "password": os.getenv("CRYPTONEWS_WP_APP_PASSWORD")
        },
        "INSIDEBITCOINS": {
            "url": os.getenv("INSIDEBITCOINS_WP_URL"),
            "username": os.getenv("INSIDEBITCOINS_WP_USERNAME"),
            "password": os.getenv("INSIDEBITCOINS_WP_APP_PASSWORD")
        }
    }
    site_options = list(sites.keys())
    selected_site = st.sidebar.selectbox("Choose a site:", site_options)
    
    wp_url = sites[selected_site]["url"]
    wp_username = sites[selected_site]["username"]
    wp_app_password = sites[selected_site]["password"]
    
    # Radio button to choose between Post or Page
    content_type_choice = st.sidebar.radio("Upload as:", ("Post", "Page"))
    
    messages_placeholder = st.empty()
    
    if st.sidebar.button("Generate Article"):
        transcripts = []
        if urls_input.strip():
            urls = [line.strip() for line in urls_input.splitlines() if line.strip()]
            for url in urls:
                extracted = extract_url_content(gemini_client, url, messages_placeholder)
                if extracted:
                    transcripts.append(extracted)
        if additional_content.strip():
            transcripts.append({
                'content': additional_content.strip(),
                'source': 'Additional Content',
                'url': ''
            })
        if not transcripts:
            st.error("Please provide either URLs or Additional Content to generate an article")
            return
        keywords = [k.strip() for k in keywords_input.splitlines() if k.strip()]
        if transcripts:
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
                
                # Log confirmation about promotional content in the Gemini prompt
                if promotional_text:
                    st.info(f"Promotional content for '{selected_promo}' was included in the Gemini prompt")
            else:
                st.error("Failed to generate article.")
        else:
            st.error("No content extracted from provided URLs.")
    
    if "article" in st.session_state and st.session_state.article:
        # Collapsed by default
        st.subheader("Generated Article (JSON)")
        with st.expander("View/Hide the JSON", expanded=False):
            try:
                article_json = json.loads(st.session_state.article)
                st.json(article_json)
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON format: {str(e)}")
                st.text_area("Raw Content", value=st.session_state.article, height=300)
        
        # Show a quick preview
        try:
            article_json = json.loads(st.session_state.article)
            
            # Display meta description above the title (with error handling)
            try:
                if 'seo' in article_json and 'metaDescription' in article_json['seo']:
                    meta_desc = article_json['seo']['metaDescription']
                    if meta_desc and meta_desc.strip():
                        st.markdown(f"**Meta Description:** *{meta_desc}*")
                        st.markdown("---")
            except Exception as e:
                logging.error(f"Error displaying meta description: {str(e)}")
                # Continue without meta description if there's an error
                
            # Display the title
            st.markdown(f"# {article_json['title']}")
            if isinstance(article_json['content']['intro'], dict):
                st.write(article_json['content']['intro'].get('Part 1', ''))
                st.write(article_json['content']['intro'].get('Part 2', ''))
            else:
                st.write(article_json['content']['intro'])
            for section in article_json['content']['sections']:
                st.write(f"### {section['heading']}")
                format_type = section.get('format', 'paragraph')
                if format_type == 'list':
                    for item in section['paragraphs']:
                        st.write(f"* {item}")
                elif format_type == 'table':
                    if isinstance(section['paragraphs'][0], str) and section['paragraphs'][0].startswith('|'):
                        for row in section['paragraphs']:
                            st.write(row)
                    elif section['paragraphs']:
                        import pandas as pd
                        if isinstance(section['paragraphs'][0], (list, tuple)):
                            df = pd.DataFrame(section['paragraphs'][1:], columns=section['paragraphs'][0])
                            st.table(df)
                        elif isinstance(section['paragraphs'][0], dict):
                            df = pd.DataFrame(section['paragraphs'])
                            st.table(df)
                else:
                    for para in section['paragraphs']:
                        st.write(para)
                st.write("")
            st.markdown("## บทสรุป")
            st.write(article_json['content']['conclusion'])
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON format: {str(e)}")
        
        # Prepare processed article
        if "article_data" not in st.session_state:
            st.session_state.article_data = {}
        if "processed_article" not in st.session_state.article_data:
            # Determine if we should add the affiliate note
            include_promotional = promotional_text is not None
            parsed = parse_article(st.session_state.article, add_affiliate_note=include_promotional)
            st.session_state.article_data["processed_article"] = parsed
        
        # Together AI Image Generation
        if "image" not in st.session_state.article_data:
            parsed_for_image = parse_article(st.session_state.article)
            image_prompt = parsed_for_image.get("image_prompt")
            alt_text = parsed_for_image.get("image_alt")
            if image_prompt:
                st.info(f"Original image prompt: '{image_prompt}'")
                image_prompt_english = re.sub(r'[\u0E00-\u0E7F]+', '', image_prompt).strip()
                if not image_prompt_english:
                    image_prompt_english = "A photo-realistic scene of cryptocurrencies floating in the air, depicting the Crypto news"
                    st.warning("Using fallback image prompt since no English text was found")
                st.info(f"Cleaned English prompt for Together AI: '{image_prompt_english}'")
            else:
                image_prompt_english = None
            
            if False: # Original condition: image_prompt_english
                pass
            else:
                st.info("Using previously generated image.")
        
        if st.button("Upload Article to WordPress"):
            if not all([wp_url, wp_username, wp_app_password]):
                st.error("Please ensure your .env file has valid credentials for the selected site.")
            else:
                try:
                    parsed = st.session_state.article_data["processed_article"]
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
                    # Prepare article data
                    article_data = {
                        "main_title": parsed.get("main_title", "No Title"),
                        "main_content": parsed.get("main_content", ""),
                        "seo_slug": parsed.get("seo_slug", ""),
                        "excerpt": parsed.get("excerpt", ""),
                        "yoast_title": parsed.get("yoast_title", ""),
                        "yoast_metadesc": parsed.get("yoast_metadesc", ""),
                        "image": st.session_state.article_data.get("image") if "image" in st.session_state.article_data else {}
                    }
                    
                    # Add featured image if available
                    if media_info and "media_id" in media_info:
                        article_data["image"]["media_id"] = media_info["media_id"]
                    
                    # Get primary keyword for upload
                    primary_keyword_upload = keywords_input.splitlines()[0] if keywords_input.strip() else ""
                    
                    # Convert markdown to HTML
                    article_data["main_content"] = markdown.markdown(article_data["main_content"])
                    
                    # Process content with promotional image and affiliate links
                    if selected_promo != "None" and selected_site:
                        try:
                            article_data["main_content"] = process_affiliate_links(
                                article_data["main_content"],
                                selected_site,
                                selected_promo
                            )
                            st.success(f"Added promotional content for {selected_promo}")
                        except Exception as e:
                            st.error(f"Error processing affiliate links: {str(e)}")
                    
                    submit_article_to_wordpress(
                        article_data, 
                        wp_url, 
                        wp_username, 
                        wp_app_password, 
                        primary_keyword=primary_keyword_upload,
                        site_name=selected_site,
                        content_type=content_type_choice
                    )
                except Exception as e:
                    st.error(f"Error during upload process: {str(e)}")

if __name__ == "__main__":
    main()
