import os
import json
import requests
from requests.auth import HTTPBasicAuth
import streamlit as st

def load_article(file_path="generated_articles.json"):
    """Load article data from a JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading article from {file_path}: {str(e)}")
        return None

def upload_image_to_wordpress(b64_data, wp_url, username, app_password, filename="generated_image.png", alt_text="Generated Image"):
    """
    Uploads an image (provided as a base64 string) to WordPress via the REST API.
    Returns the media ID if successful.
    """
    # Decode the base64 image data to binary
    try:
        image_bytes = base64.b64decode(b64_data)
    except Exception as e:
        print(f"Failed to decode base64 image data: {str(e)}")
        return None

    # Define the WordPress media endpoint
    media_endpoint = f"{wp_url.rstrip('/')}/wp-json/wp/v2/media"

    # Set up headers with a proper filename and content type (assuming PNG)
    headers = {
        'Content-Disposition': f'attachment; filename={filename}',
        'Content-Type': 'image/png'
    }

    # Post the binary data to WordPress
    response = requests.post(
        media_endpoint,
        headers=headers,
        data=image_bytes,
        auth=HTTPBasicAuth(username, app_password)
    )
    
    if response.status_code in (200, 201):
        media_data = response.json()
        print(f"Image uploaded successfully. Media ID: {media_data.get('id')}")
        return media_data.get('id')
    else:
        print(f"Image upload failed. Status code: {response.status_code}, Response: {response.text}")
        return None

def submit_article_to_wordpress(article, wp_url, username, app_password):
    """
    Submits a single article to WordPress as a draft.
    If the article contains an image, it uploads the image and attaches it as the featured image.
    """
    endpoint = f"{wp_url.rstrip('/')}/wp-json/wp/v2/posts"
    
    # If the article includes an image, upload it first
    featured_media = None
    if "image" in article:
        image_data = article["image"].get("b64_data")
        alt_text = article["image"].get("alt_text", "Generated Image")
        if image_data:
            featured_media = upload_image_to_wordpress(image_data, wp_url, username, app_password, alt_text=alt_text)
    
    # Prepare the post payload
    data = {
        "title": article.get("title", "Untitled"),
        "content": article.get("content", ""),
        "status": "draft",  # This ensures the post is saved as a draft.
        "slug": article.get("slug", ""),
        "excerpt": article.get("excerpt", ""),
        "meta": {
            "meta_description": article.get("meta_description", "")
        }
    }
    if featured_media:
        data["featured_media"] = featured_media
    
    # Submit the article to WordPress
    response = requests.post(endpoint, json=data, auth=HTTPBasicAuth(username, app_password))
    if response.status_code in (200, 201):
        post = response.json()
        print(f"Article '{article.get('title')}' submitted successfully! Post ID: {post.get('id')}")
        return post
    else:
        print(f"Failed to submit article '{article.get('title')}'. Status Code: {response.status_code}")
        print("Response:", response.text)
        return None

def main():
    # Load WordPress credentials from Streamlit secrets
    try:
        wp_url = st.secrets["wp_url"]
        username = st.secrets["wp_username"]
        app_password = st.secrets["wp_app_password"]
    except KeyError as e:
        print(f"Error: Missing WordPress credentials in Streamlit secrets: {e}")
        return

    # Load the article data from the JSON file
    article = load_article()
    if not article:
        print("No article data found.")
        return

    # Submit the article to WordPress
    submit_article_to_wordpress(article, wp_url, username, app_password)

if __name__ == "__main__":
    main()
