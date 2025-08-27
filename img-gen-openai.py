import streamlit as st
import os
from dotenv import load_dotenv
import base64
from io import BytesIO
from PIL import Image
from openai import OpenAI
import requests
from requests.auth import HTTPBasicAuth
import json
import re
import time

# Load environment variables from .env file
load_dotenv()

st.set_page_config(page_title="AI Image Generator & Uploader", layout="wide")

# ==============================
# --- WordPress Upload Functions
# ==============================
def construct_endpoint(wp_url, endpoint_path):
    wp_url = wp_url.rstrip('/')
    # Specific logic from search2.py for /th path
    if not any(domain in wp_url for domain in ["bitcoinist.com", "newsbtc.com"]) and "/th" not in wp_url:
        wp_url += "/th"
    return f"{wp_url}{endpoint_path}"

def upload_image_to_wordpress(image_bytes, wp_url, username, wp_app_password, filename="generated_image.jpg", alt_text="Generated Image"):
    """
    Uploads image bytes to WordPress via the REST API.
    image_bytes: Raw bytes of the image (JPEG format expected).
    """
    media_endpoint = construct_endpoint(wp_url, "/wp-json/wp/v2/media")
    try:
        files = {'file': (filename, image_bytes, 'image/jpeg')}
        data_payload = {'alt_text': alt_text, 'title': alt_text}

        response = requests.post(
            media_endpoint,
            files=files,
            data=data_payload,
            auth=HTTPBasicAuth(username, wp_app_password)
        )

        if response.status_code in (200, 201):
            media_data = response.json()
            media_id = media_data.get('id')
            source_url = media_data.get('source_url', '')

            # Optional metadata update if initial POST didn't set alt/title
            update_payload = {}
            if alt_text and media_data.get('alt_text') != alt_text:
                update_payload['alt_text'] = alt_text
            if alt_text and media_data.get('title', {}).get('raw') != alt_text:
                update_payload['title'] = alt_text

            if update_payload:
                update_endpoint = f"{media_endpoint}/{media_id}"
                update_response = requests.post(
                    update_endpoint,
                    json=update_payload,
                    auth=HTTPBasicAuth(username, wp_app_password)
                )
                if update_response.status_code in (200, 201, 200):
                    st.success(f"Image uploaded and metadata updated for {wp_url}. Media ID: {media_id}")
                else:
                    st.warning(f"Image uploaded to {wp_url} (ID: {media_id}), but metadata update failed. "
                               f"Status: {update_response.status_code}, Response: {update_response.text}")
            else:
                st.success(f"Image uploaded to {wp_url}. Media ID: {media_id}")
            return {"media_id": media_id, "source_url": source_url}
        else:
            st.error(f"Image upload to {wp_url} failed. Status: {response.status_code}, Response: {response.text}")
            return None
    except Exception as e:
        st.error(f"Exception during image upload to {wp_url}: {e}")
        return None

# ===================================================
# --- Image Processing (resize/optimize for WordPress)
# ===================================================
def process_image_for_wordpress(image_bytes, final_quality=80):
    """
    Intelligently resizes images for WordPress based on their aspect ratio and orientation.
    - Landscape images: Max width 1200px
    - Portrait images: Max height 675px (16:9 ratio)
    - Very wide banners: Max width 1400px, min height 300px
    - Very tall images: Max height 800px, min width 400px
    """
    try:
        img = Image.open(BytesIO(image_bytes))
