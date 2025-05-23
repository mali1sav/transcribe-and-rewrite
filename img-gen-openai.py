import streamlit as st
import os
from dotenv import load_dotenv
import base64
from io import BytesIO
from PIL import Image
from openai import OpenAI
import requests
from requests.auth import HTTPBasicAuth

# Load environment variables from .env file
load_dotenv()

st.set_page_config(page_title="OpenAI Image Generator & Uploader", layout="wide")

# --- WordPress Upload Functions (adapted from search2.py) ---
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
    # st.write(f"[Upload] Attempting to upload {filename} to {media_endpoint} with alt text: {alt_text}")
    try:
        files = {'file': (filename, image_bytes, 'image/jpeg')} # Ensure MIME type is jpeg
        # WordPress uses the 'title' field from the file upload for the media item's title,
        # and 'alt_text' can be set in the 'data' payload for some setups, or updated later.
        # For robust alt text, it's often better to update post-upload.
        data_payload = {'alt_text': alt_text, 'title': alt_text, 'caption': alt_text} 

        response = requests.post(
            media_endpoint, 
            files=files, 
            data=data_payload, # Send alt_text and title here
            auth=HTTPBasicAuth(username, wp_app_password)
        )
        
        # st.write(f"[Upload] Initial upload response status: {response.status_code}")
        if response.status_code in (200, 201):
            media_data = response.json()
            media_id = media_data.get('id')
            source_url = media_data.get('source_url', '')
            # st.write(f"[Upload] Received Media ID: {media_id}, Source URL: {source_url}")

            # Optional: Update alt text, title, caption via PATCH if initial POST doesn't set them reliably
            # Some WordPress setups might require this, others might not.
            # The original script had a PATCH, let's keep it for consistency but make it more targeted.
            update_payload = {}
            if alt_text and media_data.get('alt_text') != alt_text:
                update_payload['alt_text'] = alt_text
            if alt_text and media_data.get('title', {}).get('raw') != alt_text: # Title is an object
                 update_payload['title'] = alt_text 
            if alt_text and media_data.get('caption', {}).get('raw') != alt_text: # Caption is an object
                 update_payload['caption'] = alt_text

            if update_payload: # Only PATCH if there's something to update
                update_endpoint = f"{media_endpoint}/{media_id}"
                # st.write(f"[Upload] Attempting to update media with payload: {update_payload}")
                update_response = requests.post( # WP often uses POST for updates on media too
                    update_endpoint, 
                    json=update_payload, 
                    auth=HTTPBasicAuth(username, wp_app_password)
                )
                # st.write(f"[Upload] Update response status: {update_response.status_code}")
                if update_response.status_code in (200, 201, 200):
                    st.success(f"Image uploaded and metadata updated for {wp_url}. Media ID: {media_id}")
                else:
                    st.warning(f"Image uploaded to {wp_url} (ID: {media_id}), but metadata update failed. Status: {update_response.status_code}, Response: {update_response.text}")
            else:
                 st.success(f"Image uploaded to {wp_url}. Media ID: {media_id}")
            return {"media_id": media_id, "source_url": source_url}
        else:
            st.error(f"Image upload to {wp_url} failed. Status: {response.status_code}, Response: {response.text}")
            return None
    except Exception as e:
        st.error(f"Exception during image upload to {wp_url}: {e}")
        return None

# --- End WordPress Upload Functions ---

# Initialize session state
if 'generated_image_data' not in st.session_state:
    st.session_state.generated_image_data = None # This will store BytesIO object
if 'generated_image_prompt' not in st.session_state:
    st.session_state.generated_image_prompt = None

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# WordPress Site Configurations
WP_SITES = {
    "cryptonews": {
        "url": os.getenv("CRYPTONEWS_WP_URL"),
        "username": os.getenv("CRYPTONEWS_WP_USERNAME"),
        "password": os.getenv("CRYPTONEWS_WP_APP_PASSWORD")
    },
    "icobench": {
        "url": os.getenv("ICOBENCH_WP_URL"),
        "username": os.getenv("ICOBENCH_WP_USERNAME"),
        "password": os.getenv("ICOBENCH_WP_APP_PASSWORD")
    },
    "bitcoinist": {
        "url": os.getenv("BITCOINIST_WP_URL"),
        "username": os.getenv("BITCOINIST_WP_USERNAME"),
        "password": os.getenv("BITCOINIST_WP_APP_PASSWORD")
    }
}

def process_image_for_wordpress(image_bytes, target_width=1200, final_quality=80):
    """Resizes an image to a target width and saves it as JPEG at a specific quality."""
    try:
        img = Image.open(BytesIO(image_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        current_width, current_height = img.size
        if current_width == 0: raise ValueError("Image width is zero.")
        aspect_ratio = float(current_height) / float(current_width)
        new_height = int(target_width * aspect_ratio)
        img_resized = img.resize((target_width, new_height), Image.Resampling.LANCZOS)
        jpeg_bytes_io = BytesIO()
        img_resized.save(jpeg_bytes_io, format="JPEG", quality=final_quality, optimize=True)
        jpeg_bytes_io.seek(0)
        return jpeg_bytes_io
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

st.title("OpenAI Image Generator & WordPress Uploader")

if not OPENAI_API_KEY:
    st.error("OpenAI API key not found. Please set OPENAI_API_KEY in .env")
else:
    openai_client = OpenAI()
    with st.form("openai_image_form"):
        prompt_text = st.text_area("Enter your image prompt (English only):", height=80, key="prompt_input")
        submitted = st.form_submit_button("Generate Image")
        
        if submitted and prompt_text.strip():
            st.session_state.generated_image_data = None # Reset on new generation
            st.session_state.generated_image_prompt = None
            with st.spinner("Generating and processing image with OpenAI..."):
                try:
                    response = openai_client.images.generate(
                        model="gpt-image-1",
                        prompt=prompt_text,
                        n=1,
                        size="1536x1024"
                    )
                    if response.data and response.data[0].b64_json:
                        b64_image_data = response.data[0].b64_json
                        image_bytes_from_api = base64.b64decode(b64_image_data)
                        final_image_bytes_io = process_image_for_wordpress(image_bytes_from_api)
                        
                        if final_image_bytes_io:
                            st.session_state.generated_image_data = final_image_bytes_io
                            st.session_state.generated_image_prompt = prompt_text
                            st.image(final_image_bytes_io, caption=prompt_text[:50] + "...")
                        else:
                            st.error("Failed to process the generated image.")
                    else:
                        st.error("No image data received from OpenAI.")
                except Exception as e:
                    st.error(f"Failed to generate or process image: {e}")

# --- UI for Download and Upload (Outside the form) ---
if st.session_state.generated_image_data and st.session_state.generated_image_prompt:
    st.markdown("---_Upload to WordPress_---")
    
    current_prompt = st.session_state.generated_image_prompt
    image_data_bytesio = st.session_state.generated_image_data # This is a BytesIO object
    
    # Ensure BytesIO is ready for reading initially
    image_data_bytesio.seek(0) 
    
    # Prepare common upload parameters
    clean_prompt_filename = "".join(c if c.isalnum() or c in (' ', '_') else '' for c in current_prompt[:30]).rstrip().replace(' ', '_')
    upload_filename = f"{clean_prompt_filename}_processed.jpg" if clean_prompt_filename else "processed_image.jpg"
    alt_text_for_upload = current_prompt[:100]

    # Create columns for buttons
    col1, col2, col3 = st.columns(3)

    # Button for Cryptonews
    with col1:
        if st.button("Upload to Cryptonews", key="upload_cryptonews"):
            site_key = "cryptonews"
            site_config = WP_SITES.get(site_key)
            if site_config and all(site_config.get(k) for k in ["url", "username", "password"]):
                st.markdown(f"**Uploading to {site_key}...**")
                image_data_bytesio.seek(0) # Rewind for this upload
                upload_image_bytes = image_data_bytesio.getvalue()
                with st.spinner(f"Uploading to {site_key}..."):
                    upload_image_to_wordpress(
                        image_bytes=upload_image_bytes,
                        wp_url=site_config["url"],
                        username=site_config["username"],
                        wp_app_password=site_config["password"],
                        filename=upload_filename,
                        alt_text=alt_text_for_upload
                    )
            else:
                st.error(f"Missing or incomplete configuration for {site_key}. Check .env file.")

    # Button for ICOBench
    with col2:
        if st.button("Upload to ICOBench", key="upload_icobench"):
            site_key = "icobench"
            site_config = WP_SITES.get(site_key)
            if site_config and all(site_config.get(k) for k in ["url", "username", "password"]):
                st.markdown(f"**Uploading to {site_key}...**")
                image_data_bytesio.seek(0) # Rewind for this upload
                upload_image_bytes = image_data_bytesio.getvalue()
                with st.spinner(f"Uploading to {site_key}..."):
                    upload_image_to_wordpress(
                        image_bytes=upload_image_bytes,
                        wp_url=site_config["url"],
                        username=site_config["username"],
                        wp_app_password=site_config["password"],
                        filename=upload_filename,
                        alt_text=alt_text_for_upload
                    )
            else:
                st.error(f"Missing or incomplete configuration for {site_key}. Check .env file.")

    # Button for Bitcoinist
    with col3:
        if st.button("Upload to Bitcoinist", key="upload_bitcoinist"):
            site_key = "bitcoinist"
            site_config = WP_SITES.get(site_key)
            if site_config and all(site_config.get(k) for k in ["url", "username", "password"]):
                st.markdown(f"**Uploading to {site_key}...**")
                image_data_bytesio.seek(0) # Rewind for this upload
                upload_image_bytes = image_data_bytesio.getvalue()
                with st.spinner(f"Uploading to {site_key}..."):
                    upload_image_to_wordpress(
                        image_bytes=upload_image_bytes,
                        wp_url=site_config["url"],
                        username=site_config["username"],
                        wp_app_password=site_config["password"],
                        filename=upload_filename,
                        alt_text=alt_text_for_upload
                    )
            else:
                st.error(f"Missing or incomplete configuration for {site_key}. Check .env file.")
