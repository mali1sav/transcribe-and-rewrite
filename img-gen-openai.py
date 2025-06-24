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

# --- WordPress Upload Functions ---
def construct_endpoint(wp_url, endpoint_path):
    wp_url = wp_url.rstrip('/')
    if not any(domain in wp_url for domain in ["bitcoinist.com", "newsbtc.com"]) and "/th" not in wp_url:
        wp_url += "/th"
    return f"{wp_url}{endpoint_path}"

def upload_image_to_wordpress(image_bytes, wp_url, username, wp_app_password, filename="generated_image.jpg", alt_text="Generated Image"):
    media_endpoint = construct_endpoint(wp_url, "/wp-json/wp/v2/media")
    try:
        files = {'file': (filename, image_bytes, 'image/jpeg')}
        data_payload = {'alt_text': alt_text, 'title': alt_text, 'caption': alt_text} 

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
            
            update_payload = {}
            if alt_text and media_data.get('alt_text') != alt_text:
                update_payload['alt_text'] = alt_text
            if alt_text and media_data.get('title', {}).get('raw') != alt_text:
                 update_payload['title'] = alt_text 
            if alt_text and media_data.get('caption', {}).get('raw') != alt_text:
                 update_payload['caption'] = alt_text

            if update_payload:
                update_endpoint = f"{media_endpoint}/{media_id}"
                update_response = requests.post(
                    update_endpoint, 
                    json=update_payload, 
                    auth=HTTPBasicAuth(username, wp_app_password)
                )
                if update_response.status_code in (200, 201):
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
if 'active_image_bytes_io' not in st.session_state:
    st.session_state.active_image_bytes_io = None
if 'active_image_alt_text' not in st.session_state:
    st.session_state.active_image_alt_text = None
if 'user_uploaded_raw_bytes' not in st.session_state:
    st.session_state.user_uploaded_raw_bytes = None
# This specific state for user_uploaded_alt_text_input might be redundant if active_image_alt_text is always the source of truth
# but can be kept for clarity during the user upload processing step.
if 'user_uploaded_alt_text_input_field_value' not in st.session_state: 
    st.session_state.user_uploaded_alt_text_input_field_value = ""


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

WP_SITES = {
    "cryptonews": {"url": os.getenv("CRYPTONEWS_WP_URL"), "username": os.getenv("CRYPTONEWS_WP_USERNAME"), "password": os.getenv("CRYPTONEWS_WP_APP_PASSWORD")},
    "cryptodnes": {"url": os.getenv("CRYPTODNES_WP_URL"), "username": os.getenv("CRYPTODNES_WP_USERNAME"), "password": os.getenv("CRYPTODNES_WP_APP_PASSWORD")},
    "icobench": {"url": os.getenv("ICOBENCH_WP_URL"), "username": os.getenv("ICOBENCH_WP_USERNAME"), "password": os.getenv("ICOBENCH_WP_APP_PASSWORD")},
    "bitcoinist": {"url": os.getenv("BITCOINIST_WP_URL"), "username": os.getenv("BITCOINIST_WP_USERNAME"), "password": os.getenv("BITCOINIST_WP_APP_PASSWORD")}
}

def process_image_for_wordpress(image_bytes, target_width=1200, final_quality=80):
    try:
        img = Image.open(BytesIO(image_bytes))
        if img.mode != 'RGB': img = img.convert('RGB')
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
    # --- Form for OpenAI Image Generation ---
    with st.form("openai_image_form"):
        st.subheader("Generate Image with OpenAI")
        prompt_text = st.text_area("Enter your image prompt (English only - for AI generation):", height=80, key="prompt_input")
        alt_text_for_generated_input = st.text_area("Enter Alt text for your generated image:", height=80, key="generated_alt_text_input_val")
        submitted_generate = st.form_submit_button("Generate Image")
        
        if submitted_generate:
            if prompt_text.strip() and alt_text_for_generated_input.strip():
                # Clear previous active image and related user upload states
                st.session_state.active_image_bytes_io = None
                st.session_state.active_image_alt_text = None
                st.session_state.user_uploaded_raw_bytes = None
                st.session_state.user_uploaded_alt_text_input_field_value = ""

                with st.spinner("Generating and processing image with OpenAI..."):
                    try:
                        response = openai_client.images.generate(
                            model="dall-e-3", 
                            prompt=prompt_text,
                            n=1,
                            size="1792x1024", 
                            response_format="b64_json"
                        )
                        if response.data and response.data[0].b64_json:
                            b64_image_data = response.data[0].b64_json
                            image_bytes_from_api = base64.b64decode(b64_image_data)
                            
                            final_image_bytes_io = process_image_for_wordpress(image_bytes_from_api)
                            
                            if final_image_bytes_io:
                                st.session_state.active_image_bytes_io = final_image_bytes_io
                                # Use the specifically provided ALT text for the generated image
                                st.session_state.active_image_alt_text = alt_text_for_generated_input 
                                st.success("AI image generated and processed. It is now the active image.")
                            else:
                                st.error("Failed to process the generated image.")
                        else:
                            st.error("No image data received from OpenAI.")
                    except Exception as e:
                        st.error(f"Failed to generate or process image: {e}")
            else:
                st.warning("Please enter an image prompt AND provide alt text for the generated image.")

# --- Section for Uploading Existing Image ---
st.markdown("---")
st.subheader("Or Upload Your Own Image")

with st.form("user_image_upload_form"):
    uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg', 'webp'])
    user_alt_text_widget_input = st.text_area("Enter alt text for your image:", height=80, key="user_alt_text_input_val")
    process_uploaded_button = st.form_submit_button("Process Uploaded Image")

    if process_uploaded_button:
        if uploaded_file is not None and user_alt_text_widget_input.strip():
            # Clear previous active image from AI generation to ensure user upload takes precedence
            st.session_state.active_image_bytes_io = None 
            st.session_state.active_image_alt_text = None

            st.session_state.user_uploaded_raw_bytes = uploaded_file.getvalue()
            st.session_state.user_uploaded_alt_text_input_field_value = user_alt_text_widget_input
            
            with st.spinner("Processing your uploaded image..."):
                processed_user_image_io = process_image_for_wordpress(st.session_state.user_uploaded_raw_bytes)
                if processed_user_image_io:
                    st.session_state.active_image_bytes_io = processed_user_image_io
                    # Use the alt text that was specifically entered for this uploaded image
                    st.session_state.active_image_alt_text = st.session_state.user_uploaded_alt_text_input_field_value
                    st.success("Uploaded image processed and is now the active image.")
                else:
                    st.error("Failed to process your uploaded image.")
                    st.session_state.active_image_bytes_io = None 
                    st.session_state.active_image_alt_text = None
        else:
            st.warning("Please upload an image AND provide alt text before processing.")


# --- UI for Displaying Active Image and Uploading to WordPress ---
if st.session_state.active_image_bytes_io and st.session_state.active_image_alt_text:
    st.markdown("---")
    st.subheader("Active Image for WordPress Upload")
    
    st.session_state.active_image_bytes_io.seek(0)
    display_caption = st.session_state.active_image_alt_text
    if len(display_caption) > 150:
        display_caption = display_caption[:150] + "..."
    st.image(st.session_state.active_image_bytes_io, caption=f"ALT Text Preview: {display_caption}")

    current_alt_text_for_wp_upload = st.session_state.active_image_alt_text
    image_data_bytesio_for_wp_upload = st.session_state.active_image_bytes_io
    
    image_data_bytesio_for_wp_upload.seek(0)
    
    clean_filename_base = "".join(c if c.isalnum() or c in (' ', '_') else '' for c in current_alt_text_for_wp_upload[:30]).rstrip().replace(' ', '_')
    upload_filename = f"{clean_filename_base}_processed.jpg" if clean_filename_base else "processed_image.jpg"
    
    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)
    
    sites_to_upload = {
        "cryptonews": row1_col1, "cryptodnes": row1_col2,
        "icobench": row2_col1, "bitcoinist": row2_col2
    }

    for site_key, column in sites_to_upload.items():
        with column:
            if st.button(f"Upload to {site_key.replace('_', ' ').title()}", key=f"upload_{site_key}", use_container_width=True):
                site_config = WP_SITES.get(site_key)
                if site_config and all(site_config.get(k) for k in ["url", "username", "password"]):
                    st.markdown(f"**Uploading to {site_key}...**")
                    image_data_bytesio_for_wp_upload.seek(0)
                    upload_image_bytes = image_data_bytesio_for_wp_upload.getvalue()
                    with st.spinner(f"Uploading to {site_key}..."):
                        upload_image_to_wordpress(
                            image_bytes=upload_image_bytes,
                            wp_url=site_config["url"],
                            username=site_config["username"],
                            wp_app_password=site_config["password"],
                            filename=upload_filename,
                            alt_text=current_alt_text_for_wp_upload # This now uses the explicitly defined ALT text
                        )
                else:
                    st.error(f"Missing or incomplete configuration for {site_key}. Check .env file.")
elif not st.session_state.active_image_bytes_io: # More general check if no active image is set
    st.info("Generate an image or upload and process your own image to make it active for WordPress upload.")
