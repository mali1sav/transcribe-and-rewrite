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

# Initialize session state for the active image to be uploaded
if 'active_image_bytes_io' not in st.session_state:
    st.session_state.active_image_bytes_io = None # BytesIO object of processed image
if 'active_image_alt_text' not in st.session_state:
    st.session_state.active_image_alt_text = None # Alt text/prompt for the image
if 'user_uploaded_raw_bytes' not in st.session_state: # Temporary store for user's raw upload
    st.session_state.user_uploaded_raw_bytes = None
if 'user_uploaded_alt_text_input' not in st.session_state:
    st.session_state.user_uploaded_alt_text_input = ""

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# WordPress Site Configurations
WP_SITES = {
    "cryptonews": {
        "url": os.getenv("CRYPTONEWS_WP_URL"),
        "username": os.getenv("CRYPTONEWS_WP_USERNAME"),
        "password": os.getenv("CRYPTONEWS_WP_APP_PASSWORD")
    },
    "cryptodnes": {
        "url": os.getenv("CRYPTODNES_WP_URL"),
        "username": os.getenv("CRYPTODNES_WP_USERNAME"),
        "password": os.getenv("CRYPTODNES_WP_APP_PASSWORD")
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
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        current_width, current_height = img.size
        if current_width == 0 or current_height == 0:
            raise ValueError("Image dimensions are invalid.")
        
        aspect_ratio = float(current_width) / float(current_height)
        
        # Determine optimal dimensions based on image characteristics
        if aspect_ratio >= 2.5:
            # Very wide banner (e.g., 2000x400) - preserve width, ensure min height
            max_width = 1400
            min_height = 300
            if current_width > max_width:
                new_width = max_width
                new_height = max(int(max_width / aspect_ratio), min_height)
            else:
                new_width = current_width
                new_height = current_height
                
        elif aspect_ratio <= 0.4:
            # Very tall image (e.g., mobile screenshot 400x1000) - limit height
            max_height = 800
            min_width = 400
            if current_height > max_height:
                new_height = max_height
                new_width = max(int(max_height * aspect_ratio), min_width)
            else:
                new_width = current_width
                new_height = current_height
                
        elif aspect_ratio >= 1.0:
            # Landscape or square image - limit width to 1200px
            max_width = 1200
            if current_width > max_width:
                new_width = max_width
                new_height = int(max_width / aspect_ratio)
            else:
                new_width = current_width
                new_height = current_height
                
        else:
            # Portrait image - limit height to 675px (16:9 ratio with 1200px width)
            max_height = 675
            if current_height > max_height:
                new_height = max_height
                new_width = int(max_height * aspect_ratio)
            else:
                new_width = current_width
                new_height = current_height
        
        # Only resize if dimensions changed
        if new_width != current_width or new_height != current_height:
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            st.info(f"Resized from {current_width}×{current_height} to {new_width}×{new_height} (aspect ratio: {aspect_ratio:.2f})")
        else:
            img_resized = img
            st.info(f"Image kept at original size: {current_width}×{current_height} (aspect ratio: {aspect_ratio:.2f})")
        
        # Save as optimized JPEG
        jpeg_bytes_io = BytesIO()
        img_resized.save(jpeg_bytes_io, format="JPEG", quality=final_quality, optimize=True)
        jpeg_bytes_io.seek(0)
        return jpeg_bytes_io
        
    except Exception as e:
        st.error(f"Error processing image: {e}")
        # Add diagnostic information
        if 'image_bytes' in locals() and image_bytes:
            try:
                st.caption(f"Debug Info: Input data type: {type(image_bytes)}, Length: {len(image_bytes)}")
                if isinstance(image_bytes, bytes):
                    st.caption(f"First 20 bytes (hex): {image_bytes[:20].hex()}")
                else:
                    st.caption("Debug Info: Input data is not in bytes format.")
            except Exception as debug_e:
                st.caption(f"Debug Info: Error displaying debug info: {debug_e}")
        return None

st.title("OpenAI Image Generator & WordPress Uploader")

if not OPENAI_API_KEY:
    st.error("OpenAI API key not found. Please set OPENAI_API_KEY in .env")
else:
    openai_client = OpenAI()
    with st.form("openai_image_form"):
        prompt_text = st.text_area("Enter your image prompt (English only).", height=80, key="prompt_input")
        ai_alt_text = st.text_area("Enter alt text for the generated image:", height=80, key="ai_alt_text_input")
        submitted = st.form_submit_button("Generate Image")
        
        if submitted and prompt_text.strip() and ai_alt_text.strip():
            # Clear any previous active image data
            st.session_state.active_image_bytes_io = None
            st.session_state.active_image_alt_text = None
            st.session_state.user_uploaded_raw_bytes = None
            st.session_state.user_uploaded_alt_text_input = ""

            with st.spinner("Generating and processing image with OpenAI..."):
                try:
                    response = openai_client.images.generate(
                        model="gpt-image-1", # As per memory fa413de7-7129-48b8-b00d-67b86f2dc54d
                        prompt=prompt_text,
                        n=1,
                        size="1536x1024" # As per memory fa413de7-7129-48b8-b00d-67b86f2dc54d
                    )
                    if response.data and response.data[0].b64_json:
                        b64_image_data = response.data[0].b64_json
                        image_bytes_from_api = base64.b64decode(b64_image_data)
                        # Process using existing function, aligning with memory fa413de7-7129-48b8-b00d-67b86f2dc54d
                        final_image_bytes_io = process_image_for_wordpress(image_bytes_from_api)
                        
                        if final_image_bytes_io:
                            st.session_state.active_image_bytes_io = final_image_bytes_io
                            st.session_state.active_image_alt_text = ai_alt_text  # Use the dedicated alt text instead of prompt
                            # Display of the image will be handled by the unified section below
                        else:
                            st.error("Failed to process the generated image.")
                    else:
                        st.error("No image data received from OpenAI.")
                except Exception as e:
                    st.error(f"Failed to generate or process image: {e}")
        elif submitted and (not prompt_text.strip() or not ai_alt_text.strip()):
            st.warning("Please provide both an image prompt AND alt text before generating.")

# --- Section for Uploading Existing Image ---
st.markdown("---_Or Upload Your Own Image_---")

with st.form("user_image_upload_form"):
    uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg', 'webp'])
    user_alt_text = st.text_area("Enter alt text for your image:", height=80, key="user_alt_text_input_val")
    process_uploaded_button = st.form_submit_button("Process Uploaded Image")

    if process_uploaded_button and uploaded_file is not None and user_alt_text.strip():
        st.session_state.user_uploaded_raw_bytes = uploaded_file.getvalue()
        st.session_state.user_uploaded_alt_text_input = user_alt_text
        
        with st.spinner("Processing your uploaded image..."):
            processed_user_image_io = process_image_for_wordpress(st.session_state.user_uploaded_raw_bytes)
            if processed_user_image_io:
                st.session_state.active_image_bytes_io = processed_user_image_io
                st.session_state.active_image_alt_text = st.session_state.user_uploaded_alt_text_input
                st.success("Uploaded image processed and ready for WordPress upload.")
            else:
                st.error("Failed to process your uploaded image.")
                st.session_state.active_image_bytes_io = None # Clear if processing failed
                st.session_state.active_image_alt_text = None
    elif process_uploaded_button and (uploaded_file is None or not user_alt_text.strip()):
        st.warning("Please upload an image AND provide alt text before processing.")

# --- UI for Displaying Active Image and Uploading to WordPress ---
if st.session_state.active_image_bytes_io and st.session_state.active_image_alt_text:
    st.markdown("---_Active Image for WordPress Upload_---")
    
    # Display the active image (either AI-generated or user-uploaded and processed)
    st.session_state.active_image_bytes_io.seek(0) # Ensure BytesIO is ready for display
    st.image(st.session_state.active_image_bytes_io, caption=st.session_state.active_image_alt_text[:100] + "...")

    current_alt_text = st.session_state.active_image_alt_text
    image_data_bytesio = st.session_state.active_image_bytes_io # This is a BytesIO object
    
    # Ensure BytesIO is ready for reading for uploads
    image_data_bytesio.seek(0)
    
    # Prepare common upload parameters
    clean_alt_text_filename = "".join(c if c.isalnum() or c in (' ', '_') else '' for c in current_alt_text[:30]).rstrip().replace(' ', '_')
    upload_filename = f"{clean_alt_text_filename}_processed.jpg" if clean_alt_text_filename else "processed_image.jpg"
    alt_text_for_upload = current_alt_text[:100]

    # Create two rows of buttons with two columns each
    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)
    
    # Row 1 - Cryptonews and CryptoDnes
    with row1_col1:
        # Button for Cryptonews
        if st.button("Upload to Cryptonews", key="upload_cryptonews", use_container_width=True):
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

    
    with row1_col2:
        # Button for CryptoDnes
        if st.button("Upload to CryptoDnes", key="upload_cryptodnes", use_container_width=True):
            site_key = "cryptodnes"
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
    
    # Row 2 - ICOBench and Bitcoinist
    with row2_col1:
        # Button for ICOBench
        if st.button("Upload to ICOBench", key="upload_icobench", use_container_width=True):
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

    with row2_col2:
        # Button for Bitcoinist
        if st.button("Upload to Bitcoinist", key="upload_bitcoinist", use_container_width=True):
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
