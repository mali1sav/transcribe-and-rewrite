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
        if img.mode != 'RGB':
            img = img.convert('RGB')

        current_width, current_height = img.size
        if current_width == 0 or current_height == 0:
            raise ValueError("Image dimensions are invalid.")

        aspect_ratio = float(current_width) / float(current_height)

        if aspect_ratio >= 2.5:
            max_width = 1400
            min_height = 300
            if current_width > max_width:
                new_width = max_width
                new_height = max(int(max_width / aspect_ratio), min_height)
            else:
                new_width = current_width
                new_height = current_height
        elif aspect_ratio <= 0.4:
            max_height = 800
            min_width = 400
            if current_height > max_height:
                new_height = max_height
                new_width = max(int(max_height * aspect_ratio), min_width)
            else:
                new_width = current_width
                new_height = current_height
        elif aspect_ratio >= 1.0:
            max_width = 1200
            if current_width > max_width:
                new_width = max_width
                new_height = int(max_width / aspect_ratio)
            else:
                new_width = current_width
                new_height = current_height
        else:
            max_height = 675
            if current_height > max_height:
                new_height = max_height
                new_width = int(max_height * aspect_ratio)
            else:
                new_width = current_width
                new_height = current_height

        if new_width != current_width or new_height != current_height:
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            st.info(f"Resized from {current_width}×{current_height} to {new_width}×{new_height} (aspect ratio: {aspect_ratio:.2f})")
        else:
            img_resized = img
            st.info(f"Image kept at original size: {current_width}×{current_height} (aspect ratio: {aspect_ratio:.2f})")

        jpeg_bytes_io = BytesIO()
        img_resized.save(jpeg_bytes_io, format="JPEG", quality=final_quality, optimize=True)
        jpeg_bytes_io.seek(0)
        return jpeg_bytes_io

    except Exception as e:
        st.error(f"Error processing image: {e}")
        # Diagnostics
        try:
            st.caption(f"Debug Info: Input data type: {type(image_bytes)}, Length: {len(image_bytes)}")
            if isinstance(image_bytes, bytes):
                st.caption(f"First 20 bytes (hex): {image_bytes[:20].hex()}")
        except Exception as debug_e:
            st.caption(f"Debug Info: Error displaying debug info: {debug_e}")
        return None

# ========================================
# --- Providers: Together, OpenAI, OpenRouter
# ========================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Optional OpenRouter ranking headers
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "")
OPENROUTER_SITE_NAME = os.getenv("OPENROUTER_SITE_NAME", "")

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

def generate_image_together_ai(prompt, width=1792, height=1024):
    """
    Generate image using Together AI's FLUX.1-schnell-Free.
    Returns image bytes or None if failed.
    """
    try:
        url = "https://api.together.xyz/v1/images/generations"
        headers = {
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "black-forest-labs/FLUX.1-schnell-Free",
            "prompt": prompt,
            "width": width,
            "height": height,
            "steps": 4,
            "n": 1,
            "response_format": "b64_json"
        }
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            if result.get("data"):
                b64_image = result["data"][0].get("b64_json")
                if b64_image:
                    return base64.b64decode(b64_image)
            st.error("No image data received from Together AI.")
            return None
        else:
            st.error(f"Together AI API error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error calling Together AI API: {e}")
        return None

def generate_image_openai(prompt, size="1536x1024"):
    """
    Generate image using OpenAI's gpt-image-1 via Images API.
    Returns image bytes or None if failed.
    """
    try:
        client = OpenAI()  # Uses OPENAI_API_KEY from env
        response = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            n=1,
            size=size
        )
        if response.data and response.data[0].b64_json:
            b64_image_data = response.data[0].b64_json
            return base64.b64decode(b64_image_data)
        else:
            st.error("No image data received from OpenAI.")
            return None
    except Exception as e:
        st.error(f"Failed to generate image with OpenAI: {e}")
        return None

# ---------- OpenRouter (Gemini 2.5 Flash Image Preview) ----------
def _extract_openrouter_image_bytes(resp_json):
    """
    Extract image bytes from OpenRouter chat/completions responses for image models.
    Supports all observed shapes:
      1) choices[0].message.images[].image_url.url  (can be https:// or data:image/...;base64,...)
      2) choices[0].message.content: list of parts with image_url / image_base64 / b64_json
      3) choices[0].message.content: str containing a data URL
    """
    try:
        choices = resp_json.get("choices", [])
        if not choices:
            return None

        msg = choices[0].get("message", {}) or {}

        # --- Case 1: explicit "images" array (Google provider often uses this) ---
        imgs = msg.get("images")
        if isinstance(imgs, list) and imgs:
            for it in imgs:
                if not isinstance(it, dict):
                    continue
                iu = it.get("image_url") or {}
                url = iu.get("url")
                if not url:
                    continue
                # data URL (base64 inline)
                if url.startswith("data:image/"):
                    # e.g. data:image/png;base64,AAAA...
                    m = re.search(r"base64,([A-Za-z0-9+/=]+)$", url)
                    if m:
                        return base64.b64decode(m.group(1))
                # remote URL
                try:
                    r = requests.get(url, timeout=30)
                    if r.status_code == 200:
                        return r.content
                except Exception:
                    pass  # move on to other shapes

        # --- Case 2: content as list of parts ---
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                # base64 variants
                for key in ("image_base64", "b64", "b64_json"):
                    if key in part and part[key]:
                        return base64.b64decode(part[key])
                # url variant
                if "image_url" in part and isinstance(part["image_url"], dict):
                    url = part["image_url"].get("url")
                    if url:
                        if url.startswith("data:image/"):
                            m = re.search(r"base64,([A-Za-z0-9+/=]+)$", url)
                            if m:
                                return base64.b64decode(m.group(1))
                        try:
                            r = requests.get(url, timeout=30)
                            if r.status_code == 200:
                                return r.content
                        except Exception:
                            pass

        # --- Case 3: content as a single string containing a data URL ---
        if isinstance(content, str) and content:
            m = re.search(r'data:image/(?:png|jpeg|jpg);base64,([A-Za-z0-9+/=]+)', content)
            if m:
                return base64.b64decode(m.group(1))

        return None
    except Exception:
        return None


def generate_image_openrouter(prompt, width=1536, height=1024):
    """
    Generate image using OpenRouter (google/gemini-2.5-flash-image-preview).
    Returns image bytes or None.
    """
    try:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        # Optional ranking headers
        if OPENROUTER_SITE_URL:
            headers["HTTP-Referer"] = OPENROUTER_SITE_URL
        if OPENROUTER_SITE_NAME:
            headers["X-Title"] = OPENROUTER_SITE_NAME

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an image generation model. Return exactly one image. Do not include extra text."
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{prompt.strip()} Create a single high-quality image. Target resolution near {width}x{height}."
                    }
                ],
            },
        ]

        payload = {
            "model": "google/gemini-2.5-flash-image-preview",
            "messages": messages,
            # These hints help some routers/providers return the image payload
            "temperature": 0.7,
        }

        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
        if resp.status_code != 200:
            st.error(f"OpenRouter API error: {resp.status_code} - {resp.text}")
            return None

        resp_json = resp.json()
        img_bytes = _extract_openrouter_image_bytes(resp_json)
        if not img_bytes:
            st.error("OpenRouter response did not include a decodable image.")
            # Show a little more context to aid debugging
            try:
                truncated = json.dumps(resp_json.get("choices", [{}])[0]).strip()
                st.caption(f"Debug (choices[0] truncated): {truncated[:1500]}...")
            except Exception:
                pass
            return None

        return img_bytes

    except Exception as e:
        st.error(f"Error calling OpenRouter API: {e}")
        return None


# ==========================================
# --- Session state for active/generated image
# ==========================================
if 'active_image_bytes_io' not in st.session_state:
    st.session_state.active_image_bytes_io = None
if 'active_image_alt_text' not in st.session_state:
    st.session_state.active_image_alt_text = None
if 'user_uploaded_raw_bytes' not in st.session_state:
    st.session_state.user_uploaded_raw_bytes = None
if 'user_uploaded_alt_text_input' not in st.session_state:
    st.session_state.user_uploaded_alt_text_input = ""
if 'current_prompt' not in st.session_state:
    st.session_state.current_prompt = ""
if 'current_alt_text' not in st.session_state:
    st.session_state.current_alt_text = ""
if 'current_blockchain_bg' not in st.session_state:
    st.session_state.current_blockchain_bg = False
if 'current_hdr_grading' not in st.session_state:
    st.session_state.current_hdr_grading = False
if 'last_used_provider' not in st.session_state:
    st.session_state.last_used_provider = None

st.title("AI Image Generator & WordPress Uploader")

# --- API key availability checks
together_available = bool(TOGETHER_API_KEY)
openai_available = bool(OPENAI_API_KEY)
openrouter_available = bool(OPENROUTER_API_KEY)

if not any([together_available, openai_available, openrouter_available]):
    st.error("No API keys found. Please set at least one of: TOGETHER_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY in .env")

# -----------------
# --- Main UI Form
# -----------------
if together_available or openai_available or openrouter_available:
    openai_client = OpenAI() if openai_available else None

    with st.form("ai_image_form"):
        col_left, col_right = st.columns([3, 2])

        with col_left:
            prompt_text = st.text_area("Enter your image prompt (English only).", height=140, key="prompt_input")
            ai_alt_text = st.text_area("Enter alt text for the generated image:", height=120, key="ai_alt_text_input")

        with col_right:
            add_blockchain_bg = st.checkbox("Add futuristic blockchain background", key="blockchain_bg_checkbox")
            add_hdr_grading = st.checkbox("High dynamic range with vivid and rich color grading", key="hdr_grading_checkbox")
            add_visually_striking_prefix = st.checkbox("Start with 'A visually striking image of...'", value=True, key="visually_striking_checkbox")
            bg_option = st.selectbox(
                "Background / Color Scheme (site presets)",
                ["None", "CryptoNews (Bitberry)", "ICOBench (Green)", "CryptoDnes (Gold)", "Bitcoinist (Blue)"],
                index=0,
                key="background_scheme_select"
            )

            # Provider options (include OpenRouter)
            provider_options = []
            default_index = 0
            if together_available:
                provider_options.append("Together AI (Free - Flux Model)")
            if openai_available:
                provider_options.append("OpenAI (Premium - GPT Image)")
            if openrouter_available:
                provider_options.append("OpenRouter (Gemini 2.5 Image)")
            # Set default to Together if present, else first available
            if provider_options:
                provider = st.radio("Choose AI Provider:", provider_options, index=0, key="provider_selection")
            else:
                provider = None

            submitted = st.form_submit_button("Generate Image", use_container_width=True)

        if submitted and prompt_text.strip() and ai_alt_text.strip():
            st.session_state.current_prompt = prompt_text
            st.session_state.current_alt_text = ai_alt_text
            st.session_state.current_blockchain_bg = add_blockchain_bg
            st.session_state.current_hdr_grading = add_hdr_grading

            st.session_state.active_image_bytes_io = None
            st.session_state.active_image_alt_text = None
            st.session_state.user_uploaded_raw_bytes = None
            st.session_state.user_uploaded_alt_text_input = ""

            # Compose final prompt
            final_prompt = prompt_text
            if add_visually_striking_prefix:
                lower_pt = final_prompt.strip().lower()
                if not lower_pt.startswith("a visually striking image of"):
                    final_prompt = f"A visually striking image of {final_prompt.lstrip()}"
            if add_blockchain_bg:
                final_prompt = f"{final_prompt.rstrip('.')}. The background features a futuristic glowing blockchain motif. Apply any color scheme only to the background elements. Keep all main subjects in their natural, realistic, untinted colors with strong contrast so they pop."
            if add_hdr_grading:
                final_prompt = f"{final_prompt.rstrip('.')}. High dynamic range with vivid and rich color grading."
            if bg_option and bg_option != "None":
                preset_map = {
                    "CryptoNews (Bitberry)": "Use a Bitberry-inspired purple colour scheme with modern gradients for the background only; keep all main subjects in natural, realistic, untinted colors that strongly contrast with the background so they pop.",
                    "ICOBench (Green)": "Background should be a green-to-dark-green gradient applied only to background elements; keep all main subjects in natural, realistic, untinted colors with strong contrast so they pop.",
                    "CryptoDnes (Gold)": "Background should be a gold-to-dark gradient applied only to background elements; keep all main subjects in natural, realistic, untinted colors with strong contrast so they pop.",
                    "Bitcoinist (Blue)": "Background should use a blue-to-light-blue colour scheme applied only to background elements; keep all main subjects in natural, realistic, untinted colors with strong contrast so they pop.",
                }
                chosen_sentence = preset_map.get(bg_option)
                if chosen_sentence:
                    text = final_prompt.strip()
                    sentence_end = r"(?<=[.!?])\s+"
                    sentences = re.split(sentence_end, text)
                    filtered = []
                    for s in sentences:
                        s_stripped = s.strip()
                        if not s_stripped:
                            continue
                        if re.search(r"^background ", s_stripped, flags=re.IGNORECASE):
                            continue
                        if re.search(r"colou?r scheme", s_stripped, flags=re.IGNORECASE):
                            continue
                        filtered.append(s_stripped)
                    cleaned = ". ".join(filtered).strip()
                    if cleaned and not cleaned.endswith(('.', '!', '?')):
                        cleaned += "."
                    final_prompt = f"{cleaned} {chosen_sentence}".strip()

            # --- Branch by provider
            image_bytes_from_api = None
            if provider == "Together AI (Free - Flux Model)" and together_available:
                st.session_state.last_used_provider = "Together AI"
                with st.spinner("Generating image with Together AI (Free)..."):
                    image_bytes_from_api = generate_image_together_ai(final_prompt)
            elif provider == "OpenAI (Premium - GPT Image)" and openai_available:
                st.session_state.last_used_provider = "OpenAI"
                with st.spinner("Generating image with OpenAI (Premium)..."):
                    image_bytes_from_api = generate_image_openai(final_prompt, size="1536x1024")
            elif provider == "OpenRouter (Gemini 2.5 Image)" and openrouter_available:
                st.session_state.last_used_provider = "OpenRouter"
                with st.spinner("Generating image with OpenRouter (Gemini 2.5 Flash Image)..."):
                    # Width/height are advisory; model may choose its own.
                    image_bytes_from_api = generate_image_openrouter(final_prompt, width=1536, height=1024)
            else:
                st.error("Selected provider is not available. Check your API keys.")
                image_bytes_from_api = None

            if image_bytes_from_api:
                final_image_bytes_io = process_image_for_wordpress(image_bytes_from_api)
                if final_image_bytes_io:
                    st.session_state.active_image_bytes_io = final_image_bytes_io
                    st.session_state.active_image_alt_text = ai_alt_text
                    st.success(f"✅ Image generated successfully with {st.session_state.last_used_provider}")
                else:
                    st.error("Failed to process the generated image.")
            else:
                if provider:
                    st.error(f"Failed to generate image with {provider}.")
        elif submitted and (not prompt_text.strip() or not ai_alt_text.strip()):
            st.warning("Please provide both an image prompt AND alt text before generating.")

# -------------------------------
# --- Section: Upload Your Image
# -------------------------------
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
                st.session_state.active_image_bytes_io = None
                st.session_state.active_image_alt_text = None
    elif process_uploaded_button and (uploaded_file is None or not user_alt_text.strip()):
        st.warning("Please upload an image AND provide alt text before processing.")

# ------------------------------------------------
# --- UI: Display active image + Upload to WP
# ------------------------------------------------
if st.session_state.active_image_bytes_io and st.session_state.active_image_alt_text:
    st.markdown("---_Active Image for WordPress Upload_---")

    st.session_state.active_image_bytes_io.seek(0)
    st.image(st.session_state.active_image_bytes_io, caption=st.session_state.active_image_alt_text[:100] + "...")

    current_alt_text = st.session_state.active_image_alt_text
    image_data_bytesio = st.session_state.active_image_bytes_io
    image_data_bytesio.seek(0)

    def generate_english_filename(alt_text):
        """Generate English filename from alt text, focusing on crypto/blockchain terms"""
        crypto_terms = [
            'bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'blockchain', 'defi', 'nft',
            'trading', 'price', 'market', 'coin', 'token', 'altcoin', 'bull', 'bear',
            'pump', 'dump', 'moon', 'hodl', 'analysis', 'chart', 'technical', 'news'
        ]
        english_words = re.findall(r'[a-zA-Z]+', alt_text.lower())
        relevant_words = []
        for word in english_words:
            if len(word) > 2:
                relevant_words.append(word)
            if len(relevant_words) >= 3:
                break
        if relevant_words:
            filename_base = '_'.join(relevant_words[:3])
        else:
            filename_base = f"crypto_image_{int(time.time())}"
        return f"{filename_base}_processed.jpg"

    upload_filename = generate_english_filename(current_alt_text)
    alt_text_for_upload = current_alt_text[:100]

    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)

    with row1_col1:
        if st.button("Upload to Cryptonews", key="upload_cryptonews", use_container_width=True):
            site_key = "cryptonews"
            site_config = WP_SITES.get(site_key)
            if site_config and all(site_config.get(k) for k in ["url", "username", "password"]):
                st.markdown(f"**Uploading to {site_key}...**")
                image_data_bytesio.seek(0)
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
        if st.button("Upload to CryptoDnes", key="upload_cryptodnes", use_container_width=True):
            site_key = "cryptodnes"
            site_config = WP_SITES.get(site_key)
            if site_config and all(site_config.get(k) for k in ["url", "username", "password"]):
                st.markdown(f"**Uploading to {site_key}...**")
                image_data_bytesio.seek(0)
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

    with row2_col1:
        if st.button("Upload to ICOBench", key="upload_icobench", use_container_width=True):
            site_key = "icobench"
            site_config = WP_SITES.get(site_key)
            if site_config and all(site_config.get(k) for k in ["url", "username", "password"]):
                st.markdown(f"**Uploading to {site_key}...**")
                image_data_bytesio.seek(0)
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
        if st.button("Upload to Bitcoinist", key="upload_bitcoinist", use_container_width=True):
            site_key = "bitcoinist"
            site_config = WP_SITES.get(site_key)
            if site_config and all(site_config.get(k) for k in ["url", "username", "password"]):
                st.markdown(f"**Uploading to {site_key}...**")
                image_data_bytesio.seek(0)
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
