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
            auth=HTTPBasicAuth(username, wp_app_password),
            timeout=60
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
                    auth=HTTPBasicAuth(username, wp_app_password),
                    timeout=60
                )
                if update_response.status_code in (200, 201):
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
def process_image_for_wordpress(image_bytes, final_quality=80, force_landscape_16_9=False, max_size_kb=100):
    """
    Optimizes for WP and (optionally) hard-enforces 16:9 landscape by center-cropping.
    Adds a hard cap on output size via iterative JPEG compression and optional downscale.

    Rules:
    - Landscape images: Max width 1200px
    - Portrait images: Max height 675px (16:9 ratio)
    - Very wide banners: Max width 1400px, min height 300px
    - Very tall images: Max height 800px, min width 400px
    - Ensure final file size <= max_size_kb via compression and downscale
    """
    try:
        img = Image.open(BytesIO(image_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        w, h = img.size
        if w == 0 or h == 0:
            raise ValueError("Image dimensions are invalid.")
        aspect_ratio = w / float(h)

        # --- Force 16:9 landscape if requested ---
        if force_landscape_16_9:
            target = 16.0 / 9.0
            eps = 0.01
            if abs(aspect_ratio - target) > eps:
                if aspect_ratio > target:
                    new_w = int(h * target)
                    left = max((w - new_w) // 2, 0)
                    img = img.crop((left, 0, left + new_w, h))
                else:
                    new_h = int(w / target)
                    top = max((h - new_h) // 2, 0)
                    img = img.crop((0, top, w, top + new_h))
                w, h = img.size
                aspect_ratio = w / float(h)

        # --- Resize rules ---
        if aspect_ratio >= 2.5:
            max_width = 1400
            min_height = 300
            if w > max_width:
                new_width = max_width
                new_height = max(int(max_width / aspect_ratio), min_height)
            else:
                new_width, new_height = w, h
        elif aspect_ratio <= 0.4:
            max_height = 800
            min_width = 400
            if h > max_height:
                new_height = max_height
                new_width = max(int(max_height * aspect_ratio), min_width)
            else:
                new_width, new_height = w, h
        elif aspect_ratio >= 1.0:
            max_width = 1200
            if w > max_width:
                new_width = max_width
                new_height = int(max_width / aspect_ratio)
            else:
                new_width, new_height = w, h
        else:
            max_height = 675
            if h > max_height:
                new_height = max_height
                new_width = int(max_height * aspect_ratio)
            else:
                new_width, new_height = w, h

        if new_width != w or new_height != h:
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            st.info(f"Resized from {w}×{h} to {new_width}×{new_height} (aspect ratio: {aspect_ratio:.2f})")
        else:
            img_resized = img
            st.info(f"Image kept at {w}×{h} (aspect ratio: {aspect_ratio:.2f})")

        target_bytes = max_size_kb * 1024
        min_quality = 20
        quality_step = 5
        quality = max(10, min(95, int(final_quality)))

        def save_to_buffer(im, q):
            buf = BytesIO()
            im.save(buf, format="JPEG", quality=q, optimize=True)
            buf.seek(0)
            return buf

        out = save_to_buffer(img_resized, quality)
        cur_size = out.getbuffer().nbytes

        # reduce quality down to min_quality
        while cur_size > target_bytes and quality > min_quality:
            quality = max(min_quality, quality - quality_step)
            out = save_to_buffer(img_resized, quality)
            cur_size = out.getbuffer().nbytes

        # downscale loop if still too big
        min_w, min_h = 640, 360
        ds_img = img_resized
        while cur_size > target_bytes and (ds_img.width > min_w and ds_img.height > min_h):
            new_w = max(min_w, int(ds_img.width * 0.9))
            new_h = max(min_h, int(ds_img.height * 0.9))
            ds_img = ds_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            trial_quality = min(quality + 5, 85)
            out = save_to_buffer(ds_img, trial_quality)
            cur_size = out.getbuffer().nbytes
            while cur_size > target_bytes and trial_quality > min_quality:
                trial_quality = max(min_quality, trial_quality - quality_step)
                out = save_to_buffer(ds_img, trial_quality)
                cur_size = out.getbuffer().nbytes
            quality = trial_quality

        final_w, final_h = ds_img.size
        final_kb = cur_size // 1024
        st.info(f"Final saved size: {final_kb} KB at quality {quality} — {final_w}×{final_h}")
        out.seek(0)
        return out

    except Exception as e:
        st.error(f"Error processing image: {e}")
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

# ---------- Prompt Engineering (NEW) ----------
FLUX_ENGINEER_SYSTEM = (
    "You are a Flux prompt engineer. Follow these rules:\n\n"
    "1. Language Handling:\n"
    "    - If the prompt is in Thai, translate it to English before applying the remaining rules.\n"
    "    - If the prompt is not in Thai, proceed to the next set of rules.\n\n"
    "2. Representing Thai Individuals:\n"
    "    - If the prompt is in Thai or requests Thai individuals, ensure that the people in the image appear ethnically Thai, reflecting typical Thai features such as facial structure, and contemporary attire appropriate to the context.\n"
    "    - If the prompt does not specify Thai individuals then proceed without enforcing Thai representation.\n\n"
    "3. Scene Enhancement:\n"
    "   - Add specific lighting and atmosphere details\n"
    "   - Include natural elements and textures\n"
    "   - Specify basic camera angle and composition\n"
    "   - Keep the enhancement focused on the main subject\n\n"
    "4. Quality Focus:\n"
    "   - Include technical details such as aperture, lens, and shot type.\n"
    "   - End the prompt with \"photo-realistic style\" unless otherwise specified.\n\n"
    "Provide only the enhanced prompt without explanations."
)

def _openrouter_headers():
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    if OPENROUTER_SITE_URL:
        headers["HTTP-Referer"] = OPENROUTER_SITE_URL
    if OPENROUTER_SITE_NAME:
        headers["X-Title"] = OPENROUTER_SITE_NAME
    return headers

def enhance_prompt_with_role(user_prompt: str) -> str:
    """
    Return engineered prompt string. Prefer OpenAI; fallback to OpenRouter.
    If both unavailable or any failure occurs, return the original user_prompt.
    """
    try:
        # Prefer OpenAI if available
        if OPENAI_API_KEY:
            client = OpenAI()
            resp = client.chat.completions.create(
                model=os.getenv("OPENAI_TEXT_MODEL", "gpt-4o-mini"),
                temperature=0.2,
                messages=[
                    {"role": "system", "content": FLUX_ENGINEER_SYSTEM},
                    {"role": "user", "content": user_prompt.strip()}
                ],
            )
            out = (resp.choices[0].message.content or "").strip()
            if out:
                return out

        # Fallback to OpenRouter
        if OPENROUTER_API_KEY:
            url = "https://openrouter.ai/api/v1/chat/completions"
            payload = {
                "model": os.getenv("OPENROUTER_TEXT_MODEL", "openai/gpt-4o-mini"),
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": FLUX_ENGINEER_SYSTEM},
                    {"role": "user", "content": user_prompt.strip()}
                ],
            }
            resp = requests.post(url, headers=_openrouter_headers(), data=json.dumps(payload), timeout=60)
            if resp.status_code == 200:
                j = resp.json()
                content = (j.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
                if content:
                    return content
            else:
                st.warning(f"OpenRouter text LLM error: {resp.status_code} - {resp.text[:300]}")

    except Exception as e:
        st.warning(f"Prompt engineering failed, using raw prompt. Reason: {e}")

    return user_prompt.strip()

# =========================================
# --- Image generation functions (unchanged)
# =========================================

def generate_image_together_ai(prompt, width=1792, height=1024):
    """
    Generate image using Together AI's FLUX.1-schnell-Free.
    Returns image bytes or None if failed.
    """
    try:
        if not TOGETHER_API_KEY:
            st.error("Together API key missing.")
            return None
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
        response = requests.post(url, headers=headers, json=payload, timeout=120)
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
        if not OPENAI_API_KEY:
            st.error("OpenAI API key missing.")
            return None
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

def _extract_openrouter_image_bytes(resp_json):
    """
    Extract image bytes from OpenRouter chat/completions responses for image models.
    Supports multiple shapes.
    """
    try:
        choices = resp_json.get("choices", [])
        if not choices:
            return None

        msg = choices[0].get("message", {}) or {}

        imgs = msg.get("images")
        if isinstance(imgs, list) and imgs:
            for it in imgs:
                if not isinstance(it, dict):
                    continue
                iu = it.get("image_url") or {}
                url = iu.get("url")
                if not url:
                    continue
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

        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                for key in ("image_base64", "b64", "b64_json"):
                    if key in part and part[key]:
                        return base64.b64decode(part[key])
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
        if not OPENROUTER_API_KEY:
            st.error("OpenRouter API key missing.")
            return None
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = _openrouter_headers()

        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are an image generation model. Return exactly one image and no extra text."}
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"{prompt.strip()}\n\n"
                            f"Create exactly one high-quality image in landscape 16:9 aspect ratio "
                            f"(approximately {width}x{height}). Do not include any text output; "
                            f"respond with the image only."
                        )
                    }
                ],
            },
        ]

        payload = {
            "model": "google/gemini-2.5-flash-image-preview",
            "messages": messages,
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
if 'last_engineered_prompt' not in st.session_state:
    st.session_state.last_engineered_prompt = ""

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
            prompt_text = st.text_area("Enter your image prompt (Thai or English).", height=140, key="prompt_input")
            ai_alt_text = st.text_area("Enter alt text for the generated image:", height=120, key="ai_alt_text_input")

        with col_right:
            use_flux_engineer = st.checkbox("Use Flux Prompt Engineer (recommended)", value=True, key="use_flux_engineer_checkbox")
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
            if together_available:
                provider_options.append("Flux model (Free)")
            if openai_available:
                provider_options.append("OpenAI ($0.30 per image)")
            if openrouter_available:
                provider_options.append("Nano Banana ( $0.03 per image)")
            provider = provider_options[0] if provider_options else None

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
            st.session_state.last_engineered_prompt = ""

            # 1) Optionally engineer the prompt via LLM role
            base_prompt = prompt_text.strip()
            if use_flux_engineer:
                with st.spinner("Engineering prompt..."):
                    engineered = enhance_prompt_with_role(base_prompt)
                    st.session_state.last_engineered_prompt = engineered
                    final_prompt = engineered
            else:
                final_prompt = base_prompt

            # Ensure optional prefix only when user didn't already engineer one with similar lead
            if add_visually_striking_prefix:
                lower_pt = final_prompt.strip().lower()
                if not lower_pt.startswith("a visually striking image of"):
                    final_prompt = f"A visually striking image of {final_prompt.lstrip()}"

            # 2) Append site-specific background / HDR, with guardrails to avoid tinting subjects
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
                    # Remove any pre-existing background/color-scheme sentences to avoid duplication
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
                        if re.search(r"blockchain motif", s_stripped, flags=re.IGNORECASE) and add_blockchain_bg:
                            # keep the motif sentence; don't drop it
                            filtered.append(s_stripped)
                            continue
                        filtered.append(s_stripped)
                    cleaned = ". ".join(filtered).strip()
                    if cleaned and not cleaned.endswith(('.', '!', '?')):
                        cleaned += "."
                    final_prompt = f"{cleaned} {chosen_sentence}".strip()
                    
            # ✅ Show the exact final prompt being sent to the image API
            st.subheader("🔍 Final prompt sent to image model")
            st.code(final_prompt)


            # --- Branch by provider
            image_bytes_from_api = None
            if provider == "Flux model (Free)" and together_available:
                st.session_state.last_used_provider = "Together AI"
                with st.spinner("Generating image with Together AI (Free)..."):
                    image_bytes_from_api = generate_image_together_ai(final_prompt)
            elif provider == "OpenAI ($0.30 per image)" and openai_available:
                st.session_state.last_used_provider = "OpenAI"
                with st.spinner("Generating image with OpenAI (Premium)..."):
                    image_bytes_from_api = generate_image_openai(final_prompt, size="1536x1024")
            elif provider == "Nano Banana ( $0.03 per image)" and openrouter_available:
                st.session_state.last_used_provider = "OpenRouter"
                with st.spinner("Generating image with OpenRouter (Gemini 2.5 Flash Image)..."):
                    image_bytes_from_api = generate_image_openrouter(final_prompt, width=1536, height=1024)
            else:
                st.error("Selected provider is not available. Check your API keys.")
                image_bytes_from_api = None

            if image_bytes_from_api:
                # Enforce 16:9 only for OpenRouter outputs
                force_169 = (st.session_state.last_used_provider == "OpenRouter")
                final_image_bytes_io = process_image_for_wordpress(
                    image_bytes_from_api,
                    force_landscape_16_9=force_169
                )
                if final_image_bytes_io:
                    st.session_state.active_image_bytes_io = final_image_bytes_io
                    st.session_state.active_image_alt_text = ai_alt_text
                    st.success(f"✅ Image generated successfully with {st.session_state.last_used_provider}")
                    if st.session_state.last_engineered_prompt:
                        with st.expander("Show engineered prompt"):
                            st.code(st.session_state.last_engineered_prompt)
                else:
                    st.error("Failed to process the generated image.")
            else:
                if provider:
                    st.error(f"Failed to generate image with {provider}.")

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
        english_words = re.findall(r'[a-zA-Z]+', alt_text.lower())
        relevant_words = []
        for word in english_words:
            if len(word) > 2:
                relevant_words.append(word)
            if len(relevant_words) >= 3:
                break
        filename_base = '_'.join(relevant_words[:3]) if relevant_words else f"crypto_image_{int(time.time())}"
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
