import streamlit as st
import os
from dotenv import load_dotenv
import base64
from io import BytesIO
from PIL import Image, ImageFilter, ImageOps
from openai import OpenAI
import requests
from requests.auth import HTTPBasicAuth
import json
import re
import time
import hashlib

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
def process_image_for_wordpress(image_bytes, final_quality=80, force_landscape_16_9=False, max_size_kb=100, crop_strategy='crop'):
    """
    Optimizes for WP and (optionally) hard-enforces 16:9 landscape by center-cropping or letterboxing.
    Adds a hard cap on output size via iterative JPEG compression and optional downscale.

    Rules:
    - Landscape images: Max width 1200px
    - Portrait images: Max height 675px (16:9 ratio)
    - Very wide banners: Max width 1400px, min height 300px
    - Very tall images: Max height 800px, min width 400px

    New: enforce <= max_size_kb by stepwise quality drops + downscale.
    When force_landscape_16_9=True and crop_strategy='fit', the image is letterboxed onto a 16:9 canvas using a blurred background instead of being center-cropped.
    """
    try:
        img = Image.open(BytesIO(image_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        w, h = img.size
        if w == 0 or h == 0:
            raise ValueError("Image dimensions are invalid.")
        aspect_ratio = w / float(h)

        def _letterbox_16x9(src_img, target_width=1200):
            target_height = int(target_width * 9 / 16)
            bg = src_img.resize((target_width, target_height), Image.Resampling.LANCZOS).filter(ImageFilter.GaussianBlur(20))
            if bg.mode != 'RGB':
                bg = bg.convert('RGB')
            fitted = ImageOps.contain(src_img, (target_width, target_height), Image.Resampling.LANCZOS)
            canvas = bg.copy()
            x = (target_width - fitted.width) // 2
            y = (target_height - fitted.height) // 2
            canvas.paste(fitted, (x, y))
            return canvas

        # --- Force 16:9 landscape if requested ---
        if force_landscape_16_9:
            target = 16.0 / 9.0
            eps = 0.01
            if crop_strategy == 'fit':
                pass
            else:
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

        # Resize if needed
        if crop_strategy == 'fit' and force_landscape_16_9:
            img_resized = _letterbox_16x9(img, target_width=1200)
            st.info("Letterboxed to 1200×675 (16:9) to avoid cropping.")
        else:
            if new_width != w or new_height != h:
                img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                st.info(f"Resized from {w}×{h} to {new_width}×{new_height} (aspect ratio: {aspect_ratio:.2f})")
            else:
                img_resized = img
                st.info(f"Image kept at {w}×{h} (aspect ratio: {aspect_ratio:.2f})")

        # --- Iterative compression under max size ---
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

        while cur_size > target_bytes and quality > min_quality:
            quality = max(min_quality, quality - quality_step)
            out = save_to_buffer(img_resized, quality)
            cur_size = out.getbuffer().nbytes

        min_w, min_h = 640, 360  # keep thumbnails reasonably sharp for 16:9
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
# --- Providers: FAL (Seedream v4), OpenAI, OpenRouter
# ========================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
FAL_API_KEY = os.getenv("FAL_API_KEY")

# Optional OpenRouter ranking headers
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "")
OPENROUTER_SITE_NAME = os.getenv("OPENROUTER_SITE_NAME", "")

# ---------- OpenRouter text headers helper ----------


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

# ---------- SEO helpers via LLM (Thai alt/title + English filename) ----------
def _llm_chat_json(messages, prefer_openai=True, timeout=60) -> dict | None:
    """
    Call OpenAI (preferred) or OpenRouter for a small JSON response.
    Returns parsed JSON dict or None.
    """
    try:
        if prefer_openai and OPENAI_API_KEY:
            client = OpenAI()
            resp = client.chat.completions.create(
                model=os.getenv("OPENAI_TEXT_MODEL", "gpt-4o-mini"),
                temperature=0.2,
                messages=messages,
                response_format={"type": "json_object"},
            )
            content = (resp.choices[0].message.content or "").strip()
            return json.loads(content) if content else None
        elif OPENROUTER_API_KEY:
            url = "https://openrouter.ai/api/v1/chat/completions"
            payload = {
                "model": os.getenv("OPENROUTER_TEXT_MODEL", "openai/gpt-4o-mini"),
                "temperature": 0.2,
                "response_format": {"type": "json_object"},
                "messages": messages,
            }
            resp = requests.post(url, headers=_openrouter_headers(), data=json.dumps(payload), timeout=timeout)
            if resp.status_code == 200:
                j = resp.json()
                content = (j.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
                return json.loads(content) if content else None
            else:
                st.warning(f"OpenRouter text LLM error: {resp.status_code} - {resp.text[:300]}")
    except Exception as e:
        st.warning(f"LLM JSON call failed: {e}")
    return None

def generate_seo_meta_from_prompt(prompt_text: str) -> tuple[str, str]:
    """
    From the user's prompt, create:
      - Thai SEO sentence (alt/title)  ➜ we append dd-mm-yyyy
      - English SEO filename slug      ➜ we append dd-mm-yyyy + short hash + .jpg
    Returns (alt_title_th_with_date, filename_with_date_ext)
    """
    date_str = time.strftime("%d-%m-%Y")
    # Ask the LLM for Thai sentence + English slug in strict JSON
    sys = (
        "You are an SEO assistant. Produce a short, human-friendly ALT/TITLE in Thai, "
        "and a concise lowercase English filename slug for an image. "
        "Return strict JSON with keys: thai_sentence, english_slug. "
        "Rules: thai_sentence 6-16 words, no site names, no hashtags; "
        "english_slug 3-6 words, kebab-case, ASCII only, no dates or extension."
    )
    user = f"Image prompt/context:\n{prompt_text.strip()}\nReturn JSON only."
    out = _llm_chat_json(
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
        prefer_openai=True,
        timeout=60,
    )
    thai_sentence = None
    english_slug = None
    if isinstance(out, dict):
        thai_sentence = (out.get("thai_sentence") or "").strip()
        english_slug = (out.get("english_slug") or "").strip()
    # Fallbacks
    if not thai_sentence:
        thai_sentence = f"ภาพประกอบข่าว: {prompt_text.strip()[:60]}"
    if not english_slug:
        english_slug = _slugify_english(prompt_text)[:40] or "image"
    # Compose outputs with date
    alt_title_with_date = f"{thai_sentence} {date_str}".strip()
    slug = _slugify_english(english_slug)
    short_hash = hashlib.md5(slug.encode('utf-8')).hexdigest()[:6]
    filename = f"{slug}-{date_str}-{short_hash}.jpg"
    return alt_title_with_date, filename

# ---------- String helpers ----------
def _slugify_english(text: str) -> str:
    # keep ascii letters/numbers/spaces, turn spaces to hyphens, collapse repeats
    text = re.sub(r'[^A-Za-z0-9\s-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    text = re.sub(r'\s|-+', '-', text)
    text = re.sub(r'-{2,}', '-', text)
    return text.strip('-') or 'image'

def seo_filename_from_alt_via_openrouter(alt_text: str) -> str | None:
    """Ask an OpenRouter text model to produce a short SEO-friendly English slug (kebab-case).
    Returns slug string or None on failure."""
    if not OPENROUTER_API_KEY:
        return None
    try:
        url = "https://openrouter.ai/api/v1/chat/completions"
        system = ("You generate concise SEO-friendly image filenames. "
                  "Output ONLY a lowercase kebab-case slug of 3-6 words, ASCII only, "
                  "no dates, no file extension, no quotes.")
        payload = {
            "model": os.getenv("OPENROUTER_TEXT_MODEL", "openai/gpt-4o-mini"),
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Alt text: {alt_text}\nReturn only the slug."}
            ],
        }
        resp = requests.post(url, headers=_openrouter_headers(), data=json.dumps(payload), timeout=45)
        if resp.status_code == 200:
            j = resp.json()
            content = (j.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
            if content:
                # sanitize in case the model adds extra words
                return _slugify_english(content)
    except Exception:
        pass
    return None

def generate_seo_image_name(alt_text: str, ext: str = "jpg") -> str:
    """Create an English, SEO-friendly, unique image filename from alt text.
    Uses OpenRouter LLM when available; falls back to local slugify.
    Appends date (dd-mm-yyyy) and a short hash for uniqueness."""
    slug = seo_filename_from_alt_via_openrouter(alt_text) or _slugify_english(alt_text)
    date_str = time.strftime("%d-%m-%Y")
    short_hash = hashlib.md5(alt_text.encode("utf-8")).hexdigest()[:6]
    return f"{slug}-{date_str}-{short_hash}.{ext}"

# Utility: generate filename from slug directly
def generate_filename_from_slug(slug: str, ext: str = "jpg") -> str:
    slug = _slugify_english(slug)
    date_str = time.strftime("%d-%m-%Y")
    short_hash = hashlib.md5(slug.encode("utf-8")).hexdigest()[:6]
    return f"{slug}-{date_str}-{short_hash}.{ext}"

# ---------- Thai-aware prompt engineering system prompt ----------
FLUX_ENGINEER_SYSTEM = (
    "You are a prompt engineer specialized in ByteDance/Seedream v4 image generation. "
    "Transform the user's request (Thai or English) into a single English prompt optimized for Seedream v4. "
    "Follow Seedream best practices: be concise but specific; clearly state subject(s), attributes, actions, setting, composition, camera/shot type, lens/focal length, lighting, atmosphere, color palette, materials/textures, and style (photographic/illustration/3D). "
    "Prefer natural, realistic renderings unless the user asks for stylization. "
    "Avoid on-image text, watermarks, UI, logos. Do NOT add brand names unless provided by the user. "
    "When the user implies an editorial/crypto news illustration, prefer impactful but realistic scenes. "
    "If the user asks in Thai, translate faithfully to English while adding missing visual details. "
    "Output: ONE paragraph in English only, no lists, no extra commentary."
)

# ---------- Prompt engineer (prefers OpenAI; falls back to OpenRouter) ----------

def enhance_prompt_with_role(user_prompt: str) -> str:
    """
    Return engineered prompt string. Prefer OpenAI; fallback to OpenRouter.
    If both unavailable or any failure occurs, return the original user_prompt.
    Accepts Thai or English; always outputs English.
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

# -----------------------------
# --- FAL API helpers (Seedream)
# -----------------------------
def _fal_post(model_path: str, payload: dict):
    """POST to FAL model and return JSON or raise."""
    url = f"https://fal.run/{model_path.strip('/')}"
    headers = {
        "Authorization": f"Key {FAL_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=180)
    if r.status_code != 200:
        raise RuntimeError(f"FAL API error {r.status_code}: {r.text[:500]}")
    return r.json()

def _extract_image_bytes_from_fal(resp_json: dict):
    """Best-effort extraction supporting common FAL response shapes."""
    # 1) images array with urls
    try:
        images = resp_json.get("images") or resp_json.get("output", {}).get("images")
        if isinstance(images, list) and images:
            first = images[0]
            if isinstance(first, dict):
                url = first.get("url") or first.get("image_url")
                if url:
                    r = requests.get(url, timeout=60)
                    if r.status_code == 200:
                        return r.content
            if isinstance(first, str) and first.startswith("http"):
                r = requests.get(first, timeout=60)
                if r.status_code == 200:
                    return r.content
    except Exception:
        pass

    # 2) direct base64 field variants
    for key in ("image", "image_base64", "b64", "b64_json"):
        b64_val = resp_json.get(key) or resp_json.get("output", {}).get(key)
        if b64_val:
            try:
                return base64.b64decode(b64_val)
            except Exception:
                pass

    return None

def generate_image_fal_seedream(prompt, width=1536, height=1024):
    """
    Generate image using FAL Seedream v4 text-to-image.
    Note: FAL Seedream expects `image_size` as one of the presets: 'square_hd', 'square', 'portrait_4_3', 'portrait_16_9', 'landscape_4_3', 'landscape_16_9'. We use 'landscape_16_9' for site thumbnails.
    Returns image bytes or None if failed.
    """
    try:
        payload = {
            "prompt": prompt,
            "image_size": "landscape_16_9",
            "num_inference_steps": 28,
            "guidance_scale": 4.0,
            "num_samples": 1
        }
        resp_json = _fal_post("fal-ai/bytedance/seedream/v4/text-to-image", payload)
        img_bytes = _extract_image_bytes_from_fal(resp_json)
        if not img_bytes:
            st.error("No image data received from FAL Seedream v4.")
            try:
                st.caption(f"Debug (resp snippet): {json.dumps(resp_json)[:1200]}...")
            except Exception:
                pass
            return None
        return img_bytes
    except Exception as e:
        st.error(f"Error calling FAL Seedream v4: {e}")
        return None

def edit_image_fal_seedream(image_bytes_list: list[bytes], prompt: str, strength: float = 0.6):
    """
    Seedream v4 edit supports multiple images via `image_urls` (up to 4).
    Pass 1–4 images as bytes; we convert to data URLs and send them in order.
    """
    try:
        if not image_bytes_list:
            raise ValueError("No images provided for edit.")
        # Build data URLs (PNG default)
        def to_data_url(b: bytes, mime: str = "image/png") -> str:
            return f"data:{mime};base64," + base64.b64encode(b).decode("utf-8")

        urls = [to_data_url(b) for b in image_bytes_list[:4]]

        payload = {
            "prompt": prompt,
            "image_urls": urls,                 # REQUIRED; list of up to 4
            "strength": max(0.0, min(1.0, float(strength))),
            "num_inference_steps": 28,
            "guidance_scale": 4.0,
            "num_samples": 1
        }

        resp_json = _fal_post("fal-ai/bytedance/seedream/v4/edit", payload)
        img_bytes = _extract_image_bytes_from_fal(resp_json)
        if not img_bytes:
            st.error("FAL edit response did not include an image.")
            try:
                st.caption(f"Debug (resp snippet): {json.dumps(resp_json)[:1200]}...")
            except Exception:
                pass
            return None
        return img_bytes

    except Exception as e:
        st.error(f"Error calling FAL Seedream v4 edit: {e}")
        return None



# --------- Ensure exact 16:9 via Seedream edit (API-compliant) ---------

def ensure_16x9_via_fal_edit(base_image_bytes: bytes, context_prompt: str = "", width: int = 1280, height: int = 720, strength: float = 0.35) -> bytes | None:
    """
    Use Seedream v4 edit API to expand the canvas to an exact 16:9 frame by outpainting left/right.
    Complies with FAL docs by passing `image_size` as an OBJECT: {"width": W, "height": H}.
    """
    try:
        data_url = "data:image/png;base64," + base64.b64encode(base_image_bytes).decode("utf-8")
        prompt = (
            "Extend the scene horizontally to fill a 16:9 landscape frame (outpaint left and right). "
            "Preserve the main subject and proportions; keep style, lighting, and depth of field consistent. "
            "Do not add on-image text, logos, or watermarks."
        )
        if context_prompt:
            prompt += f" Context: {context_prompt.strip()}"
        payload = {
            "prompt": prompt,
            "image_urls": [data_url],
            "image_size": {"width": int(width), "height": int(height)},
            "strength": max(0.0, min(1.0, float(strength))),
            "num_inference_steps": 24,
            "guidance_scale": 4.0,
            "num_samples": 1
        }
        resp_json = _fal_post("fal-ai/bytedance/seedream/v4/edit", payload)
        return _extract_image_bytes_from_fal(resp_json)
    except Exception as e:
        st.warning(f"16:9 expansion failed: {e}")
        return None


# ---------- OpenRouter (Gemini 2.5 Flash Image Preview) ----------
def _extract_openrouter_image_bytes(resp_json):
    """
    Extract image bytes from OpenRouter chat/completions responses for image models.
    Supports shapes:
      1) choices[0].message.images[].image_url.url
      2) choices[0].message.content: list of parts with image_url / image_base64 / b64_json
      3) choices[0].message.content: str containing a data URL
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
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
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
                        "text": "You are an image generation model. Return exactly one image and no extra text."
                    }
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
if 'use_flux_engineer' not in st.session_state:
    st.session_state.use_flux_engineer = True
if 'last_engineered_prompt' not in st.session_state:
    st.session_state.last_engineered_prompt = ""

# --- Add pending_edit_bytes and jump_to_edit state ---
if 'pending_edit_bytes' not in st.session_state:
    st.session_state.pending_edit_bytes = None
if 'jump_to_edit' not in st.session_state:
    st.session_state.jump_to_edit = False

# --- Add safe margins and prevent cropping state ---
if 'add_safe_margins' not in st.session_state:
    st.session_state.add_safe_margins = True
if 'prevent_cropping' not in st.session_state:
    st.session_state.prevent_cropping = True

st.title("AI Image Generator & WordPress Uploader")

# --- API key availability checks
openai_available = bool(OPENAI_API_KEY)
openrouter_available = bool(OPENROUTER_API_KEY)
fal_available = bool(FAL_API_KEY)

if not any([fal_available, openai_available, openrouter_available]):
    st.error("No API keys found. Please set at least one of: FAL_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY in .env")

# -----------------
# --- Main UI Tabs
# -----------------
if fal_available or openai_available or openrouter_available:
    openai_client = OpenAI() if openai_available else None

    tab_generate, tab_edit, tab_upload = st.tabs(["Generate", "Edit / Compose", "Upload Your Own"])

    # =========================
    # TAB 1: GENERATE (TXT->IMG)
    # =========================
    with tab_generate:
        with st.form("ai_image_form"):
            col_left, col_right = st.columns([3, 2])

            with col_left:
                prompt_text = st.text_area("Enter your image prompt (Thai or English).", height=140, key="prompt_input")
                ai_alt_text = st.text_area("Enter alt text for the generated image:", height=120, key="ai_alt_text_input")

            with col_right:
                use_flux_engineer = st.checkbox("Use prompt engineer (Thai ➜ English via LLM)", value=st.session_state.use_flux_engineer, help="Accept Thai/Eng input; LLM rewrites to polished English prompt for the image model.")
                st.session_state.use_flux_engineer = use_flux_engineer
                add_visually_striking_prefix = st.checkbox("Start with 'A visually striking image of...'", value=True, key="visually_striking_checkbox")
                bg_option = st.selectbox(
                    "Background / Color Scheme (site presets)",
                    ["None", "CryptoNews (Bitberry)", "ICOBench (Green)", "CryptoDnes (Gold)", "Bitcoinist (Blue)"],
                    index=0,
                    key="background_scheme_select"
                )
                st.session_state.prevent_cropping = st.checkbox("Prevent cropping (AI expand to 16:9)", value=st.session_state.prevent_cropping, help="Use Seedream edit to extend left/right so the image fits 16:9 without blur padding.")
                st.session_state.add_safe_margins = st.checkbox("Add 'safe margins' note to prompt", value=st.session_state.add_safe_margins, help="Ask the model to leave headroom/side margins for later crops.")
                prevent_cropping = st.session_state.prevent_cropping
                add_safe_margins = st.session_state.add_safe_margins

                # Provider options (FAL default)
                provider_options = []
                if fal_available:
                    provider_options.append("FAL Seedream v4 ($0.03 per image)")
                if openrouter_available:
                    provider_options.append("OpenRouter ($0.03 per image)")
                if openai_available:
                    provider_options.append("OpenAI ($0.30 per image)")  # moved last
                provider = st.radio("Choose AI Provider:", provider_options, index=0 if provider_options else 0, key="provider_selection")

                submitted = st.form_submit_button("Generate Image", use_container_width=True)

            if submitted and prompt_text.strip() and ai_alt_text.strip():
                st.session_state.current_prompt = prompt_text
                st.session_state.current_alt_text = ai_alt_text

                st.session_state.active_image_bytes_io = None
                st.session_state.active_image_alt_text = None
                st.session_state.user_uploaded_raw_bytes = None
                st.session_state.user_uploaded_alt_text_input = ""

                # 1) Optionally engineer the prompt via LLM role (Thai OK)
                base_prompt = prompt_text.strip()
                if use_flux_engineer:
                    with st.spinner("Engineering prompt..."):
                        engineered = enhance_prompt_with_role(base_prompt)
                        st.session_state.last_engineered_prompt = engineered
                        final_prompt = engineered
                else:
                    final_prompt = base_prompt

                if add_safe_margins:
                    final_prompt = f"{final_prompt.rstrip('.')}. Compose with generous headroom and side margins; keep the main subject centered and clear of edges; suitable for 16:9 crops."

                if add_visually_striking_prefix:
                    lower_pt = final_prompt.strip().lower()
                    if not lower_pt.startswith("a visually striking image of"):
                        final_prompt = f"A visually striking image of {final_prompt.lstrip()}"
                if bg_option and bg_option != "None":
                    preset_map = {
                        "CryptoNews (Bitberry)": "Use a purple colour scheme with modern gradients for the background",
                        "ICOBench (Green)": "Background should be a green-to-dark-green gradient applied only to background elements",
                        "CryptoDnes (Gold)": "Background should be a gold-to-dark gradient applied only to background elements",
                        "Bitcoinist (Blue)": "Background should use a blue-to-light-blue colour scheme applied only to background elements",
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

                with st.expander("Show final prompt sent to image model", expanded=False):
                    st.code(final_prompt, language="markdown")

                # --- Branch by provider
                image_bytes_from_api = None
                if provider.startswith("FAL") and fal_available:
                    st.session_state.last_used_provider = "FAL Seedream v4"
                    with st.spinner("Generating image with FAL Seedream v4..."):
                        image_bytes_from_api = generate_image_fal_seedream(final_prompt, width=1536, height=1024)
                elif provider.startswith("OpenRouter") and openrouter_available:
                    st.session_state.last_used_provider = "OpenRouter"
                    with st.spinner("Generating image with OpenRouter (Gemini 2.5 Flash Image)..."):
                        image_bytes_from_api = generate_image_openrouter(final_prompt, width=1536, height=1024)
                elif provider.startswith("OpenAI") and openai_available:
                    st.session_state.last_used_provider = "OpenAI"
                    with st.spinner("Generating image with OpenAI (Premium)..."):
                        try:
                            client = OpenAI()
                            response = client.images.generate(
                                model="gpt-image-1",
                                prompt=final_prompt,
                                n=1,
                                size="1536x1024"
                            )
                            if response.data and response.data[0].b64_json:
                                image_bytes_from_api = base64.b64decode(response.data[0].b64_json)
                            else:
                                st.error("No image data received from OpenAI.")
                        except Exception as e:
                            st.error(f"Failed to generate image with OpenAI: {e}")
                else:
                    st.error("Selected provider is not available. Check your API keys.")
                    image_bytes_from_api = None

                if image_bytes_from_api:
                    # If requested, expand to exact 16:9 using Seedream edit (API-compliant image_size object)
                    if image_bytes_from_api and prevent_cropping and fal_available:
                        with st.spinner("Expanding canvas to 16:9 (AI outpaint)..."):
                            expanded = ensure_16x9_via_fal_edit(image_bytes_from_api, context_prompt=final_prompt, width=1280, height=720, strength=0.35)
                            if expanded:
                                image_bytes_from_api = expanded
                            else:
                                st.info("16:9 expansion failed; proceeding without it.")
                    final_image_bytes_io = process_image_for_wordpress(
                        image_bytes_from_api,
                        force_landscape_16_9=True,
                        crop_strategy='crop'
                    )

                    if final_image_bytes_io:
                        st.session_state.active_image_bytes_io = final_image_bytes_io
                        st.session_state.active_image_alt_text = ai_alt_text
                        st.success(f"✅ Image generated successfully with {st.session_state.last_used_provider}")
                    else:
                        st.error("Failed to process the generated image.")
                else:
                    if provider:
                        st.error(f"Failed to generate image with {provider}.")

    # =========================
    # TAB 2: EDIT / COMPOSE
    # =========================
    with tab_edit:
        st.write("Use Seedream v4's advanced edit/composition. Upload a base image and optional secondary image, then describe the change.")
        # --- Show preview/notice if an image is staged for editing ---
        if st.session_state.pending_edit_bytes:
            st.success("A generated image from the **Generate** tab is ready to edit below.")
            st.image(BytesIO(st.session_state.pending_edit_bytes), caption="Staged for Edit", use_column_width=True)
        elif st.session_state.active_image_bytes_io:
            # If no explicit staging, we can still offer the last active image as a convenience
            try:
                st.info("Tip: You can use the currently active image for editing (see checkbox in the form).")
            except Exception:
                pass
        with st.form("fal_edit_form"):
            use_current_generated = st.checkbox(
                "Use generated image from **Generate** tab",
                value=bool(st.session_state.pending_edit_bytes or st.session_state.active_image_bytes_io),
                help="If checked, the image you generated will be used as the base. Otherwise, upload a file."
            )
            st.caption("Provide up to 4 images (1–4). If 'Use generated image' is checked, it counts as the first.")
            r1c1, r1c2 = st.columns(2)
            r2c1, r2c2 = st.columns(2)
            with r1c1:
                st.markdown("**Image 1**")
                img1_file = st.file_uploader("", type=['png', 'jpg', 'jpeg', 'webp'], key="fal_img1", label_visibility="collapsed")
            with r1c2:
                st.markdown("**Image 2**")
                img2_file = st.file_uploader("", type=['png', 'jpg', 'jpeg', 'webp'], key="fal_img2", label_visibility="collapsed")
            with r2c1:
                st.markdown("**Image 3**")
                img3_file = st.file_uploader("", type=['png', 'jpg', 'jpeg', 'webp'], key="fal_img3", label_visibility="collapsed")
            with r2c2:
                st.markdown("**Image 4**")
                img4_file = st.file_uploader("", type=['png', 'jpg', 'jpeg', 'webp'], key="fal_img4", label_visibility="collapsed")
            edit_prompt = st.text_area("Edit/Compose prompt (Thai or English):", height=140, key="fal_edit_prompt")
            use_engineer_edit = st.checkbox(
                "Use prompt engineer (Thai ➜ English via LLM)",
                value=True,
                help="Accept Thai/Eng input; LLM rewrites to polished English prompt tailored for Seedream edit."
            )
            st.session_state.add_safe_margins = st.checkbox("Add 'safe margins' note to prompt", value=st.session_state.add_safe_margins)
            st.session_state.prevent_cropping = st.checkbox("Prevent cropping (AI expand to 16:9)", value=st.session_state.prevent_cropping)
            add_safe_margins_edit = st.session_state.add_safe_margins
            prevent_cropping_edit = st.session_state.prevent_cropping
            strength = st.slider("Edit strength (lower = subtle, higher = stronger)", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
            make_16x9 = st.checkbox("Force landscape 16:9 after edit", value=True)
            alt_text_edit = st.text_input("Alt text for edited image:", value="Edited image")
            run_edit = st.form_submit_button("Run", use_container_width=True)

            if run_edit:
                if not fal_available:
                    st.error("FAL_API_KEY missing. Add it to your .env.")
                elif not edit_prompt.strip():
                    st.warning("Please enter an edit/composition prompt.")
                else:
                    # Engineer the edit prompt if enabled (Thai OK)
                    base_edit_prompt = edit_prompt.strip()
                    if use_engineer_edit:
                        with st.spinner("Engineering edit prompt..."):
                            engineered_edit = enhance_prompt_with_role(base_edit_prompt)
                    else:
                        engineered_edit = base_edit_prompt
                    final_edit_prompt = engineered_edit
                    if add_safe_margins_edit:
                        final_edit_prompt = f"{final_edit_prompt.rstrip('.')}. Compose with generous headroom and side margins; keep the main subject centered and clear of edges; suitable for 16:9 crops."
                    with st.expander("Show final edit prompt sent to Seedream", expanded=True):
                        st.code(final_edit_prompt, language="markdown")

                    images_list: list[bytes] = []
                    # Prefer explicitly staged bytes from Generate tab as first image
                    if use_current_generated and st.session_state.pending_edit_bytes:
                        images_list.append(st.session_state.pending_edit_bytes)
                    elif use_current_generated and st.session_state.active_image_bytes_io:
                        try:
                            st.session_state.active_image_bytes_io.seek(0)
                            images_list.append(st.session_state.active_image_bytes_io.getvalue())
                        except Exception:
                            pass
                    # Add uploaded files (in order) until we have at most 4
                    for f in [img1_file, img2_file, img3_file, img4_file]:
                        if f and len(images_list) < 4:
                            images_list.append(f.getvalue())

                    if not images_list:
                        st.warning("No base image available. Either check the 'Use generated image' option or upload 1–4 images.")
                    else:
                        with st.spinner("Running Seedream v4 edit..."):
                            out_bytes = edit_image_fal_seedream(images_list, final_edit_prompt, strength=strength)
                        if out_bytes:
                            # Expand to exact 16:9 using Seedream edit when requested
                            if out_bytes and prevent_cropping_edit and fal_available:
                                with st.spinner("Expanding canvas to 16:9 (AI outpaint)..."):
                                    expanded = ensure_16x9_via_fal_edit(out_bytes, context_prompt=final_edit_prompt, width=1280, height=720, strength=0.35)
                                    if expanded:
                                        out_bytes = expanded
                                    else:
                                        st.info("16:9 expansion failed; proceeding without it.")
                            processed = process_image_for_wordpress(
                                out_bytes,
                                force_landscape_16_9=make_16x9,
                                crop_strategy='crop'
                            )
                            if processed:
                                st.session_state.active_image_bytes_io = processed
                                st.session_state.active_image_alt_text = alt_text_edit
                                st.session_state.pending_edit_bytes = None  # clear staging after success
                                st.success("✅ Edit finished and ready for upload.")
                            else:
                                st.error("Failed to process edited image.")

    # =========================
    # TAB 3: UPLOAD YOUR OWN
    # =========================
    with tab_upload:
        st.markdown("--- _Or Upload Your Own Image_ ---")
        with st.form("user_image_upload_form"):
            uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg', 'webp'])
            user_alt_text = st.text_area("Enter alt text for your image:", height=80, key="user_alt_text_input_val")
            st.session_state.add_safe_margins = st.checkbox("Add 'safe margins' note to prompt", value=st.session_state.add_safe_margins)
            st.session_state.prevent_cropping = st.checkbox("Prevent cropping (AI expand to 16:9)", value=st.session_state.prevent_cropping)
            process_uploaded_button = st.form_submit_button("Process Uploaded Image")

            if process_uploaded_button and uploaded_file is not None and user_alt_text.strip():
                st.session_state.user_uploaded_raw_bytes = uploaded_file.getvalue()
                st.session_state.user_uploaded_alt_text_input = user_alt_text

                raw_bytes = st.session_state.user_uploaded_raw_bytes
                with st.spinner("Processing your uploaded image..."):
                    processed_user_image_io = process_image_for_wordpress(
                        raw_bytes,
                        crop_strategy='crop',
                        force_landscape_16_9=False
                    )
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
    st.markdown("--- _Active Image for WordPress Upload_ ---")

    st.session_state.active_image_bytes_io.seek(0)
    st.image(st.session_state.active_image_bytes_io, caption=st.session_state.active_image_alt_text[:100] + "...")

    # --- Edit button to stage image for Edit tab ---
    if st.button("Edit this image in Seedream", key="send_to_edit", use_container_width=True):
        try:
            image_data_bytesio_copy = BytesIO(st.session_state.active_image_bytes_io.getvalue())
        except Exception:
            st.session_state.active_image_bytes_io.seek(0)
            image_data_bytesio_copy = BytesIO(st.session_state.active_image_bytes_io.read())
        image_data_bytesio_copy.seek(0)
        st.session_state.pending_edit_bytes = image_data_bytesio_copy.getvalue()
        st.session_state.jump_to_edit = True
        st.success("Sent to **Edit / Compose** tab. Click that tab to continue.")

    current_alt_text = st.session_state.active_image_alt_text
    image_data_bytesio = st.session_state.active_image_bytes_io
    image_data_bytesio.seek(0)


    # Build SEO meta from the best available context (engineered prompt preferred)
    def pick_best_prompt_context() -> str:
        if st.session_state.get("last_engineered_prompt"):
            return st.session_state.last_engineered_prompt
        if st.session_state.get("current_prompt"):
            return st.session_state.current_prompt
        if st.session_state.get("user_uploaded_alt_text_input"):
            return st.session_state.user_uploaded_alt_text_input
        return st.session_state.active_image_alt_text or ""

    context_for_seo = pick_best_prompt_context()
    alt_text_for_upload, upload_filename = generate_seo_meta_from_prompt(context_for_seo)
    alt_text_for_upload = alt_text_for_upload[:100]

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
