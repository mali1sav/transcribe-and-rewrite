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
from urllib.parse import urlparse, unquote
import streamlit.components.v1 as components

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
# --- Providers: FAL (Seedream v4, Imagen 4 Preview), OpenAI, OpenRouter
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


def _filename_from_url(url: str) -> str:
    """Best-effort filename extraction from a remote image URL."""
    parsed = urlparse(url)
    if not parsed.path:
        return "dropped-image"
    candidate = os.path.basename(parsed.path)
    candidate = unquote(candidate)
    if not candidate:
        return "dropped-image"
    return candidate.split("?")[0] or "dropped-image"


def _build_external_drop_component(component_key: str = "external-drop"):
    """Render a custom drop zone capable of receiving browser-dragged remote images."""
    component_id = component_key.replace(" ", "-")
    drop_component = f"""
        <style>
            .external-drop-zone {{
                border: 2px dashed var(--secondary-color, #e0e0e0);
                border-radius: 10px;
                padding: 1.5rem;
                text-align: center;
                color: var(--text-color, #555);
                font-family: 'Source Sans Pro', sans-serif;
                background-color: rgba(0, 0, 0, 0.02);
                transition: border-color 0.2s ease, background-color 0.2s ease;
                cursor: pointer;
            }}
            .external-drop-zone.dragover {{
                border-color: #ff4b4b;
                background-color: rgba(255, 75, 75, 0.08);
                color: #ff4b4b;
            }}
            .external-drop-zone small {{
                display: block;
                margin-top: 0.5rem;
                color: inherit;
            }}
        </style>
        <div id="{component_id}-drop" class="external-drop-zone">
            <strong>Drag an image from another tab or click to pick a file</strong>
            <small>Images from other webpages will be fetched automatically.</small>
        </div>
        <input type="file" id="{component_id}-file" accept="image/*" style="display:none" />
        <script>
            const resolveStreamlit = () => {{
                const candidates = [
                    window.parent?.Streamlit,
                    window.Streamlit,
                    window.parent?.parent?.Streamlit,
                ];
                for (const candidate of candidates) {{
                    if (candidate && typeof candidate.setComponentValue === 'function') {{
                        return candidate;
                    }}
                }}
                return null;
            }};

            const doc = window.document;
            const dropZone = doc.getElementById('{component_id}-drop');
            const fileInput = doc.getElementById('{component_id}-file');

            const sendPayload = (payload) => {{
                const Streamlit = resolveStreamlit();
                if (!Streamlit || typeof Streamlit.setComponentValue !== 'function') {{
                    return;
                }}
                Streamlit.setComponentValue(JSON.stringify(payload));
            }};

            const readFileAsDataUrl = (file) => new Promise((resolve, reject) => {{
                const reader = new FileReader();
                reader.onload = () => resolve(reader.result);
                reader.onerror = (err) => reject(err);
                reader.readAsDataURL(file);
            }});

            const handleFile = async (file) => {{
                if (!file) return;
                try {{
                    const dataUrl = await readFileAsDataUrl(file);
                    const base64 = dataUrl.split(',')[1];
                    sendPayload({{
                        kind: 'file',
                        name: file.name,
                        type: file.type,
                        data: base64,
                        lastModified: file.lastModified || null
                    }});
                }} catch (err) {{
                    sendPayload({{ kind: 'error', message: 'Unable to read dropped file.' }});
                }}
            }};

            const extractUrlFromHtml = (html) => {{
                if (!html) return '';
                const match = html.match(/<img[^>]+src\s*=\s*"([^"]+)"/i) || html.match(/<img[^>]+src\s*=\s*'([^']+)'/i);
                if (match && match[1]) return match[1];
                const background = html.match(/url\((['"]?)(.*?)\1\)/i);
                if (background && background[2]) return background[2];
                return '';
            }};

            const resolveStringItems = async (items) => {{
                const strings = await Promise.all(Array.from(items || []).filter((item) => item.kind === 'string').map((item) => new Promise((resolve) => {{
                    try {{
                        item.getAsString((data) => resolve({{ type: item.type, data }}));
                    }} catch (err) {{
                        resolve(null);
                    }}
                }})));
                return strings.filter(Boolean);
            }};

            const findDroppedUrl = async (dt) => {{
                let url = dt.getData('text/uri-list') || dt.getData('text/plain');
                const htmlData = dt.getData('text/html');
                if (!url && htmlData) {{
                    url = extractUrlFromHtml(htmlData);
                }}

                if (url) return url.trim();

                const resolved = await resolveStringItems(dt.items);
                for (const entry of resolved) {{
                    if (entry.type === 'text/uri-list' && entry.data) return entry.data.split('\n')[0].trim();
                }}
                for (const entry of resolved) {{
                    if (entry.type === 'text/html' && entry.data) {{
                        const candidate = extractUrlFromHtml(entry.data);
                        if (candidate) return candidate;
                    }}
                }}
                for (const entry of resolved) {{
                    if (entry.data) {{
                        const maybeUrl = entry.data.trim();
                        if (maybeUrl.startsWith('http')) return maybeUrl;
                    }}
                }}
                return '';
            }};

            if (dropZone && !dropZone.dataset.bound) {{
                dropZone.dataset.bound = 'true';

                dropZone.addEventListener('click', () => fileInput && fileInput.click());

                dropZone.addEventListener('dragover', (event) => {{
                    event.preventDefault();
                    event.stopPropagation();
                    if (event.dataTransfer) {{
                        event.dataTransfer.dropEffect = 'copy';
                    }}
                    dropZone.classList.add('dragover');
                }});

                dropZone.addEventListener('dragleave', (event) => {{
                    event.preventDefault();
                    event.stopPropagation();
                    dropZone.classList.remove('dragover');
                }});

                dropZone.addEventListener('drop', async (event) => {{
                    event.preventDefault();
                    event.stopPropagation();
                    dropZone.classList.remove('dragover');

                    const dt = event.dataTransfer;
                    if (!dt) return;

                    const fileItem = Array.from(dt.items || []).find((item) => item.kind === 'file' && item.type.startsWith('image/'));
                    if (fileItem) {{
                        const file = fileItem.getAsFile();
                        if (file) {{
                            await handleFile(file);
                            return;
                        }}
                    }}

                    if (dt.files && dt.files.length > 0) {{
                        await handleFile(dt.files[0]);
                        return;
                    }}

                    const url = await findDroppedUrl(dt);
                    if (url) {{
                        sendPayload({{ kind: 'url', url }});
                    }} else {{
                        sendPayload({{ kind: 'error', message: 'Unable to detect an image from the drop event.' }});
                    }}
                }});

                fileInput && fileInput.addEventListener('change', async (event) => {{
                    const files = event.target.files;
                    if (files && files.length > 0) {{
                        await handleFile(files[0]);
                        event.target.value = '';
                    }}
                }});
            }}

            const markReady = () => {{
                const Streamlit = resolveStreamlit();
                if (Streamlit && typeof Streamlit.setComponentReady === 'function') {{
                    Streamlit.setComponentReady();
                    if (typeof Streamlit.setFrameHeight === 'function') {{
                        Streamlit.setFrameHeight(190);
                    }}
                }} else {{
                    setTimeout(markReady, 50);
                }}
            }};

            markReady();
        </script>
    """
    return components.html(drop_component, height=200, key=component_key)

# ---------- Thai-aware prompt engineering system prompt ----------
IMAGE_ENGINEER_SYSTEM = (
    "You are a prompt engineer specialized in image prompt generation. Your prompts result in stunning, high-quality visuals that capture attentions and clicks."
    "Transform the user's request (Thai or English) into a single English prompt optimized for Seedream v4. "
    "Follow Seedream best practices: be concise but specific; clearly state subject(s), attributes, actions, setting, composition, camera/shot type, lens/focal length, lighting, atmosphere, color palette, materials/textures, and style (photographic/illustration/3D). "
    "\n\n"
    "CRITICAL QUALITY ENHANCEMENTS - Always add these elements for professional, cinematic results:\n"
    "- LIGHTING: Specify dramatic, volumetric, or studio lighting with bloom effects, lens flares, and natural light rays where appropriate\n"
    "- COLOR GRADING: Add cinematic color palettes (teal and orange, blue and gold, warm sunset tones) for visual impact\n"
    "- MATERIALS & TEXTURES: Describe photorealistic materials (metallic surfaces with reflections, glass with refraction, water with caustics, fabric textures)\n"
    "- DEPTH & FOCUS: Include shallow depth of field with bokeh for professional photography look (e.g., 'f/1.8 aperture', 'cinematic bokeh')\n"
    "- DYNAMIC RANGE: Add 'HDR', 'high contrast', 'rich shadows', 'vibrant highlights' for premium quality\n"
    "- RESOLUTION & DETAIL: Mention '8K', 'ultra detailed', 'hyperrealistic textures', 'sharp focus'\n"
    "- ATMOSPHERE: Describe environmental effects (volumetric fog, light rays, particle effects, atmospheric haze)\n"
    "\n"
    "Prefer natural, realistic renderings with professional photography quality unless the user asks for stylization. "
    "When the user implies an editorial/crypto news illustration, emphasize cinematic lighting, metallic/glass materials, and teal-orange color grading for modern financial aesthetics. "
    "If the user asks in Thai, translate faithfully to English while enriching with these cinematic visual details. "
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
                    {"role": "system", "content": IMAGE_ENGINEER_SYSTEM},
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
                    {"role": "system", "content": IMAGE_ENGINEER_SYSTEM},
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

def _get_image_size(image_bytes: bytes) -> tuple[int, int] | None:
    """Return (width, height) for given image bytes or None on failure."""
    try:
        with Image.open(BytesIO(image_bytes)) as img:
            return img.width, img.height
    except Exception:
        return None


def _is_near_aspect_ratio(image_bytes: bytes, target_ratio: float = 16/9, tolerance: float = 0.03) -> bool:
    """
    Check whether the provided image bytes are already close to the target aspect ratio.
    tolerance is the relative difference allowed (e.g. 0.03 == ±3%).
    """
    size = _get_image_size(image_bytes)
    if not size:
        return False
    width, height = size
    if width <= 0 or height <= 0:
        return False
    ratio = width / float(height)
    return abs(ratio - target_ratio) <= target_ratio * tolerance


def generate_image_fal_seedream(prompt, width=1536, height=864):
    """
    Generate image using FAL Seedream v4 text-to-image.
    We request a precise 16:9 frame using the documented object form for `image_size`.
    Returns image bytes or None if failed.
    """
    try:
        payload = {
            "prompt": prompt,
            "image_size": {"width": int(width), "height": int(height)},
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


def generate_image_fal_imagen4_preview(prompt, width=1536, height=1024):
    """
    Generate image using FAL Imagen 4 (Preview).
    Tries the same payload shape as Seedream (prompt + image_size object).
    Returns image bytes or None if failed.
    """
    try:
        # Attempt 1: image_size object (preferred by many FAL pipelines)
        attempts = []
        attempts.append({
            "prompt": prompt,
            "image_size": {"width": int(width), "height": int(height)}
        })
        # Attempt 2: explicit width/height
        attempts.append({
            "prompt": prompt,
            "width": int(width),
            "height": int(height)
        })
        # Attempt 3: size string
        attempts.append({
            "prompt": prompt,
            "size": f"{int(width)}x{int(height)}"
        })

        last_error = None
        for payload in attempts:
            try:
                resp_json = _fal_post("fal-ai/imagen4/preview", payload)
                img_bytes = _extract_image_bytes_from_fal(resp_json)
                if img_bytes:
                    return img_bytes
                # If no bytes but no exception, continue to next shape
            except Exception as inner_e:
                last_error = inner_e
                continue

        if last_error:
            st.error(f"Error calling FAL Imagen 4 (Preview): {last_error}")
        else:
            st.error("No image data received from FAL Imagen 4 (Preview).")
        return None
    except Exception as e:
        st.error(f"Error calling FAL Imagen 4 (Preview): {e}")
        return None

def edit_image_fal_seedream(image_bytes_list: list[bytes], prompt: str, strength: float = 0.6, preserve_subject_lighting: bool = False, negative_prompt: str = "", image_size: dict | None = None):
    """
    Seedream v4 edit supports multiple images via `image_urls` (up to 4).
    Pass 1–4 images as bytes; we convert to data URLs and send them in order.
    
    Args:
        image_bytes_list: List of image bytes to edit/compose (1-4 images)
        prompt: Edit/composition instructions
        strength: Edit strength (0.0-1.0); lower preserves more of original
        preserve_subject_lighting: If True, adds instructions to maintain subject brightness/vibrancy
        negative_prompt: Negative prompt to avoid unwanted characteristics
        image_size: Optional dict with 'width' and 'height' keys (e.g., {"width": 3840, "height": 2160})
    """
    try:
        if not image_bytes_list:
            raise ValueError("No images provided for edit.")
        # Build data URLs (PNG default)
        def to_data_url(b: bytes, mime: str = "image/png") -> str:
            return f"data:{mime};base64," + base64.b64encode(b).decode("utf-8")

        urls = [to_data_url(b) for b in image_bytes_list[:4]]
        
        # Enhance prompt to preserve subject lighting if requested
        final_prompt = prompt
        final_negative_prompt = ""
        
        if preserve_subject_lighting:
            final_prompt = (
                f"{prompt.strip()} "
                "The main subject should retain its bright, vibrant lighting and colors. "
                "Keep the subject well-lit. "
                "Maintain vivid colors on the subject."
            )
        
        # Only add negative prompt if explicitly requested (separate from preserve_subject_lighting)
        if negative_prompt:  # If user passed a custom negative prompt
            final_negative_prompt = negative_prompt

        payload = {
            "prompt": final_prompt,
            "image_urls": urls,                 # REQUIRED; list of up to 4
            "strength": max(0.0, min(1.0, float(strength))),
            "num_inference_steps": 28,
            "guidance_scale": 4.0,  # FAL default - lower values allow more creative interpretation
            "num_samples": 1
        }
        
        # Add image size if specified (matches FAL playground behavior)
        if image_size and isinstance(image_size, dict) and "width" in image_size and "height" in image_size:
            payload["image_size"] = image_size
        
        # Add negative prompt if provided (Seedream 4.0 supports this)
        if final_negative_prompt:
            payload["negative_prompt"] = final_negative_prompt

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
if 'editable_prompt' not in st.session_state:
    st.session_state.editable_prompt = ""
if 'prompt_edited' not in st.session_state:
    st.session_state.prompt_edited = False
if 'original_prompt_for_reset' not in st.session_state:
    st.session_state.original_prompt_for_reset = ""
if 'prompt_ready' not in st.session_state:
    st.session_state.prompt_ready = False
if 'editable_edit_prompt' not in st.session_state:
    st.session_state.editable_edit_prompt = ""
if 'edit_prompt_edited' not in st.session_state:
    st.session_state.edit_prompt_edited = False
if 'last_engineered_edit_prompt' not in st.session_state:
    st.session_state.last_engineered_edit_prompt = ""
if 'edit_original_prompt_for_reset' not in st.session_state:
    st.session_state.edit_original_prompt_for_reset = ""
if 'edit_prompt_ready' not in st.session_state:
    st.session_state.edit_prompt_ready = False
if 'edit_images_bytes' not in st.session_state:
    st.session_state.edit_images_bytes = []
if 'edit_strength' not in st.session_state:
    st.session_state.edit_strength = 0.35
if 'edit_preserve_lighting' not in st.session_state:
    st.session_state.edit_preserve_lighting = False  # Changed to False - let user opt-in
if 'edit_use_negative_prompt' not in st.session_state:
    st.session_state.edit_use_negative_prompt = False
if 'edit_output_resolution' not in st.session_state:
    st.session_state.edit_output_resolution = "3840x2160"  # Match FAL playground default
if 'edit_force_169' not in st.session_state:
    st.session_state.edit_force_169 = True
if 'edit_alt_text_value' not in st.session_state:
    st.session_state.edit_alt_text_value = "Edited image"
if 'edit_prevent_cropping' not in st.session_state:
    st.session_state.edit_prevent_cropping = True
if 'edit_add_safe_margins' not in st.session_state:
    st.session_state.edit_add_safe_margins = False
if 'edit_use_prompt_engineer' not in st.session_state:
    st.session_state.edit_use_prompt_engineer = False
if 'edit_use_generated_image' not in st.session_state:
    st.session_state.edit_use_generated_image = False
if 'edit_raw_prompt' not in st.session_state:
    st.session_state.edit_raw_prompt = ""

# --- Add pending_edit_bytes and jump_to_edit state ---
if 'pending_edit_bytes' not in st.session_state:
    st.session_state.pending_edit_bytes = None
if 'jump_to_edit' not in st.session_state:
    st.session_state.jump_to_edit = False

# --- Add safe margins and prevent cropping state ---
if 'add_safe_margins' not in st.session_state:
    st.session_state.add_safe_margins = False
if 'prevent_cropping' not in st.session_state:
    st.session_state.prevent_cropping = True
if 'external_drop_bytes' not in st.session_state:
    st.session_state.external_drop_bytes = None
if 'external_drop_name' not in st.session_state:
    st.session_state.external_drop_name = ""
if 'external_drop_source' not in st.session_state:
    st.session_state.external_drop_source = ""
if 'external_drop_token' not in st.session_state:
    st.session_state.external_drop_token = ""
if 'external_drop_error' not in st.session_state:
    st.session_state.external_drop_error = ""

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
        provider_options: list[str] = []
        if fal_available:
            provider_options.append("FAL Seedream v4 ($0.03 per image)")
            provider_options.append("FAL Imagen 4 (Preview)")
        if openrouter_available:
            provider_options.append("OpenRouter ($0.03 per image)")
        if openai_available:
            provider_options.append("OpenAI ($0.30 per image)")  # moved last

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

                prepare_prompt = st.form_submit_button("Prepare Prompt", use_container_width=True)

            if prepare_prompt:
                if not prompt_text.strip() or not ai_alt_text.strip():
                    st.warning("Please enter both a prompt and alt text before preparing.")
                else:
                    st.session_state.current_prompt = prompt_text
                    st.session_state.current_alt_text = ai_alt_text

                    # Reset previously generated assets
                    st.session_state.active_image_bytes_io = None
                    st.session_state.active_image_alt_text = None
                    st.session_state.user_uploaded_raw_bytes = None
                    st.session_state.user_uploaded_alt_text_input = ""

                    # 1) Optionally engineer the prompt via LLM role (Thai OK)
                    base_prompt = prompt_text.strip()
                    engineered = base_prompt
                    if use_flux_engineer:
                        with st.spinner("Engineering prompt..."):
                            engineered = enhance_prompt_with_role(base_prompt)
                        st.session_state.last_engineered_prompt = engineered
                    else:
                        st.session_state.last_engineered_prompt = base_prompt

                    final_prompt = engineered

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

                    st.session_state.editable_prompt = final_prompt
                    st.session_state.original_prompt_for_reset = final_prompt
                    st.session_state.prompt_edited = False
                    # Default provider selection after preparation
                    if provider_options:
                        default_provider = st.session_state.get("selected_provider_value", provider_options[0])
                        if default_provider not in provider_options:
                            default_provider = provider_options[0]
                        st.session_state.selected_provider_value = default_provider
                    else:
                        st.session_state.selected_provider_value = ""
                    st.session_state.prepared_prevent_cropping = prevent_cropping
                    st.session_state.prompt_ready = True

                    st.success("Prompt prepared. Review and edit below before generating the image.")

        if st.session_state.get("prompt_ready"):
            selected_provider_default = st.session_state.get("selected_provider_value", provider_options[0] if provider_options else "")
            if provider_options:
                selected_index = provider_options.index(selected_provider_default) if selected_provider_default in provider_options else 0
                selected_provider = st.radio(
                    "Choose AI Provider:",
                    provider_options,
                    index=selected_index,
                    key="provider_selection_ready"
                )
            else:
                selected_provider = ""

            st.session_state.selected_provider_value = selected_provider

            st.subheader("✏️ Edit Enhanced Prompt (Optional)")
            st.caption("Modify the enhanced English prompt below. The final version shown will be sent to the image model when you generate.")

            edited_prompt = st.text_area(
                "Enhanced Prompt (editable):",
                value=st.session_state.editable_prompt,
                height=160,
                key="editable_prompt_editor"
            )

            if edited_prompt != st.session_state.editable_prompt:
                st.session_state.editable_prompt = edited_prompt
                st.session_state.prompt_edited = edited_prompt.strip() != st.session_state.original_prompt_for_reset.strip()

            col_actions = st.columns([1, 1, 1]) if provider_options else [st.container(), st.container(), st.container()]

            with col_actions[0]:
                reset_clicked = st.button("🔄 Reset Prompt", key="reset_prompt_outside", use_container_width=True)
            with col_actions[1]:
                st.empty()
            with col_actions[2]:
                generate_clicked = st.button("🚀 Generate Image", key="generate_image_button", use_container_width=True, type="primary")

            if reset_clicked:
                st.session_state.editable_prompt = st.session_state.original_prompt_for_reset
                st.session_state.prompt_edited = False
                st.experimental_rerun()

            if generate_clicked:
                final_prompt = (st.session_state.editable_prompt or "").strip()
                if not final_prompt:
                    st.error("Final prompt is empty. Please revise it before generating.")
                else:
                    if not st.session_state.current_alt_text:
                        st.error("Alt text is missing. Please prepare the prompt again with alt text.")
                    else:
                        image_bytes_from_api = None
                        provider_choice = selected_provider or ""

                        if fal_available and ("Seedream" in provider_choice):
                            st.session_state.last_used_provider = "FAL Seedream v4"
                            with st.spinner("Generating image with FAL Seedream v4..."):
                                image_bytes_from_api = generate_image_fal_seedream(final_prompt, width=1536, height=1024)
                        elif fal_available and ("Imagen 4" in provider_choice):
                            st.session_state.last_used_provider = "FAL Imagen 4 (Preview)"
                            with st.spinner("Generating image with FAL Imagen 4 (Preview)..."):
                                image_bytes_from_api = generate_image_fal_imagen4_preview(final_prompt, width=1536, height=1024)
                        elif provider_choice.startswith("OpenRouter") and openrouter_available:
                            st.session_state.last_used_provider = "OpenRouter"
                            with st.spinner("Generating image with OpenRouter (Gemini 2.5 Flash Image)..."):
                                image_bytes_from_api = generate_image_openrouter(final_prompt, width=1536, height=1024)
                        elif provider_choice.startswith("OpenAI") and openai_available:
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
                            if provider_options:
                                st.error("Selected provider is not available. Check your API keys or choose another provider.")
                            image_bytes_from_api = None

                        if image_bytes_from_api:
                            prevent_cropping_flag = st.session_state.get("prepared_prevent_cropping", False)
                            needs_outpaint = not _is_near_aspect_ratio(image_bytes_from_api) and prevent_cropping_flag and fal_available
                            if needs_outpaint:
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
                                st.session_state.active_image_alt_text = st.session_state.current_alt_text
                                st.session_state.prompt_ready = False
                                st.success(f"✅ Image generated successfully with {st.session_state.last_used_provider}")
                            else:
                                st.error("Failed to process the generated image.")
                        else:
                            if provider_choice:
                                st.error(f"Failed to generate image with {provider_choice}.")

    # =========================
    # TAB 2: EDIT / COMPOSE
    # =========================
    with tab_edit:
        st.write("Use Seedream v4's advanced edit/composition. Upload a base image and optional secondary image, then describe the change.")

        if st.session_state.pending_edit_bytes:
            st.success("A generated image from the **Generate** tab is staged for editing.")
        elif st.session_state.active_image_bytes_io:
            st.info("Tip: enable **Use generated image** to start from your most recent output.")

        with st.form("fal_edit_form"):
            use_current_generated = st.checkbox(
                "Use generated image from **Generate** tab",
                value=st.session_state.edit_use_generated_image,
                help="If checked, the last generated image becomes the first base image."
            )
            st.caption("Provide up to 4 images (1–4). If 'Use generated image' is checked, it counts as the first input.")

            r1c1, r1c2 = st.columns(2)
            with r1c1:
                st.markdown("**Image 1**")
                img1_file = st.file_uploader("", type=['png', 'jpg', 'jpeg', 'webp'], key="fal_img1", label_visibility="collapsed")
            with r1c2:
                st.markdown("**Image 2**")
                img2_file = st.file_uploader("", type=['png', 'jpg', 'jpeg', 'webp'], key="fal_img2", label_visibility="collapsed")

            r2c1, r2c2 = st.columns(2)
            with r2c1:
                st.markdown("**Image 3**")
                img3_file = st.file_uploader("", type=['png', 'jpg', 'jpeg', 'webp'], key="fal_img3", label_visibility="collapsed")
            with r2c2:
                st.markdown("**Image 4**")
                img4_file = st.file_uploader("", type=['png', 'jpg', 'jpeg', 'webp'], key="fal_img4", label_visibility="collapsed")

            edit_prompt_input = st.text_area(
                "Edit/Compose prompt (Thai or English):",
                height=140,
                value=st.session_state.edit_raw_prompt
            )

            use_engineer_edit = st.checkbox(
                "Use prompt engineer (Thai ➜ English via LLM)",
                value=st.session_state.edit_use_prompt_engineer,
                help="Leave off to preserve your exact instructions. Enable if you need translation/expansion."
            )

            prepare_edit = st.form_submit_button("Prepare Edit", use_container_width=True)

        if prepare_edit:
            if not fal_available:
                st.error("FAL_API_KEY missing. Add it to your .env.")
            elif not edit_prompt_input.strip():
                st.warning("Please enter an edit/composition prompt.")
            else:
                images_list: list[bytes] = []
                if use_current_generated and st.session_state.pending_edit_bytes:
                    images_list.append(st.session_state.pending_edit_bytes)
                elif use_current_generated and st.session_state.active_image_bytes_io:
                    try:
                        st.session_state.active_image_bytes_io.seek(0)
                        images_list.append(st.session_state.active_image_bytes_io.getvalue())
                    except Exception:
                        pass
                for f in [img1_file, img2_file, img3_file, img4_file]:
                    if f and len(images_list) < 4:
                        images_list.append(f.getvalue())

                if not images_list:
                    st.warning("No base image available. Either enable 'Use generated image' or upload 1–4 files.")
                else:
                    base_edit_prompt = edit_prompt_input.strip()
                    engineered_edit = base_edit_prompt
                    if use_engineer_edit:
                        with st.spinner("Engineering edit prompt..."):
                            engineered_edit = enhance_prompt_with_role(base_edit_prompt)
                        st.session_state.last_engineered_edit_prompt = engineered_edit
                    else:
                        st.session_state.last_engineered_edit_prompt = ""

                    st.session_state.edit_use_generated_image = use_current_generated
                    st.session_state.edit_use_prompt_engineer = use_engineer_edit
                    st.session_state.edit_images_bytes = images_list
                    st.session_state.edit_raw_prompt = base_edit_prompt
                    st.session_state.editable_edit_prompt = engineered_edit
                    st.session_state.edit_original_prompt_for_reset = engineered_edit
                    st.session_state.edit_prompt_editor_ready = engineered_edit
                    st.session_state.edit_prompt_edited = False
                    st.session_state.edit_prompt_ready = True
                    st.session_state.edit_add_safe_margins = False
                    st.success("Edit prompt prepared. Refine it below, then run the edit when ready.")

        if st.session_state.get("edit_prompt_ready"):
            staged_count = len(st.session_state.edit_images_bytes)
            st.markdown("### Step 2 · Refine & Run the Edit")
            st.caption(f"{staged_count} image{'s' if staged_count != 1 else ''} staged for Seedream edit.")

            if "edit_prompt_editor_ready" not in st.session_state:
                st.session_state.edit_prompt_editor_ready = st.session_state.editable_edit_prompt

            edited_prompt = st.text_area(
                "Enhanced Edit Prompt (editable):",
                key="edit_prompt_editor_ready",
                height=180
            )

            if edited_prompt != st.session_state.editable_edit_prompt:
                st.session_state.editable_edit_prompt = edited_prompt
                st.session_state.edit_prompt_edited = edited_prompt.strip() != st.session_state.edit_original_prompt_for_reset.strip()

            action_cols = st.columns([1, 1, 2])
            with action_cols[0]:
                if st.button("🔄 Reset Prompt", key="reset_edit_prompt_ready", use_container_width=True):
                    st.session_state.editable_edit_prompt = st.session_state.edit_original_prompt_for_reset
                    st.session_state.edit_prompt_editor_ready = st.session_state.edit_original_prompt_for_reset
                    st.session_state.edit_prompt_edited = False
                    st.experimental_rerun()
            with action_cols[1]:
                st.caption("Use advanced options to adjust margins, cropping, and strength.")

            with st.expander("Advanced edit options", expanded=False):
                safe_margin_toggle = st.checkbox(
                    "Add 'safe margins' note to prompt",
                    value=st.session_state.edit_add_safe_margins,
                    key="edit_safe_margins_toggle"
                )
                prevent_cropping_toggle = st.checkbox(
                    "Prevent cropping (AI expand to 16:9)",
                    value=st.session_state.edit_prevent_cropping,
                    key="edit_prevent_cropping_toggle"
                )
                force_landscape_toggle = st.checkbox(
                    "Force landscape 16:9 after edit",
                    value=st.session_state.edit_force_169,
                    key="edit_force_169_toggle"
                )
                strength_value = st.slider(
                    "Edit strength (lower = subtle, higher = stronger)",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.edit_strength,
                    step=0.05,
                    key="edit_strength_slider",
                    help="For compositing bright subjects onto dark backgrounds, use 0.25-0.35. Higher values (0.6-0.8) allow more blending but may darken subjects."
                )
                preserve_lighting_toggle = st.checkbox(
                    "Preserve subject brightness/lighting",
                    value=st.session_state.edit_preserve_lighting,
                    key="edit_preserve_lighting_toggle",
                    help="Adds explicit instructions to maintain subject vibrancy. May make composition less natural. Try without this first."
                )
                use_negative_prompt_toggle = st.checkbox(
                    "Use negative prompt (aggressive preservation)",
                    value=st.session_state.edit_use_negative_prompt,
                    key="edit_use_negative_prompt_toggle",
                    help="Adds negative prompt to explicitly prevent darkening. Very strong - only enable if results are still too dark without it."
                )
                
                output_resolution = st.selectbox(
                    "Output resolution",
                    ["3840x2160 (4K)", "1920x1080 (Full HD)", "1280x720 (HD)", "Auto (let model decide)"],
                    index=0,
                    key="edit_output_resolution_select",
                    help="Higher resolution improves lighting detail and composition quality. FAL playground uses 4K by default."
                )

            st.session_state.edit_add_safe_margins = safe_margin_toggle
            st.session_state.edit_prevent_cropping = prevent_cropping_toggle
            st.session_state.edit_force_169 = force_landscape_toggle
            st.session_state.edit_strength = strength_value
            st.session_state.edit_preserve_lighting = preserve_lighting_toggle
            st.session_state.edit_use_negative_prompt = use_negative_prompt_toggle
            st.session_state.edit_output_resolution = output_resolution

            st.session_state.edit_alt_text_value = st.text_input(
                "Alt text for edited image:",
                value=st.session_state.edit_alt_text_value,
                key="edit_alt_text_ready"
            )

            with action_cols[2]:
                run_edit_now = st.button("🚀 Run Edit", key="run_edit_now", use_container_width=True, type="primary")

            if run_edit_now:
                images_list = st.session_state.edit_images_bytes
                if not images_list:
                    st.warning("No base images staged. Prepare the edit again.")
                else:
                    final_edit_prompt = st.session_state.editable_edit_prompt.strip()
                    if st.session_state.edit_add_safe_margins and final_edit_prompt:
                        final_edit_prompt = f"{final_edit_prompt.rstrip('.')}. Compose with generous headroom and side margins; keep the main subject centered and clear of edges; suitable for 16:9 crops."

                    with st.spinner("Running Seedream v4 edit..."):
                        # Build negative prompt only if explicitly enabled
                        neg_prompt = ""
                        if st.session_state.edit_use_negative_prompt:
                            neg_prompt = "dark face, dull face, underexposed subject, dim lighting on subject, muddy colors, loss of vibrancy, shadowy face"
                        
                        # Parse resolution setting
                        img_size = None
                        res_str = st.session_state.edit_output_resolution
                        if "3840x2160" in res_str:
                            img_size = {"width": 3840, "height": 2160}
                        elif "1920x1080" in res_str:
                            img_size = {"width": 1920, "height": 1080}
                        elif "1280x720" in res_str:
                            img_size = {"width": 1280, "height": 720}
                        # else: None = let model decide
                        
                        out_bytes = edit_image_fal_seedream(
                            images_list, 
                            final_edit_prompt, 
                            strength=st.session_state.edit_strength,
                            preserve_subject_lighting=st.session_state.edit_preserve_lighting,
                            negative_prompt=neg_prompt,
                            image_size=img_size
                        )

                    if out_bytes:
                        needs_outpaint_edit = st.session_state.edit_prevent_cropping and fal_available and not _is_near_aspect_ratio(out_bytes)
                        if needs_outpaint_edit:
                            with st.spinner("Expanding canvas to 16:9 (AI outpaint)..."):
                                expanded = ensure_16x9_via_fal_edit(out_bytes, context_prompt=final_edit_prompt, width=1280, height=720, strength=0.35)
                                if expanded:
                                    out_bytes = expanded
                                else:
                                    st.info("16:9 expansion failed; proceeding without it.")

                        processed = process_image_for_wordpress(
                            out_bytes,
                            force_landscape_16_9=st.session_state.edit_force_169,
                            crop_strategy='crop'
                        )

                        if processed:
                            st.session_state.active_image_bytes_io = processed
                            st.session_state.active_image_alt_text = st.session_state.edit_alt_text_value
                            st.session_state.pending_edit_bytes = None
                            st.success("✅ Edit finished and ready for WordPress upload.")
                        else:
                            st.error("Failed to process the edited image.")
                    else:
                        st.error("Seedream did not return an edited image. Try adjusting the prompt or strength.")

    # =========================
    # TAB 3: UPLOAD YOUR OWN
    # =========================
    with tab_upload:
        st.markdown("--- _Or Upload Your Own Image_ ---")

        drop_payload_raw = _build_external_drop_component("user-upload-drop")
        drop_payload = None
        if drop_payload_raw:
            if isinstance(drop_payload_raw, str):
                try:
                    drop_payload = json.loads(drop_payload_raw)
                except json.JSONDecodeError:
                    drop_payload = None
            elif isinstance(drop_payload_raw, dict):
                drop_payload = drop_payload_raw

        if isinstance(drop_payload, dict):
            kind = drop_payload.get("kind")
            token_value = ""
            if kind == "file":
                data_b64 = drop_payload.get("data")
                token_value = f"file:{drop_payload.get('name', '')}:{drop_payload.get('lastModified', '')}:{len(data_b64 or '')}"
                if data_b64 and token_value != st.session_state.external_drop_token:
                    try:
                        raw_bytes = base64.b64decode(data_b64)
                    except Exception:
                        st.session_state.external_drop_error = "Dropped file could not be decoded as an image."
                        st.session_state.external_drop_token = token_value
                    else:
                        st.session_state.external_drop_bytes = raw_bytes
                        st.session_state.external_drop_name = drop_payload.get("name") or "dropped-image"
                        st.session_state.external_drop_source = "Drag-and-drop file"
                        st.session_state.external_drop_token = token_value
                        st.session_state.external_drop_error = ""
                        st.success(f"Image '{st.session_state.external_drop_name}' staged from drag-and-drop.")
            elif kind == "url":
                url = (drop_payload.get("url") or "").strip()
                token_value = f"url:{url}"
                if url and token_value != st.session_state.external_drop_token:
                    try:
                        if url.startswith("data:image"):
                            header, data_part = url.split(",", 1)
                            raw_bytes = base64.b64decode(data_part)
                            mime = header.split(";")[0].split(":")[1] if ":" in header else "image/png"
                            ext = mime.split("/")[-1] if "/" in mime else "png"
                            filename_guess = f"dropped-image.{ext}" if ext else "dropped-image"
                        else:
                            headers = {"User-Agent": "Mozilla/5.0 (compatible; Streamlit-Drop/1.0)"}
                            response = requests.get(url, timeout=15, headers=headers)
                            response.raise_for_status()
                            content_type = response.headers.get("Content-Type", "")
                            if "image" not in content_type.lower():
                                raise ValueError("Dropped URL does not point to an image.")
                            content_length = response.headers.get("Content-Length")
                            max_remote_size = 15 * 1024 * 1024
                            if content_length:
                                try:
                                    if int(content_length) > max_remote_size:
                                        raise ValueError("Dropped image is larger than 15MB.")
                                except ValueError:
                                    pass
                            raw_bytes = response.content
                            if len(raw_bytes) > max_remote_size:
                                raise ValueError("Dropped image is larger than 15MB.")
                            if not raw_bytes:
                                raise ValueError("Dropped image returned no data.")
                            filename_guess = _filename_from_url(url)
                            if "." not in filename_guess:
                                ext = content_type.split("/")[-1].split(";")[0] if "/" in content_type else "jpg"
                                if ext and not filename_guess.lower().endswith(ext.lower()):
                                    filename_guess = f"{filename_guess}.{ext}"
                        st.session_state.external_drop_bytes = raw_bytes
                        st.session_state.external_drop_name = filename_guess
                        st.session_state.external_drop_source = url
                        st.session_state.external_drop_token = token_value
                        st.session_state.external_drop_error = ""
                        st.success("Image fetched from dropped URL.")
                    except Exception as drop_err:
                        message = str(drop_err) or "Unknown error"
                        st.session_state.external_drop_error = f"Failed to fetch dropped image: {message}"
                        st.session_state.external_drop_token = token_value
                elif not url:
                    st.session_state.external_drop_error = "Dropped item did not include an image URL."
                    st.session_state.external_drop_token = token_value
            elif kind == "error":
                token_value = f"error:{drop_payload.get('message', '')}"
                if token_value != st.session_state.external_drop_token:
                    st.session_state.external_drop_error = drop_payload.get("message", "Unable to drop the image.")
                    st.session_state.external_drop_token = token_value

        if st.session_state.external_drop_error:
            st.warning(st.session_state.external_drop_error)

        if st.session_state.external_drop_bytes:
            preview_col, clear_col = st.columns([4, 1])
            with preview_col:
                st.image(BytesIO(st.session_state.external_drop_bytes), caption=st.session_state.external_drop_name or "Dropped image", use_column_width=True)
                if st.session_state.external_drop_source and not st.session_state.external_drop_source.startswith("Drag"):
                    st.caption(f"Source: {st.session_state.external_drop_source}")
            with clear_col:
                if st.button("Clear drop", key="clear_external_drop"):
                    st.session_state.external_drop_bytes = None
                    st.session_state.external_drop_name = ""
                    st.session_state.external_drop_source = ""
                    st.session_state.external_drop_token = ""
                    st.session_state.external_drop_error = ""
                    st.experimental_rerun()

        with st.form("user_image_upload_form"):
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['png', 'jpg', 'jpeg', 'webp'],
                help="You can also drag an image from another tab into the drop zone above."
            )
            if uploaded_file is None and st.session_state.external_drop_bytes:
                st.caption(f"Using dropped image: {st.session_state.external_drop_name or 'dropped image'}")
            user_alt_text = st.text_area("Enter alt text for your image:", height=80, key="user_alt_text_input_val")
            st.session_state.add_safe_margins = st.checkbox("Add 'safe margins' note to prompt", value=st.session_state.add_safe_margins)
            st.session_state.prevent_cropping = st.checkbox("Prevent cropping (AI expand to 16:9)", value=st.session_state.prevent_cropping)
            process_uploaded_button = st.form_submit_button("Process Uploaded Image")

            selected_bytes = None
            if uploaded_file is not None:
                selected_bytes = uploaded_file.getvalue()
            elif st.session_state.external_drop_bytes:
                selected_bytes = st.session_state.external_drop_bytes

            if process_uploaded_button and selected_bytes is not None and user_alt_text.strip():
                st.session_state.user_uploaded_raw_bytes = selected_bytes
                st.session_state.user_uploaded_alt_text_input = user_alt_text

                with st.spinner("Processing your uploaded image..."):
                    processed_user_image_io = process_image_for_wordpress(
                        selected_bytes,
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
            elif process_uploaded_button and (selected_bytes is None or not user_alt_text.strip()):
                st.warning("Please upload or drop an image AND provide alt text before processing.")

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
