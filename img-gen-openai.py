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
from urllib.parse import urlparse
import csv

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
def process_image_for_wordpress(image_bytes, final_quality=80, force_landscape_16_9=False, max_size_kb=100):
    """
    Optimizes for WP and (optionally) hard-enforces 16:9 landscape by center-cropping.
    Adds a hard cap on output size via iterative JPEG compression and optional downscale.

    Rules (unchanged):
    - Landscape images: Max width 1200px
    - Portrait images: Max height 675px (16:9 ratio)
    - Very wide banners: Max width 1400px, min height 300px
    - Very tall images: Max height 800px, min width 400px

    New:
    - Ensure final file size <= max_size_kb by reducing quality stepwise, then downscaling if necessary.
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
                    # too wide -> crop width
                    new_w = int(h * target)
                    left = max((w - new_w) // 2, 0)
                    img = img.crop((left, 0, left + new_w, h))
                else:
                    # too tall -> crop height
                    new_h = int(w / target)
                    top = max((h - new_h) // 2, 0)
                    img = img.crop((0, top, w, top + new_h))
                w, h = img.size
                aspect_ratio = w / float(h)

        # --- Resize rules (same as before) ---
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

        # do initial resize (if needed)
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

        # 1) reduce quality down to min_quality
        while cur_size > target_bytes and quality > min_quality:
            quality = max(min_quality, quality - quality_step)
            out = save_to_buffer(img_resized, quality)
            cur_size = out.getbuffer().nbytes

        # 2) if still too big, progressively downscale by 90% until under cap or min dims reached
        min_w, min_h = 640, 360  # keep thumbnails reasonably sharp for 16:9
        ds_img = img_resized
        while cur_size > target_bytes and (ds_img.width > min_w and ds_img.height > min_h):
            new_w = max(min_w, int(ds_img.width * 0.9))
            new_h = max(min_h, int(ds_img.height * 0.9))
            ds_img = ds_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            # after downscale, try a slightly higher quality first for better visuals, then drop if needed
            trial_quality = min(quality + 5, 85)
            out = save_to_buffer(ds_img, trial_quality)
            cur_size = out.getbuffer().nbytes
            # if still too big, drop quality again
            while cur_size > target_bytes and trial_quality > min_quality:
                trial_quality = max(min_quality, trial_quality - quality_step)
                out = save_to_buffer(ds_img, trial_quality)
                cur_size = out.getbuffer().nbytes
            # update quality for next loop
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


# ===================================================
# --- Bulk Optimization Helpers (No admin/plugin required)
# ===================================================
def is_image_url(url):
    try:
        return bool(re.search(r"\.(?:jpg|jpeg|png|webp)(?:\?|$)", url, flags=re.IGNORECASE))
    except Exception:
        return False


def get_filename_from_url(url):
    try:
        path = urlparse(url).path
        name = os.path.basename(path)
        return name or f"image_{int(time.time())}.jpg"
    except Exception:
        return f"image_{int(time.time())}.jpg"


def download_image(url, timeout=30):
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200 and r.content:
            return r.content
        else:
            st.warning(f"Failed to download image: {url} (status {r.status_code})")
            return None
    except Exception as e:
        st.error(f"Error downloading image {url}: {e}")
        return None


def wp_auth(site_config):
    return HTTPBasicAuth(site_config["username"], site_config["password"])


def wp_get(endpoint, site_config, params=None):
    try:
        url = construct_endpoint(site_config["url"], endpoint)
        resp = requests.get(url, params=params or {}, auth=wp_auth(site_config), timeout=30)
        return resp
    except Exception as e:
        st.error(f"GET {endpoint} failed: {e}")
        return None


def wp_post(endpoint, site_config, json_body=None):
    try:
        url = construct_endpoint(site_config["url"], endpoint)
        resp = requests.post(url, json=json_body or {}, auth=wp_auth(site_config), timeout=30)
        return resp
    except Exception as e:
        st.error(f"POST {endpoint} failed: {e}")
        return None


def replace_in_posts_pages_blocks(site_config, old_to_new_map, dry_run=True, max_pages=25):
    """
    Searches posts, pages, and reusable blocks for occurrences of old image URLs
    and replaces them with new URLs. Returns a log list of actions.
    """
    logs = []

    def process_collection(collection):
        page = 1
        while page <= max_pages:
            resp = wp_get(f"/wp-json/wp/v2/{collection}", site_config, params={"per_page": 50, "page": page, "orderby": "date", "order": "desc"})
            if not resp or resp.status_code not in (200, 201):
                break
            items = resp.json()
            if not items:
                break
            for item in items:
                item_id = item.get("id")
                content_obj = item.get("content") or {}
                original = content_obj.get("rendered", "")
                if not original:
                    continue
                updated = original
                replaced_any = False
                for old_url, new_url in old_to_new_map.items():
                    if old_url in updated:
                        updated = updated.replace(old_url, new_url)
                        replaced_any = True
                if replaced_any and updated != original:
                    if dry_run:
                        logs.append({
                            "type": collection,
                            "id": item_id,
                            "action": "would_update",
                            "replacements": [k for k, v in old_to_new_map.items() if k in original]
                        })
                    else:
                        upd = wp_post(f"/wp-json/wp/v2/{collection}/{item_id}", site_config, json_body={"content": updated})
                        if upd and upd.status_code in (200, 201):
                            logs.append({"type": collection, "id": item_id, "action": "updated"})
                        else:
                            logs.append({"type": collection, "id": item_id, "action": "failed", "status": getattr(upd, 'status_code', None), "resp": getattr(upd, 'text', None)})
            page += 1

    # Process posts, pages, and blocks (reusable blocks)
    for col in ("posts", "pages", "blocks"):
        process_collection(col)

    return logs


# ==========================================
# --- CSV-driven bulk (posts/pages only)
# ==========================================
def _load_csv_records_from_str(csv_text):
    """
    Load CSV text and return list of dict rows with required headers:
    Page_or_Post, Image_URL, Image_Size
    """
    try:
        reader = csv.DictReader(csv_text.splitlines())
        required = {"Page_or_Post", "Image_URL", "Image_Size"}
        if not required.issubset(set(reader.fieldnames or [])):
            missing = required - set(reader.fieldnames or [])
            raise ValueError(f"CSV missing required columns: {', '.join(sorted(missing))}")
        rows = []
        for r in reader:
            rows.append({
                "Page_or_Post": (r.get("Page_or_Post") or "").strip(),
                "Image_URL": (r.get("Image_URL") or "").strip(),
                "Image_Size": r.get("Image_Size")
            })
        return rows
    except Exception as e:
        st.error(f"Failed to parse CSV: {e}")
        return []


def bulk_optimize_from_csv(site_key, csv_rows, dry_run=True, row_limit=5, min_size_kb=150):
    """
    Build a targeted run from CSV rows.
    - Keep rows where Image_Size >= min_size_kb
    - Deduplicate images by URL path
    - Process first N unique images (row_limit)
    - Targeted pages only: use Page_or_Post URLs from those selected rows
    Reuses bulk_optimize_urls() underneath.
    """
    # Filter rows by size and basic URL checks
    filtered = []
    for r in csv_rows:
        try:
            size_val = r.get("Image_Size")
            size_int = int(str(size_val).strip()) if size_val is not None and str(size_val).strip().isdigit() else None
        except Exception:
            size_int = None
        img_url = r.get("Image_URL") or ""
        page_url = r.get("Page_or_Post") or ""
        if not (img_url.startswith("http") and page_url.startswith("http")):
            continue
        if size_int is not None and size_int < (min_size_kb * 1024):
            continue
        # Skip obvious archives/taxonomy for targeted list
        if not slug_from_url(page_url):
            continue
        if not is_image_url(img_url):
            continue
        filtered.append({"Image_URL": img_url, "Page_or_Post": page_url})

    # Deduplicate images by path, retain insertion order
    seen_paths = set()
    selected_images = []  # ordered unique images
    image_to_pages = {}
    for r in filtered:
        path = urlparse(r["Image_URL"]).path
        if path not in seen_paths:
            if len(selected_images) < row_limit:
                selected_images.append(r["Image_URL"])  # preserve order
                seen_paths.add(path)
        # Map pages only for images we decided to handle
        if r["Image_URL"] in selected_images:
            image_to_pages.setdefault(r["Image_URL"], set()).add(r["Page_or_Post"])

    # Build union of targeted pages for chosen images
    targeted_pages = []
    for img in selected_images:
        targeted_pages.extend(sorted(image_to_pages.get(img, [])))
    targeted_pages = sorted(set(targeted_pages))

    if not selected_images:
        st.info("CSV produced no candidates after filtering. Adjust size threshold or row limit.")
        return {}, []

    # Reuse the core pipeline
    return bulk_optimize_urls(
        site_key,
        input_urls=selected_images,
        dry_run=dry_run,
        targeted_page_urls=targeted_pages,
        only_targeted=True,
    )
def bulk_optimize_urls(site_key, input_urls, dry_run=True, targeted_page_urls=None, only_targeted=False):
    """
    - Deduplicate image URLs
    - Download, optimize, upload
    - Build old->new URL map
    - Replace references in posts/pages/blocks
    Returns (map, logs)
    """
    site_config = WP_SITES.get(site_key)
    if not site_config or not all(site_config.get(k) for k in ["url", "username", "password"]):
        st.error(f"Missing or incomplete configuration for {site_key}. Check .env file.")
        return {}, []

    # Separate image URLs vs page URLs; for pages we won't process directly, references are handled by replacements.
    image_urls = []
    for u in input_urls:
        if is_image_url(u):
            image_urls.append(u)

    # De-duplicate by final filename path
    dedup_map = {}
    for u in image_urls:
        dedup_map[urlparse(u).path] = u
    unique_image_urls = list(dedup_map.values())

    old_to_new = {}
    for url in unique_image_urls:
        st.write(f"Processing image: {url}")
        img_bytes = download_image(url)
        if not img_bytes:
            continue
        processed = process_image_for_wordpress(img_bytes)
        if not processed:
            continue
        if dry_run:
            # In dry-run, estimate filename but do not upload
            suggested_name = get_filename_from_url(url)
            st.info(f"Dry-run: would upload optimized image as {suggested_name}")
            # We cannot know new URL without upload; skip mapping creation in dry-run
            continue
        processed.seek(0)
        # Upload optimized image
        filename = get_filename_from_url(url)
        up = upload_image_to_wordpress(
            image_bytes=processed.getvalue(),
            wp_url=site_config["url"],
            username=site_config["username"],
            wp_app_password=site_config["password"],
            filename=filename,
            alt_text=os.path.splitext(filename)[0].replace('-', ' ').replace('_', ' ')
        )
        if up and up.get("source_url"):
            old_to_new[url] = up["source_url"]
        else:
            st.warning(f"Upload failed for {url}")

    logs = []
    if old_to_new:
        if only_targeted and targeted_page_urls:
            st.write("Replacing references in specified pages only...")
            logs = replace_in_specific_urls(site_config, targeted_page_urls, old_to_new, dry_run=dry_run)
        elif not dry_run:
            st.write("Replacing references across posts/pages/blocks...")
            logs = replace_in_posts_pages_blocks(site_config, old_to_new, dry_run=False)

    return old_to_new, logs


# ================================
# --- Targeted replacement helpers
# ================================
def normalize_wp_path(path):
    # strip trailing slashes and pagination
    p = re.sub(r"/page/\d+/?$", "/", path.rstrip('/'))
    if not p.endswith('/'):
        p += '/'
    return p


def is_archive_or_taxonomy_path(path):
    # Heuristics: category/tag/author/date archives
    return bool(re.search(r"/(category|tag|author|date|\d{4}/\d{2})/", path))


def slug_from_url(url):
    try:
        parsed = urlparse(url)
        path = normalize_wp_path(parsed.path)
        # Remove optional /th prefix
        parts = [p for p in path.split('/') if p]
        if not parts:
            return None
        if parts[0] == 'th' and len(parts) >= 2:
            parts = parts[1:]
        # skip obvious archives
        if is_archive_or_taxonomy_path('/' + '/'.join(parts) + '/'):
            return None
        # last part is slug for posts/pages like /th/my-post/
        return parts[-1]
    except Exception:
        return None


def fetch_item_by_slug(site_config, slug):
    # Try pages then posts by slug
    for collection in ("pages", "posts"):
        resp = wp_get(f"/wp-json/wp/v2/{collection}", site_config, params={"slug": slug})
        if resp and resp.status_code == 200:
            items = resp.json() or []
            if items:
                item = items[0]
                return collection, item
    return None, None


def replace_in_specific_urls(site_config, page_urls, old_to_new_map, dry_run=True):
    logs = []
    for url in page_urls:
        s = slug_from_url(url)
        if not s:
            logs.append({"url": url, "action": "skipped_non_editable_or_no_slug"})
            continue
        collection, item = fetch_item_by_slug(site_config, s)
        if not item:
            logs.append({"url": url, "slug": s, "action": "not_found"})
            continue
        item_id = item.get("id")
        original = (item.get("content") or {}).get("rendered", "")
        if not original:
            logs.append({"url": url, "slug": s, "type": collection, "id": item_id, "action": "no_rendered_content"})
            continue
        updated = original
        replaced_any = False
        for old_url, new_url in old_to_new_map.items():
            if old_url in updated:
                updated = updated.replace(old_url, new_url)
                replaced_any = True
        if not replaced_any:
            logs.append({"url": url, "slug": s, "type": collection, "id": item_id, "action": "no_matches"})
            continue
        if dry_run:
            logs.append({"url": url, "slug": s, "type": collection, "id": item_id, "action": "would_update"})
        else:
            upd = wp_post(f"/wp-json/wp/v2/{collection}/{item_id}", site_config, json_body={"content": updated})
            if upd and upd.status_code in (200, 201):
                logs.append({"url": url, "slug": s, "type": collection, "id": item_id, "action": "updated"})
            else:
                logs.append({"url": url, "slug": s, "type": collection, "id": item_id, "action": "failed", "status": getattr(upd, 'status_code', None)})
    return logs


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
                provider_options.append("Flux model (Free)")
            if openai_available:
                provider_options.append("OpenAI ($0.30 per image)")
            if openrouter_available:
                provider_options.append("Nano Banana ( $0.03 per image)")
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
                    # Width/height are advisory; model may choose its own.
                    image_bytes_from_api = generate_image_openrouter(final_prompt, width=1536, height=1024)
            else:
                st.error("Selected provider is not available. Check your API keys.")
                image_bytes_from_api = None

            if image_bytes_from_api:
                # Enforce 16:9 only for OpenRouter outputs
                force_169 = (
                    st.session_state.last_used_provider == "OpenRouter"
                    or provider == "OpenRouter (Gemini 2.5 Image)"
                )
                final_image_bytes_io = process_image_for_wordpress(
                    image_bytes_from_api,
                    force_landscape_16_9=force_169
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

# ----------------------------------------------
# --- Section: Bulk Optimize Existing Media URLs
# ----------------------------------------------
st.markdown("---")
st.header("Bulk Optimize Existing Media (No plugin)")
st.caption("Paste image and/or page URLs. Images will be optimized and uploaded; references in posts/pages/blocks will be updated to the new URLs. Use Dry run first.")

with st.form("bulk_optimize_form"):
    site_keys = list(WP_SITES.keys())
    default_site_index = site_keys.index("cryptonews") if "cryptonews" in site_keys else 0
    selected_site = st.selectbox("Target WordPress Site", site_keys, index=default_site_index)
    urls_input = st.text_area(
        "Image or mixed URLs (one per line)",
        height=150,
        placeholder="https://cimg.co/wp-content/uploads/.../big-image.jpg\nhttps://cimg.co/wp-content/uploads/.../another.jpg"
    )
    targeted_pages_input = st.text_area(
        "Targeted Page/Post URLs to update (optional, one per line)",
        height=120,
        placeholder="https://cryptonews.com/th/cryptocurrency/solaxy-launch-date/\nhttps://cryptonews.com/th/fantasy-games-2025/"
    )
    only_targeted = st.checkbox("Only update specified pages (skip site-wide scan)", value=True)
    dry_run_mode = st.checkbox("Dry run (analyze only, no uploads or edits)", value=True)
    run_bulk = st.form_submit_button("Run Bulk Optimize", use_container_width=True)

if run_bulk:
    raw_lines = [line.strip() for line in urls_input.splitlines() if line.strip()]
    # Also split lines by spaces or commas to be forgiving
    all_urls = []
    for line in raw_lines:
        for part in re.split(r"[\s,]+", line):
            if part and part.startswith("http"):
                all_urls.append(part)
    targeted_pages = [l.strip() for l in targeted_pages_input.splitlines() if l.strip().startswith("http")]
    if not all_urls:
        st.warning("Please provide at least one valid http(s) URL.")
    else:
        with st.spinner("Running bulk optimization..."):
            mapping, logs = bulk_optimize_urls(
                selected_site,
                all_urls,
                dry_run=dry_run_mode,
                targeted_page_urls=targeted_pages,
                only_targeted=only_targeted
            )
        if dry_run_mode:
            st.success("Dry-run complete. No uploads or content edits were made.")
            if mapping:
                st.write("(Dry-run) Old -> New mapping (if any)")
                st.json(mapping)
            if logs:
                st.write("Planned content updates:")
                st.json(logs)
            else:
                if only_targeted:
                    st.info("No matches found in targeted pages.")
        else:
            with st.spinner("Running CSV-driven optimization..."):
                mapping, logs = bulk_optimize_from_csv(
                    site_key=csv_site,
                    csv_rows=rows,
                    dry_run=csv_dry_run,
                    row_limit=int(row_limit),
                    min_size_kb=int(min_kb),
                )
            if csv_dry_run:
                st.success("Dry-run complete. No uploads or content edits were made.")
                if mapping:
                    st.write("(Dry-run) Old -> New mapping (if any)")
                    st.json(mapping)
                if logs:
                    st.write("Planned content updates:")
                    st.json(logs)
            else:
                st.success("CSV-driven optimization complete.")
                if mapping:
                    st.write("Old -> New image URL mapping:")
                    st.json(mapping)
                if logs:
                    st.write("Content update logs:")
                    st.json(logs)
