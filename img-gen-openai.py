import streamlit as st
import os
from dotenv import load_dotenv
import base64
from io import BytesIO
from PIL import Image, ImageOps
from openai import OpenAI
import requests
from requests.auth import HTTPBasicAuth

# Load environment variables
load_dotenv()

st.set_page_config(page_title="OpenAI Image Generator & Uploader", layout="wide")

# ---------- WordPress helpers ----------
def construct_endpoint(wp_url: str, endpoint_path: str) -> str:
    wp_url = wp_url.rstrip('/')
    if not any(d in wp_url for d in ("bitcoinist.com", "newsbtc.com")) and "/th" not in wp_url:
        wp_url += "/th"
    return f"{wp_url}{endpoint_path}"

def upload_image_to_wordpress(
    image_bytes: bytes,
    wp_url: str,
    username: str,
    wp_app_password: str,
    filename: str = "generated_image.jpg",
    alt_text: str = "Generated Image"
):
    media_endpoint = construct_endpoint(wp_url, "/wp-json/wp/v2/media")
    try:
        files = {"file": (filename, image_bytes, "image/jpeg")}
        data_payload = {"alt_text": alt_text, "title": alt_text, "caption": alt_text}
        response = requests.post(media_endpoint, files=files, data=data_payload,
                                 auth=HTTPBasicAuth(username, wp_app_password))

        if response.status_code in (200, 201):
            media_data = response.json()
            media_id = media_data.get("id")
            update_payload = {}
            if media_data.get("alt_text") != alt_text:
                update_payload["alt_text"] = alt_text
            if media_data.get("title", {}).get("raw") != alt_text:
                update_payload["title"] = alt_text
            if media_data.get("caption", {}).get("raw") != alt_text:
                update_payload["caption"] = alt_text

            if update_payload:
                update_endpoint = f"{media_endpoint}/{media_id}"
                r2 = requests.post(update_endpoint, json=update_payload,
                                   auth=HTTPBasicAuth(username, wp_app_password))
                if r2.status_code not in (200, 201):
                    st.warning(f"Meta-update failed ({r2.status_code})")

            st.success(f"Image uploaded to {wp_url} (ID {media_id})")
            return {"media_id": media_id, "source_url": media_data.get("source_url", "")}
        else:
            st.error(f"Upload failed ({response.status_code}): {response.text}")
            return None
    except Exception as e:
        st.error(f"Upload error: {e}")
        return None
# ---------------------------------------

# Streamlit state
for key in (
    "active_image_bytes_io", "active_image_alt_text", "user_uploaded_raw_bytes",
    "user_uploaded_alt_text_input_field_value"
):
    st.session_state.setdefault(key, None if "bytes" in key else "")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WP_SITES = {
    "cryptonews": {"url": os.getenv("CRYPTONEWS_WP_URL"),
                   "username": os.getenv("CRYPTONEWS_WP_USERNAME"),
                   "password": os.getenv("CRYPTONEWS_WP_APP_PASSWORD")},
    "cryptodnes": {"url": os.getenv("CRYPTODNES_WP_URL"),
                   "username": os.getenv("CRYPTODNES_WP_USERNAME"),
                   "password": os.getenv("CRYPTODNES_WP_APP_PASSWORD")},
    "icobench": {"url": os.getenv("ICOBENCH_WP_URL"),
                 "username": os.getenv("ICOBENCH_WP_USERNAME"),
                 "password": os.getenv("ICOBENCH_WP_APP_PASSWORD")},
    "bitcoinist": {"url": os.getenv("BITCOINIST_WP_URL"),
                   "username": os.getenv("BITCOINIST_WP_USERNAME"),
                   "password": os.getenv("BITCOINIST_WP_APP_PASSWORD")}
}

def process_image_for_wordpress(image_bytes: bytes,
                                target_size=(1200, 800),
                                final_quality=80) -> BytesIO | None:
    """Centre-crop/resize to exactly 1200×800 JPEG."""
    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        img_fitted = ImageOps.fit(img, target_size,
                                  Image.Resampling.LANCZOS, centering=(0.5, 0.5))
        buf = BytesIO()
        img_fitted.save(buf, format="JPEG", quality=final_quality, optimize=True)
        buf.seek(0)
        return buf
    except Exception as e:
        st.error(f"Image processing failed: {e}")
        return None

st.title("OpenAI Image Generator & WordPress Uploader")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY missing in .env")
    st.stop()

client = OpenAI()

# ---------- OpenAI generation ----------
with st.form("openai_image_form"):
    st.subheader("Generate Image with OpenAI")
    prompt_text = st.text_area("Prompt (English only):", height=80, key="prompt_input")
    alt_text_gen = st.text_area("Alt text for generated image:", height=80,
                                key="generated_alt_text_input_val")
    if st.form_submit_button("Generate Image"):
        if prompt_text.strip() and alt_text_gen.strip():
            st.session_state.active_image_bytes_io = None
            with st.spinner("Generating via GPT-Image-1…"):
                try:
                    result = client.images.generate(
                        model="gpt-image-1",
                        prompt=prompt_text,
                        size="1536x1024",
                        output_format="jpeg"  # defaults to jpeg anyway
                    )
                    b64_data = result.data[0].b64_json
                    image_bytes = base64.b64decode(b64_data)
                    img_io = process_image_for_wordpress(image_bytes)
                    if img_io:
                        st.session_state.active_image_bytes_io = img_io
                        st.session_state.active_image_alt_text = alt_text_gen
                        st.success("Image ready (1200×800).")
                    else:
                        st.error("Post-process failed")
                except Exception as e:
                    st.error(f"Generation failed: {e}")
        else:
            st.warning("Prompt and alt-text are both required.")

# ---------- User upload ----------
st.markdown("---")
st.subheader("Or Upload Your Own Image")
with st.form("user_upload_form"):
    file_up = st.file_uploader("Choose image", type=("png", "jpg", "jpeg", "webp"))
    alt_text_up = st.text_area("Alt text:", height=80, key="user_alt_text_input_val")
    if st.form_submit_button("Process Uploaded Image"):
        if file_up and alt_text_up.strip():
            img_io = process_image_for_wordpress(file_up.getvalue())
            if img_io:
                st.session_state.active_image_bytes_io = img_io
                st.session_state.active_image_alt_text = alt_text_up
                st.success("Upload processed (1200×800).")
            else:
                st.error("Processing failed")
        else:
            st.warning("File and alt-text required.")

# ---------- Active image preview & WP upload ----------
if st.session_state.active_image_bytes_io and st.session_state.active_image_alt_text:
    st.markdown("---")
    st.subheader("Active Image Preview / Upload")
    st.image(st.session_state.active_image_bytes_io,
             caption=st.session_state.active_image_alt_text[:150] + "…"
             if len(st.session_state.active_image_alt_text) > 150
             else st.session_state.active_image_alt_text)

    alt_text = st.session_state.active_image_alt_text
    img_io = st.session_state.active_image_bytes_io
    filename_base = "".join(
        c if c.isalnum() or c in (" ", "_") else "" for c in alt_text[:30]
    ).strip().replace(" ", "_") or "image"
    upload_filename = f"{filename_base}_processed.jpg"

    col_map = {"cryptonews": 0, "cryptodnes": 1, "icobench": 2, "bitcoinist": 3}
    cols = st.columns(4)
    for site, cfg in WP_SITES.items():
        with cols[col_map[site]]:
            if st.button(f"Upload to {site.title()}", key=f"btn_{site}",
                         use_container_width=True):
                if all(cfg.values()):
                    with st.spinner(f"Uploading to {site}…"):
                        img_io.seek(0)
                        upload_image_to_wordpress(
                            image_bytes=img_io.getvalue(),
                            wp_url=cfg["url"],
                            username=cfg["username"],
                            wp_app_password=cfg["password"],
                            filename=upload_filename,
                            alt_text=alt_text
                        )
                else:
                    st.error(f"{site}: missing creds in .env")
else:
    st.info("Generate or upload an image first.")
