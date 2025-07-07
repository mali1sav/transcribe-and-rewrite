import streamlit as st
import os
from dotenv import load_dotenv
import base64
from io import BytesIO
from PIL import Image, ImageOps
from openai import OpenAI
import requests
from requests.auth import HTTPBasicAuth

# --------------------------------------------------------------------
#  Initial setup
# --------------------------------------------------------------------
load_dotenv()
st.set_page_config(page_title="OpenAI Image Generator & Uploader", layout="wide")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --------------------------------------------------------------------
#  WordPress helpers
# --------------------------------------------------------------------
def construct_endpoint(wp_url: str, endpoint_path: str) -> str:
    wp_url = wp_url.rstrip("/")
    if not any(d in wp_url for d in ("bitcoinist.com", "newsbtc.com")) and "/th" not in wp_url:
        wp_url += "/th"
    return f"{wp_url}{endpoint_path}"

def upload_image_to_wordpress(
    image_bytes: bytes,
    wp_url: str,
    username: str,
    wp_app_password: str,
    filename: str,
    alt_text: str
):
    media_endpoint = construct_endpoint(wp_url, "/wp-json/wp/v2/media")
    files = {"file": (filename, image_bytes, "image/jpeg")}
    data_payload = {"alt_text": alt_text, "title": alt_text, "caption": alt_text}

    try:
        r = requests.post(media_endpoint, files=files, data=data_payload,
                          auth=HTTPBasicAuth(username, wp_app_password))
        if r.status_code not in (200, 201):
            st.error(f"Upload failed ({r.status_code}): {r.text}")
            return None

        media = r.json()
        media_id = media.get("id")
        st.success(f"Image uploaded to {wp_url} (ID {media_id})")
        return {"media_id": media_id, "source_url": media.get("source_url", "")}
    except Exception as e:
        st.error(f"Upload error: {e}")
        return None

# --------------------------------------------------------------------
#  Image post-processing (1200 × 800 exact)
# --------------------------------------------------------------------
def process_image_for_wordpress(
    image_bytes: bytes,
    target_size=(1200, 800),
    quality=80
) -> BytesIO | None:
    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        fitted = ImageOps.fit(img, target_size, Image.Resampling.LANCZOS, centering=(0.5, 0.5))
        buf = BytesIO()
        fitted.save(buf, format="JPEG", quality=quality, optimize=True)
        buf.seek(0)
        return buf
    except Exception as e:
        st.error(f"Image processing failed: {e}")
        return None

# --------------------------------------------------------------------
#  Streamlit session state defaults
# --------------------------------------------------------------------
for key, default in (
    ("active_image_bytes_io", None),
    ("active_image_alt_text", ""),
    ("user_uploaded_raw_bytes", None),
    ("user_uploaded_alt_text_input", "")
):
    if key not in st.session_state:
        st.session_state[key] = default

# --------------------------------------------------------------------
#  WordPress site configs
# --------------------------------------------------------------------
WP_SITES = {
    "cryptonews":  {"url": os.getenv("CRYPTONEWS_WP_URL"),  "username": os.getenv("CRYPTONEWS_WP_USERNAME"),  "password": os.getenv("CRYPTONEWS_WP_APP_PASSWORD")},
    "cryptodnes":  {"url": os.getenv("CRYPTODNES_WP_URL"),  "username": os.getenv("CRYPTODNES_WP_USERNAME"),  "password": os.getenv("CRYPTODNES_WP_APP_PASSWORD")},
    "icobench":    {"url": os.getenv("ICOBENCH_WP_URL"),    "username": os.getenv("ICOBENCH_WP_USERNAME"),    "password": os.getenv("ICOBENCH_WP_APP_PASSWORD")},
    "bitcoinist":  {"url": os.getenv("BITCOINIST_WP_URL"),  "username": os.getenv("BITCOINIST_WP_USERNAME"),  "password": os.getenv("BITCOINIST_WP_APP_PASSWORD")},
}

# --------------------------------------------------------------------
#  Page title
# --------------------------------------------------------------------
st.title("OpenAI Image Generator & WordPress Uploader")

# --------------------------------------------------------------------
#  Guard for missing API key
# --------------------------------------------------------------------
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found in .env")
    st.stop()

client = OpenAI()

# --------------------------------------------------------------------
#  Image generation form
# --------------------------------------------------------------------
with st.form("gen_form"):
    prompt = st.text_area("Image prompt (English only):", height=80)
    alt_text_gen = st.text_area("Alt text for this generated image:", height=60)
    if st.form_submit_button("Generate Image"):
        if not prompt.strip() or not alt_text_gen.strip():
            st.warning("Both prompt and alt-text are required.")
        else:
            st.session_state.active_image_bytes_io = None
            with st.spinner("Generating image…"):
                try:
                    res = client.images.generate(
                        model="gpt-image-1",
                        prompt=prompt,
                        size="1536x1024",           # 3:2 – crops cleanly to 1200×800
                        output_format="jpeg"        # API returns base-64 JPEG
                    )
                    img_b64 = res.data[0].b64_json
                    img_bytes = base64.b64decode(img_b64)
                    img_io = process_image_for_wordpress(img_bytes)
                    if not img_io:
                        raise RuntimeError("Post-processing failed")

                    st.session_state.active_image_bytes_io = img_io
                    st.session_state.active_image_alt_text = alt_text_gen
                    st.success("Image ready → see preview below.")
                except Exception as e:
                    st.error(f"Generation failed: {e}")

# --------------------------------------------------------------------
#  User upload form
# --------------------------------------------------------------------
st.markdown("---")
st.subheader("Or upload your own image")
with st.form("upload_form"):
    up_file = st.file_uploader("Choose image", type=("png", "jpg", "jpeg", "webp"))
    alt_text_up = st.text_area("Alt text for uploaded image:", height=60)
    if st.form_submit_button("Process Upload"):
        if not up_file or not alt_text_up.strip():
            st.warning("File and alt-text are required.")
        else:
            img_io = process_image_for_wordpress(up_file.getvalue())
            if img_io:
                st.session_state.active_image_bytes_io = img_io
                st.session_state.active_image_alt_text = alt_text_up
                st.success("Upload processed → see preview below.")
            else:
                st.error("Processing failed.")

# --------------------------------------------------------------------
#  Preview & upload controls
# --------------------------------------------------------------------
img_io = st.session_state.active_image_bytes_io
alt_text = st.session_state.active_image_alt_text

if img_io and alt_text:
    st.markdown("---")
    st.subheader("Preview & WordPress upload")
    img_io.seek(0)
    st.image(img_io, caption=alt_text[:150] + ("…" if len(alt_text) > 150 else ""))

    fname_root = "".join(c if c.isalnum() or c in (" ", "_") else ""
                         for c in alt_text[:30]).rstrip().replace(" ", "_") or "img"
    wp_filename = f"{fname_root}_processed.jpg"

    cols = st.columns(4)
    for ix, site_key in enumerate(WP_SITES):
        cfg = WP_SITES[site_key]
        with cols[ix]:
            if st.button(f"Upload to {site_key.title()}", use_container_width=True):
                if not all(cfg.values()):
                    st.error(f"{site_key}: credentials missing in .env")
                    continue
                img_io.seek(0)
                upload_image_to_wordpress(
                    image_bytes=img_io.getvalue(),
                    wp_url=cfg["url"],
                    username=cfg["username"],
                    wp_app_password=cfg["password"],
                    filename=wp_filename,
                    alt_text=alt_text
                )
else:
    st.info("Generate or upload an image to continue.")
