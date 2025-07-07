import streamlit as st
import os
from dotenv import load_dotenv
import base64
import requests
from io import BytesIO
from PIL import Image, ImageOps
from openai import OpenAI
from requests.auth import HTTPBasicAuth

# ─────────────────────────  Init  ─────────────────────────
load_dotenv()
st.set_page_config(page_title="OpenAI Image Generator & Uploader", layout="wide")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY missing in .env")
    st.stop()

client = OpenAI()

# ──────────────────────  Session defaults  ──────────────────────
for key, default in (("active_image_bytes_io", None),
                     ("active_image_alt_text", "")):
    st.session_state.setdefault(key, default)

# ─────────────────────────  WP helpers  ─────────────────────────
def construct_endpoint(wp_url: str, path: str) -> str:
    wp_url = wp_url.rstrip("/")
    if not any(d in wp_url for d in ("bitcoinist.com", "newsbtc.com")) and "/th" not in wp_url:
        wp_url += "/th"
    return f"{wp_url}{path}"

def upload_image_to_wordpress(img_bytes: bytes, wp_cfg: dict,
                              filename: str, alt_text: str):
    media_ep = construct_endpoint(wp_cfg["url"], "/wp-json/wp/v2/media")
    files = {"file": (filename, img_bytes, "image/jpeg")}
    data = {"alt_text": alt_text, "title": alt_text, "caption": alt_text}
    try:
        r = requests.post(media_ep, files=files, data=data,
                          auth=HTTPBasicAuth(wp_cfg["username"], wp_cfg["password"]))
        if r.status_code not in (200, 201):
            st.error(f"{wp_cfg['url']} upload failed ({r.status_code}): {r.text}")
        else:
            st.success(f"✅ Uploaded to {wp_cfg['url']} (ID {r.json().get('id')})")
    except Exception as e:
        st.error(f"{wp_cfg['url']} upload error: {e}")

# ─────────────────────  Image post-processing  ─────────────────────
def to_wp_size(raw_bytes: bytes, size=(1200, 800), quality=80) -> BytesIO | None:
    try:
        img = Image.open(BytesIO(raw_bytes)).convert("RGB")
        fitted = ImageOps.fit(img, size, Image.Resampling.LANCZOS, centering=(.5, .5))
        buf = BytesIO()
        fitted.save(buf, format="JPEG", quality=quality, optimize=True)
        buf.seek(0)
        return buf
    except Exception as e:
        st.error(f"Processing failed: {e}")
        return None

# ─────────────────────  WP site config  ─────────────────────
WP_SITES = {
    "cryptonews": {"url": os.getenv("CRYPTONEWS_WP_URL"),
                   "username": os.getenv("CRYPTONEWS_WP_USERNAME"),
                   "password": os.getenv("CRYPTONEWS_WP_APP_PASSWORD")},
    "cryptodnes": {"url": os.getenv("CRYPTODNES_WP_URL"),
                   "username": os.getenv("CRYPTODNES_WP_USERNAME"),
                   "password": os.getenv("CRYPTODNES_WP_APP_PASSWORD")},
    "icobench":   {"url": os.getenv("ICOBENCH_WP_URL"),
                   "username": os.getenv("ICOBENCH_WP_USERNAME"),
                   "password": os.getenv("ICOBENCH_WP_APP_PASSWORD")},
    "bitcoinist": {"url": os.getenv("BITCOINIST_WP_URL"),
                   "username": os.getenv("BITCOINIST_WP_USERNAME"),
                   "password": os.getenv("BITCOINIST_WP_APP_PASSWORD")},
}

st.title("OpenAI Image Generator & WordPress Uploader")

# ─────────────────────  Image generation form  ─────────────────────
with st.form("gen_form"):
    g_prompt = st.text_area("Prompt (English only):", height=100)
    g_alt    = st.text_area("Alt text for WordPress:", height=100)
    gen_btn  = st.form_submit_button("Generate Image")

if gen_btn:
    if not g_prompt.strip() or not g_alt.strip():
        st.warning("Prompt AND alt-text required.")
    else:
        try:
            with st.spinner("Generating image…"):
                res  = client.images.generate(model="gpt-image-1",
                                              prompt=g_prompt,
                                              n=1,
                                              size="1536x1024")   # 3:2 ratio → crops to 1200×800
                datum = res.data[0]
                raw   = base64.b64decode(datum.b64_json) if getattr(datum, "b64_json", None) \
                        else requests.get(datum.url).content
                img_io = to_wp_size(raw)
                if not img_io:
                    st.error("Post-processing failed.")
                else:
                    st.session_state.active_image_bytes_io = img_io
                    st.session_state.active_image_alt_text = g_alt
                    st.success("Image ready ↓")
        except Exception as e:
            st.error(f"Generation error: {e}")

# ─────────────────────  User upload form  ─────────────────────
st.markdown("---")
with st.form("upload_form"):
    up_file = st.file_uploader("Upload image", type=("png", "jpg", "jpeg", "webp"))
    up_alt  = st.text_area("Alt text for WordPress:", height=100)
    up_btn  = st.form_submit_button("Process Upload")

if up_btn:
    if not up_file or not up_alt.strip():
        st.warning("File AND alt-text required.")
    else:
        img_io = to_wp_size(up_file.getvalue())
        if img_io:
            st.session_state.active_image_bytes_io = img_io
            st.session_state.active_image_alt_text = up_alt
            st.success("Upload processed ↓")
        else:
            st.error("Processing failed.")

# ─────────────────────  Preview + WP upload  ─────────────────────
img_io = st.session_state.active_image_bytes_io
alt    = st.session_state.active_image_alt_text

if img_io and alt:
    st.markdown("---")
    st.subheader("Preview & upload")
    img_io.seek(0)
    st.image(img_io, caption=alt[:150] + ("…" if len(alt) > 150 else ""))

    # filename from alt text
    base = "".join(c if c.isalnum() or c in (" ", "_") else "" for c in alt[:30]).strip().replace(" ", "_") or "img"
    wp_filename = f"{base}_processed.jpg"

    cols = st.columns(len(WP_SITES))
    for i, (site, cfg) in enumerate(WP_SITES.items()):
        with cols[i]:
            if st.button(f"Upload → {site.title()}", key=f"btn_{site}", use_container_width=True):
                if not all(cfg.values()):
                    st.error(f"{site}: missing .env creds"); continue
                img_io.seek(0)
                upload_image_to_wordpress(img_io.getvalue(), cfg, wp_filename, alt)
else:
    st.info("Generate or upload an image first.")
