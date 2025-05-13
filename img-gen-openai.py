import streamlit as st
import os
import requests

st.set_page_config(page_title="OpenAI Image Generator", layout="wide")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_IMAGE_URL = "https://api.openai.com/v1/images/generations"

if not OPENAI_API_KEY:
    st.error("OpenAI API key not found. Please set OPENAI_API_KEY in your environment variables.")
else:
    with st.form("openai_image_form"):
        prompt = st.text_area("Enter your image prompt (English only):", height=80)
        submitted = st.form_submit_button("Generate Image")
        if submitted and prompt.strip():
            with st.spinner("Generating image with OpenAI..."):
                try:
                    headers = {
                        "Authorization": f"Bearer {OPENAI_API_KEY}",
                        "Content-Type": "application/json"
                    }
                    payload = {
                        "model": "dall-e-3",
                        "prompt": prompt,
                        "n": 1,
                        "size": "1792x1024"
                    }
                    response = requests.post(OPENAI_IMAGE_URL, headers=headers, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    image_url = data["data"][0]["url"]
                    # Download the image and convert to JPEG with optimized quality
                    from io import BytesIO
                    from PIL import Image
                    import requests as rq

                    img_response = rq.get(image_url)
                    img_response.raise_for_status()
                    img_bytes = BytesIO(img_response.content)
                    img = Image.open(img_bytes)
                    # Convert to JPEG and resize (if needed, but DALL-E should return 1200x800)
                    # Google Discover: At least 1200px wide, keep aspect ratio
                    img = img.convert("RGB")
                    width = 1200
                    aspect = img.height / img.width
                    height = int(width * aspect)
                    img = img.resize((width, height))
                    jpeg_bytes = BytesIO()
                    img.save(jpeg_bytes, format="JPEG", quality=80, optimize=True)
                    jpeg_bytes.seek(0)
                    st.image(jpeg_bytes, caption=prompt[:50] + "...")
                except Exception as e:
                    st.error(f"Failed to generate image: {e}")
