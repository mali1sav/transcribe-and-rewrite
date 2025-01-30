import streamlit as st
from docx import Document
import os
import tempfile
from PIL import Image
import requests
import json

# Configuration
IMAGE_SIZE = (1200, 800)
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]

def process_docx(file_path):
    # Extract content and images from DOCX
    doc = Document(file_path)
    content = []
    images = []
    
    for para in doc.paragraphs:
        content.append(para.text)
    
    for rel in doc.part.rels.values():
        if "image" in rel.target_ref:
            img_path = os.path.basename(rel.target_ref)
            images.append((img_path, rel.target_part.blob))
    
    return "\n".join(content), images

def resize_image(image_data, size):
    img = Image.open(io.BytesIO(image_data))
    img = img.convert("RGB")
    img.thumbnail(size)
    output = io.BytesIO()
    img.save(output, format="JPEG", quality=85)
    return output.getvalue()

def get_llm_suggestion(text):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "deepseek/deepseek-r1",
        "messages": [{
            "role": "user",
            "content": f"""Analyze this content and suggest:
            1. Appropriate HTML tags (H1, H2, paragraph, list)
            2. Image filename based on content
            3. URL slug
            Content: {text[:2000]}"""
        }]
    }
    
    response = requests.post("https://openrouter.ai/api/v1/chat/completions",
                             headers=headers,
                             data=json.dumps(payload))
    
    return response.json()["choices"][0]["message"]["content"]

def main():
    st.title("DOCX to WordPress Converter")
    
    uploaded_file = st.file_uploader("Upload DOCX file", type=["docx"])
    
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        content, images = process_docx(tmp_path)
        
        # Process images
        image_data = []
        for idx, (img_name, img_blob) in enumerate(images):
            resized_img = resize_image(img_blob, IMAGE_SIZE)
            img_filename = f"image_{idx+1}.jpg"
            image_data.append((img_filename, resized_img))
        
        # Get LLM suggestions
        llm_response = get_llm_suggestion(content)
        
        # Generate HTML
        html_output = f"<!-- LLM Suggestions:\n{llm_response}\n-->\n\n"
        html_output += "<div class='wp-content'>\n"
        
        # Basic conversion (enhance this with proper parsing)
        for line in content.split("\n"):
            if line.strip():
                html_output += f"<p>{line}</p>\n"
        
        html_output += "</div>"
        
        # Display results
        st.subheader("Processed HTML")
        st.code(html_output)
        
        st.subheader("Processed Images")
        for img_name, img_data in image_data:
            st.image(img_data, caption=img_name)
            st.download_button(f"Download {img_name}", img_data, file_name=img_name)

if __name__ == "__main__":
    main()