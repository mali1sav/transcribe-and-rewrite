import streamlit as st
import os
from dotenv import load_dotenv
from together import Together
import google.generativeai as genai

load_dotenv()  # Load environment variables from .env

st.set_page_config(page_title="Together AI Image Generator", layout="wide")

def init_gemini_client():
    """Initialize Google Gemini client."""
    try:
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            st.error("Gemini API key not found. Please set GEMINI_API_KEY in your environment variables.")
            return None
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        return {
            'model': model,
            'name': 'gemini-2.0-flash'
        }
    except Exception as e:
        st.error(f"Failed to initialize Gemini client: {str(e)}")
        return None

def make_gemini_request(client, prompt):
    """Translate Thai prompts to clean English for image generation"""
    try:
        # Explicit translation instruction with output format constraint
        translation_prompt = f"If the prompt is not in English, translate the prompt text to English for an image generation prompt. Adhere to image prompt best practices. Return only the translation without any additional text or explanation: {prompt}"
        response = client['model'].generate_content(translation_prompt)
        if response and response.text:
            return response.text
    except Exception as e:
        st.error(f"Error making Gemini request: {str(e)}")
        return None
# A function to show translated prompt in English and start generating image

def generate_image(prompt):
    try:
        client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
        
        with st.spinner("Generating image..."):
            response = client.images.generate(
                prompt=prompt,
                model="black-forest-labs/FLUX.1-schnell-free",
                width=1200,
                height=800, 
                steps=4,
                n=1,
                response_format="b64_json"
            )
            
            image_b64 = response.data[0].b64_json
            st.image(f"data:image/png;base64,{image_b64}", 
                    caption=prompt[:50] + "...")
            
    except Exception as e:
        st.error(f"Error generating image: {str(e)}")

with st.form("image_form"):
    prompt = st.text_area("Enter your image prompt in any language:", height=80)
    
    if st.form_submit_button("Generate Image"):
        gemini_client = init_gemini_client()
        if gemini_client:
            translated_prompt = make_gemini_request(gemini_client, prompt)
            if translated_prompt:
                # Clean any residual formatting from Gemini response
                clean_prompt = translated_prompt.strip('"').split("\n")[0].strip()
                st.subheader("Translated Prompt in English:")
                st.code(clean_prompt, language="text")
                generate_image(clean_prompt)
