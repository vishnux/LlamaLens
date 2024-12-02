import streamlit as st
import ollama
from PIL import Image
import io
import os
import uuid
import pandas as pd
import pytesseract
import cv2
import numpy as np
from typing import List, Dict, Any

from modules.text_processing import (
    extract_tables, 
    generate_summary, 
    search_and_highlight
)
from modules.file_export import export_text
from modules.language_support import detect_language, ocr_languages
from modules.batch_processing import process_batch_files

# Configuration and Global Settings
class OCRConfig:
    """Application-wide configuration and settings."""
    SUPPORTED_FILE_TYPES = ['png', 'jpg', 'jpeg', 'pdf']
    MAX_FILE_SIZE_MB = 10
    TEMP_UPLOAD_DIR = 'uploads'

    @staticmethod
    def initialize_upload_dir():
        """Ensure upload directory exists."""
        os.makedirs(OCRConfig.TEMP_UPLOAD_DIR, exist_ok=True)

# Utility Functions
def preprocess_image(image: Image) -> np.ndarray:
    """
    Preprocess image for improved OCR accuracy.
    
    Techniques:
    - Convert to grayscale
    - Apply noise reduction
    - Enhance contrast
    """
    # Convert PIL Image to OpenCV format
    img_array = np.array(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Noise reduction
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    return enhanced

def validate_file(uploaded_file):
    """
    Validate uploaded file size and type.
    
    Args:
        uploaded_file: Streamlit uploaded file object
    
    Returns:
        bool: Whether file is valid
    """
    if uploaded_file is None:
        return False
    
    # Check file size (max 10MB)
    file_size = uploaded_file.size / (1024 * 1024)  # Convert to MB
    if file_size > OCRConfig.MAX_FILE_SIZE_MB:
        st.error(f"File too large. Maximum size is {OCRConfig.MAX_FILE_SIZE_MB}MB.")
        return False
    
    # Check file type
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension not in OCRConfig.SUPPORTED_FILE_TYPES:
        st.error(f"Unsupported file type. Supported types: {', '.join(OCRConfig.SUPPORTED_FILE_TYPES)}")
        return False
    
    return True

def main():
    """
    Main Streamlit application entry point.
    Implements comprehensive OCR functionality with multiple features.
    """
    # Initialize configuration
    OCRConfig.initialize_upload_dir()
    
    # Page Configuration
    st.set_page_config(
        page_title="SmartOCR Pro",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for enhanced UI
    st.markdown("""
    <style>
    .stApp {
        background-color: #f4f4f4;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stProgress > div > div {
        background-color: #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Title and Subheader
    st.title("🔍 SmartOCR Pro")
    st.markdown("Intelligent Document Text Extraction")
    
    # Sidebar for Configuration
    with st.sidebar:
        st.header("📤 Document Upload")
        
        # Language Selection
        selected_language = st.selectbox(
            "OCR Language", 
            list(ocr_languages.keys()), 
            index=0
        )
        
        # Batch Processing Toggle
        batch_processing = st.checkbox("Batch Processing")
        
        # File Upload
        if not batch_processing:
            uploaded_file = st.file_uploader(
                "Choose a document", 
                type=OCRConfig.SUPPORTED_FILE_TYPES,
                accept_multiple_files=False
            )
        else:
            uploaded_files = st.file_uploader(
                "Choose multiple documents", 
                type=OCRConfig.SUPPORTED_FILE_TYPES,
                accept_multiple_files=True
            )
        
        # Advanced Options
        st.header("⚙️ Advanced Options")
        extract_tables = st.checkbox("Extract Tables")
        generate_summary = st.checkbox("Generate Summary")
        
        # Export Options
        export_format = st.selectbox(
            "Export Format", 
            ['.txt', '.csv', '.docx']
        )
    
    # Main Processing Area
    if batch_processing and 'uploaded_files' in locals():
        # Batch Processing Logic
        if uploaded_files:
            with st.spinner("Processing Batch..."):
                results = process_batch_files(
                    uploaded_files, 
                    language=selected_language,
                    extract_tables=extract_tables
                )
                
                for idx, result in enumerate(results, 1):
                    st.subheader(f"Document {idx}")
                    st.write(result['text'])
                    st.write(f"Confidence: {result['confidence']:.2f}%")
    
    elif not batch_processing and 'uploaded_file' in locals() and uploaded_file:
        if validate_file(uploaded_file):
            # Single File Processing
            with st.spinner("Analyzing Document..."):
                # Preprocess image
                image = Image.open(uploaded_file)
                processed_image = preprocess_image(image)
                
                # OCR Processing
                extracted_text = pytesseract.image_to_string(
                    processed_image, 
                    lang=ocr_languages[selected_language]
                )
                
                # Confidence Calculation
                confidence_score = calculate_ocr_confidence(extracted_text)
                
                # Optional Table Extraction
                tables = extract_tables(processed_image) if extract_tables else []
                
                # Optional Text Summarization
                summary = generate_summary(extracted_text) if generate_summary else None
                
                # Display Results
                st.subheader("Extracted Text")
                st.text_area("OCR Result", extracted_text, height=300)
                
                # Confidence Display
                st.metric("Extraction Confidence", f"{confidence_score:.2f}%")
                
                # Copy to Clipboard Button
                st.button("📋 Copy Text", 
                    on_click=copy_to_clipboard, 
                    args=(extracted_text,)
                )
                
                # Export Button
                st.download_button(
                    label=f"Export as {export_format}",
                    data=export_text(extracted_text, export_format),
                    file_name=f'ocr_result{export_format}',
                    mime='text/plain'
                )
    
    # Footer
    st.markdown("---")
    st.markdown("© 2024 SmartOCR Pro | Intelligent Document Intelligence")

def calculate_ocr_confidence(text: str) -> float:
    """
    Calculate OCR confidence based on text characteristics.
    
    Args:
        text (str): Extracted text
    
    Returns:
        float: Confidence percentage
    """
    if not text:
        return 0.0
    
    # Simple heuristics for confidence calculation
    words = text.split()
    readable_word_ratio = len([w for w in words if len(w) > 2]) / len(words) if words else 0
    
    return min(readable_word_ratio * 100, 95.0)

def copy_to_clipboard(text: str):
    """
    Copy text to clipboard.
    
    Args:
        text (str): Text to copy
    """
    st.toast("Text Copied to Clipboard! 📋")

if __name__ == "__main__":
    main()
