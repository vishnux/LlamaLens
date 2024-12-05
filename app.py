import streamlit as st
from llama_ocr import LlamaOCR
import fitz  # PyMuPDF for PDF handling
from PIL import Image
import io

# Initialize Llama OCR
ocr_engine = LlamaOCR()

def process_image(file):
    """Process an uploaded image file and extract text."""
    image = Image.open(file)
    text = ocr_engine.extract_text(image)
    return text

def process_pdf(file):
    """Process an uploaded PDF file and extract text from all pages."""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def main():
    st.title("OCR Text Extraction App")
    st.write("Upload an image or PDF file to extract text.")

    # File upload widget
    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "png", "pdf"])

    if uploaded_file is not None:
        file_type = uploaded_file.type
        st.write(f"Uploaded file: {uploaded_file.name} ({file_type})")

        if "image" in file_type:
            processing_function = process_image
            file_format = "image"
        elif "application/pdf" in file_type:
            processing_function = process_pdf
            file_format = "pdf"
        else:
            st.error("Unsupported file type. Please upload a JPG, PNG, or PDF file.")
            return

        # Process the file with a loading spinner
        with st.spinner(f"Processing {file_format}..."):
            try:
                extracted_text = processing_function(uploaded_file)
            except Exception as e:
                st.error(f"Error processing the file: {e}")
                return

        # Display the extracted text
        st.subheader("Extracted Text")
        st.text_area("Text", extracted_text, height=300)

        # Offer to download the text
        st.download_button(
            label="Download Text",
            data=extracted_text,
            file_name='extracted_text.txt',
            mime='text/plain'
        )

if __name__ == "__main__":
    main()
