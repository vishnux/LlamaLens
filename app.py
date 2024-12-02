import os
import tempfile
import streamlit as st
from llama_ocr import LlamaOCR
import PyPDF2

def validate_file(uploaded_file):
    """
    Validate the uploaded file type and size.
    
    Args:
        uploaded_file (UploadedFile): Streamlit uploaded file object
    
    Returns:
        bool: True if file is valid, False otherwise
    """
    # Check file size (limit to 10MB)
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    
    # Supported file types
    SUPPORTED_TYPES = ['jpg', 'jpeg', 'png', 'pdf']
    
    if uploaded_file is None:
        st.warning("No file uploaded.")
        return False
    
    # Check file size
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error(f"File size exceeds {MAX_FILE_SIZE / (1024 * 1024)}MB limit.")
        return False
    
    # Check file extension
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension not in SUPPORTED_TYPES:
        st.error(f"Unsupported file type. Please upload {', '.join(SUPPORTED_TYPES)} files.")
        return False
    
    return True

def process_pdf(uploaded_file):
    """
    Extract text from PDF file.
    
    Args:
        uploaded_file (UploadedFile): Uploaded PDF file
    
    Returns:
        str: Extracted text from PDF
    """
    # Create a temporary file to save the uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
        temp_pdf.write(uploaded_file.getbuffer())
        temp_pdf_path = temp_pdf.name
    
    try:
        # Open the PDF file
        with open(temp_pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Extract text from all pages
            extracted_text = ""
            for page in pdf_reader.pages:
                extracted_text += page.extract_text() + "\n\n"
        
        return extracted_text.strip()
    
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return ""
    finally:
        # Clean up the temporary file
        os.unlink(temp_pdf_path)

def extract_text_from_image(uploaded_file):
    """
    Extract text from image using Llama OCR.
    
    Args:
        uploaded_file (UploadedFile): Uploaded image file
    
    Returns:
        str: Extracted text from image
    """
    # Create a temporary file to save the uploaded image
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_img:
        temp_img.write(uploaded_file.getbuffer())
        temp_img_path = temp_img.name
    
    try:
        # Initialize Llama OCR
        ocr = LlamaOCR()
        
        # Extract text from image
        extracted_text = ocr.ocr(temp_img_path)
        
        return extracted_text
    
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return ""
    finally:
        # Clean up the temporary file
        os.unlink(temp_img_path)

def main():
    """
    Main Streamlit application function.
    Sets up the UI and handles file processing.
    """
    # Set page configuration
    st.set_page_config(
        page_title="Llama OCR Text Extractor",
        page_icon="📄",
        layout="centered"
    )
    
    # Application title and description
    st.title("📄 Llama OCR Text Extractor")
    st.markdown("""
    ### Extract text from images and PDFs
    - Supports JPG, PNG, and PDF files
    - Maximum file size: 10MB
    - Fast and accurate text extraction
    """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image or PDF file",
        type=['jpg', 'jpeg', 'png', 'pdf'],
        help="Upload an image or PDF to extract text"
    )
    
    # Process file if uploaded
    if uploaded_file is not None:
        # Validate file
        if not validate_file(uploaded_file):
            return
        
        # Show loading spinner
        with st.spinner('Extracting text...'):
            # Determine file type and process accordingly
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'pdf':
                extracted_text = process_pdf(uploaded_file)
            else:
                extracted_text = extract_text_from_image(uploaded_file)
        
        # Display extracted text
        if extracted_text:
            st.subheader("Extracted Text")
            st.text_area("", value=extracted_text, height=300)
            
            # Download button for extracted text
            st.download_button(
                label="Download Extracted Text",
                data=extracted_text,
                file_name="extracted_text.txt",
                mime="text/plain"
            )
        else:
            st.warning("No text could be extracted from the file.")

if __name__ == "__main__":
    main()
