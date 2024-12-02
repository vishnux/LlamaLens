import os
import tempfile
import streamlit as st
import ollama
import PIL.Image
import pytesseract
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
                # First try extracting text directly
                page_text = page.extract_text()
                
                # If direct extraction fails, convert page to image and use OCR
                if not page_text.strip():
                    # You might want to add PDF to image conversion logic here
                    # For now, we'll just add a note
                    page_text = "Could not extract text directly from this page."
                
                extracted_text += page_text + "\n\n"
        
        return extracted_text.strip()
    
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return ""
    finally:
        # Clean up the temporary file
        os.unlink(temp_pdf_path)

def extract_text_with_pytesseract(image_path):
    """
    Extract text from image using Tesseract OCR.
    
    Args:
        image_path (str): Path to the image file
    
    Returns:
        str: Extracted text from image
    """
    try:
        # Open the image
        image = PIL.Image.open(image_path)
        
        # Extract text using Tesseract
        extracted_text = pytesseract.image_to_string(image)
        
        return extracted_text.strip()
    except Exception as e:
        st.error(f"Error with Tesseract OCR: {e}")
        return ""

def enhance_ocr_with_llm(raw_text):
    """
    Use Ollama to enhance or clean up the OCR extracted text.
    
    Args:
        raw_text (str): Raw text extracted by OCR
    
    Returns:
        str: Enhanced or cleaned text
    """
    try:
        # Prompt to clean up and enhance OCR text
        prompt = f"""You are an expert in cleaning up OCR text. 
        Carefully review the following text and correct any obvious OCR errors, 
        preserve the original formatting, and return the most accurate version:

        ```
        {raw_text}
        ```

        Return the cleaned text, maintaining the original structure as much as possible."""

        # Use Ollama to process the text
        response = ollama.chat(
            model='llava:13b',  # Multimodal model that can help with text understanding
            messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
        )
        
        return response['message']['content']
    
    except Exception as e:
        st.error(f"Error enhancing text with Ollama: {e}")
        return raw_text

def main():
    """
    Main Streamlit application function.
    Sets up the UI and handles file processing.
    """
    # Set page configuration
    st.set_page_config(
        page_title="Ollama OCR Text Extractor",
        page_icon="📄",
        layout="centered"
    )
    
    # Application title and description
    st.title("📄 Ollama OCR Text Extractor")
    st.markdown("""
    ### Extract and Enhance Text from Images and PDFs
    - Supports JPG, PNG, and PDF files
    - Maximum file size: 10MB
    - Powered by Tesseract OCR and Ollama
    """)
    
    # Model selection
    st.sidebar.header("OCR Settings")
    ocr_model = st.sidebar.selectbox(
        "Select OCR Enhancement Model",
        ["llava:13b", "mistral", "llama2"]
    )
    
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
        with st.spinner('Extracting and Enhancing Text...'):
            # Determine file type and process accordingly
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            # Create a temporary file to save the uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_file_path = temp_file.name
            
            try:
                # Extract text based on file type
                if file_extension == 'pdf':
                    extracted_text = process_pdf(uploaded_file)
                else:
                    # Use Tesseract for initial OCR
                    extracted_text = extract_text_with_pytesseract(temp_file_path)
                
                # Enhance text with Ollama
                if extracted_text:
                    enhanced_text = enhance_ocr_with_llm(extracted_text)
                else:
                    enhanced_text = "No text could be extracted."
            
            except Exception as e:
                st.error(f"Processing error: {e}")
                enhanced_text = ""
            
            finally:
                # Clean up temporary file
                os.unlink(temp_file_path)
        
        # Display extracted and enhanced text
        if enhanced_text:
            st.subheader("Extracted and Enhanced Text")
            st.text_area("", value=enhanced_text, height=300)
            
            # Download button for extracted text
            st.download_button(
                label="Download Extracted Text",
                data=enhanced_text,
                file_name="extracted_text.txt",
                mime="text/plain"
            )
        else:
            st.warning("No text could be extracted from the file.")

if __name__ == "__main__":
    main()
