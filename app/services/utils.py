import PyPDF2
import os
from typing import List

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """
    Extract text content from PDF bytes.
    """
    import io
    try:
        pdf_file = io.BytesIO(pdf_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text
    except Exception as e:
        # Log this error properly in a real app
        print(f"Error extracting text from PDF bytes: {str(e)}")
        raise Exception(f"Error extracting text from PDF: {str(e)}")

def extract_text_from_pdf_path(pdf_file_path: str) -> str:
    """
    Extract text content from a PDF file path.
    """
    try:
        with open(pdf_file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from PDF path {pdf_file_path}: {str(e)}")
        raise Exception(f"Error extracting text from PDF: {str(e)}")


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks.
    """
    if not text or len(text) <= chunk_size:
        return [text] if text else []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        # Ensure end does not exceed text length
        actual_end = min(end, len(text))
        
        chunk = text[start:actual_end]
        chunks.append(chunk)
        
        if actual_end == len(text):
            break
            
        # Move start, ensuring overlap doesn't push it beyond reasonable limits
        start += chunk_size - overlap
        if start >= len(text): # Should not happen if text > chunk_size
             break
    
    return chunks

# Example of how to use with uploaded file:
# from werkzeug.utils import secure_filename
# import uuid
# def save_uploaded_file(file_storage, upload_folder):
#     if file_storage and file_storage.filename != '':
#         filename = secure_filename(file_storage.filename)
#         # Create a unique filename to avoid overwrites
#         unique_filename = f"{uuid.uuid4().hex}_{filename}"
#         file_path = os.path.join(upload_folder, unique_filename)
#         file_storage.save(file_path)
#         return file_path
#     return None

