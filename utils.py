
import pdfplumber
import docx
import os
import re
from typing import List, Union

def extract_text(file_path: str) -> Union[str, None]:
    """Extracts text from PDF, DOCX, or TXT."""
    print(f"Extracting text from: {file_path}")
    _, extension = os.path.splitext(file_path)
    
    try:
        if extension == '.pdf':
            with pdfplumber.open(file_path) as pdf:
                return "\n".join(page.extract_text() for page in pdf.pages)
        elif extension == '.docx':
            doc = docx.Document(file_path)
            return "\n".join(para.text for para in doc.paragraphs)
        elif extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            print(f"Warning: Unsupported file type: {file_path}")
            return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def simple_tokenizer(text: str) -> List[str]:
    """A simple tokenizer for BM25."""
    text = re.sub(r'[^\w\s]', '', text).lower()
    return text.split()

print("File 'utils.py' created.")
