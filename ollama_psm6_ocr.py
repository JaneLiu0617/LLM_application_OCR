import base64
import requests
import os
import pytesseract
from PIL import Image

SYSTEM_PROMPT = """Act as an OCR assistant. Analyze the provided image and:
1. Recognize **all visible text** in the image with high accuracy.
2. **Preserve the original structure, spacing, and formatting** of the text, including multiple lines.
3. If any words or phrases are unclear, indicate this with [unclear] in your transcription.
4. If the text is skewed, rotated, or distorted, attempt to normalize it while maintaining its structure.

Here is the raw OCR output extracted from the image:
-----
{ocr_text}
-----
Please clean up and correct any mistakes while keeping the original format.
Provide only the transcription without any additional comments."""

def encode_image_to_base64(image_path):
    """Convert an image file to a base64 encoded string."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Error: File not found -> {image_path}")
    
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def perform_tesseract_ocr(image_path, psm_mode=6):
    """Perform OCR using Tesseract with a specified PSM mode."""
    pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract" 
    image = Image.open(image_path)
    custom_config = f"--psm {psm_mode}"
    ocr_text = pytesseract.image_to_string(image, config=custom_config)
    return ocr_text.strip()

def perform_ocr_with_llama(image_path):
    """Perform OCR using Tesseract and refine output using LLaMA 3.2-Vision via Ollama API."""
    
    # Step 1: Extract text using Tesseract OCR
    ocr_text = perform_tesseract_ocr(image_path, psm_mode=6)
    
    # Step 2: Convert image to base64 for LLaMA processing
    base64_image = encode_image_to_base64(image_path)
    
    # Step 3: Send OCR text + image to LLaMA 3.2 for correction
    url = "http://localhost:****/api/chat"  # Ensure your Ollama server is running
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "llama3.2-vision",
        "messages": [
            {
                "role": "user",
                "content": SYSTEM_PROMPT.format(ocr_text=ocr_text),
                "images": [base64_image],
            }
        ],
        "stream": False,
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=1800)
        response.raise_for_status()  # Raises an error for non-2xx responses
        return response.json().get("message", {}).get("content", "")
    except requests.exceptions.RequestException as e:
        print("Error connecting to Ollama:", e)
        return None

if __name__ == "__main__":
    image_path = "/Users/jiayiliu/Desktop/llama-ocr/source/image.jpg"  # Replace with your actual image path
    result = perform_ocr_with_llama(image_path)
    
    if result:
        print("\nOCR Recognition Result:\n")
        print(result)
    else:
        print("OCR failed.")
