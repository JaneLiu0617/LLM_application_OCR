import base64
import requests
import os

# SYSTEM_PROMPT =  """Act as an OCR assistant. Analyze the provided image and:
# 1. Recognize **all visible text** in the image with high accuracy.
# 2. **Preserve the original structure, spacing, and formatting** of the text, including multiple lines.
# 3. If any words or phrases are unclear, indicate this with [unclear] in your transcription.
# 4. If the text is skewed, rotated, or distorted, attempt to normalize it while maintaining its structure.

# Provide only the transcription without any additional comments."""
SYSTEM_PROMPT = """Act as an OCR assistant. Analyze the provided image and:
1. Recognize **all visible text** in the image with high accuracy.
2. **Preserve the original structure, spacing, and formatting** of the text, including multiple lines. **Specifically, output newline characters (\\n) to indicate line breaks.**
3. If any words or phrases are unclear, indicate this with [unclear] in your transcription.
4. If the text is skewed, rotated, or distorted, attempt to normalize it while maintaining its structure.

Example:
Input Image:
Line 1: This is the first line.
Line 2: And this is the second line.

Output:
This is the first line.\\nAnd this is the second line.

Provide only the transcription without any additional comments."""

def encode_image_to_base64(image_path):
    """Convert an image file to a base64 encoded string."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Error: File not found -> {image_path}")
    
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def perform_ocr(image_path):
    """Perform OCR using Llama 3.2-Vision via Ollama API."""
    base64_image = encode_image_to_base64(image_path)
    
    url = "http://localhost:****/api/chat"  # Ensure your Ollama server is running
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "llama3.2-vision",
        "messages": [
            {
                "role": "user",
                "content": SYSTEM_PROMPT,
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
    result = perform_ocr(image_path)
    
    if result:
        print("\nOCR Recognition Result:\n")
        print(result)
    else:
        print("OCR failed.")
