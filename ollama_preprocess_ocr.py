import base64
import requests
import os
import cv2
import numpy as np

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

def deskew(image):
    """Deskews the image using OpenCV."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coords = np.column_stack(np.where(gray > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def preprocess_image(image_path):
    """Preprocesses the image using OpenCV."""
    image = cv2.imread(image_path)
    deskewed = deskew(image)
    upscaled = cv2.resize(deskewed, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(upscaled, kernel, iterations=1)
    return dilated

def encode_image_to_base64(image_path):
    """Convert an image file to a base64 encoded string after preprocessing."""
    try:
        preprocessed_image = preprocess_image(image_path)
        _, encoded_image = cv2.imencode('.jpg', preprocessed_image) # Encode the preprocessed image to jpg format.
        return base64.b64encode(encoded_image).decode('utf-8')
    except Exception as e:
        print(f"Error during preprocessing or encoding: {e}")
        return None

def perform_ocr(image_path):
    """Perform OCR using Llama 3.2-Vision via Ollama API."""
    base64_image = encode_image_to_base64(image_path)

    if base64_image is None:
        return None  # Return None if image encoding fails

    url = "http://localhost:11434/api/chat"  # Ensure your Ollama server is running
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
        response = requests.post(url, json=payload, headers=headers, timeout=8000)
        response.raise_for_status()  # Raises an error for non-2xx responses
        return response.json().get("message", {}).get("content", "")
    except requests.exceptions.RequestException as e:
        print("Error connecting to Ollama:", e)
        return None

if __name__ == "__main__":
    image_path = "/Users/jiayiliu/Desktop/llama-ocr/source/stock_gs200.jpg"  # Replace with your actual image path
    result = perform_ocr(image_path)
    
    if result:
        print("\nOCR Recognition Result:\n")
        print(result)
    else:
        print("OCR failed.")