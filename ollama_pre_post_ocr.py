import base64
import requests
import os
import cv2
import numpy as np
import re
from spellchecker import SpellChecker

SYSTEM_PROMPT = """Act as an OCR assistant. Analyze the provided image, which contains stock information in a table-like format with columns. 
1. Recognize all text with high accuracy, including the text in each column.
2. Preserve the original structure, spacing, and formatting of the text, including multiple lines and columns.
3. If any words or phrases are unclear, indicate this with [unclear] in your transcription.
4. If the text is skewed, rotated, or distorted, attempt to normalize it while maintaining its structure.
5. Output the text in a structured manner, indicating the columns.
Example:
Column 1: Value 1, Value 2
Column 2: Value 3, Value 4
"""

def upscale_image(image, scale_factor=2):
    width = int(image.shape[1] * scale_factor)
    height = int(image.shape[0] * scale_factor)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)

def binarize_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

def reduce_noise(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def dilate_image(image, kernel_size=(2, 2)):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.dilate(image, kernel, iterations=1)

def preprocess_image(image_path):
    try:
        image = cv2.imread(image_path)
        upscaled = upscale_image(image)
        denoised = reduce_noise(upscaled)
        binarized = binarize_image(denoised)
        dilated = dilate_image(binarized)
        return dilated
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None

def encode_image_to_base64(image_path):
    try:
        preprocessed_image = preprocess_image(image_path)
        if preprocessed_image is None:
            return None
        _, encoded_image = cv2.imencode('.jpg', preprocessed_image)
        return base64.b64encode(encoded_image).decode('utf-8')
    except Exception as e:
        print(f"Error during encoding: {e}")
        return None

def perform_ocr(image_path):
    base64_image = encode_image_to_base64(image_path)

    if base64_image is None:
        return None

    url = "http://localhost:*****/api/chat" # change into your own sever
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
        response.raise_for_status()
        return response.json().get("message", {}).get("content", "")
    except requests.exceptions.RequestException as e:
        print("Error connecting to Ollama:", e)
        return None

def clean_text(text):
    """Cleans the text by removing extra spaces and non-alphanumeric characters."""
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\w\s\.\+\-]', '', text)
    return text

def extract_stock_data(ocr_text):
    """Extracts stock data from the OCR text, assuming space-separated columns."""
    lines = ocr_text.split('\n')
    stock_data = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 5:
            stock_data.append(parts)
    return stock_data

def correct_spelling(text):
    """Corrects spelling errors in the text."""
    spell = SpellChecker()
    words = text.split()
    corrected_words = []
    for word in words:
        corrected_words.append(spell.correction(word) or word)
    return ' '.join(corrected_words)

def post_process(ocr_text):
    """Applies post-processing steps to the OCR text."""
    cleaned_text = clean_text(ocr_text)
    stock_data = extract_stock_data(cleaned_text)
    corrected_text = correct_spelling(cleaned_text)
    return stock_data, corrected_text #return both structured and corrected text.

if __name__ == "__main__":
    image_path = "/Users/jiayiliu/Desktop/llama-ocr/source/stock_gs200.jpg"
    result = perform_ocr(image_path)
    
    if result:
        stock_data, corrected_text = post_process(result)
        print("\nStructured Stock Data:\n", stock_data)
        print("\nCorrected Text:\n", corrected_text)
    else:
        print("OCR failed.")
