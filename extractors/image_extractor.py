
from PIL import Image
import pytesseract
import cv2
import numpy as np

def preprocess_image(file):
    image = Image.open(file).convert("RGB")
    img_array = np.array(image)

    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    denoised = cv2.fastNlMeansDenoising(thresh, h=30)

    return Image.fromarray(denoised)

def extract_text_from_image(file, handwritten=False):
    if handwritten:
        image = preprocess_image(file)
    else:
        image = Image.open(file).convert("RGB")

    return pytesseract.image_to_string(image, config='--psm 6')