"""
OCR the number plate detected by the YOLO model
"""

# %pip install -q opencv-python-headless numpy easyocr tabulate

import cv2
import numpy as np
import easyocr
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
import re


# my current image status (brightness, sharpness, )
# brightness
def check_img_status(chk_image):

  brightness = np.mean(chk_image) # normal : 80-170
  print(f"Image Brightness: {brightness:.2f}")

  sharpness = cv2.Laplacian(chk_image, cv2.CV_64F).var() # higher is better
  print(f"Image Sharpness: {sharpness:.2f}")

  noise_level = np.std(chk_image) # less is better
  print(f"Background Noise Level: {noise_level:.2f}")

# easyocr

def easy_ocr(ocr_image):
    reader = easyocr.Reader(['en'])
    results = reader.readtext(ocr_image, detail=1, paragraph=False)
    return results

# format detected text to indian number plate

def interpret_indian_number_plate(detected_text_list):
    """
    Indian number plate format : [StateCode][RTOCode][Series][Number]
    """

    raw = ''.join(detected_text_list)
    cleaned = re.sub(r'[^A-Za-z0-9]', '', raw).upper()  # Remove hyphens, spaces, symbols

    # Remove 'IND' if present anywhere
    cleaned = re.sub(r'ind', '', cleaned, flags=re.IGNORECASE)

    # Match pattern: 1-2 letters (state) + 1-2 digits (RTO) + 1-3 letters (series) + 1-4 digits (number)
    pattern = re.compile(r'^([A-Z]{2})(\d{1,2})([A-Z]{1,3})(\d{1,4})$')
    match = pattern.match(cleaned)

    if match:
        # Extract and return without padding
        state = match.group(1)
        rto = match.group(2)
        series = match.group(3)
        number = match.group(4)
        print(f"{state} {rto}{series} {number}")
        return f"{state} {rto}{series} {number}"

    # If pattern doesn't match, return empty
    print("does not match indian numberplate format")
    return cleaned

# Process OCR results: draw bounding boxes, display text & probability, and print a table.
def extract_text(image):
    table_data = []
    ocr_image = image.copy()
    results = easy_ocr(ocr_image)

    for i, (bbox, text, prob) in enumerate(results):
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))

        # Draw bounding box
        cv2.rectangle(ocr_image, top_left, bottom_right, (0, 255, 0), 2)

        # Prepare text and probability
        text = text.upper()
        prob_text = f"{prob:.2f}"

        # Define positions
        text_position = (top_left[0], top_left[1] - 10)
        prob_position = (text_position[0] + len(text) * 15, text_position[1])

        # Get size for background
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        (prob_w, prob_h), _ = cv2.getTextSize(prob_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        # Draw background rectangles
        cv2.rectangle(ocr_image,
                    (text_position[0], text_position[1] - text_h),
                    (text_position[0] + text_w, text_position[1] + 5),
                    (0, 0, 0), -1)

        cv2.rectangle(ocr_image,
                    (prob_position[0], prob_position[1] - prob_h),
                    (prob_position[0] + prob_w, prob_position[1] + 5),
                    (0, 0, 0), -1)

        # Put colored text
        cv2.putText(ocr_image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)   # Green
        cv2.putText(ocr_image, prob_text, prob_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)  # Blue

    # Show result
    # showImage(ocr_image)


    # Print extracted text
    extracted_text = [text for (_, text, _) in results]
    print("\nDetected Text:", extracted_text)
    detected_number = interpret_indian_number_plate(extracted_text)

    # Create a DataFrame for the table
    # df = pd.DataFrame(table_data, columns=["Region", "Detected Text", "conf score"])

    # Display table
    # print("\nOCR Results Table:")
    # print(tabulate(df, headers="keys", tablefmt="fancy_grid", showindex=False))
    
    return extracted_text,detected_number


def apply_ocr(ocr_image):
  return extract_text(ocr_image)

# image preprocessing function

def toGray(image):
  image_g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  return image_g

# Noise Removal Filters
def apply_gaussian_blur(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0) # smoothens the image, reduce random noise
    return blurred

def apply_median_blur(image):
    blurred = cv2.medianBlur(image, 5) # replacing each pixel with the median of neighbors
    return blurred

def apply_bilateral_filter(image):
    filtered = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75) # Reduces noise while preserving edges
    return filtered

def apply_fast_nl_means_denoising(image):
    image_denoised = cv2.fastNlMeansDenoising(image, h=30) # Removes noise while maintaining structural details
    return image_denoised

# contrast enhancement
def apply_clahe(image):
  """Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to a grayscale image."""
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  image_clahe = clahe.apply(image)
  return image_clahe

# Applies thresholding
def apply_threshold(image):
    _, image_th = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # Converts the grayscale image to black & white
    return image_th

def apply_adaptive_threshold(image):
    image_th = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2) # handles uneven lighting
    return image_th

# Enlarges images to improve text visibility
def upscale_with_interpolation(image, scale=2, method='cubic'):
    interpolation_methods = {
        'nearest': cv2.INTER_NEAREST,
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4
    }
    h, w = image.shape[:2]
    resized_image = cv2.resize(image, (w * scale, h * scale), interpolation=interpolation_methods.get(method, cv2.INTER_CUBIC))
    return resized_image

# Define morphological function
def apply_morphology(image, kernel_size=(3, 3), operation='dilate'):
    # either expanding (dilate) or shrinking (erode) white regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    if operation == 'dilate':
        return cv2.dilate(image, kernel, iterations=1)
    elif operation == 'erode':
        return cv2.erode(image, kernel, iterations=1)
    return image


def showImage(img):
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow("detected Image", image_rgb)
    cv2.waitKey(0)    # Wait for key press to close window
    cv2.destroyAllWindows()

# crop image as per bbox

def crop_image_with_bbox(image, bbox):
    x_min, y_min, x_max, y_max = map(int, bbox)  # Convert to integers
    cropped = image[y_min:y_max, x_min:x_max]
    return cropped


# find visibility of detected text

def find_probability(plate_text):
  cleaned_text = plate_text.replace(" ", "")
  no_of_chars = cleaned_text.__len__()
  print(no_of_chars)
  print("visibility : ", (no_of_chars/10)*100 , "%")
  return no_of_chars/10


# return number plate in indian number plate forma and use '*' as filler - return np_dict
def indian_number_plate_format(detected_text_list):
    """
    Format Indian number plate with expected pattern:
    2 letters + 2 digits + 2 letters + 4 digits.
    Output: dict with index 0-9 as keys and plate characters (or '*') as values.
    """

    # Step 1: Clean input
    chars = []
    for part in detected_text_list:
        cleaned = re.sub(r'[^A-Za-z0-9]', '', part).upper()
        cleaned = re.sub(r'IND', '', cleaned, flags=re.IGNORECASE)
        chars.extend(list(cleaned))

    pattern = ['letter', 'letter', 'digit', 'digit', 'letter', 'letter', 'digit', 'digit', 'digit', 'digit']

    plate_dict = {}  # final map: {0: 'D', 1: 'L', ..., 9: '4'}
    i = 0  # pointer to input chars

    for idx, expected in enumerate(pattern):
        while i < len(chars):
            c = chars[i]
            if (expected == 'letter' and c.isalpha()) or (expected == 'digit' and c.isdigit()):
                plate_dict[idx] = c
                i += 1
                break
            else:
                plate_dict[idx] = '*'
                break  # try same input char for next expected
        else:
            plate_dict[idx] = '*'  # if input chars are exhausted

    # print("Plate as dict:", plate_dict)
    return plate_dict


# match and predict number-plate char in correct order from two occluded image extracted text
"""
np_dict1 : {0: 'H', 1: 'R', 2: '2', 3: '6', 4: '*', 5: '*', 6: '6', 7: '9', 8: '8', 9: '6'}
np_dict2 : {0: 'H', 1: 'R', 2: '2', 3: '6', 4: 'T', 5: 'C', 6: '6', 7: '*', 8: '*', 9: '*'}
"""
def predict_np(np_dict1, np_dict2):
    # Define NP structure: index ranges per section
    sections = {
        'state': [0, 1],
        'rto': [2, 3],
        'series': [4, 5],
        'number': [6, 7, 8, 9]
    }

    final_np = {}

    for section, indices in sections.items():
        # Extract characters for this section
        chars1 = [np_dict1[i] for i in indices]
        chars2 = [np_dict2[i] for i in indices]

        # Count valid (non-*) entries
        valid1 = sum(c != '*' for c in chars1)
        valid2 = sum(c != '*' for c in chars2)

        # Choose better section
        chosen = chars1 if valid1 >= valid2 else chars2

        # Fill final dict with chosen section values
        for i, idx in enumerate(indices):
            final_np[idx] = chosen[i]

    # Join to get final plate
    final_plate = ''.join(final_np[i] for i in range(10))
    print("Predicted Plate:", final_plate)
    return final_plate

    