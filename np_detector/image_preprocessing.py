# Image preprocessing defined function (utility function)

import numpy as np
import cv2


# my current image status (brightness, sharpness, )
def check_img_status(chk_image):

  brightness = np.mean(chk_image) # normal : 80-170
  print(f"Image Brightness: {brightness:.2f}")

  sharpness = cv2.Laplacian(chk_image, cv2.CV_64F).var() # higher is better
  print(f"Image Sharpness: {sharpness:.2f}")

  noise_level = np.std(chk_image) # less is better
  print(f"Background Noise Level: {noise_level:.2f}")

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

# crop image as per bbox
def crop_image_with_bbox(image, bbox):
    x_min, y_min, x_max, y_max = map(int, bbox)  # Convert to integers
    cropped = image[y_min:y_max, x_min:x_max]
    return cropped

