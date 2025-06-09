from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
import base64
from typing import List

from anpr import detect_number_plate
from ocr_numberplate import ( apply_ocr, indian_number_plate_format, predict_np, find_probability)
from image_preprocessing import (crop_image_with_bbox, toGray, upscale_with_interpolation, apply_fast_nl_means_denoising, apply_clahe, apply_threshold, apply_morphology)

app = FastAPI()

# Allow CORS from all origins (for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_test():
    return {"test": "Hello world"}

class ImageData(BaseModel):
    image_base64: str

class PredictRequest(BaseModel):
    images: List[ImageData]  # Expecting exactly 2 images

def base64_to_cv2_img(base64_str):
    try:
        img_data = base64.b64decode(base64_str)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Decoded image is None")
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {e}")

def extract_text(image):
    result = detect_number_plate(image)
    if not result:
        raise HTTPException(status_code=404, detail="No number plate detected in image.")
    
    bbox = result[0]['box']
    image_c = crop_image_with_bbox(image, bbox)
    image_g = toGray(image_c)
    image_upscale = upscale_with_interpolation(image_g)
    image_bilateral = apply_fast_nl_means_denoising(image_upscale)
    image_clahe = apply_clahe(image_bilateral)
    image_th = apply_threshold(image_clahe)
    image_morphology = apply_morphology(image_clahe, operation='dilate')
    
    extracted_text, detected_number = apply_ocr(image_morphology)
    formatted_np = indian_number_plate_format(extracted_text)  # dict or formatted plate
    occlusion = find_probability(detected_number)
    return extracted_text, detected_number, formatted_np, occlusion

@app.post("/predict_number_plate")
def predict_number_plate_base64(request: PredictRequest):                                               
    if len(request.images) != 2:
        img1 = base64_to_cv2_img(request.images[0].image_base64)
        extracted_text1, detected_number1, format_np1, visibility1 = extract_text(img1)
        return {
            "extracted_text1": extracted_text1,
            "detected_number1":detected_number1,
            "visibility1":visibility1
        }

    # Decode images from base64
    img1 = base64_to_cv2_img(request.images[0].image_base64)
    img2 = base64_to_cv2_img(request.images[1].image_base64)

    # Extract text and format number plate from both images
    extracted_text1, detected_number1, format_np1, visibility1 = extract_text(img1)
    extracted_text2, detected_number2, format_np2, visibility2 = extract_text(img2)

    # Predict final number plate by merging both dicts
    predicted_np = predict_np(format_np1, format_np2)

    return {
        "extracted_text1": extracted_text1,
        "detected_number1":detected_number1,
        "visibility1":visibility1,
        "visibility2":visibility2,
        "detected_number2":detected_number2,
        "extracted_text2": extracted_text2,
        "predicted_number_plate": predicted_np
    }

