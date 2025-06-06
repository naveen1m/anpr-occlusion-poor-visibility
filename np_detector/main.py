
import cv2
from anpr import detect_number_plate, trained_model, test_code
from ocr_numberplate import (
    crop_image_with_bbox, toGray, showImage, upscale_with_interpolation,
    apply_fast_nl_means_denoising, apply_clahe, apply_threshold,
    apply_morphology, apply_ocr, detected_number, find_probability
)

def main():
    test_image_path = "../data/blur-test-images/blur-car1A.jpeg"
    result = detect_number_plate(test_image_path)
    print("YOLO Detections: ", result)

    bbox = result[0]['box']

    image = cv2.imread(test_image_path)
    # showImage(image)
    image_c = crop_image_with_bbox(image,bbox)
    image_g = toGray(image_c)
    image_upscale = upscale_with_interpolation(image_g)
    image_bilateral = apply_fast_nl_means_denoising(image_upscale)
    image_clahe = apply_clahe(image_bilateral)
    image_th = apply_threshold(image_clahe)
    image_morphology = apply_morphology(image_th, operation='erode')
    # showImage(image_morphology)

    # apply ocr
    apply_ocr(image_morphology)

# def main():
#     test_code()

if __name__ == "__main__":
    main()
