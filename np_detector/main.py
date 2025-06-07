
import cv2
from anpr import detect_number_plate, test_code
from ocr_numberplate import ( apply_ocr, indian_number_plate_format, predict_np)
from image_preprocessing import (crop_image_with_bbox, toGray, upscale_with_interpolation, apply_fast_nl_means_denoising, apply_clahe, apply_threshold, apply_morphology)


def main():
    test_image_path1 = "../data/blur-test-images/blur-car6.jpeg"
    test_image_path2 = "../data/blur-test-images/blur-car6A.jpeg"
    # result = detect_number_plate(test_image_path)
    # print("YOLO Detections: ", result)

    def extract_text(test_image_path):
        result = detect_number_plate(test_image_path)
        bbox = result[0]['box']
        image = cv2.imread(test_image_path)
        # showImage(image)
        image_c = crop_image_with_bbox(image,bbox)
        image_g = toGray(image_c)
        image_upscale = upscale_with_interpolation(image_g)
        image_bilateral = apply_fast_nl_means_denoising(image_upscale)
        image_clahe = apply_clahe(image_bilateral)
        image_th = apply_threshold(image_clahe)
        image_morphology = apply_morphology(image_clahe, operation='dilate')
        # showImage(image_th)
        extracted_text,detected_number = apply_ocr(image_morphology)
        format_np = indian_number_plate_format(extracted_text) # return a dictionary
        return extracted_text, detected_number, format_np

    _, _, format_np1 = extract_text(test_image_path1) 
    _, _, format_np2 = extract_text(test_image_path2) 
    print("format_np1 : ", format_np1)
    print("format_np2 : ", format_np2)

    predicted_np = predict_np(format_np1, format_np2)
    print("correct number plate predicted in occluded license plate", predicted_np)

    # # apply ocr
    # print("extracted_text: ", extracted_text)
    # print("detected_number: ", detected_number)
    # format_np = indian_number_plate_format(extracted_text)
    # print("format_np", format_np)

def test():
    # extracted_text = ['HR 26 TC 6']
    # print(extracted_text)
    # detected_number = interpret_indian_number_plate(extracted_text)
    # print("detected_number: ", detected_number)
    # format_np = indian_number_plate_format(extracted_text)
    # print("format_np", format_np)

    np_dict1 = {0: 'H', 1: 'R', 2: '2', 3: '6', 4: '*', 5: '*', 6: '6', 7: '9', 8: '8', 9: '6'}
    np_dict2 = {0: 'H', 1: 'R', 2: '2', 3: '6', 4: 'T', 5: 'C', 6: '6', 7: '*', 8: '*', 9: '*'}
    predict_np(np_dict1, np_dict2)
    test_code()

if __name__ == "__main__":
    main()
    # test()
