import numpy as np
import cv2
import pytesseract
import re
import sys

from PIL import Image
from collections import defaultdict

sys.path.append('EAST')
from eval import *


def four_point_transform(img, pts):
    '''
    Takes in the original image with the object of interest and
    extracts and transforms the view of the object.
    As seen on https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    :param img: The complete image.
    :param pts: The four points surrounding the object of interest.
    :return: Returns object of interest from a "birds eye view".
    '''

    pts = pts.astype('float32')
    (tl, tr, br, bl) = pts

    # compute the width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # construct set of destination points to obtain a "birds eye view",
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    # return the warped image, as well as the coordinates and the transformation matrix
    return warped, dst, M

    
def get_ocr_data(img, psm=6, oem=1, whitelist=None):
    '''
    Reads the text.
    :param img: An image containing text.
    :param psm: The "page segmentation method". If set to 6 we assume a single uniform block of text.
                More information at https://github.com/tesseract-ocr/tesseract/wiki/ImproveQuality
    :param oem: The engine to use. 1 sets the engine to use the LSTM.
    :param whitelist: Characters to allow in the result.
    :return: Data containing information about the text, confidence and more.
    '''

    ocr_data = pytesseract.image_to_data(
        Image.fromarray(img), output_type='dict',
        config='--psm {} --oem {} -c load_system_dawg=0 load_freq_dawg=0'.format(psm, oem))

    # as whitelist doesn't work for LSTM engine
    ocr_data['text'] = [re.sub(r'[^{}]+'.format(whitelist), '', text) if whitelist is not None else text 
                        for text in ocr_data['text']]
    return ocr_data


def filter_ocr_result(ocr_result, threshold=50):
    '''
    Filters the results obtained by `get_ocr_data` regarding the confidence of the OCR.
    :param ocr_result: The original results.
    :param threshold: Threshold determining the minimal confidence for a text part.
    :return: Filtered OCR data.
    '''

    ocr_filtered = defaultdict(list)
    valid_confs = [i for i, (conf, text) in enumerate(zip(ocr_result['conf'], ocr_result['text'])) if conf > 0]

    for key, value in ocr_result.items():
        ocr_filtered[key] = list(np.array(value)[valid_confs])
    ocr_filtered['valid_colors'] = [(0, 0, 255) if val > threshold 
                                   else (255, 0, 0) for val in list(np.array(ocr_result['conf'])[valid_confs])]
    return ocr_filtered
    
    
def reverse_ocr_polys(data, sr_shape, ocr, canvas_factor):
    '''
    The OCR not only reads the text but also specifies the location of the parts that it read.
    This function shows the surrounding polygons returned by the OCR.
    :param data: Data containing the image,
                coordinates and transformation matrix obtained from the four-point-transformation.
    :param sr_shape: Shape of the super-resolved image.
    :param ocr: The OCR results.
    :param canvas_factor: The factor which was used to increase the canvas of the text images.
    :return:
    '''

    img = data[0]
    dst = data[1]
    matrix = data[2]
    ratios = (sr_shape[0] / img.shape[0], sr_shape[1] / img.shape[1])

    # Get the positions of all parts of the text and transform them to the size of the original image.
    text_parts = [
        np.array([
            [dst[0][0] + (left - sr_shape[1] * canvas_factor) / ratios[1], 
             dst[0][1] + (top - sr_shape[0] * canvas_factor) / ratios[0]],
            [dst[0][0] + (left - sr_shape[1] * canvas_factor) / ratios[1] + width / ratios[1], 
             dst[0][1] + (top - sr_shape[0] * canvas_factor) / ratios[0]],
            [dst[0][0] + (left - sr_shape[1] * canvas_factor) / ratios[1] + width / ratios[1], 
             dst[0][1] + (top - sr_shape[0] * canvas_factor) / ratios[0] + height / ratios[0]],
            [dst[0][0] + (left - sr_shape[1] * canvas_factor) / ratios[1], 
             dst[0][1] + (top - sr_shape[0] * canvas_factor) / ratios[0] + height / ratios[0]]],
            dtype='float32')
        for left, top, width, height in zip(ocr['left'], ocr['top'], ocr['width'], ocr['height'])]

    # Reverse the bird-eye-view transformation
    M_inv = np.linalg.inv(matrix)
    reversed_polys = [np.concatenate([[np.dot(M_inv, np.append(pt, 1)) for pt in part]]) for part in text_parts]

    return [np.row_stack([(pt / pt[-1])[:2] for pt in poly]).astype('int32') for poly in reversed_polys]
    
    
def run_east_model(imgs, model, verbose=0):
    '''
    Runs the detection of the EAST model
    :param imgs: Images to detect text polygons in.
    :param model: Loaded model
    :return: Returns the polygons for all images that were fed in.
    '''

    resize_data = [resize_image(img) for img in imgs]
    imgs_resized = [data[0] for data in resize_data]
    ratios = [data[1] for data in resize_data]

    imgs_resized = np.array([(img / 127.5) - 1 for img in imgs_resized])

    # feed image into model
    # if images have different shapes input one image after another
    if len(imgs_resized.shape) == 1:
        east_outputs = [model.predict(np.expand_dims(img, axis=0)) for img in imgs_resized]
        score_maps = [np.expand_dims(data[0][0], axis=0) for data in east_outputs]
        geo_maps = [np.expand_dims(data[1][0], axis=0) for data in east_outputs]
    else:
        east_outputs = model.predict(imgs_resized)        
        score_maps = [np.expand_dims(data, axis=0) for data in east_outputs[0]]
        geo_maps = [np.expand_dims(data, axis=0) for data in east_outputs[1]]

    polyss = [detect(score_map=score_map, geo_map=geo_map, verbose=verbose)
              for score_map, geo_map in zip(score_maps, geo_maps)]
    
    for i, (polys, ratio) in enumerate(zip(polyss, ratios)):
        if polys is not None:
            polyss[i] = polyss[i][:, :8].reshape((-1, 4, 2))
            polyss[i][:, :, 0] /= ratio[1]
            polyss[i][:, :, 1] /= ratio[0]

    return polyss


def dilate_img(img, kernel_size=2, iterations=1):
    '''
    Thins out the characters in the text. Thinning characters seperate them which usually leads to better OCR performance.
    :param img: Image containing the text.
    :param kernel_size: Kernel to use for the transformation. The higher the kernel size the thinner the result.
    :param iterations: Number of iterations.
    :return:
    '''

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(img, kernel, iterations=iterations)