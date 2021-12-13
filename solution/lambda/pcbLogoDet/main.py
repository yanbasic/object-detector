import base64
from os import environ

import torch
from pylibdmtx.pylibdmtx import decode

from utils import letterbox, non_max_suppression

import json

import cv2
import concurrent.futures
from base64image import Base64Image
import time

from main_ocr import *

environ['MODEL_NAME'] = 'standard'
environ['MODEL_PATH'] = '/opt/program/model/standard/'
MODEL_FULL_PATH = '/opt/program/yolov5s/weights/best.torchscript.pt'

device = 'cpu'
class_names = [
    "2D_CODE", "Caution", "3C", "EAC", "UL2", "WEEE", "KC", "ATEX",
    "FM2", "Failsafe", "RCM", "FM", "CE", "UL1"]

# load yolo-v5 pre-trained model weights
detector = torch.jit.load(MODEL_FULL_PATH, map_location=device)
detector.eval()
print("Load object detection model successfully.")

# load data matrix code recognizer
data_matrix_code_decoder = decode


def recognize_data_matrix_code(roi):
    """
    recognize the content of the given data matrix

    :param roi: BGR image data, which is loaded with cv2.imread interface
    :return: code content, string
    """
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    height, width, _ = roi.shape
    target_width = 100.0
    scale = target_width / width
    dim = (int(target_width), int(height * scale))  # (width, height)
    resized_roi = cv2.resize(roi, dim, interpolation=cv2.INTER_AREA)
    code = data_matrix_code_decoder(resized_roi)
    code = code[0].data.decode('utf-8') if len(code) != 0 else ""

    return code


def detection_handler(event, context):
    request = json.loads(event['body'])
    image_base64_seq = request.get('image_base64_enc', None)

    # decode the base64 image
    im_bytes = base64.b64decode(image_base64_seq)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)
    image = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)  # BGR

    # resize the input image
    height, width, channels = image.shape
    resized_img, ratio, (dw, dh) = letterbox(image, new_shape=(640, 640), stride=32, auto=False)

    # pre-process
    image_chw_rgb = resized_img.transpose((2, 0, 1))[::-1]  # convert HWC to CHW, BGR to RGB
    image_chw_rgb = np.ascontiguousarray(image_chw_rgb)
    image_chw_rgb = torch.from_numpy(image_chw_rgb).to(device)
    image_chw_rgb = image_chw_rgb.float()  # convert uint8 to float32
    image_chw_rgb /= 255.0  # 0 - 255 to 0.0 - 1.0
    image_chw_rgb = image_chw_rgb[None]  # expand for batch dim
    model_input = image_chw_rgb.to(device)

    # object detection inference
    outputs, _ = detector(model_input)

    # post-process
    pred = non_max_suppression(
        prediction=outputs, conf_thres=0.50, iou_thres=0.50, classes=None, agnostic=False, max_det=100)
    detections = pred[0].cpu().numpy()
    print("detections.shape = {}".format(detections.shape))

    # resize back to original image scale
    ret_detections = {
        "channels": channels,
        "height": height,
        "width": width,
        "detections": list()
    }
    scale_w, scale_h = ratio
    for det in detections:
        x_min = int((det[0] - dw) / scale_w)
        y_min = int((det[1] - dh) / scale_h)
        x_max = int((det[2] - dw) / scale_w)
        y_max = int((det[3] - dh) / scale_h)

        confidence = float(det[4])
        cls_id = int(det[5])
        cls_name = class_names[cls_id]

        if cls_id == 0:  # 2D_CODE
            margin = 15
            roi_y_min = max(0, y_min - margin)
            roi_x_min = max(0, x_min - margin)
            roi_y_max = min(height, y_max + margin)
            roi_x_max = min(width, x_max + margin)

            roi = image[roi_y_min:roi_y_max, roi_x_min:roi_x_max]
            code = recognize_data_matrix_code(roi=roi)

            ret_detections["detections"].append({
                "bbox": [x_min, y_min, x_max, y_max],
                "confidence": confidence,
                "cls_name": cls_name,
                "data_matrix_code": code
            })
        else:
            ret_detections["detections"].append({
                "bbox": [x_min, y_min, x_max, y_max],
                "confidence": confidence,
                "cls_name": cls_name
            })

    print("ret_detections = {}".format(ret_detections))
    return ret_detections


def handler(event, context):
    if 'body' not in event:
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Headers': '*',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': '*'
            }
        }
    start_time = time.time()
    api_response = {
    }
    if isinstance(event['body'], str):
        body = json.loads(event['body'])
    else:
        body = event['body']
    with concurrent.futures.ThreadPoolExecutor() as executor:
        if 'ocr' in body and body['ocr']:
            future_ocr = executor.submit(ocr_handler, event, context)
        if 'detection' in body and body['detection']:
            future_detection = executor.submit(detection_handler, event, context)

    if 'ocr' in body and body['ocr']:
        api_response['ocr'] = future_ocr.result()
    if 'detection' in body and body['detection']:
        api_response['detection'] = future_detection.result()

    api_response['duration'] = time.time() - start_time

    response = {
        'statusCode': 200,
        'body': json.dumps(api_response, ensure_ascii=False),
        "headers":
            {
                "Content-Type": "application/json",
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'OPTIONS,POST',
            }
    }
    return response


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes


class TextSystem():
    def __init__(self):
        self.text_detector = TextDetector()
        self.text_recognizer = TextRecognizer()
        self.drop_score = 0.3
        self.text_classifier = TextClassifier()

    def get_rotate_crop_image(self, img, points):
        '''
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        '''
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    def __call__(self, img):
        ori_im = img.copy()
        dt_boxes = self.text_detector(img)
        if dt_boxes is None:
            return None, None
        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = self.get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        img_crop_list, angle_list = self.text_classifier(
            img_crop_list)

        rec_res = self.text_recognizer(img_crop_list)
        filter_boxes, filter_rec_res = [], []
        for box, rec_reuslt in zip(dt_boxes, rec_res):
            text, score = rec_reuslt
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_reuslt)
        return filter_boxes, filter_rec_res


text_sys = TextSystem()


def ocr_handler(event, context):
    if 'body' not in event:
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Headers': '*',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': '*'
            }
        }
    if isinstance(event['body'], str):
        body = json.loads(event['body'])
    else:
        body = event['body']
    base64_image = Base64Image.from_base64_image_string(body['image_base64_enc'])

    pil_image = base64_image.get_pil_image()
    img = np.array(pil_image)[:, :, :3][:, :, ::-1]
    dt_boxes, rec_res = text_sys(img)

    boxes = dt_boxes
    dt_results = list(zip(dt_boxes, rec_res))
    dt_results.sort(key=lambda x: (x[0].min(0)[1]))

    result = []
    for row in dt_results:
        row = {
            "words": row[1][0],
            "location": [int(row[0][0][1]),
                         int(row[0][0][0]),
                         int(row[0][2][0] - row[0][0][0]),
                         int(row[0][2][1] - row[0][0][1])
                         ],
            "confidence": float(row[1][1])
        }
        result.append(row)

    return result
