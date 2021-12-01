import numpy as np
import cv2
import json
import base64
from pylibdmtx.pylibdmtx import decode
from utils import letterbox, non_max_suppression
import torch

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


def handler(event, context):
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

    # return response
    response = {
        'statusCode': 200,
        'body': json.dumps(ret_detections, ensure_ascii=False),
        "headers":
            {
                "Content-Type": "application/json",
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'OPTIONS,POST,GET',
            }
    }

    return response


