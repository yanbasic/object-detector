import torch
import cv2
import numpy as np
import os
from utils import letterbox, non_max_suppression


class ObjectDetector(object):
    """
    Reference:
    - https://pypi.org/project/pylibdmtx/
    - http://libdmtx.sourceforge.net/

    For MacOS:
        # brew install libdmtx
        # pip3 install pylibdmtx==0.1.9

    For Linux:
        # sudo apt-get install libdmtx0a
        # pip3 install pylibdmtx==0.1.9
    """
    def __init__(self, model_full_path, vis_root_dir):
        """
        Constructor

        :param model_full_path: model weights of yolo-v5
        :param vis_root_dir: root directory which stores the visualized images
        """
        self._model_full_path = model_full_path

        # create visualization directory
        self._vis_root_dir = vis_root_dir
        if not os.path.exists(self._vis_root_dir):
            os.makedirs(self._vis_root_dir)

        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._class_names = ["2D_CODE", "Caution", "3C", "EAC", "UL2", "WEEE", "KC", "ATEX", "FM2", "Failsafe",
                             "RCM", "FM", "CE", "UL1"]
        self._vis_colors = {
            "2D_CODE": (255, 51, 51), "Caution": (255, 0, 255), "3C": (255, 128, 0), "EAC": (0, 153, 0),
            "UL2": (255, 153, 153), "WEEE": (0, 128, 255), "KC": (255, 204, 255), "ATEX": (0, 153, 153),
            "FM2": (204, 204, 0), "Failsafe": (153, 0, 153), "RCM": (255, 255, 153), "FM": (153, 255, 51),
            "CE": (178, 102, 255), "UL1": (93, 156, 192)
        }

        # load yolo-v5 pre-trained model weights
        self._detector = torch.jit.load(self._model_full_path, map_location=self._device)
        self._detector.eval()       # configure it as inference mode

    def run(self, image_full_path):
        """
        inference on given image, detect all interested objects, including

        :param image_full_path: full path of given image
        :return:
        """
        raw_image = cv2.imread(image_full_path, cv2.IMREAD_COLOR)  # BGR
        resized_img, ratio, (dw, dh) = letterbox(raw_image, new_shape=(640, 640), stride=32, auto=False)

        img = resized_img.transpose((2, 0, 1))[::-1]                # convert HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self._device)
        img = img.float()                                           # convert uint8 to float32
        img /= 255                                                  # 0 - 255 to 0.0 - 1.0
        img = img[None]                                             # expand for batch dim

        model_input = img.to(self._device)
        pred, _ = self._detector(model_input)

        pred = non_max_suppression(
            prediction=pred,
            conf_thres=0.50,
            iou_thres=0.50,
            classes=None,
            agnostic=False,
            max_det=100
        )
        detections = pred[0].cpu().numpy()

        # visualization
        scale_w, scale_h = ratio
        for det in detections:
            x_min = (det[0] - dw) / scale_w
            y_min = (det[1] - dh) / scale_h
            x_max = (det[2] - dw) / scale_w
            y_max = (det[3] - dh) / scale_h

            color = self._vis_colors[self._class_names[int(det[-1])]]
            color = (color[2], color[1], color[0])
            label_info = '{} {:.3f}'.format(self._class_names[int(det[5])], det[4])
            vis_image = cv2.rectangle(raw_image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 5)
            vis_image = cv2.putText(
                vis_image, label_info, (int(x_min), int(y_min) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

        cv2.imwrite(os.path.join(self._vis_root_dir, 'vis_' + image_full_path.split("test_imgs/")[1]), vis_image)


if __name__ == '__main__':
    detector = ObjectDetector(
        model_full_path='../models/yolov5s/weights/best.torchscript.pt',
        vis_root_dir="../visualization/"
    )
    detector.run(image_full_path="../test_imgs/IMG_SIMENS_0001.jpg")
