import torch
import cv2
import numpy as np
import os
import time
from PIL import Image
from pylibdmtx.pylibdmtx import decode
from utils import letterbox, non_max_suppression


class PerfEstimator(object):
    def __init__(self, model_full_path, vis_root_dir, dataset_root_dir, train_txt_full_path, val_txt_full_path):
        """
        Performance Estimator Constructor

        :param model_full_path: model weights of yolo-v5
        :param vis_root_dir: root directory which stores the visualized images
        :param dataset_root_dir: root directory of original dataset
        :param train_txt_full_path: training samples
        :param val_txt_full_path: validation samples
        """
        self._model_full_path = model_full_path
        self._dataset_root_dir = dataset_root_dir
        self._train_txt_full_path = train_txt_full_path
        self._val_txt_full_path = val_txt_full_path

        # create visualization directory
        self._vis_root_dir = vis_root_dir
        if not os.path.exists(self._vis_root_dir):
            os.system("mkdir -p ../vis/failed/")
            os.system("mkdir -p ../vis/success/")

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

        # load data matrix code recognizer
        self._data_matrix_code_decoder = decode

    def load_samples(self, subset):
        """
        Load samples

        :param subset: 'trainval', 'train' or 'val'
        :return:
        """
        if subset == 'train':
            all_lines = open(self._train_txt_full_path, 'r').readlines()
        elif subset == 'trainval':
            train_lines = open(self._train_txt_full_path, 'r').readlines()
            val_lines = open(self._val_txt_full_path, 'r').readlines()
            all_lines = list()
            all_lines.extend(train_lines)
            all_lines.extend(val_lines)
        elif subset == 'val':
            all_lines = open(self._val_txt_full_path, 'r').readlines()
        else:
            raise RuntimeError("Subset {} not supported.".format(subset))

        samples = list()
        for line in all_lines:
            line = line.strip()
            sample_name = line.split("images/")[1].split(".jpg")[0]
            label_name = sample_name + '.txt'
            label = np.loadtxt(os.path.join(self._dataset_root_dir, 'labels', label_name))
            samples.append({
                "image_full_path": os.path.join(self._dataset_root_dir, 'images', sample_name + '.jpg'),
                "label": label
            })

        return samples

    def estimate(self, subset='trainval', enable_vis=False):
        """
        Estimate the performance of given dataset

        :param subset: 'trainval', 'train' or 'val'
        :return: None
        """
        assert subset in ['train', 'val', 'trainval']
        samples = self.load_samples(subset)

        detections = list()
        for index, sample in enumerate(samples):
            print("Processing sample {} ({}/{})...".format(sample['image_full_path'], index+1, len(samples)))

            image = cv2.imread(sample['image_full_path'], cv2.IMREAD_COLOR)
            height, width, channels = image.shape
            res = self.inference(
                image,
                dump_image_name=sample['image_full_path'].split("/images/")[1],
                enable_vis=enable_vis)
            labels = sample['label']
            detections.append(res)

        # statistic performance
        self.analyze_performance(detections)

    @staticmethod
    def analyze_performance(detections):
        """
        analyze performance between predictions and ground truth

        :param detections: detected results, list
        :return:
        """
        print("Totally {} samples:".format(len(detections)))
        data_matrix_count = 0
        data_matrix_count_decode_success = 0

        for detection in detections:
            for det in detection:
                if det['cls_name'] == '2D_CODE':
                    data_matrix_count += 1

                if det['cls_name'] == '2D_CODE' and det['data_matrix_code'] != "":
                    data_matrix_count_decode_success += 1

        print("Totally {} 2D Codes, decode {} successfully. Accuracy = {}".format(
            data_matrix_count,
            data_matrix_count_decode_success,
            float(data_matrix_count_decode_success)/data_matrix_count))

    def inference(self, image, dump_image_name, enable_vis=False):
        """
        inference on given image, detect all interested objects, including

        :param image: BGR data, which is loaded using cv2.imread
        :param dump_image_name: dumped image name
        :return:
        """
        height, width, channels = image.shape
        resized_img, ratio, (dw, dh) = letterbox(image, new_shape=(640, 640), stride=32, auto=False)

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

        # resize back to original image scale
        ret_detections = list()
        scale_w, scale_h = ratio
        for det in detections:
            x_min = int((det[0] - dw) / scale_w)
            y_min = int((det[1] - dh) / scale_h)
            x_max = int((det[2] - dw) / scale_w)
            y_max = int((det[3] - dh) / scale_h)

            confidence = det[4]
            cls_id = int(det[5])
            cls_name = self._class_names[cls_id]

            if cls_id == 0:     # 2D_CODE
                margin = 15
                roi_y_min = max(0, y_min-margin)
                roi_x_min = max(0, x_min-margin)
                roi_y_max = min(height, y_max+margin)
                roi_x_max = min(width, x_max+margin)

                roi = image[roi_y_min:roi_y_max, roi_x_min:roi_x_max]
                code = self.recognize_data_matrix_code(
                    roi=roi, sample_name=dump_image_name.split(".jpg")[0], enable_vis=enable_vis)

                ret_detections.append({
                    "bbox": [x_min, y_min, x_max, y_max],
                    "confidence": confidence,
                    "cls_name": cls_name,
                    "data_matrix_code": code
                })
            else:
                ret_detections.append({
                    "bbox": [x_min, y_min, x_max, y_max],
                    "confidence": confidence,
                    "cls_name": cls_name
                })

            # visualization
            if enable_vis:
                color = self._vis_colors[self._class_names[cls_id]]
                color = (color[2], color[1], color[0])
                label_info = '{} {:.3f}'.format(self._class_names[int(det[5])], det[4])
                vis_image = cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 5)
                vis_image = cv2.putText(
                    vis_image, label_info, (int(x_min), int(y_min) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

        if enable_vis:
            cv2.imwrite(os.path.join(self._vis_root_dir, dump_image_name), vis_image)
        return ret_detections

    def recognize_data_matrix_code(self, roi, sample_name, enable_vis):
        """
        recognize the content of the given data matrix

        :param roi: BGR image data, which is loaded with cv2.imread interface
        :return: code content, string
        """
        # t1 = time.time()
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        height, width, _ = roi.shape
        target_width = 90.0
        scale = target_width / width
        dim = (int(target_width), int(height * scale))            # (width, height)
        resized_roi = cv2.resize(roi, dim, interpolation=cv2.INTER_AREA)
        pil_image = Image.fromarray(resized_roi)

        for degree in np.arange(0, 360, 45):
            input_roi = pil_image.rotate(degree, expand=True)
            code = self._data_matrix_code_decoder(input_roi)
            code = code[0].data.decode('utf-8') if len(code) != 0 else ""

            if code != "":
                break

        if enable_vis:
            if code == "":
                dump_full_path = os.path.join(self._vis_root_dir, "failed", sample_name + "_2D_CODE.jpg")
            else:
                dump_full_path = os.path.join(self._vis_root_dir, "success", sample_name + "_2D_CODE.jpg")

            cv2.imwrite(dump_full_path, cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))

        # t2 = time.time()
        # print("Time cost = {} ms".format(1000 * (t2 - t1)))
        return code


if __name__ == '__main__':
    estimator = PerfEstimator(
        model_full_path='../models/yolov5s/weights/best.torchscript.pt',
        vis_root_dir="../vis/",
        dataset_root_dir="../dataset/factory/",
        train_txt_full_path="../dataset/factory/train.txt",
        val_txt_full_path="../dataset/factory/val.txt"
    )

    estimator.estimate(subset='val', enable_vis=False)
