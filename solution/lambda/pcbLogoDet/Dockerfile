FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive
ARG FUNCTION_DIR="/opt/program"

RUN sed -i s@/security.ubuntu.com@/mirrors.tuna.tsinghua.edu.cn@g /etc/apt/sources.list
RUN sed -i s@/archive.ubuntu.com@/mirrors.tuna.tsinghua.edu.cn@g /etc/apt/sources.list
RUN cat /etc/apt/sources.list

RUN apt-get -y update && \
    apt-get install -y libdmtx0a \
    wget \
    unzip \
    python3 \
    python3-pip \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip==21.3.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install --no-cache  torch==1.7.1 \
    numpy==1.19.5 \
    torchvision==0.8.2 \
    pylibdmtx==0.1.9 \
    requests==2.25.1 \
    six==1.16.0 \
    opencv-python-headless==4.5.3.56 \
    onnxruntime==1.8.1 \
    Pillow==8.1.1 \
    pyclipper==1.3.0 \
    Shapely==1.7.1 \
    base64image==0.5.1 \
    urllib3==1.26.6 \
    python-dateutil==2.8.2 \
    certifi==2021.5.30 \
    idna==2.10 \
    chardet==4.0.0 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN mkdir -p ${FUNCTION_DIR}
COPY main.py ${FUNCTION_DIR}/.
COPY utils.py ${FUNCTION_DIR}/.

COPY main_ocr.py ${FUNCTION_DIR}/.
COPY infer_ocr_app.py ${FUNCTION_DIR}/.
COPY imaug/ ${FUNCTION_DIR}/imaug/
COPY postprocess/ ${FUNCTION_DIR}/postprocess/



RUN pip3 install --target ${FUNCTION_DIR} awslambdaric
WORKDIR ${FUNCTION_DIR}

RUN wget -c https://simens.s3.amazonaws.com/models/yolov5s.zip -O ${FUNCTION_DIR}/yolov5s.zip
RUN cd ${FUNCTION_DIR} && unzip yolov5s.zip && rm -rf yolov5s.zip

# get OCR model
RUN mkdir -p ${FUNCTION_DIR}/model/standard
RUN wget -c https://aws-gcr-solutions-assets.s3.cn-northwest-1.amazonaws.com.cn/ai-solution-kit/infer-ocr-model/standard/v1.0.0/classifier.onnx -O ${FUNCTION_DIR}/model/standard/classifier.onnx
RUN wget -c https://aws-gcr-solutions-assets.s3.cn-northwest-1.amazonaws.com.cn/ai-solution-kit/infer-ocr-model/standard/v1.0.0/det_standard.onnx -O ${FUNCTION_DIR}/model/standard/det_standard.onnx
RUN wget -c https://aws-gcr-solutions-assets.s3.cn-northwest-1.amazonaws.com.cn/ai-solution-kit/infer-ocr-model/standard/v1.0.0/keys_v1.txt -O ${FUNCTION_DIR}/model/standard/keys_v1.txt
RUN wget -c https://aws-gcr-solutions-assets.s3.cn-northwest-1.amazonaws.com.cn/ai-solution-kit/infer-ocr-model/standard/v1.0.0/rec_standard.onnx -O ${FUNCTION_DIR}/model/standard/rec_standard.onnx

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PYTHONIOENCODING="utf8"

ENTRYPOINT [ "python3", "-m", "awslambdaric" ]
CMD [ "main.handler" ]

