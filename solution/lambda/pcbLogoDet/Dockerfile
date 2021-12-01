FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive
ARG FUNCTION_DIR="/opt/program"

RUN apt-get -y update && \
    apt-get install -y libdmtx0a \
    wget \
    unzip \
    python3 \
    python3-pip \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip==21.3.1
RUN pip3 install --no-cache  torch==1.7.1 \
    numpy==1.19.5 \
    torchvision==0.8.2 \
    pylibdmtx==0.1.9

RUN mkdir -p ${FUNCTION_DIR}
COPY main.py ${FUNCTION_DIR}/.
COPY utils.py ${FUNCTION_DIR}/.
RUN pip3 install --target ${FUNCTION_DIR} awslambdaric
WORKDIR ${FUNCTION_DIR}

RUN wget -c https://simens.s3.amazonaws.com/models/yolov5s.zip -O ${FUNCTION_DIR}/yolov5s.zip
RUN cd ${FUNCTION_DIR} && unzip yolov5s.zip && rm -rf yolov5s.zip

ENTRYPOINT [ "python3", "-m", "awslambdaric" ]
CMD [ "main.handler" ]

