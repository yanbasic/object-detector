FROM public.ecr.aws/lambda/python

RUN apt-get -y update && \
    apt-get yum -y libdmtx0a \
    wget \
    unzip \
    python3-pip

RUN pip3 install --upgrade pip==21.3.1
RUN pip3 install --no-cache  torch==1.7.1 \
    numpy==1.19.5 \
    torchvision==0.8.2 \
    pylibdmtx==0.1.9 \
    opencv-python==4.5.4.60

COPY main.py ./
COPY utils.py ./

RUN mkdir -p /opt/ml/model
RUN wget -c https://simens.s3.amazonaws.com/models/yolov5s.zip -O /opt/ml/model/yolov5s.zip
RUN cd /opt/ml/model/ && unzip yolov5s.zip && rm -rf yolov5s.zip

CMD [ "main.handler" ]
