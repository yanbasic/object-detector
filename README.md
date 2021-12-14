## 基于YOLO-V5二维码目标检测与识别


### 1. 基于Lambda容器镜像的解决方案部署

用户可以按照下述几个步骤进行解决方案部署：

#### 步骤一：启动EC2 (OS为Ubuntu 18.04, 机型为t3.xlarge)，安装Docker依赖项
```angular2html
sudo apt-get update && 
sudo apt-get install -y git unzip zip awscli ca-certificates curl gnupg lsb-release && 
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg &&
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null &&
sudo apt-get update &&
sudo apt-get install -y docker-ce docker-ce-cli containerd.io &&
sudo chmod 666 /var/run/docker.sock
```


#### 步骤二：安装nodejs
```angular2html
sudo apt-get update &&
sudo apt-get -y upgrade &&
curl -fsSL https://deb.nodesource.com/setup_16.x | sudo bash - &&
sudo apt-get install -y nodejs
```

#### 步骤三：配置AWS用户
在命令行aws configure，输入AWS账号的的 Access Key ID，Secret Access Key字段，
Default output format [None]字段输入您所在的region，如cn-north-1。

#### 步骤四：克隆代码并基于CDK进行部署
```angular2html
git clone https://github.com/gaowexu/object-detector.git
cd object-detector/solution
npm install
npm run cdk deploy -- --require-approval never
```


### REST API Reference

- HTTP 方法: `POST`

- Body 参数

| **Name**  | **Type**  | **Optional** |  **Description**  |
|----------|-----------|------------|------------|
|ocr       |*bool*          |No | 是否包含目标检测结果，默认为false |
|detection |*bool*          |No | 是否包含OCR文本识别结果，默认为false |
|image_base64_enc |*String* |No |Base64-encoded image data|

- 请求 Body 示例

``` json
{
    "ocr": true,
    "detection": true,
    "image_base64_enc": "/9j/4AAQSkZJRgABAQEAeAB4AAD..."
}
```

- 返回 示例

``` json
{
    "ocr": [
        {
            "words": "xxxx",
            "location": [
                295,
                932,
                472,
                99
            ],
            "confidence": 0.9855925440788269
        }
    ],
    "detection": {
        "channels": 3,
        "height": 3225,
        "width": 2419,
        "detections": [
            {
                "bbox": [
                    876,
                    1615,
                    1141,
                    1891
                ],
                "confidence": 0.9268702268600464,
                "cls_name": "xxx",
                "data_matrix_code": "xxx"
            }
        ]
    },
    "duration": 0.8072800636291504
}
```



