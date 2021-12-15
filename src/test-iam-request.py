import base64
import json
import requests
from aws_requests_auth.boto_utils import BotoAWSRequestsAuth

host_id = "2si6jze1q4"
auth = BotoAWSRequestsAuth(
    aws_host=host_id + ".execute-api.cn-northwest-1.amazonaws.com.cn",
    aws_region="cn-northwest-1",
    aws_service="execute-api",
)


def main():
    with open("../test_imgs/IMG_SIMENS_0001.jpg", "rb") as f:  # 转为二进制格式
        base64_data = base64.b64encode(f.read())  # 使用base64 编码
        url = (
            "https://"
            + host_id
            + ".execute-api.cn-northwest-1.amazonaws.com.cn/prod/detect"
        )
        payload = json.dumps(
            {
                "ocr": True,
                "detection": True,
                "image_base64_enc": str(base64_data, encoding="utf-8"),
            }
        )
        headers = {"Content-Type": "application/json"}
        print("----------------sending request----------------")
        response = requests.request(
            "POST", url, headers=headers, data=payload, auth=auth
        )
        print(json.loads(response.text))
        print("----------------finished----------------")


if __name__ == "__main__":
    main()
