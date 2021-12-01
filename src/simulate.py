import base64
import requests
import time
import json


def get_base64_encoding(full_path):
    with open(full_path, "rb") as f:
        data = f.read()
        image_base64_enc = base64.b64encode(data)
        image_base64_enc = str(image_base64_enc, 'utf-8')

    return image_base64_enc


if __name__ == '__main__':
    endpoint = 'https://q3kvov2kc0.execute-api.cn-north-1.amazonaws.com.cn/prod/detect'
    image_base64_enc = get_base64_encoding(full_path="../test_imgs/IMG_SIMENS_0001.jpg")

    request_body = {
        "image_base64_enc": image_base64_enc
    }

    t1 = time.time()
    response = requests.post(endpoint, data=json.dumps(request_body))
    t2 = time.time()
    print('Time cost = {}'.format(t2 - t1))

    # Step 3: visualization
    response = json.loads(response.text)
    print('Response = {}'.format(response))
