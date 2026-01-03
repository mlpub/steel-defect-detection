from PIL import Image
import base64
from io import BytesIO
import json
import requests
import sys


if len(sys.argv) > 1:
    IMAGE_NAME = sys.argv[1]
else:
    print("Argument missing: image file name")
    sys.exit(1)

with Image.open(IMAGE_NAME) as img:
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

url = "https://steel-defect-app-901723644506.asia-southeast1.run.app/predict"
headers = {"Content-Type": "application/json"}
data = json.dumps({"image": img_base64})

response = requests.post(url, headers=headers, data=data)
print("Status code:", response.status_code)
print("Raw response:", response.text)
# Uncomment below to try parsing JSON if response looks correct
# try:
#     print("Response:", response.json())
# except Exception as e:
#     print("JSON decode error:", e)
