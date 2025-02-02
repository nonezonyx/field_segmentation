import requests
from config import *
import base64

url = "http://localhost:8000/process-land/"

files = {'image': open(RAW_TEST_IMAGES_DIR + 'IMG_TEST.png', 'rb')}
data = {'width': 22, 'height': 12}

response = requests.post(url, files=files, data=data)
result = response.json()

image_data = base64.b64decode(result['processed_image'].split(',')[1])
with open('processed_image.jpg', 'wb') as f:
    f.write(image_data)

print(f"Growing Land: {result['growing_land']}")
print(f"Resting Land: {result['resting_land']}")