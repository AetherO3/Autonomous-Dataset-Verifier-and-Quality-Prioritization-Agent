from PIL import Image
import requests
from io import BytesIO

def load_image_from_url(url: str):
    response = requests.get(url)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")