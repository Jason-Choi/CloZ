from config import S3_URL
from requests import get
from PIL import Image
from io import BytesIO


def get_image(image_id: str) -> Image.Image:
    image_url = f"{S3_URL}/{image_id}"
    res = get(image_url)
    image = Image.open(BytesIO(res.content)).convert("RGB")
    return image
