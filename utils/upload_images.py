from PIL.Image import Image
from typing import List
import boto3
from botocore.client import Config as S3Config
from datetime import datetime
import io
from config import NEXT_PUBLIC_S3_ACCESS_KEY_ID, NEXT_PUBLIC_S3_BUCKET, NEXT_PUBLIC_S3_SECRET_ACCESS_KEY

s3_config = S3Config(signature_version="s3v4")

s3 = boto3.resource(
    "s3",
    aws_access_key_id=NEXT_PUBLIC_S3_ACCESS_KEY_ID,
    aws_secret_access_key=NEXT_PUBLIC_S3_SECRET_ACCESS_KEY,
    config=s3_config
)


def get_timestamp() -> str:
    now = datetime.now()
    return now.strftime("%Y%m%d%H%M%S%f")[:-5]


def upload_images(images: List[Image]) -> List[str]:
    filenames: List[str] = []
    for i, image in enumerate(images):
        byte_IO = io.BytesIO()
        image.save(byte_IO, format="JPEG")
        byte_array = byte_IO.getvalue()

        filename = f"{get_timestamp()}{i}.jpg"
        filenames.append(filename)

        s3.Bucket(
            NEXT_PUBLIC_S3_BUCKET
        ).put_object(
            Key=f"assets/{filename}",
            Body=byte_array,
            ContentType=f"image/jpg"
        )
    return filenames
