from fastapi import FastAPI

from config import *
from items import EditItem, GenerateItem, ResponseItem
from pipelines import pipe_edit, pipe_generate
from utils import upload_images, get_image
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/v1/generate", response_model=ResponseItem)
async def generate(item: GenerateItem):
    return {"images": [
        "20230127220712340.jpg",
        "20230127220712341.jpg",
        "20230127220712342.jpg",
        "20230127220712343.jpg",
        "20230127220712344.jpg",
        "20230127220712345.jpg",
    ]}


@app.post("/v1/edit", response_model=ResponseItem)
async def edit(item: EditItem):
    return {"images": [
        "20230127220712340.jpg"
    ]}


@app.post("/v2/generate", response_model=ResponseItem)
async def generatev2(item: GenerateItem):
    images = pipe_generate(item.get_prompt())
    filenames = upload_images(images)
    return {"images": filenames}


@app.post("/v2/edit", response_model=ResponseItem)
async def editv2(item: EditItem):
    source_image = get_image(item.image)
    images = pipe_edit(source_image, item.source.get_prompt(),
                       item.target.get_prompt())
    filenames = upload_images(images)
    return {"images": filenames}
