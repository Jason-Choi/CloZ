import torch
from diffusers import CycleDiffusionPipeline, DDIMScheduler
from config import DEVICE_ID, NUM_IMAGES_PER_PROMPT, NUM_INFERENCE_STEPS, DIFFUSION_PATH
from PIL.Image import Image
from typing import List


scheduler = DDIMScheduler.from_pretrained(DIFFUSION_PATH, subfolder="scheduler")
pipe = CycleDiffusionPipeline.from_pretrained(
    DIFFUSION_PATH, torch_dtype=torch.float16, scheduler=scheduler
)
pipe.to(f"cuda:{DEVICE_ID}")



def edit(source_image: Image, source_prompt: str, target_prompt: str) -> List[Image]:
    images: List[Image] = pipe(
        prompt=target_prompt,
        source_prompt=source_prompt,
        image=source_image,
        num_inference_steps=NUM_INFERENCE_STEPS,
        eta=0.1
    ).images
    return images
