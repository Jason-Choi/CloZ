import torch
from diffusers import StableDiffusionPipeline
from config import DEVICE_ID, NUM_IMAGES_PER_PROMPT, NUM_INFERENCE_STEPS, DIFFUSION_PATH
from PIL.Image import Image
from typing import List
import deepspeed

pipe = StableDiffusionPipeline.from_pretrained(
    DIFFUSION_PATH, torch_dtype=torch.float16
)
deepspeed.init_inference(
    model=getattr(pipe,"model", pipe),      # Transformers models
    mp_size=1,        # Number of GPU
    dtype=torch.float16, # dtype of the weights (fp16)
    replace_method="auto", # Lets DS autmatically identify the layer to replace
    replace_with_kernel_inject=True, # replace the model with the kernel injector
)



def generate(prompt: str) -> List[Image]:
    images: List[Image] = pipe(
        prompt=prompt,
        num_images_per_prompt=NUM_IMAGES_PER_PROMPT,
        num_inference_steps=NUM_INFERENCE_STEPS
    ).images
    return images
