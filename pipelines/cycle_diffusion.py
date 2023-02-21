import torch
from diffusers import CycleDiffusionPipeline, DDIMScheduler
from config import NUM_IMAGES_PER_PROMPT, NUM_INFERENCE_STEPS, DIFFUSION_PATH
from PIL.Image import Image
from typing import List
import deepspeed

scheduler = DDIMScheduler.from_pretrained(DIFFUSION_PATH, subfolder="scheduler")
pipe = CycleDiffusionPipeline.from_pretrained(
    DIFFUSION_PATH, torch_dtype=torch.float16, scheduler=scheduler
)
deepspeed.init_inference(
    model=getattr(pipe,"model", pipe),      # Transformers models
    mp_size=1,        # Number of GPU
    dtype=torch.float16, # dtype of the weights (fp16)
    replace_method="auto", # Lets DS autmatically identify the layer to replace
    replace_with_kernel_inject=True, # replace the model with the kernel injector
)



def edit(source_image: Image, source_prompt: str, target_prompt: str) -> List[Image]:
    images: List[Image] = pipe(
        prompt=target_prompt,
        source_prompt=source_prompt,
        image=source_image,
        num_inference_steps=NUM_INFERENCE_STEPS,
        eta=0.1
    ).images
    return images
