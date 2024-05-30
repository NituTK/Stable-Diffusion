# Install required libraries
#pip install torch==2.2.1, transformers diffusers, accelerate, xformers, matplotlib;
# Import necessary modules

import torch
import json
import os
import random
import matplotlib.pyplot as plt

from torch import autocast
from diffusers import StableDiffusionPipeline, DDIMScheduler, EulerDiscreteScheduler
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import StableDiffusionXLPipeline
from PIL import Image
#from IPython.display import display

model_sd = "runwayml/stable-diffusion-v1-5"
output_dir = "ImgOutput/"

# Create a Stable Diffusion pipeline
#pipeline = StableDiffusionPipeline()
#pipeline = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4').to('cuda')
pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
#,torch_dtype=torch.float16)
pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)

def grid_img(imgs, rows=1, cols=3, scale=1):
  assert len(imgs) == rows * cols

  w, h = imgs[0].size
  w, h = int(w * scale), int(h * scale)

  grid = Image.new('RGB', size = (cols * w, rows * h))
  grid_w, grid_h = grid.size

  for i, img in enumerate(imgs):
    img = img.resize((w, h), Image.LANCZOS)
    grid.paste(img, box=(i % cols * w, i // cols * h))
  return grid

num_imgs = 3
prompt = 'photograph of an old car'
imgs = pipeline(prompt, num_images_per_prompt=num_imgs).images
grid = grid_img(imgs, rows=1, cols=3, scale=0.75)
grid.save("carrslt.png")
