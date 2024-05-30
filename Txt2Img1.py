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
#from IPython.display import display

model_sd = "runwayml/stable-diffusion-v1-5"
output_dir = "ImgOutput/"

# Create a Stable Diffusion pipeline
#pipeline = StableDiffusionPipeline()
#pipeline = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4').to('cuda')
pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
