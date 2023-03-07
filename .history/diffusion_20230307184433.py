#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
prompt_slug = prompt.replace(" ", "_")
image = pipe(prompt).images[0]  
    
image.save(f"{prompt_slug}.png")