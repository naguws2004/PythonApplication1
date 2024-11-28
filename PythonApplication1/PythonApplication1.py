from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
from diffusers import DiffusionPipeline
import gradio as gr

pipe = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")

def img2img(image, prompt, strength):
    images = pipe(prompt=prompt, image=image, strength=strength, guidance_scale=7.5).images
    return images[0]

iface = gr.Interface(
    fn=img2img,
    inputs=[
        gr.Image(type="pil"), 
        gr.Textbox(label="Prompt"),
        gr.Slider(label="Strength", minimum=0.0, maximum=1.0, value=0.75, step=0.05),
    ],
    outputs=gr.Image(type="pil"),
    title="Stable Diffusion img2img",
    description="Modify an image using Stable Diffusion",
)

iface.launch(share=True)
