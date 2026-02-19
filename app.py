import os
import torch
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from diffusers import AutoPipelineForText2Image

# Ensure images folder exists
os.makedirs("images", exist_ok=True)

app = FastAPI()

# Static files (to serve generated images)
app.mount("/images", StaticFiles(directory="images"), name="images")
templates = Jinja2Templates(directory="templates")

# Load SDXL Turbo (CPU)
pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float32
).to("cpu")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "image_url": None}
    )

@app.post("/generate", response_class=HTMLResponse)
async def generate_image(
    request: Request,
    prompt: str = Form(...)
):
    image = pipe(
        prompt=prompt,
        num_inference_steps=2,
        guidance_scale=0.0
    ).images[0]

    output_path = "images/output.png"
    image.save(output_path)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "image_url": f"/images/output.png",
            "prompt": prompt
        }
    )
