from fastapi import FastAPI, File, UploadFile, Form
from config import *
from fastapi.responses import JSONResponse
from src.models.DeepLabv3.model import get_model
from PIL import Image
import io
import base64
import torch
import os
from src.big_image import process_image
from src.data.utils import get_color_mask, calc_percentage

app = FastAPI()
model = get_model(SAVE_DIR + 'DeepLabv3_epoch_24.pth')

@app.post("/process-land/")
async def process_land(
    image: UploadFile = File(..., description="Upload an image of the land"),
    width: float = Form(..., gt=0, description="Width of the land area"),
    height: float = Form(..., gt=0, description="Height of the land area")
):
    image_data = await image.read()
    img = Image.open(io.BytesIO(image_data))

    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mask = process_image(img, (width, height), model.to(device), device)

    class_percentage = calc_percentage(mask)

    img_np = get_color_mask(mask)
    img = Image.fromarray(img_np)

    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    encoded_image = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
    
    return JSONResponse(content={
        "processed_image": f"data:image/jpeg;base64,{encoded_image}",
        "growing_land": width * height * class_percentage[1].item(),
        "resting_land": width * height * class_percentage[0].item()
    })