import torch
import torch.nn.functional as F
from torchvision import transforms

def process_image(image, image_size_km, model, device='cuda'):
    # image is a PIL Image
    image = image.convert('RGB')
    image = transforms.ToTensor()(image)
    image = image.to(device)
    C, H, W = image.shape
    width_km, height_km = image_size_km
    km_per_pixel_x = width_km / W
    km_per_pixel_y = height_km / H

    tile_size_km = 2.1
    x_steps_km = torch.arange(0, width_km + tile_size_km, tile_size_km)
    y_steps_km = torch.arange(0, height_km + tile_size_km, tile_size_km)
    
    x_steps_pix = (x_steps_km / km_per_pixel_x).round().long().clamp(max=W)
    y_steps_pix = (y_steps_km / km_per_pixel_y).round().long().clamp(max=H)

    full_mask = torch.zeros((3, H, W), device=device)
    
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        for i in range(len(y_steps_pix) - 1):
            for j in range(len(x_steps_pix) - 1):
                y_start, y_end = y_steps_pix[i], y_steps_pix[i+1]
                x_start, x_end = x_steps_pix[j], x_steps_pix[j+1]
                
                if y_end <= y_start or x_end <= x_start:
                    continue
                
                tile = image[:, y_start:y_end, x_start:x_end]
                tile_resized = F.interpolate(tile.unsqueeze(0), size=(512, 512), 
                                             mode='bilinear', align_corners=False)
                
                mask_pred = model(tile_resized)
                if not torch.is_tensor(mask_pred):
                    mask_pred = mask_pred['out']
                mask_resized = F.interpolate(mask_pred, size=(y_end - y_start, x_end - x_start),
                                             mode='bilinear', align_corners=False)
                
                full_mask[:, y_start:y_end, x_start:x_end] = mask_resized.squeeze(0)
    
    return torch.argmax(full_mask.cpu(), dim=0)