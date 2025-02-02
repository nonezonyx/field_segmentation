import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms.functional as TF
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random


def create_dataloaders(full_dataset, batch_size=4, num_workers=4, test_ratio=0.2):
    test_size = int(len(full_dataset) * test_ratio)
    train_size = len(full_dataset) - test_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True
    )
    
    return train_loader, test_loader


def show_dataloader_examples(dataloader, num_examples=5):
    images, masks = next(iter(dataloader))
    num_examples = min(num_examples, len(images))
    fig, axes = plt.subplots(num_examples, 2, figsize=(10, 3*num_examples))
    
    for i in range(num_examples):
        img = images[i].numpy().transpose(1, 2, 0) 
        img = (img * 255).astype(np.uint8)

        mask = masks[i].numpy()
        mask = np.transpose((mask * 255).astype(np.uint8), (1, 2, 0))
        
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Image {i+1}")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mask)
        axes[i, 1].set_title(f"Mask {i+1}")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()


def show_prediction(dataloader, model):
    images, masks = next(iter(dataloader))

    image = images[0]
    true_mask = masks[0]

    show_prediction_image(image, model, true_mask)


def show_prediction_image(image, model, true_mask=None):
    model.eval()

    device = next(model.parameters()).device
    image_tensor = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        if not torch.is_tensor(output):
            output = output['out']
        pred_mask = torch.argmax(output, dim=1).squeeze(0)
        
    img_np = image.numpy().transpose(1, 2, 0)
    img_np = (img_np * 255).astype(np.uint8)

    if true_mask is not None:
        true_np = true_mask.numpy()
        true_np = np.transpose((true_np * 255).astype(np.uint8), (1, 2, 0))

    class_colors = np.array([
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255]
    ], dtype=np.uint8)

    pred_mask_np = pred_mask.cpu().numpy()
    color_mask = class_colors[pred_mask_np]
    
    fig, axes = plt.subplots(1, 2 + (true_mask is not None), figsize=(15, 5))

    axes[0].imshow(img_np)
    axes[0].set_title("Input Image")
    axes[0].axis('off')

    axes[1].imshow(color_mask)
    axes[1].set_title("Predicted Mask")
    axes[1].axis('off')

    if true_mask is not None:
        axes[2].imshow(true_np)
        axes[2].set_title("Ground Truth Mask")
        axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_history(history):
    if 'train_loss' not in history or 'val_loss' not in history:
        print("The history dictionary must contain 'train_loss' and 'val_loss' keys.")
        return

    train_loss = history['train_loss']
    val_loss = history['val_loss']

    epochs = range(1, len(train_loss) + 1)

    plt.plot(epochs, train_loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')

    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


def get_color_mask(mask):
    class_colors = np.array([
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255]
    ], dtype=np.uint8)
    
    mask = mask.cpu().numpy()
    return class_colors[mask]


def show_mask(mask):
    plt.imshow(get_color_mask(mask))


def show_pair(img1, img2):
    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.subplot(1, 2, 2)
    plt.imshow(img2)


def show_image_mask(image, mask):
    show_pair(image, get_color_mask(mask))


def calc_percentage(mask):
    mask = mask.flatten()
    class_counts = torch.bincount(mask, minlength=3)
    class_counts = class_counts / mask.numel()
    return class_counts
    
    