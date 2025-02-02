import xml.etree.ElementTree as ET
import cv2
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path


def create_masks(annotations_path, images_dir, masks_dir, color_mapping):
    os.makedirs(masks_dir, exist_ok=True)
    tree = ET.parse(annotations_path)
    root = tree.getroot()

    for image_elem in root.findall('image'):
        image_name = image_elem.get('name')
        width = int(image_elem.get('width'))
        height = int(image_elem.get('height'))
        mask = np.zeros((height, width, 3), dtype=np.uint8)
        mask[:, :, 0] = 255
        for polygon_elem in image_elem.findall('polygon'):
            label = polygon_elem.get('label')
            points_str = polygon_elem.get('points')
            assert(label in color_mapping)
            points = []
            for pair in points_str.split(';'):
                if pair.strip():
                    x, y = map(float, pair.split(','))
                    points.append([x, y])
            assert(len(points) >= 3)
            pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], color_mapping[label])
            
        mask_filename = os.path.splitext(image_name)[0] + '.png'
        cv2.imwrite(os.path.join(masks_dir, mask_filename), mask)


def split_and_save(image, mask, counter, images_dir, masks_dir, num_rows, num_cols):
    assert(image.shape[:2] == mask.shape[:2])
    img_height, img_width = image.shape[:2]
    
    patch_height = img_height // num_rows
    patch_width = img_width // num_cols

    for i in range(num_rows):
        for j in range(num_cols):
            y_start = i * patch_height
            y_end = (i + 1) * patch_height if i < num_rows - 1 else img_height
            x_start = j * patch_width
            x_end = (j + 1) * patch_width if j < num_cols - 1 else img_width

            img_patch = image[y_start:y_end, x_start:x_end]
            mask_patch = mask[y_start:y_end, x_start:x_end]

            cv2.imwrite(os.path.join(images_dir, f"IMG_{counter}.png"), img_patch)
            cv2.imwrite(os.path.join(masks_dir, f"IMG_{counter}.png"), mask_patch)
            counter += 1
            
    return counter


# 7 and 15 is values to cut images to aproximately 3 by 3 km
def slice_images(raw_images_dir, raw_masks_dir, images_dir, masks_dir, num_rows=7, num_cols=15):
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    counter = 0
    image_paths = [p for p in Path(raw_images_dir).iterdir() if p.is_file()]
    
    for img_path in tqdm(image_paths, desc="Processing images"):
        mask_path = Path(raw_masks_dir) / img_path.name
        
        if not mask_path.exists():
            print(f"Mask missing for {img_path.name}, skipping")
            continue
            
        image = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path))
        
        if image is None or mask is None:
            print(f"Failed to read {img_path.name} or its mask, skipping")
            continue
            
        if image.shape[:2] != mask.shape[:2]:
            print(f"Size mismatch in {img_path.name}, skipping")
            continue

        counter = split_and_save(image, mask, counter, images_dir, masks_dir, num_rows, num_cols)

    print(f"Created {counter} image-mask pairs")