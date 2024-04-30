import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np

class VineyardDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        # Prepare list of files, checking if both image and mask exist
        self.images = []
        for image in os.listdir(images_dir):
            if image.endswith('.png'):
                img_path = os.path.join(images_dir, image)
                mask_name = image.replace('.png', '_instanceIds.png')
                mask_path = os.path.join(masks_dir, mask_name)
                if os.path.exists(mask_path):  # Check if mask file exists
                    self.images.append(img_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_name = os.path.basename(img_path).replace('.png', '_instanceIds.png')
        mask_path = os.path.join(self.masks_dir, mask_name)

        #try:
        #    image = Image.open(img_path).convert("RGB")
        #    mask = Image.open(mask_path).convert("L")
        #except FileNotFoundError:
        #    return None  # Return None if there's an issue opening the file

        image = np.array(Image.open(img_path).convert("RGB")) # we are using np array since we will be using Albumentations library which req np array
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform:
            augmentations = self.transform(image = image, mask = mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask
