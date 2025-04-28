import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset

class DenoisingDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir, transform=None):
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.transform = transform

        self.image_files = os.listdir(clean_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        clean_image = Image.open(os.path.join(self.clean_dir, self.image_files[idx])).convert('RGB')
        noisy_image = Image.open(os.path.join(self.noisy_dir, self.image_files[idx])).convert('RGB')

        if self.transform:
            clean_image = self.transform(clean_image)
            noisy_image = self.transform(noisy_image)

        return noisy_image, clean_image
