import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DenoiseAutoEncoder(nn.Module):
    def __init__(self):
        super(DenoiseAutoEncoder, self).__init__()
        #encoder
        self.enc1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.enc4 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.enc5 = nn.Conv2d(8, 4, kernel_size=3, padding=1)  # Added this layer
        self.pool = nn.MaxPool2d(2, 2)
        
        #decoder
        self.dec1 = nn.Conv2d(4, 8, kernel_size=3, padding=1)  # Added this layer
        self.dec2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.dec3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.dec4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Add an additional upsampling step in the decoder
        self.dec5 = nn.Conv2d(64, 128, kernel_size=4, padding=1)
        self.out = nn.Conv2d(128, 3, kernel_size=4, padding=1)
        
    def forward(self, x):
        #encode
        x = F.relu(self.enc1(x))
        x = self.pool(x)
        x = F.relu(self.enc2(x))
        x = self.pool(x)
        x = F.relu(self.enc3(x))
        x = self.pool(x)
        x = F.relu(self.enc4(x))
        x = self.pool(x)
        x = F.relu(self.enc5(x))
        x = self.pool(x)  # the latent space representation
        
        #decode
        x = F.upsample(x, scale_factor=2, mode='nearest')
        x = F.relu(self.dec1(x))
        x = F.upsample(x, scale_factor=2, mode='nearest')
        x = F.relu(self.dec2(x))
        x = F.upsample(x, scale_factor=2, mode='nearest')
        x = F.relu(self.dec3(x))
        x = F.upsample(x, scale_factor=2, mode='nearest')
        x = F.relu(self.dec4(x))
        # Add an additional upsampling step in the decoder
        x = F.upsample(x, scale_factor=2, mode='nearest')
        x = F.relu(self.dec5(x))
        x = F.sigmoid(self.out(x))
        
        return x

# Check if CUDA is available and set PyTorch to use GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

# Dataset
class DenoisingDataset(torch.utils.data.Dataset):
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

# PSNR function
def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

# Transform
transform = transforms.Compose([
    transforms.Resize((16, 16)),
    transforms.ToTensor(),
])

# Datasets and dataloaders
train_data = DenoisingDataset(clean_dir='./train', noisy_dir='./train_noisy', transform=transform)
val_data = DenoisingDataset(clean_dir='./val', noisy_dir='./val_noisy', transform=transform)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False)

# Model, loss function, optimizer
model = DenoiseAutoEncoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
n_epochs = 10
for epoch in range(1, n_epochs+1):
    train_loss = 0.0
    for data in train_loader:
        noisy_images, clean_images = data
        noisy_images, clean_images = noisy_images.to(device), clean_images.to(device)
        optimizer.zero_grad()
        outputs = model(noisy_images)
        loss = criterion(outputs, clean_images)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*noisy_images.size(0)
    train_loss = train_loss/len(train_loader)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
    import datetime
    print(datetime.datetime.now())

# Testing
from matplotlib import pyplot as plt
model.eval()
with torch.no_grad():
    for i, data in enumerate(val_loader):
        noisy_images, clean_images = data
        noisy_images, clean_images = noisy_images.to(device), clean_images.to(device)
        outputs = model(noisy_images)
        psnrs = []
        for j in range(len(outputs)):
            clean_image = clean_images[j].cpu().numpy()
            denoised_image = outputs[j].cpu().numpy()
            psnrs.append(psnr(clean_image, denoised_image))
        print(f'Image {i}, PSNR: {np.mean(psnrs)}')
