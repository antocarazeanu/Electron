import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from model import DenoiseAutoEncoder
from dataset import DenoisingDataset

# Check if CUDA is available and set PyTorch to use GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Load the datasets
train_data = DenoisingDataset(clean_dir='./train', noisy_dir='./train_noisy', transform=transform)
val_data = DenoisingDataset(clean_dir='./val', noisy_dir='./val_noisy', transform=transform)

# Create the dataloaders
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
val_loader = DataLoader(val_data, batch_size=8, shuffle=False)

# Initialize the model
model = DenoiseAutoEncoder()
model = model.to(device)

# Loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
def train_model(model, criterion, optimizer, n_epochs):
    model.train()
    for epoch in range(n_epochs):
        train_loss = 0.0
        for data in train_loader:
            noisy_images, clean_images = data
            optimizer.zero_grad()
            outputs = model(noisy_images.to(device))
            loss = criterion(outputs, clean_images.to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * noisy_images.size(0)
        train_loss = train_loss / len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

# Test the model
def test_model(model, val_loader):
    model.eval()
    for i, data in enumerate(val_loader):
        noisy_images, clean_images = data
        outputs = model(noisy_images.to(device))
        # ... (your testing loop)

if __name__ == "__main__":
    # Train the model
    train_model(model, criterion, optimizer, n_epochs=20)

    # Save the model
    torch.save(model.state_dict(), 'model_fin.pth')

    # Test the model
    test_model(model, val_loader)
