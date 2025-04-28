import torch.nn as nn
import torch.nn.functional as F

class DenoiseAutoEncoder(nn.Module):
    def __init__(self):
        super(DenoiseAutoEncoder, self).__init__()
        # Encoder
        self.enc1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.enc4 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        # Remove one layer from the encoder
        self.pool = nn.MaxPool2d(2, 2)

        # Decoder
        # Remove one layer from the decoder
        self.dec1 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.dec2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.dec3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Add an additional upsampling step in the decoder
        self.dec4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.out = nn.Conv2d(128, 3, kernel_size=3, padding=1)

    def forward(self, x):
        # Encode
        x = F.relu(self.enc1(x))
        x = self.pool(x)
        x = F.relu(self.enc2(x))
        x = self.pool(x)
        x = F.relu(self.enc3(x))
        x = self.pool(x)
        x = F.relu(self.enc4(x))
        x = self.pool(x)

        # Decode
        x = F.upsample(x, scale_factor=2, mode='nearest')
        x = F.relu(self.dec1(x))
        x = F.upsample(x, scale_factor=2, mode='nearest')
        x = F.relu(self.dec2(x))
        x = F.upsample(x, scale_factor=2, mode='nearest')
        x = F.relu(self.dec3(x))
        x = F.upsample(x, scale_factor=2, mode='nearest')
        # Add an additional upsampling step in the decoder
        x = F.relu(self.dec4(x))
        x = F.sigmoid(self.out(x))

        return x
