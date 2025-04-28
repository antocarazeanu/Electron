import os
import torch
from torchvision import transforms
from PIL import Image
from model import DenoiseAutoEncoder  # Update with the correct file name

def denoise_image(model, input_path, output_path):
    # Load and preprocess the input image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    input_image = Image.open(input_path).convert('RGB')
    input_tensor = transform(input_image).unsqueeze(0).to(device)

    # Denoise the image using the model
    with torch.no_grad():
        denoised_tensor = model(input_tensor)

    # Convert the output tensor to a PIL Image and resize it to the original image size
    denoised_image = transforms.ToPILImage()(denoised_tensor.cpu().squeeze(0))
    denoised_image = denoised_image.resize(input_image.size)

    # Save the denoised image
    denoised_image.save(output_path)

# Load the trained model
model = DenoiseAutoEncoder()
model.load_state_dict(torch.load('model_fin.pth'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Denoise each image in the 'images/noisy' directory
noisy_dir = 'images/noisy'
output_dir = 'images/denoised'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for filename in os.listdir(noisy_dir):
    if filename.endswith(".jpg"):
        input_path = os.path.join(noisy_dir, filename)
        output_path = os.path.join(output_dir, f'denoised_{filename}')
        denoise_image(model, input_path, output_path)

print("Denoising completed.")
