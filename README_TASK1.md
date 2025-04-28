Image Denoising with Autoencoders
This project is about training a Convolutional Autoencoder for the task of image denoising. The autoencoder is trained to map noisy images to clean images.

Requirements
Python 3.6 or above
PyTorch
torchvision
PIL
numpy
matplotlib
Usage
Initialize the model:
Train the model:
The model is trained with clean JPEG files from the train folder and noisy JPEG files from the train_noisy folder.

Save the model:
After training, the model is saved to a file named model_fin.pth.

Test the model:
Load the trained model and test it with your validation data. The model will print a comparison between the noisy and the denoised images in the results folder.

Model Architecture
The model is a Convolutional Autoencoder with 4 layers in the encoder and 4 layers in the decoder. The encoder downsamples the input image while the decoder upsamples and tries to recreate the original image.

Dataset
The dataset used for training and validation should be placed in ./train, ./train_noisy, ./val, ./val_noisy directories respectively. The dataset is not included in this repository.

Results
The model saves the denoised images in the ./rez_fin directory. It also calculates and prints the average PSNR (Peak Signal-to-Noise Ratio) for each batch of validation data.

Note
This is a basic implementation of an image denoising autoencoder. For more complex and noisy images, you might need a more complex model or more training epochs.