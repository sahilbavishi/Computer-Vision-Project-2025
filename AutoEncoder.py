# Authors: Sahil Mehul Bavishi (s2677266) and Matteo Spadaccia (its a me Mario)
# Subject: Computer Vision Coursework. Auto-Encoder 
# Date: 21.03.2025


# Global Constants
genVISUALS = False  # set to False in order to avoid time-consuming visualizations' genaration (images are instead displayed as pre-saved in 'Output/Visuals' folder)
dpi = 500           # dpi for .pdf-saved images visualization (with genVISUALS = False)
rePREPROC = False   # if True, the input images' resizing and augmentation are run, otherwise the saved outcomes are used
random_seed = 42


# IMPORTS
import os
from pathlib import Path
from pdf2image import convert_from_path  # (also install !apt-get install poppler-utils)
from IPython.display import display
import numpy as np
from PIL import Image
import seaborn as sns
import cv2
import tensorflow as tf
import albumentations as A
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import pickle as pkl
from torch.cuda.amp import autocast, GradScaler

if torch.cuda.is_available():
    print("using GPU")
else:
    print("using CPU")

# Loading input data
input_folder_trainval = 'Data/Input/TrainVal'
input_trainval = [f for f in os.listdir(input_folder_trainval+'/color') if f.lower().endswith(('.jpg', '.jpeg'))]
input_trainval_labels = [f for f in os.listdir(input_folder_trainval+'/label') if f.lower().endswith(('.png'))]

input_folder_test = 'Data/Input/Test'
input_test = [f for f in os.listdir(input_folder_test+'/color') if f.lower().endswith(('.jpg', '.jpeg'))]
input_test_labels = [f for f in os.listdir(input_folder_test+'/label') if f.lower().endswith(('.png'))]

output_folder = 'Data/Output'
output_folder_resized = os.path.join(output_folder, 'Resized')
output_folder_augmented = os.path.join(output_folder, 'Augmented')

output_folder_resized_color = os.path.join(output_folder_resized, 'color')
output_folder_resized_label = os.path.join(output_folder_resized, 'label')
output_folder_augmented_color = os.path.join(output_folder_augmented, 'color')
output_folder_augmented_label = os.path.join(output_folder_augmented, 'label')

"""### 1. Dataset preprocessing and augmentation

The images are resized to the dimensions (H<sub>min</sub>, W<sub>min</sub>), thus to take the Q3 height and width size over all the images in the dataset; the instances will be processed in this format, then the output resized back to the original dimensions...

Furthermore...(data augmentation)
"""

# Printing stats
print(f"Input set size: {len(input_trainval)}\n")
print("Q3 width and height values (both 500 pixels) were chosen for resizing.")

"""#### a) Resizing"""

# Resizing images (to Q3 width, Q3 height)
print("Using previously resized images and labels (500x500).")
imgResize = (500, 500)

"""#### b) Augmenting dataset"""
print("Using previously augmented data.")


"""#### c) Preparing datasets"""

# Preparing train, valid and test sets
validSize = 0.2
batchSize = 4
imgChannels = 3 ## this was 3
inputSize = (imgResize[0], imgResize[1], imgChannels)

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Define preprocessed size
preprocessedSize = (256, 256)  # Desired size for input images and masks
inputSize = (256,256,3)

trainval_images = [os.path.join(output_folder_augmented_color, f) for f in os.listdir(output_folder_augmented_color) if f.endswith('.jpg')]
trainval_masks = [os.path.join(output_folder_augmented_label, f) for f in os.listdir(output_folder_augmented_label) if f.endswith('.png')]

train_images, val_images, train_masks, val_masks = train_test_split(trainval_images, trainval_masks, test_size=validSize, random_state=random_seed)

test_images = [os.path.join(input_folder_test, 'color', f) for f in os.listdir(os.path.join(input_folder_test, 'color')) if f.endswith('.jpg')]
test_masks = [os.path.join(input_folder_test, 'label', f) for f in os.listdir(os.path.join(input_folder_test, 'label')) if f.endswith('.png')]


print(f"Train set size: {len(train_images)} ({(1-validSize)*100}%)")
print(f"Valid set size: {len(val_images)} ({(validSize)*100}%)")
print(f" Test set size: {len(test_images)}")
print(f"\nInput dimension: {preprocessedSize + (imgChannels,)}")  # Adjusted for the preprocessed size
print(f"Batches' size: {batchSize}")

classesNum = 4 # number of output classes
epochsNum = 100 # number of training epochs
batchSize = 32



##### DEFINING THE ENCODER PART OF THE AUTO-ENCODER (PRE TRAINING) #####

scaler = GradScaler()
# Define the Autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(0.2),
        nn.MaxPool2d(kernel_size=2, stride=2),  # 500 -> 250
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(0.2),
        nn.MaxPool2d(kernel_size=2, stride=2),  # 250 -> 125
        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(0.2),
        nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),  # Add another layer for a richer encoding
        nn.LeakyReLU(0.2),
    )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    
  # Below is the code to train the autoencoder  
    
'''
# # Custom Dataset for loading images
class ImageDataset(Dataset):
    def __init__(self, image_files, transform=None, max_images=100):
        self.image_files = image_files  # Limit to first 100 images
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# Transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1,1]
])


# Load dataset with first 100 images
dataset = ImageDataset(image_files=train_images, transform=transform, max_images=100)
dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Initialize model, loss, and optimizer
autoencoder = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3) 

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight)  # Xavier initialization
        if m.bias is not None:
            nn.init.zeros_(m.bias)

autoencoder.apply(weights_init)

# Training the Autoencoder
num_epochs = 30
for epoch in range(num_epochs):
    print(f"Epoch [{epoch+1}/{num_epochs}] Running:")
    torch.cuda.empty_cache()
    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()

        with autocast():  # Use autocast only here
            outputs = autoencoder(batch)
            loss = criterion(outputs, batch)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Save the model after each epoch
    torch.save(autoencoder.state_dict(), f'Data/Output/Models/updated_working_autoencoder_epoch_{epoch+1}.pth')
'''


###### NOW THAT THE ENCODER IS PRE-TRAINED, WE CAN CREATE A GOOD SEGMENTER #######

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual  # Skip connection
        return self.relu(x)

# Enhanced Segmentation Decoder
class SegmentationDecoder(nn.Module):
    def __init__(self, encoder):
        super(SegmentationDecoder, self).__init__()
        
        # Partially freeze encoder (freeze first few layers, fine-tune deeper ones)
        # for param in list(encoder.parameters())[:4]:  # Adjust layers to freeze
        #     param.requires_grad = False
        for param in encoder.parameters():
            param.requires_grad = True
        
        self.encoder = encoder

        # Decoder with residual blocks and skip connections
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            ResidualBlock(128, 128),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            ResidualBlock(64, 64),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, classesNum, kernel_size=1)  # Output logits
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



# Custom Dataset for loading images and labels
class SegmentationDataset(Dataset):
    def __init__(self, image_files, label_files, transform=None):
        self.image_files = image_files
        self.label_files = label_files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)


    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label_path = self.image_files[idx].replace('.jpg', '.png').replace('color', 'label')
        image = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('L')  # Load label as grayscale

        if self.transform:
            image = self.transform(image)
            
        label = label.resize((256, 256), Image.NEAREST)

        # Convert label to tensor
        label = torch.tensor(np.array(label), dtype=torch.long)  # Shape: (H, W)

        # # Map pixel values to class indices
        label = torch.where(label == 38, 1, label)
        label = torch.where(label == 75, 2, label)
        label = torch.where(label == 255, 3, label)
        label = torch.where(label == 0, 0, label)  # Ensure 0 stays as class 0



        

        return image, label  # No one-hot encoding




# Transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


# Load segmentation dataset with first 100 images
segmentation_dataset = SegmentationDataset(
    image_files=train_images,
    label_files=train_masks,
    transform=transform
)
segmentation_dataloader = DataLoader(segmentation_dataset, batch_size=batchSize, shuffle=True)

# Load the pre-trained autoencoder
autoencoder = Autoencoder()
autoencoder.load_state_dict(torch.load('Data/Output/Models/updated_working_autoencoder_epoch_30.pth'))

# Initialize the segmentation model
# segmentation_model = SegmentationDecoder(autoencoder.encoder)

# Define loss and optimizer for segmentation
segmentation_model = SegmentationDecoder(autoencoder.encoder)
#class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0])  # Adjust these values as needed
segmentation_criterion = nn.CrossEntropyLoss()#(weight=class_weights)
# Example of class weights (higher weight for less frequent classes)
segmentation_optimizer = optim.Adam(segmentation_model.decoder.parameters(), lr=1e-3, weight_decay=1e-4)

# Training the Segmentation Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
segmentation_model.to(device)
# scheduler = torch.optim.lr_scheduler.StepLR(segmentation_optimizer, step_size=3, gamma=0.5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(segmentation_optimizer, T_max=10, eta_min=1e-5)

scaler = torch.cuda.amp.GradScaler()

num_epochs = 100
for epoch in tqdm(range(num_epochs), desc="Training Progress"):
    print(f"Epoch [{epoch+1}/{num_epochs}] Running:")
    
    # Wrap the dataloader with tqdm for batch progress
    for images, labels in tqdm(segmentation_dataloader, desc=f"Epoch {epoch+1}", leave=False):
        images, labels = images.to(device), labels.to(device)

        segmentation_optimizer.zero_grad()

        # Use autocast for mixed precision training
        with autocast():  
            outputs = segmentation_model(images)  # Forward pass
            loss = segmentation_criterion(outputs, labels)  # Compute loss

        # Scale the loss and backpropagate
        scaler.scale(loss).backward()

        # Step the optimizer with scaled gradients
        scaler.step(segmentation_optimizer)
        
        # Update the scaler for next iteration
        scaler.update()
        
    scheduler.step()

      
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    torch.save(segmentation_model.state_dict(), f'Data/Output/Models/Better_Final_Segmentation_Model_epoch_{epoch+1}.pth')


