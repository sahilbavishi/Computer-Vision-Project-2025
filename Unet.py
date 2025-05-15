# Authors: Sahil Mehul Bavishi (s2677266) and Matteo Spadaccia (s2748897)
# Subject: Computer Vision Coursework. U-Net 
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
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import pickle as pkl


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
imgResize = (256, 256)

"""#### b) Augmenting dataset"""
print("Using previously augmented data.")


"""#### c) Preparing datasets"""

# Preparing train, valid and test sets
validSize = 0.2
batchSize = 16
imgChannels = 3 ## this was 3
inputSize = (imgResize[0], imgResize[1], imgChannels)

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Define preprocessed size
preprocessedSize = imgResize  # Desired size for input images and masks
inputSize = (128,128,3)

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
batchSize = 16


# Define the UNet architecture
class UNet(nn.Module):
    def __init__(self, num_classes=classesNum):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(3, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = self.conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = self.conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = self.conv_block(256, 512)

        # Decoder
        self.up5 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec5 = self.conv_block(512, 256)

        self.up6 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec6 = self.conv_block(256, 128)

        self.up7 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec7 = self.conv_block(128, 64)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1) # output layer
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2)
        )
    
    def forward(self, x):
        # Encoder
        c1 = self.enc1(x)
        p1 = self.pool1(c1)

        c2 = self.enc2(p1)
        p2 = self.pool2(c2)

        c3 = self.enc3(p2)
        p3 = self.pool3(c3)

        c4 = self.enc4(p3)

        # Decoder
        u5 = self.up5(c4)
        u5 = torch.cat([u5, c3], dim=1)
        c5 = self.dec5(u5)

        u6 = self.up6(c5)
        u6 = torch.cat([u6, c2], dim=1)
        c6 = self.dec6(u6)

        u7 = self.up7(c6)
        u7 = torch.cat([u7, c1], dim=1)
        c7 = self.dec7(u7)

        outputs = self.final_conv(c7)

        return outputs

# Define hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model, optimizer, and loss function
model = UNet(classesNum).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# criterion = nn.CrossEntropyLoss()

def dice_loss(pred, target, smooth=1e-6):
    pred = torch.softmax(pred, dim=1)
    target = torch.nn.functional.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def focal_loss(pred, target, alpha=0.8, gamma=2.0):
    pred = torch.softmax(pred, dim=1)
    target = torch.nn.functional.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2)
    logpt = -torch.nn.functional.cross_entropy(pred, target, reduction='none')
    pt = torch.exp(logpt)
    focal_loss = -((1 - pt) ** gamma) * logpt
    return focal_loss.mean()

criterion = lambda outputs, masks: nn.CrossEntropyLoss()(outputs, masks) + dice_loss(outputs, masks)

# criterion = lambda outputs, masks: 0.7 * dice_loss(outputs, masks) + 0.3 * focal_loss(outputs, masks)


# Evaluation
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == masks).sum().item()
            total += masks.numel()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")


class UNetDataset(Dataset):
    def __init__(self, image_files, mask_files, transform=None, target_transform=None, max_images=100):
        self.image_files = image_files  # Limit to first 100 images
        self.mask_files = mask_files  # Limit to first 100 masks
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label_path = self.image_files[idx].replace('.jpg', '.png').replace('color', 'label')
        image = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('L')  # Load label as grayscale

        if self.transform:
            image = self.transform(image)
            
        label = label.resize(imgResize, Image.NEAREST)

        # Convert label to tensor
        label = torch.tensor(np.array(label), dtype=torch.long)  # Shape: (H, W)

        # # Map pixel values to class indices
        label = torch.where(label == 38, 1, label)
        label = torch.where(label == 75, 2, label)
        label = torch.where(label == 255, 3, label)
        label = torch.where(label == 0, 0, label)  # Ensure 0 stays as class 0

        # mapping = {38: 1, 75: 2, 255: 0, 0: 0}
        # label = torch.tensor(np.vectorize(mapping.get)(np.array(label)), dtype=torch.long)

        

        return image, label  # No one-hot encoding

# Define transformations
image_transform = transforms.Compose([
    transforms.Resize(imgResize),
    transforms.ToTensor()
])



# Load dataset with first 100 images
train_dataset = UNetDataset(train_images, train_masks, transform=image_transform, target_transform=None, max_images=100)
train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)

val_dataset = UNetDataset(val_images, val_masks, transform=image_transform, target_transform=None, max_images=100)
val_loader = DataLoader(val_dataset, batch_size=batchSize, shuffle=False)

test_dataset = UNetDataset(test_images, test_masks, transform=image_transform, target_transform=None, max_images=100)
test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False)

# Define the training function
def train_unet(model, train_loader, val_loader, epochs, model_save_path="/home/s2677266/CVis/Data/Output/Models/", device="cuda" if torch.cuda.is_available() else "cpu"):
    model.to(device)
    
    # criterion = lambda outputs, masks: 0.7 * dice_loss(outputs, masks) + 0.3 * focal_loss(outputs, masks)
    # criterion = lambda outputs, masks: nn.CrossEntropyLoss()(outputs, masks) + dice_loss(outputs, masks)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, masks = images.to(device), masks.float().squeeze(1).to(device)  # Ensure correct shape
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {epoch_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.long().squeeze(1).to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss / len(val_loader):.4f}")

        # Save model after each epoch
        # epoch_model_path = f"{model_save_path}Better_Final_Outline_unet_model_epoch_{epoch+1}.pth"  # with 32, 128
        epoch_model_path = f"{model_save_path}UNET_CROSS_LOSS_EPOCH_{epoch+1}.pth"
        torch.save(model.state_dict(), epoch_model_path)
        print(f"Model saved at {epoch_model_path}")

    print("Training complete!")

# Define the testing function
def test_unet(model, test_loader, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.to(device)
    model.eval()
    test_loss = 0
    
    # criterion = lambda outputs, masks: 0.7 * dice_loss(outputs, masks) + 0.3 * focal_loss(outputs, masks)
    # criterion = lambda outputs, masks: nn.CrossEntropyLoss()(outputs, masks) + dice_loss(outputs, masks)
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.float().squeeze(1).to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            test_loss += loss.item()

    print(f"Test Loss: {test_loss / len(test_loader):.4f}")

# def train_unet(model, train_loader, val_loader, epochs, model_save_path="/home/s2677266/CVis/Data/Output/Models/", device="cuda" if torch.cuda.is_available() else "cpu"):
#     model.to(device)
    
#     # Define the criterion (with separate conversion for each loss)
#     criterion = lambda outputs, masks: 0.7 * dice_loss(outputs, masks.long()) + 0.3 * focal_loss(outputs, masks.float())

#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     for epoch in range(epochs):
#         model.train()
#         epoch_loss = 0
#         for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
#             images, masks = images.to(device), masks.squeeze(1).to(device)  # Keep masks as they are for loss function

#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, masks)
#             loss.backward()
#             optimizer.step()
#             epoch_loss += loss.item()

#         print(f"Epoch {epoch+1}/{epochs}, Training Loss: {epoch_loss / len(train_loader):.4f}")

#         # Validation
#         model.eval()
#         val_loss = 0
#         with torch.no_grad():
#             for images, masks in val_loader:
#                 images, masks = images.to(device), masks.squeeze(1).to(device)
#                 outputs = model(images)
#                 loss = criterion(outputs, masks)
#                 val_loss += loss.item()
        
#         print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss / len(val_loader):.4f}")

#         # Save model after each epoch
#         epoch_model_path = f"{model_save_path}UNET_DICE_FOCAL_LOSS_EPOCH_{epoch+1}.pth"
#         torch.save(model.state_dict(), epoch_model_path)
#         print(f"Model saved at {epoch_model_path}")

#     print("Training complete!")

# # Define the testing function
# def test_unet(model, test_loader, device="cuda" if torch.cuda.is_available() else "cpu"):
#     model.to(device)
#     model.eval()
#     test_loss = 0

#     criterion = lambda outputs, masks: 0.7 * dice_loss(outputs, masks.long()) + 0.3 * focal_loss(outputs, masks.float())

#     with torch.no_grad():
#         for images, masks in test_loader:
#             images, masks = images.to(device), masks.squeeze(1).to(device)
#             outputs = model(images)
#             loss = criterion(outputs, masks)
#             test_loss += loss.item()

#     print(f"Test Loss: {test_loss / len(test_loader):.4f}")


# Call the training function
train_unet(model, train_loader, val_loader, epochs=epochsNum)

# Evaluate on the test set
test_unet(model, test_loader)

print('Execution Completed')
