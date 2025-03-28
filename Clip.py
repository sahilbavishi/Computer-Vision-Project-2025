# Authors: Sahil Mehul Bavishi (s2677266) and Matteo Spadaccia (its a me Mario)
# Subject: Computer Vision Coursework. Clip
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
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import open_clip  # Alternative to Hugging Face's CLIP


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
batchSize = 16

## CLIP MODEL, first part is feature extraction



# Load CLIP Model from open_clip
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
model = model.to(device)
tokenizer = open_clip.get_tokenizer("ViT-B-32")

# Directory to save CLIP features
clip_features_dir = "Data/Output/Models/clipTrainFeatures"
# os.makedirs(clip_features_dir, exist_ok=True)

# # Extract and save CLIP features
# def extract_clip_features(image_path):
#     image = Image.open(image_path).convert("RGB")
#     image_tensor = preprocess_val(image).unsqueeze(0).to(device)  # Use open_clip's preprocessing
#     with torch.no_grad():
#         features = model.encode_image(image_tensor)  # Extract image features
#     features = features / features.norm(p=2, dim=-1, keepdim=True)  # Normalize
#     return features.cpu()

# clip_features_trainval = {}
# print("Extracting Clip Features")
# for image_name in trainval_images:
#     image_path = image_name
#     clip_features = extract_clip_features(image_path)

#     # Save each feature tensor separately
#     feature_path = os.path.join(clip_features_dir, f"{image_name.split('/')[-1].split('.')[0]}.pt")
#     torch.save(clip_features, feature_path)

#     clip_features_trainval[image_name] = feature_path  # Store path reference

clip_features_trainval = {
    image_name: torch.load(os.path.join(clip_features_dir, f"{image_name.split('/')[-1].split('.')[0]}.pt"))
    for image_name in trainval_images
}

print("*****************************************************")
print(f"Loaded {len(clip_features_trainval)} CLIP feature tensors.")
# print(f"Loaded {clip_features_trainval} CLIP feature tensors.")
print("*****************************************************")


# Custom Dataset for Segmentation
class SegmentationDataset(Dataset):
    def __init__(self, image_files, label_files, clip_features_dict, transform=None):
        self.image_files = image_files
        self.label_files = label_files
        self.clip_features_dict = clip_features_dict  # Store CLIP features
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label_path = img_path.replace('.jpg', '.png').replace('color', 'label')

        # Load image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)  # Apply transformations
        else:
            image = transforms.ToTensor()(image)  # Convert manually if transform is None

        # Load CLIP feature tensor
        image_name = img_path
        clip_feature = self.clip_features_dict[image_name]  # Shape: (512,)

        # Load and process label
        label = Image.open(label_path).convert('L')
        label = label.resize((256, 256), Image.NEAREST)
        label = torch.tensor(np.array(label), dtype=torch.long)

        # Convert label classes to match expected format
        label = torch.where(label == 38, 1, label)
        label = torch.where(label == 75, 2, label)
        label = torch.where(label == 255, 3, label)
        label = torch.where(label == 0, 0, label)
        # label = torch.where(label >= num_classes, 0, label)
        return image, clip_feature, label  # Return both image & CLIP feature




# Define Transformations
transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=Image.NEAREST),
    transforms.ToTensor()
])

# Load Dataset & DataLoader
batch_size = batchSize


segmentation_dataset = SegmentationDataset(
    image_files=train_images,
    label_files=train_masks,
    clip_features_dict=clip_features_trainval,  # Use CLIP features
    transform=None  # No need for image transformations
)


segmentation_dataloader = DataLoader(segmentation_dataset, batch_size=batch_size, shuffle=True)



# Segmentation Model
class SegmentationDecoder(nn.Module):
    def __init__(self, clip_feature_dim=512, num_classes=classesNum):
        super(SegmentationDecoder, self).__init__()

        # Project CLIP features
        self.clip_fc = nn.Linear(clip_feature_dim, 128)  

        # CNN Feature Extractor
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128 + 128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True),  # Ensure fixed output size
            nn.ConvTranspose2d(64, num_classes, kernel_size=3, stride=1, padding=1),
        )


    def forward(self, image, clip_features):
        cnn_features = self.encoder(image)
        clip_features = self.clip_fc(clip_features).view(clip_features.shape[0], 128, 1, 1)
        clip_features = clip_features.expand(-1, -1, cnn_features.shape[2], cnn_features.shape[3])

        fusion = torch.cat([cnn_features, clip_features], dim=1)
        segmentation_output = self.decoder(fusion)
        return segmentation_output



# Initialize Model, Loss, Optimizer
segmentation_model = SegmentationDecoder().to(device)
segmentation_criterion = nn.CrossEntropyLoss()
segmentation_optimizer = optim.Adam(segmentation_model.parameters(), lr=1e-3)

scheduler = torch.optim.lr_scheduler.StepLR(segmentation_optimizer, step_size=5, gamma=0.5)

# Mixed Precision Training
scaler = torch.cuda.amp.GradScaler()


# Training Loop
for epoch in tqdm(range(epochsNum), desc="Training Progress"):
    segmentation_model.train()
    running_loss = 0.0

    print(f"Epoch [{epoch+1}/{epochsNum}] Running:")
    
    for images, clip_features, labels in segmentation_dataloader:  # Unpack correctly
        # print(torch.unique(labels))
        images, clip_features, labels = images.to(device), clip_features.to(device), labels.to(device)

        segmentation_optimizer.zero_grad()

        # Mixed precision training
        with torch.cuda.amp.autocast(): 
            outputs = segmentation_model(images, clip_features)  # Now correctly passing 2 args
            loss = segmentation_criterion(outputs, labels)  # Compute loss

        # Scale the loss and backpropagate
        scaler.scale(loss).backward()
        scaler.step(segmentation_optimizer)
        scaler.update()

        running_loss += loss.item()


    scheduler.step()
    
    avg_loss = running_loss / len(segmentation_dataloader)
    print(f"Epoch [{epoch+1}/{epochsNum}], Loss: {avg_loss:.4f}")

    torch.save(segmentation_model.state_dict(), f'Data/Output/Models/Better_Final_clip_segmentation_model_epoch_{epoch+1}.pth')



print("Training complete!")