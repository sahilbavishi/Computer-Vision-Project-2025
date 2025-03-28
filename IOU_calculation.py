import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm  # Progress bar for better visibility
import open_clip

choose_autoencoder_epoch = 30
choose_segmentation_decoder_epoch = 67
unet_epoch = 94


print("Autoencoder IOU")


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
                nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),  # No activation function
                nn.Sigmoid()
            )


        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x
        
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
        for param in list(encoder.parameters())[:4]:  # Adjust layers to freeze
            param.requires_grad = False
        
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
            nn.Conv2d(32, 3, kernel_size=1)  # Output logits
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

autoencoder = Autoencoder().to(device)
autoencoder.load_state_dict(torch.load(f'Data/Output/Models/updated_working_autoencoder_epoch_{choose_autoencoder_epoch}.pth', map_location=device))

# Load segmentation model weights
segmentation_model = SegmentationDecoder(autoencoder.encoder).to(device)
segmentation_model.load_state_dict(torch.load(f'Data/Output/Models/final_segmentation_model1_epoch_{choose_segmentation_decoder_epoch}.pth', map_location=device))

autoencoder.eval()
segmentation_model.eval()

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to match model input size
    transforms.ToTensor(),
])

def load_image_and_label(image_path, label_path):
    """Load an image and its corresponding label, apply necessary transformations."""
    image = Image.open(image_path).convert('RGB')
    label = Image.open(label_path).convert('L')  # Load label in grayscale
    label = label.resize((256, 256), Image.NEAREST)

    input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    label_array = np.array(label)  # Convert label to numpy array

    # Replace pixels with value 255 with 0
    label_array[label_array == 255] = 0

    return input_tensor, label_array


def compute_iou(pred, target, num_classes=3):
    """
    Compute Intersection over Union (IoU) for multi-class segmentation using accumulated dataset-wide counts.
    
    :param pred: Predicted segmentation map (H, W), with class indices.
    :param target: Ground truth segmentation map (H, W), with class indices.
    :param num_classes: Number of classes in segmentation.
    :return: IoU scores per class and mean IoU.
    """
    intersection_counts = np.zeros(num_classes)
    union_counts = np.zeros(num_classes)
    
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        
        intersection_counts[cls] += torch.logical_and(pred_inds, target_inds).sum().item()
        union_counts[cls] += torch.logical_or(pred_inds, target_inds).sum().item()
    
    iou_per_class = [
        intersection_counts[cls] / union_counts[cls] if union_counts[cls] > 0 else float('nan')
        for cls in range(num_classes)
    ]
    
    mean_iou = np.nanmean(iou_per_class)
    return iou_per_class, mean_iou


def compute_dice_coefficient(pred, target, num_classes=3):
    """
    Compute Dice Coefficient for multi-class segmentation.
    
    :param pred: Predicted segmentation map (H, W), with class indices.
    :param target: Ground truth segmentation map (H, W), with class indices.
    :param num_classes: Number of classes in segmentation.
    :return: Dice scores per class and mean Dice score.
    """
    dice_per_class = []
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = 2.0 * torch.logical_and(pred_inds, target_inds).sum().item()
        union = pred_inds.sum().item() + target_inds.sum().item()
        
        if union == 0:
            dice_per_class.append(float('nan'))
        else:
            dice_per_class.append(intersection / union)
    
    mean_dice = np.nanmean(dice_per_class)
    return dice_per_class, mean_dice


def compute_pixel_accuracy(pred, target):
    """
    Compute Pixel Accuracy.
    :param pred: Predicted segmentation map (H, W), with class indices.
    :param target: Ground truth segmentation map (H, W), with class indices.
    :return: Pixel accuracy score.
    """
    correct = (pred == target).sum().item()
    total = target.numel()
    return correct / total

# Store results
all_intersection = np.zeros(3)
all_union = np.zeros(3)
all_dice = []
all_pixel_accs = []

# Load Images
# Define file paths
image_dir = "/home/s2677266/CVis/Data/Input/Test/color/"
label_dir = "/home/s2677266/CVis/Data/Input/Test/label/"

# Get list of all images
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]

# Mapping between class indices and pixel values
# Mapping between class indices and pixel values
class_to_pixel = {0: 0, 1: 38, 2: 75, 3: 255}  # Ensure unique keys

# Ensure 255 maps to class 0
pixel_to_class = {v: (0 if v == 255 else k) for k, v in class_to_pixel.items()}

# Evaluate each image
for image_file in tqdm(image_files, desc="Processing images"):
    image_path = os.path.join(image_dir, image_file)
    label_path = os.path.join(label_dir, image_file.replace(".jpg", ".png"))
    
    input_tensor, label_image = load_image_and_label(image_path, label_path)
    
    with torch.no_grad():
        output_tensor = segmentation_model(input_tensor)
    
    output_tensor = F.softmax(output_tensor, dim=1)
    predicted_classes = torch.argmax(output_tensor, dim=1).squeeze(0).cpu()
    
    label_tensor = torch.tensor(label_image, dtype=torch.uint8)
    label_class_map = torch.zeros_like(label_tensor)
    
    for pixel_value, class_idx in pixel_to_class.items():
        label_class_map[label_tensor == pixel_value] = class_idx
    
    # Compute metrics
    iou_per_class, _ = compute_iou(predicted_classes, label_class_map)
    _, mean_dice = compute_dice_coefficient(predicted_classes, label_class_map)
    pixel_acc = compute_pixel_accuracy(predicted_classes, label_class_map)
    
    for cls in range(3):
        if not np.isnan(iou_per_class[cls]):
            all_intersection[cls] += torch.logical_and(predicted_classes == cls, label_class_map == cls).sum().item()
            all_union[cls] += torch.logical_or(predicted_classes == cls, label_class_map == cls).sum().item()
    
    all_dice.append(mean_dice)
    all_pixel_accs.append(pixel_acc)

# Compute final IoU per class and overall mean IoU
final_iou_per_class = [
    all_intersection[cls] / all_union[cls] if all_union[cls] > 0 else float('nan')
    for cls in range(3)
]
overall_mean_iou = np.nanmean(final_iou_per_class)
overall_mean_dice = np.nanmean(all_dice)
overall_pixel_accuracy = np.mean(all_pixel_accs)

print(f"Per-Class IoU: {final_iou_per_class}")
print(f"Overall Mean IoU: {overall_mean_iou:.4f}")
print(f"Overall Mean Dice Coefficient: {overall_mean_dice:.4f}")
print(f"Overall Pixel Accuracy: {overall_pixel_accuracy:.4f}")


print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print("DOING UNET NOW")
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class UNet(nn.Module):
    def __init__(self, num_classes=3):
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

        # Final output layer
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
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
    
model = UNet().to(device)
model.load_state_dict(torch.load(f'Data/Output/Models/final_unet_model_epoch_{unet_epoch}.pth', map_location=device))

model.eval()

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to match model input size
    transforms.ToTensor(),
])

def load_image_and_label(image_path, label_path):
    """Load an image and its corresponding label, apply necessary transformations."""
    image = Image.open(image_path).convert('RGB')
    label = Image.open(label_path).convert('L')  # Load label in grayscale
    label = label.resize((256, 256), Image.NEAREST)

    input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    label_array = np.array(label)  # Convert label to numpy array

    # Replace pixels with value 255 with 0
    label_array[label_array == 255] = 0

    return input_tensor, label_array

def compute_iou(pred, target, num_classes=4):
    """
    Compute the Intersection over Union (IoU) score for multi-class segmentation.
    
    :param pred: Predicted segmentation map (H, W), with class indices.
    :param target: Ground truth segmentation map (H, W), with class indices.
    :param num_classes: Number of classes in segmentation.
    :return: IoU scores per class and mean IoU.
    """
    iou_per_class = []
    
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        
        intersection = torch.logical_and(pred_inds, target_inds).sum().item()
        union = torch.logical_or(pred_inds, target_inds).sum().item()

        if union == 0:
            iou_per_class.append(float('nan'))  # Ignore classes not present in the image
        else:
            iou_per_class.append(intersection / union)

    mean_iou = np.nanmean(iou_per_class)  # Compute mean IoU ignoring NaNs
    return iou_per_class, mean_iou

# Define file paths
image_dir = "/home/s2677266/CVis/Data/Input/Test/color/"
label_dir = "/home/s2677266/CVis/Data/Input/Test/label/"

# Get list of all images
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]

# Mapping between class indices and pixel values
# Mapping between class indices and pixel values
class_to_pixel = {0: 0, 1: 38, 2: 75, 3: 255}  # Ensure unique keys

# Ensure 255 maps to class 0
pixel_to_class = {v: (0 if v == 255 else k) for k, v in class_to_pixel.items()}

# Store IoU results
all_ious = []

# Loop through all images
for image_file in tqdm(image_files, desc="Processing images"):
    # Construct full paths
    image_path = os.path.join(image_dir, image_file)
    label_path = os.path.join(label_dir, image_file.replace(".jpg", ".png"))

    # Load image and label
    input_tensor, label_image = load_image_and_label(image_path, label_path)

    # Run inference
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # Apply softmax and get predicted class indices
    output_tensor = F.softmax(output_tensor, dim=1)
    predicted_classes = torch.argmax(output_tensor, dim=1).squeeze(0).cpu()  # (H, W)

    # Convert predicted class indices to pixel values
    output_pixel_values = torch.zeros_like(predicted_classes, dtype=torch.uint8)
    for class_idx, pixel_value in class_to_pixel.items():
        output_pixel_values[predicted_classes == class_idx] = pixel_value

    # Convert label image to class indices
    label_tensor = torch.from_numpy(label_image).to(dtype=torch.uint8)
    label_class_map = torch.zeros_like(label_tensor, dtype=torch.uint8)

    for pixel_value, class_idx in pixel_to_class.items():
        label_class_map[label_tensor == pixel_value] = class_idx

    # Compute IoU
    _, mean_iou = compute_iou(output_pixel_values, label_class_map)
    all_ious.append(mean_iou)

# Compute overall mean IoU
overall_mean_iou = np.nanmean(all_ious)

print(f"Overall Mean IoU for Unet: {overall_mean_iou:.4f}")





### IOU for CLIP



class Clip_Decoder(nn.Module):
    def __init__(self, clip_feature_dim=512, num_classes=3):
        super(Clip_Decoder, self).__init__()

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
    

# Function to compute IoU
def compute_iou(pred_mask, gt_mask, num_classes):
    iou_per_class = []
    
    for cls in range(num_classes):
        pred_class = (pred_mask == cls).astype(np.uint8)
        gt_class = (gt_mask == cls).astype(np.uint8)

        intersection = np.logical_and(pred_class, gt_class).sum()
        union = np.logical_or(pred_class, gt_class).sum()

        if union == 0:
            iou_per_class.append(float('nan'))  # Avoid division by zero
        else:
            iou_per_class.append(intersection / union)

    mean_iou = np.nanmean(iou_per_class)  # Ignore NaNs when computing mean
    return iou_per_class, mean_iou

# Load pre-trained CLIP model using open_clip
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
model = model.to(device)
tokenizer = open_clip.get_tokenizer("ViT-B-32")

# Define paths
images_dir = "/home/s2677266/CVis/Data/Input/Test/color/"
masks_dir = "/home/s2677266/CVis/Data/Input/Test/label/"  # Path to ground truth masks
image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

# Select a random image
random_image_file = random.choice(image_files)
img_path = os.path.join(images_dir, random_image_file)
mask_path = os.path.join(masks_dir, random_image_file.replace('.jpg', '.png'))  # Adjust if needed

# Load and preprocess image for CLIP feature extraction
image_clip = Image.open(img_path).convert("RGB")
image_clip_tensor = preprocess_val(image_clip).unsqueeze(0).to(device)  # Use open_clip preprocessing

# Extract CLIP features
with torch.no_grad():
    clip_features = model.encode_image(image_clip_tensor)
clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)  # Normalize
clip_features = clip_features.to(device)

# Load and preprocess image for segmentation model
image_seg = Image.open(img_path).convert("RGB")
image_tensor = transforms.ToTensor()(image_seg).unsqueeze(0).to(device)

# Load ground truth mask
gt_mask = Image.open(mask_path)
gt_mask = np.array(gt_mask)



# Load segmentation model
clip_decoder_epoch = 87
num_classes = 3
clip_segmentation_model = Clip_Decoder().to(device)
clip_segmentation_model.load_state_dict(torch.load(f'Data/Output/Models/final_clip_segmentation_model_epoch_{clip_decoder_epoch}.pth', map_location=device))
clip_segmentation_model.eval()

# Perform inference
with torch.no_grad():
    model_output = clip_segmentation_model(image_tensor, clip_features)

# Convert model output to predicted segmentation mask
pred_mask = model_output.squeeze(0)  # Shape: (num_classes, H, W)
pred_mask = torch.argmax(pred_mask, dim=0).cpu().numpy()

gt_mask = Image.open(mask_path).convert("L")  # Convert to grayscale
gt_mask = np.array(gt_mask.resize(pred_mask.shape[::-1], Image.NEAREST))

# Compute IoU
iou_per_class, mean_iou = compute_iou(pred_mask, gt_mask, num_classes)

# Print IoU values
print(f"IoU per class: {iou_per_class}")
print(f"Mean IoU: {mean_iou:.4f}")
