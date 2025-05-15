# Authors: Sahil Mehul Bavishi (s2677266) and Matteo Spadaccia (s2748897)
# Subject: Computer Vision Coursework. Robustness exploration on CLIP model
# Date: 31.03.2025


"""0. Preliminary code"""

print("ROBUSTNESS EXPLORATION WITH DIFFERENT PERTURBATION TYPES (CLIP):")

# Setting code-behaviour varaibles
classNum = 4
pixel_to_class = {0: 0, 38: 1, 75: 2, 255: 3}
exludedClasses = [3]    # classes excludeed from metric calculation (3 not to consider the outline)
maxImages = -1           # number of input images to load from the defined set (0 to consider them all)
digits = 3              # digits to approximate decimal results in, when printed

# Defining paths
model_dir = f'Data/Output/Models/New_clip_segmentation_model_epoch_41.pth'
images_dir = 'Data/Input/Test/color/'
masks_dir = 'Data/Input/Test/label/'
output_plots_dir = 'Data/Output/Visuals/CLIP_Robustness/'

# Importing useful libraries
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import open_clip
import matplotlib.pyplot as plt
import cv2
from skimage.util import random_noise

device = "cuda" if torch.cuda.is_available() else "cpu"

image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

# Loading pre-trained CLIP model
CLIP_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
CLIP_model = CLIP_model.to(device)
tokenizer = open_clip.get_tokenizer("ViT-B-32")

class CLipDecoder(nn.Module):
    def __init__(self, clip_feature_dim=512, num_classes=classNum):
        super(CLipDecoder, self).__init__()

        # Project CLIP features
        self.clip_fc = nn.Linear(clip_feature_dim, 256)  

        # CNN Feature Extractor
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(384, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(64, num_classes, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, image, clip_features):
        cnn_features = self.encoder(image)
        clip_features = self.clip_fc(clip_features).view(clip_features.shape[0], 256, 1, 1)
        clip_features = clip_features.expand(-1, -1, cnn_features.shape[2], cnn_features.shape[3])
        fusion = torch.cat([cnn_features, clip_features], dim=1)
        segmentation_output = self.decoder(fusion)
        return segmentation_output

# Loading segmentation model
model = CLipDecoder().to(device)
model.load_state_dict(torch.load(model_dir, map_location=device))
model.eval()

def compute_dice_coefficient(pred, target, num_classes=classNum, exclude_classes=[]):
    """
    Compute Dice Coefficient for multi-class segmentation with the option to exclude specific classes.
    
    :param pred: Predicted segmentation map (H, W), with class indices.
    :param target: Ground truth segmentation map (H, W), with class indices.
    :param num_classes: Number of classes in segmentation.
    :param exclude_classes: List of class indices to exclude from Dice calculation.
    :return: Dice scores per class and mean Dice score.
    """

    dice_per_class = []
    for cls in range(num_classes):
        if cls in exclude_classes:
            continue
            
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = 2.0 * torch.logical_and(pred_inds, target_inds).sum().item()
        union = pred_inds.sum().item() + target_inds.sum().item()
        
        if union == 0:
            dice_per_class.append(float('nan'))
        else:
            dice_per_class.append(intersection / union)
    
    mean_dice = np.nanmean(dice_per_class) if dice_per_class else float('nan')

    return dice_per_class, mean_dice


"""a. Gaussian noise"""

def apply_gaussian_noise(image, std_dev):
    """Applies Gaussian noise to an image with a given standard deviation."""
    noise = np.random.normal(0, std_dev, image.shape)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image

def evaluate_gaussian_noise(model, images_dir, masks_dir, std_dev_levels, device, save_as:str="a_gaussian_noise"):
    """Evaluates the model with Gaussian noise perturbation and plots Dice score vs. noise level."""
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    dice_scores = []
    
    for std_dev in std_dev_levels:
        total_dice = []
        counter = 0
        for img_file in image_files:
            counter += 1
            if counter == maxImages:
                break
            img_path = os.path.join(images_dir, img_file)
            mask_path = os.path.join(masks_dir, img_file.replace('.jpg', '.png'))
            
            image = np.array(Image.open(img_path).convert("RGB"))
            noisy_image = apply_gaussian_noise(image, std_dev)
            image_tensor = transforms.ToTensor()(Image.fromarray(noisy_image)).unsqueeze(0).to(device)
            
            gt_mask = Image.open(mask_path).convert("L")
            gt_mask = gt_mask.resize((256, 256), Image.NEAREST)
            label_array = np.array(gt_mask)

            label_tensor = torch.from_numpy(label_array).to(dtype=torch.uint8)
            gt_mask_class = torch.zeros_like(label_tensor, dtype=torch.uint8)

            for pixel_value, class_idx in pixel_to_class.items():
                gt_mask_class[label_tensor == pixel_value] = class_idx

            image_clip_tensor = preprocess_val(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
    
            with torch.no_grad():
                clip_features = CLIP_model.encode_image(image_clip_tensor)
                clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
                model_output = model(image_tensor, clip_features)
                pred_mask = model_output.squeeze(0)
                pred_mask = torch.argmax(pred_mask, dim=0).cpu()

            _, mean_dice = compute_dice_coefficient(pred_mask, gt_mask_class, exclude_classes = exludedClasses)  
            
            total_dice.append(mean_dice)
        
        dice_scores.append(np.mean(total_dice))
    
    plt.figure()
    plt.plot(std_dev_levels, dice_scores, marker='o')
    plt.xlabel("Gaussian Noise Standard Deviation")
    plt.ylabel("Mean Dice Score")
    plt.title("Gaussian Noise")
    plt.grid()
    plt.savefig(output_plots_dir + save_as + ".pdf", format="pdf")
    
    return dice_scores

print("\na) Gaussian Noise...", end=' ')
std_dev_levels = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
dice_scores = evaluate_gaussian_noise(model, images_dir, masks_dir, std_dev_levels, device)
print("Done!")
print(" -"*49)
print("| st. deviations  =", " | ".join([f"{value:{digits+2}}" for value in std_dev_levels]), "|")
print("| mean Dice score =", " | ".join([f"{d:{digits+2}.{digits}f}" for d in dice_scores]), "|")
print(" -"*49,"\n")


"""b. Gaussian blurring"""

def apply_gaussian_blur(image, ksize):
    """Applies Gaussian blur to an image with a given kernel size."""
    if ksize % 2 == 0:
        ksize += 1
    kernel = np.array([
        [1/16, 2/16, 1/16],
        [2/16, 4/16, 2/16],
        [1/16, 2/16, 1/16]
    ])
    blurred_image = cv2.filter2D(image, -1, kernel)
    return blurred_image

def evaluate_gaussian_blur(model, images_dir, masks_dir, max_blur_level, device, save_as:str="b_gaussian_blurring"):
    """Evaluates the model with Gaussian blurring perturbation and plots Dice score vs. blur level."""
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    dice_scores = []
    
    for blur_level in range(max_blur_level + 1):
        total_dice = []
        
        counter = 0
        for img_file in image_files:
            counter += 1
            if counter == maxImages:
                break
            img_path = os.path.join(images_dir, img_file)
            mask_path = os.path.join(masks_dir, img_file.replace('.jpg', '.png'))
            
            blurred_image = np.array(Image.open(img_path).convert("RGB"))
            for i in range(blur_level):
                blurred_image = apply_gaussian_blur(blurred_image, 3)
            
            image_tensor = transforms.ToTensor()(Image.fromarray(blurred_image)).unsqueeze(0).to(device)

            gt_mask = Image.open(mask_path).convert("L")
            gt_mask = gt_mask.resize((256, 256), Image.NEAREST)
            label_array = np.array(gt_mask)

            label_tensor = torch.from_numpy(label_array).to(dtype=torch.uint8)
            gt_mask_class = torch.zeros_like(label_tensor, dtype=torch.uint8)

            for pixel_value, class_idx in pixel_to_class.items():
                gt_mask_class[label_tensor == pixel_value] = class_idx

            image_clip_tensor = preprocess_val(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
    
            with torch.no_grad():
                clip_features = CLIP_model.encode_image(image_clip_tensor)
                clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
                model_output = model(image_tensor, clip_features)
                pred_mask = model_output.squeeze(0)
                pred_mask = torch.argmax(pred_mask, dim=0).cpu()
                
            _, mean_dice = compute_dice_coefficient(pred_mask, gt_mask_class, exclude_classes = exludedClasses)  
            total_dice.append(mean_dice)
        
        dice_scores.append(np.mean(total_dice))
    
    plt.figure()
    plt.plot(range(max_blur_level + 1), dice_scores, marker='o')
    plt.xlabel("Gaussian Blur Level (number of iterative 3x3 mask applications)")
    plt.ylabel("Mean Dice Score")
    plt.title("Gaussian Blurring")
    plt.grid()
    plt.savefig(output_plots_dir + save_as + ".pdf", format="pdf")
    
    return dice_scores

print("\nb) Gaussian Blurring...", end=' ')
max_blur_level = 9
dice_scores = evaluate_gaussian_blur(model, images_dir, masks_dir, max_blur_level, device)
print("Done!")
print(" -"*49)
print("| blurring steps  =", " | ".join([f"{value:{digits+2}}" for value in list(range(max_blur_level+1))]), "|")
print("| mean Dice score =", " | ".join([f"{d:{digits+2}.{digits}f}" for d in dice_scores]), "|")
print(" -"*49,"\n")


"""c. Contrast increase"""

def apply_contrast_change(image, contrast_factor):
    """Applies contrast increase to an image by multiplying pixel values by a contrast factor."""
    contrast_image = image * contrast_factor
    contrast_image = np.clip(contrast_image, 0, 255).astype(np.uint8)
    return contrast_image

def evaluate_contrast_change(model, images_dir, masks_dir, contrast_factors, device, save_as:str="c_contrast_increase", invert_axis:bool=False):
    """Evaluates the model with increased contrast perturbation and plots Dice score vs. contrast factor."""
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    dice_scores = []
    
    for contrast_factor in contrast_factors:
        total_dice = []
        
        counter = 0
        for img_file in image_files:
            counter += 1
            if counter == maxImages:
                break
            img_path = os.path.join(images_dir, img_file)
            mask_path = os.path.join(masks_dir, img_file.replace('.jpg', '.png'))
            
            image = np.array(Image.open(img_path).convert("RGB"))
            contrast_image = apply_contrast_change(image, contrast_factor)
            
            image_tensor = transforms.ToTensor()(Image.fromarray(contrast_image)).unsqueeze(0).to(device)
            
            gt_mask = Image.open(mask_path).convert("L")
            gt_mask = gt_mask.resize((256, 256), Image.NEAREST)
            label_array = np.array(gt_mask)

            label_tensor = torch.from_numpy(label_array).to(dtype=torch.uint8)
            gt_mask_class = torch.zeros_like(label_tensor, dtype=torch.uint8)

            for pixel_value, class_idx in pixel_to_class.items():
                gt_mask_class[label_tensor == pixel_value] = class_idx

            image_clip_tensor = preprocess_val(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
    
            with torch.no_grad():
                clip_features = CLIP_model.encode_image(image_clip_tensor)
                clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
                model_output = model(image_tensor, clip_features)
                pred_mask = model_output.squeeze(0)
                pred_mask = torch.argmax(pred_mask, dim=0).cpu()
                
            _, mean_dice = compute_dice_coefficient(pred_mask, gt_mask_class, exclude_classes = exludedClasses)  
            total_dice.append(mean_dice)
        
        dice_scores.append(np.mean(total_dice))
    
    plt.figure()
    plt.plot(contrast_factors, dice_scores, marker='o')
    plt.xlabel("Contrast Factor")
    plt.ylabel("Mean Dice Score")
    plt.title("Contrast Decrease" if invert_axis else "Contrast Increase")
    plt.grid()
    if invert_axis: plt.gca().invert_xaxis()
    plt.savefig(output_plots_dir + save_as + ".pdf", format="pdf")
    
    return dice_scores

print("\nc) Contrast Increase...", end=' ')
contrast_factors = [1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.1, 1.15, 1.2, 1.25]
dice_scores = evaluate_contrast_change(model, images_dir, masks_dir, contrast_factors, device)
print("Done!")
print(" -"*49)
print("| contrast factor =", " | ".join([f"{value:{digits+2}}" for value in contrast_factors]), "|")
print("| mean Dice score =", " | ".join([f"{d:{digits+2}.{digits}f}" for d in dice_scores]), "|")
print(" -"*49,"\n")


"""d. Contrast decrease"""
# (using contrast increase functions)

print("\nd) Contrast Decrease...", end=' ')
contrast_factors = [1.0, 0.95, 0.90, 0.85, 0.80, 0.60, 0.40, 0.30, 0.20, 0.10]
dice_scores = evaluate_contrast_change(model, images_dir, masks_dir, contrast_factors, device, save_as="d_contrast_decrease", invert_axis=True)
print("Done!")
print(" -"*49)
print("| contrast factor =", " | ".join([f"{value:{digits+2}}" for value in contrast_factors]), "|")
print("| mean Dice score =", " | ".join([f"{d:{digits+2}.{digits}f}" for d in dice_scores]), "|")
print(" -"*49,"\n")


"""e. Brightness Increase"""

def apply_brightness_change(image, brightness_offset):
    """Applies brightness change by adding an offset to pixel values."""
    bright_image = image.astype(np.int16) + brightness_offset
    bright_image = np.clip(bright_image, 0, 255).astype(np.uint8)
    return bright_image

def evaluate_brightness_change(model, images_dir, masks_dir, brightness_offsets, device, save_as:str="e_brightness_increase", invert_axis:bool=False):
    """Evaluates the model with brightness perturbation and plots Dice score vs. brightness offset."""
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    dice_scores = []
    
    for brightness_offset in brightness_offsets:
        total_dice = []
        
        counter = 0
        for img_file in image_files:
            counter += 1
            if counter == maxImages:
                break
            img_path = os.path.join(images_dir, img_file)
            mask_path = os.path.join(masks_dir, img_file.replace('.jpg', '.png'))
            
            image = np.array(Image.open(img_path).convert("RGB"))
            bright_image = apply_brightness_change(image, brightness_offset)
            
            image_tensor = transforms.ToTensor()(Image.fromarray(bright_image)).unsqueeze(0).to(device)
            
            gt_mask = Image.open(mask_path).convert("L")
            gt_mask = gt_mask.resize((256, 256), Image.NEAREST)
            label_array = np.array(gt_mask)

            label_tensor = torch.from_numpy(label_array).to(dtype=torch.uint8)
            gt_mask_class = torch.zeros_like(label_tensor, dtype=torch.uint8)
            
            for pixel_value, class_idx in pixel_to_class.items():
                gt_mask_class[label_tensor == pixel_value] = class_idx
            
            image_clip_tensor = preprocess_val(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
    
            with torch.no_grad():
                clip_features = CLIP_model.encode_image(image_clip_tensor)
                clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
                model_output = model(image_tensor, clip_features)
                pred_mask = model_output.squeeze(0)
                pred_mask = torch.argmax(pred_mask, dim=0).cpu()
                
            _, mean_dice = compute_dice_coefficient(pred_mask, gt_mask_class, exclude_classes=exludedClasses)  
            total_dice.append(mean_dice)
        
        dice_scores.append(np.mean(total_dice))
    
    plt.figure()
    plt.plot(brightness_offsets, dice_scores, marker='o')
    plt.xlabel("Brightness Offset")
    plt.ylabel("Mean Dice Score")
    plt.title("Brightness Decrease" if invert_axis else "Brightness Increase")
    plt.grid()
    if invert_axis: plt.gca().invert_xaxis()
    plt.savefig(output_plots_dir + save_as + ".pdf", format="pdf")
    
    return dice_scores

print("\ne) Brightness Increase...", end=' ')
brightness_offsets = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
dice_scores = evaluate_brightness_change(model, images_dir, masks_dir, brightness_offsets, device)
print("Done!")
print(" -"*49)
print("| bright. offset  =", " | ".join([f"{value:{digits+2}}" for value in brightness_offsets]), "|")
print("| mean Dice score =", " | ".join([f"{d:{digits+2}.{digits}f}" for d in dice_scores]), "|")
print(" -"*49,"\n")


"""f. Brightness Decrease"""
# (using brightness increase functions)

print("\nf) Brightness Decrease...", end=' ')
brightness_offsets = [0, -5, -10, -15, -20, -25, -30, -35, -40, -45]
dice_scores = evaluate_brightness_change(model, images_dir, masks_dir, brightness_offsets, device, save_as="f_brightness_decrease", invert_axis=True)
print("Done!")
print(" -"*49)
print("| bright. offset  =", " | ".join([f"{value:{digits+2}}" for value in brightness_offsets]), "|")
print("| mean Dice score =", " | ".join([f"{d:{digits+2}.{digits}f}" for d in dice_scores]), "|")
print(" -"*49,"\n")


"""g. Square occlusion"""

def apply_occlusion(image, square_size):
    """Applies occlusion by replacing a random square region with black pixels."""
    h, w, _ = image.shape
    if square_size > 0:
        x = np.random.randint(0, w - square_size + 1)
        y = np.random.randint(0, h - square_size + 1)
        image[y:y+square_size, x:x+square_size] = 0
    return image

def evaluate_occlusion(model, images_dir, masks_dir, occlusion_sizes, device, save_as:str="g_square_occlusion"):
    """Evaluates the model under increasing occlusion perturbation."""
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    dice_scores = []

    for square_size in occlusion_sizes:
        total_dice = []

        counter = 0
        for img_file in image_files:
            counter += 1
            if counter == maxImages:
                break
            img_path = os.path.join(images_dir, img_file)
            mask_path = os.path.join(masks_dir, img_file.replace('.jpg', '.png'))

            image = np.array(Image.open(img_path).convert("RGB"))
            occluded_image = apply_occlusion(image, square_size)

            image_tensor = transforms.ToTensor()(Image.fromarray(occluded_image)).unsqueeze(0).to(device)

            gt_mask = Image.open(mask_path).convert("L")
            gt_mask = gt_mask.resize((256, 256), Image.NEAREST)
            label_array = np.array(gt_mask)

            label_tensor = torch.from_numpy(label_array).to(dtype=torch.uint8)
            gt_mask_class = torch.zeros_like(label_tensor, dtype=torch.uint8)

            for pixel_value, class_idx in pixel_to_class.items():
                gt_mask_class[label_tensor == pixel_value] = class_idx

            image_clip_tensor = preprocess_val(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)

            with torch.no_grad():
                clip_features = CLIP_model.encode_image(image_clip_tensor)
                clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
                model_output = model(image_tensor, clip_features)
                pred_mask = model_output.squeeze(0)
                pred_mask = torch.argmax(pred_mask, dim=0).cpu()

            _, mean_dice = compute_dice_coefficient(pred_mask, gt_mask_class, exclude_classes=exludedClasses)
            total_dice.append(mean_dice)

        dice_scores.append(np.mean(total_dice))

    plt.figure()
    plt.plot(occlusion_sizes, dice_scores, marker='o')
    plt.xlabel("Occlusion Size (edge length in pixels)")
    plt.ylabel("Mean Dice Score")
    plt.title("Square Occlusion")
    plt.grid()
    plt.savefig(output_plots_dir + save_as + ".pdf", format="pdf")
    return dice_scores

print("\ng) Square Occlusion...", end=' ')
occlusion_sizes = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
dice_scores = evaluate_occlusion(model, images_dir, masks_dir, occlusion_sizes, device)
print("Done!")
print(" -" * 49)
print("| oc. square size =", " | ".join([f"{value:{digits+2}}" for value in occlusion_sizes]), "|")
print("| mean Dice score =", " | ".join([f"{d:{digits+2}.{digits}f}" for d in dice_scores]), "|")
print(" -" * 49, "\n")


"""h. Salt&Pepper Noise"""

def apply_salt_and_pepper_noise(image, noise_amount):
    """Applies salt and pepper noise to an image."""
    noisy_image = random_noise(image, mode='s&p', amount=noise_amount)
    noisy_image = (noisy_image * 255).astype(np.uint8)
    return noisy_image

def evaluate_salt_and_pepper_noise(model, images_dir, masks_dir, noise_levels, device, save_as:str="h_SaP_noise"):
    """Evaluates the model with salt-and-pepper noise perturbation and plots Dice score vs. noise level."""
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    dice_scores = []
    
    for noise_amount in noise_levels:
        total_dice = []
        
        counter = 0
        for img_file in image_files:
            counter += 1
            if counter == maxImages:
                break
            img_path = os.path.join(images_dir, img_file)
            mask_path = os.path.join(masks_dir, img_file.replace('.jpg', '.png'))
            
            image = np.array(Image.open(img_path).convert("RGB"))
            noisy_image = apply_salt_and_pepper_noise(image, noise_amount)
            
            image_tensor = transforms.ToTensor()(Image.fromarray(noisy_image)).unsqueeze(0).to(device)
            
            gt_mask = Image.open(mask_path).convert("L")
            gt_mask = gt_mask.resize((256, 256), Image.NEAREST)
            label_array = np.array(gt_mask)

            label_tensor = torch.from_numpy(label_array).to(dtype=torch.uint8)
            gt_mask_class = torch.zeros_like(label_tensor, dtype=torch.uint8)

            for pixel_value, class_idx in pixel_to_class.items():
                gt_mask_class[label_tensor == pixel_value] = class_idx

            image_clip_tensor = preprocess_val(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
    
            with torch.no_grad():
                clip_features = CLIP_model.encode_image(image_clip_tensor)
                clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
                model_output = model(image_tensor, clip_features)
                pred_mask = model_output.squeeze(0)
                pred_mask = torch.argmax(pred_mask, dim=0).cpu()
                
            _, mean_dice = compute_dice_coefficient(pred_mask, gt_mask_class, exclude_classes=exludedClasses)  
            total_dice.append(mean_dice)
        
        dice_scores.append(np.mean(total_dice))
    
    plt.figure()
    plt.plot(noise_levels, dice_scores, marker='o')
    plt.xlabel("Salt & Pepper Noise Level")
    plt.ylabel("Mean Dice Score")
    plt.title("Salt & Pepper Noise")
    plt.grid()
    plt.savefig(output_plots_dir + save_as + ".pdf", format="pdf")
    
    return dice_scores

print("\nh) Salt & Pepper Noise...", end=' ')
noise_levels = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18]
dice_scores = evaluate_salt_and_pepper_noise(model, images_dir, masks_dir, noise_levels, device)
print("Done!")
print(" -"*49)
print("| S&P noise level =", " | ".join([f"{value:{digits+2}}" for value in noise_levels]), "|")
print("| mean Dice score =", " | ".join([f"{d:{digits+2}.{digits}f}" for d in dice_scores]), "|")
print(" -"*49,"\n")