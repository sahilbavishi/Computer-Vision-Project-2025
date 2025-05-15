# Authors: Sahil Mehul Bavishi (s2677266) and Matteo Spadaccia (s2748897)
# Subject: Computer Vision Coursework. Performance evaluation
# Date: 21.03.2025


""""0. Preliminary code"""
print("PERFORMANCE EVALUATION:")

# Setting code-behaviour varaibles
classNum = 4
exludedClasses = [3]
class_to_pixel = {0: 0, 1: 38, 2: 75, 3: 255}
pixel_to_class = {v: k for k, v in class_to_pixel.items()}

image_dir = 'Data/Input/Test/color/'
label_dir = 'Data/Input/Test/label/'
heatmap_dir = 'Data/Input/Test/heatmaps/'

# (to skip testing of a model, set any relevant desired epoch to 0)
models_dir = 'Data/Output/Models/'
unet_epoch = 0 # CROSS : 29, DICE: 22
autoencoder_epoch = 0 # 82
segmentation_decoder_epoch = 0 #42
clip_decoder_epoch =0 #42
prompted_clip_decoder_epoch = 0 #42

# Importing useful libraries
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import open_clip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]

# Defining data loading
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
def load_image_and_label(image_path, label_path):
    """Load an image and its corresponding label, apply necessary transformations."""
    image = Image.open(image_path).convert('RGB')
    label = Image.open(label_path).convert('L') # label(grayscale)
    input_tensor = transform(image).unsqueeze(0).to(device)
    label_array = np.array(label)
    return input_tensor, label_array

# Defining IoU function
def compute_iou(pred, target, num_classes=classNum, exclude_classes=[]):
    """
    Compute Intersection over Union (IoU) for multi-class segmentation using accumulated dataset-wide counts,
    with the option to exclude specific classes from the computation.
    
    :param pred: Predicted segmentation map (H, W), with class indices.
    :param target: Ground truth segmentation map (H, W), with class indices.
    :param num_classes: Number of classes in segmentation.
    :param exclude_classes: List of class indices to exclude from IoU calculation.
    :return: IoU scores per class and mean IoU.
    """

    intersection_counts = np.zeros(num_classes)
    union_counts = np.zeros(num_classes)

    for cls in range(num_classes):
        if cls in exclude_classes:
            continue
        
        pred_inds = pred == cls
        target_inds = target == cls
        
        intersection_counts[cls] += torch.logical_and(pred_inds, target_inds).sum().item()
        union_counts[cls] += torch.logical_or(pred_inds, target_inds).sum().item()

    iou_per_class = [
        intersection_counts[cls] / union_counts[cls] if union_counts[cls] > 0 else float('nan')
        for cls in range(num_classes)
        if cls not in exclude_classes
    ]
    
    mean_iou = np.nanmean(iou_per_class) if iou_per_class else float('nan')

    return iou_per_class, mean_iou

# Defining Dice coefficient function
def compute_dice_coefficient(pred, target, num_classes=classNum, exclude_classes=[]):
    """
    Compute Dice Coefficient for multi-class segmentation,
    with the option to exclude specific classes from the computation.
    
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

# Defining pixel accuracy function
def compute_pixel_accuracy(pred, target, exclude_classes=[]):
    """
    Compute Pixel Accuracy,
    with the option to exclude specific classes from the computation.
    
    :param pred: Predicted segmentation map (H, W), with class indices.
    :param target: Ground truth segmentation map (H, W), with class indices.
    :param exclude_classes: List of class indices to exclude from accuracy calculation.
    :return: Pixel accuracy score.
    """

    exclude_mask = torch.zeros_like(target, dtype=torch.bool)
    for cls in exclude_classes:
        exclude_mask |= (target == cls)

    valid_pixels = ~exclude_mask
    correct = (pred[valid_pixels] == target[valid_pixels]).sum().item()
    total = valid_pixels.sum().item()

    return correct / total if total > 0 else float('nan')


"""1. UNet testing"""
if unet_epoch!=0:
    print("\nTesting UNet...")

    # Loading model
    class UNet(nn.Module):
        def __init__(self, num_classes=classNum):
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
        
    model = UNet().to(device)
    model.load_state_dict(torch.load(models_dir+f'UNET_DICE_LOSS_EPOCH_{unet_epoch}.pth', map_location=device))
    model.eval()

    print("(model loaded)")

    # Evaluating image by image
    all_intersection = np.zeros(3)
    all_union = np.zeros(3)
    all_dice = []
    all_pixel_accs = []
    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(image_dir, image_file)
        label_path = os.path.join(label_dir, image_file.replace(".jpg", ".png"))
        input_tensor, label_image = load_image_and_label(image_path, label_path)

        # Inference
        with torch.no_grad():
            output_tensor = model(input_tensor)
        output_tensor = F.softmax(output_tensor, dim=1) # softmax to get predicted class indices
        predicted_classes = torch.argmax(output_tensor, dim=1).squeeze(0).cpu()
        predicted_classes = F.interpolate( # resizing predicted mask to original label size
            predicted_classes.unsqueeze(0).unsqueeze(0).float(),
            size=(label_image.shape[0], label_image.shape[1]),
            mode='nearest'
        ).squeeze(0).squeeze(0).to(dtype=torch.uint8)
        label_tensor = torch.from_numpy(label_image).to(dtype=torch.uint8) # convertion of label image to class indices
        label_class_map = torch.zeros_like(label_tensor, dtype=torch.uint8)
        for pixel_value, class_idx in pixel_to_class.items():
            label_class_map[label_tensor == pixel_value] = class_idx

        # Computing metrics
        iou_per_class, _ = compute_iou(predicted_classes, label_class_map, exclude_classes=exludedClasses)
        _, mean_dice = compute_dice_coefficient(predicted_classes, label_class_map, exclude_classes = exludedClasses)
        pixel_acc = compute_pixel_accuracy(predicted_classes, label_class_map, exclude_classes = exludedClasses)

        for cls in range(classNum):
            if cls in exludedClasses:
                continue
            if not np.isnan(iou_per_class[cls]):
                all_intersection[cls] += torch.logical_and(predicted_classes == cls, label_class_map == cls).sum().item()
                all_union[cls] += torch.logical_or(predicted_classes == cls, label_class_map == cls).sum().item()
        
        all_dice.append(mean_dice)
        all_pixel_accs.append(pixel_acc)

    # Displaying overall scores
    final_iou_per_class = [
        all_intersection[cls] / all_union[cls] if all_union[cls] > 0 else float('nan')
        for cls in range(classNum) if cls not in exludedClasses
    ]
    overall_mean_iou = np.nanmean(final_iou_per_class)
    overall_mean_dice = np.nanmean(all_dice)
    overall_pixel_accuracy = np.mean(all_pixel_accs)

    print(f"Per-Class IoU: {final_iou_per_class}")
    print(f"Overall Mean IoU: {overall_mean_iou:.4f}")
    print(f"Overall Mean Dice Coefficient: {overall_mean_dice:.4f}")
    print(f"Overall Pixel Accuracy: {overall_pixel_accuracy:.4f}")


"""2. Autoencoder testing"""
if autoencoder_epoch*segmentation_decoder_epoch!=0:
    print("\nTesting Autoencoder...")

    # Loading model
    class Autoencoder(nn.Module):
        def __init__(self):
            super(Autoencoder, self).__init__()

            # Encoder
            self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),  # /2 size
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),  # /2 size
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),  # /2 size
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )

            # Decoder
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(128, 256, kernel_size=3, stride=2, padding=1, output_padding=1), # x2 size
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), # x2 size
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # x2 size
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1), # output layer
                nn.Sigmoid()
            )

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x

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
            x += residual # skip connection
            return self.relu(x)
        
    class SegmentationDecoder(nn.Module):
        def __init__(self, encoder):
            super(SegmentationDecoder, self).__init__()
            
            for param in encoder.parameters():
                param.requires_grad = True
            
            self.encoder = encoder

            # Decoder (with residual blocks and skip connections)
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1), # x2 size
                ResidualBlock(128, 128),
                nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # x2 size
                ResidualBlock(64, 64),
                nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),   # x2 size
                ResidualBlock(32, 32),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 4, kernel_size=1)
            )
        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x

    autoencoder = Autoencoder().to(device)
    autoencoder.load_state_dict(torch.load(models_dir+f'updated_complex_working_autoencoder_epoch_{autoencoder_epoch}.pth', map_location=device))
    autoencoder.eval()

    # Loading segmentation model
    segmentation_model = SegmentationDecoder(autoencoder.encoder).to(device)
    segmentation_model.load_state_dict(torch.load(models_dir+f'Finetuned_Segmentation_82_Decoder_DICE_epoch_{segmentation_decoder_epoch}.pth', map_location=device))
    segmentation_model.eval()
    
    print("(model loaded)")
    
    # Evaluating image by image
    all_intersection = np.zeros(3)
    all_union = np.zeros(3)
    all_dice = []
    all_pixel_accs = []
    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(image_dir, image_file)
        label_path = os.path.join(label_dir, image_file.replace(".jpg", ".png"))
        input_tensor, label_image = load_image_and_label(image_path, label_path)
        
        # Inference
        with torch.no_grad():
            output_tensor = segmentation_model(input_tensor)
        output_tensor = F.softmax(output_tensor, dim=1)
        predicted_classes = torch.argmax(output_tensor, dim=1).squeeze(0).cpu()
        predicted_classes = F.interpolate( # resizing predicted mask to original label size
            predicted_classes.unsqueeze(0).unsqueeze(0).float(),
            size=(label_image.shape[0], label_image.shape[1]),
            mode='nearest'
        ).squeeze(0).squeeze(0).to(dtype=torch.uint8)
        label_tensor = torch.tensor(label_image, dtype=torch.uint8)
        label_class_map = torch.zeros_like(label_tensor)
        for pixel_value, class_idx in pixel_to_class.items():
            label_class_map[label_tensor == pixel_value] = class_idx
        
        # Computing metrics
        iou_per_class, _ = compute_iou(predicted_classes, label_class_map, exclude_classes=exludedClasses)
        _, mean_dice = compute_dice_coefficient(predicted_classes, label_class_map, exclude_classes = exludedClasses)
        pixel_acc = compute_pixel_accuracy(predicted_classes, label_class_map, exclude_classes = exludedClasses)
        
        for cls in range(classNum):
            if cls in exludedClasses:
                continue
            if not np.isnan(iou_per_class[cls]):
                all_intersection[cls] += torch.logical_and(predicted_classes == cls, label_class_map == cls).sum().item()
                all_union[cls] += torch.logical_or(predicted_classes == cls, label_class_map == cls).sum().item()
        
        all_dice.append(mean_dice)
        all_pixel_accs.append(pixel_acc)

    # Displaying overall scores
    final_iou_per_class = [
        all_intersection[cls] / all_union[cls] if all_union[cls] > 0 else float('nan')
        for cls in range(classNum) if cls not in exludedClasses
    ]
    overall_mean_iou = np.nanmean(final_iou_per_class)
    overall_mean_dice = np.nanmean(all_dice)
    overall_pixel_accuracy = np.mean(all_pixel_accs)

    print(f"Per-Class IoU: {final_iou_per_class}")
    print(f"Overall Mean IoU: {overall_mean_iou:.4f}")
    print(f"Overall Mean Dice Coefficient: {overall_mean_dice:.4f}")
    print(f"Overall Pixel Accuracy: {overall_pixel_accuracy:.4f}")


"""3. CLIP testing"""
if clip_decoder_epoch!=0:
    print("\nTesting CLIP...")

    # Loading model
    class ClipDecoder(nn.Module):
        def __init__(self, clip_feature_dim=512, num_classes=classNum):
            super(ClipDecoder, self).__init__()

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

    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    # Loading segmentation model
    clip_segmentation_model = ClipDecoder().to(device)
    clip_segmentation_model.load_state_dict(torch.load(models_dir+f'New_clip_segmentation_model_epoch_{clip_decoder_epoch}.pth', map_location=device))
    clip_segmentation_model.eval()

    print("(model loaded)")

    # Evaluating image by image
    all_intersection = {cls: 0 for cls in range(classNum)}
    all_union = {cls: 0 for cls in range(classNum)}
    all_dice = []
    all_pixel_accs = []
    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(image_dir, image_file)
        label_path = os.path.join(label_dir, image_file.replace('.jpg', '.png'))
        _, label_image = load_image_and_label(image_path, label_path)

        # CLIP feature extraction
        image_clip = Image.open(image_path).convert("RGB") 
        image_clip_tensor = preprocess_val(image_clip).unsqueeze(0).to(device)
        with torch.no_grad():
            clip_features = model.encode_image(image_clip_tensor)
        clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)

        # Inference
        image_seg = Image.open(image_path).convert("RGB")
        image_tensor = transforms.ToTensor()(image_seg).unsqueeze(0).to(device)
        with torch.no_grad():
            output_tensor = clip_segmentation_model(image_tensor, clip_features)
        pred_mask = output_tensor.squeeze(0) # output conversion to predicted segmentation mask
        pred_mask = torch.argmax(pred_mask, dim=0).cpu()
        pred_mask = F.interpolate( # resizing predicted mask to original label size
            pred_mask.unsqueeze(0).unsqueeze(0).float(),
            size=(label_image.shape[0], label_image.shape[1]),
            mode='nearest'
        ).squeeze(0).squeeze(0).to(dtype=torch.uint8)
        label_tensor = torch.from_numpy(label_image).to(dtype=torch.uint8)
        gt_mask_class = torch.zeros_like(label_tensor, dtype=torch.uint8)
        for pixel_value, class_idx in pixel_to_class.items():
            gt_mask_class[label_tensor == pixel_value] = class_idx
        
        # Computing metrics
        iou_per_class, _ = compute_iou(pred_mask, gt_mask_class, classNum, exclude_classes=exludedClasses)
        _, mean_dice = compute_dice_coefficient(pred_mask, gt_mask_class, exclude_classes=exludedClasses)
        mean_dice = float(mean_dice)
        pixel_acc = compute_pixel_accuracy(pred_mask, gt_mask_class, exclude_classes=exludedClasses)

        for cls in range(classNum):
            if cls in exludedClasses:
                continue
            if not np.isnan(iou_per_class[cls]):
                all_intersection[cls] += torch.logical_and(pred_mask == cls, gt_mask_class == cls).sum().item()
                all_union[cls] += torch.logical_or(pred_mask == cls, gt_mask_class == cls).sum().item()

        all_dice.append(mean_dice)
        all_pixel_accs.append(pixel_acc)

    # Displaying overall scores
    final_iou_per_class = [
        all_intersection[cls] / all_union[cls] if all_union[cls] > 0 else float('nan')
        for cls in range(classNum) if cls not in exludedClasses
    ]
    overall_mean_iou = np.nanmean(final_iou_per_class)
    overall_mean_dice = np.nanmean(all_dice)
    overall_pixel_accuracy = np.mean(all_pixel_accs)

    print(f"IoU per class: {final_iou_per_class}")
    print(f"Overall Mean IoU: {overall_mean_iou:.4f}")
    print(f"Overall Mean Dice Coefficient: {overall_mean_dice:.4f}")
    print(f"Overall Pixel Accuracy: {overall_pixel_accuracy:.4f}")


"""4. Prompted CLIP testing"""
if prompted_clip_decoder_epoch!=0:
    print("\nTesting prompted CLIP...")

    # Loading model
    class PromptedClipDecoder(nn.Module):
        def __init__(self, clip_feature_dim=512, num_classes=classNum):
            super(PromptedClipDecoder, self).__init__()

            # Project CLIP features
            self.clip_fc = nn.Linear(clip_feature_dim, 256)  

            # CNN Feature Extractor (with 4 input channels: RGB + Heatmap)
            self.encoder = nn.Sequential(
                nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
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

    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    # Loading segmentation model
    prompted_clip_segmentation_model = PromptedClipDecoder().to(device)
    prompted_clip_segmentation_model.load_state_dict(torch.load(models_dir+f'PROMPTED_CLIP_CROSS_epoch_{prompted_clip_decoder_epoch}.pth', map_location=device))
    prompted_clip_segmentation_model.eval()

    print("(model loaded)")

    # Evaluating image by image
    all_intersection = {cls: 0 for cls in range(classNum)}
    all_union = {cls: 0 for cls in range(classNum)}
    all_dice = []
    all_pixel_accs = []
    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(image_dir, image_file)
        label_path = os.path.join(label_dir, image_file.replace('.jpg', '.png'))
        heatmap_path = os.path.join(heatmap_dir, image_file.split('.')[0]+'_heatmap.png')
        _, label_image = load_image_and_label(image_path, label_path)

        # CLIP feature extraction
        image_clip = Image.open(image_path).convert("RGB") 
        image_clip_tensor = preprocess_val(image_clip).unsqueeze(0).to(device)
        with torch.no_grad():
            clip_features = model.encode_image(image_clip_tensor)
        clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)

        # Inference
        image_seg = Image.open(image_path).convert("RGB").resize((500, 500))
        heatmap = Image.open(heatmap_path).convert('L')
        heatmap = transforms.ToTensor()(heatmap).unsqueeze(0).to(device)
        image_tensor = transforms.ToTensor()(image_seg).unsqueeze(0).to(device)
        image_tensor = torch.cat([image_tensor, heatmap], dim=1)
        with torch.no_grad():
            output_tensor = prompted_clip_segmentation_model(image_tensor, clip_features)
        pred_mask = output_tensor.squeeze(0) # output conversion to predicted segmentation mask
        pred_mask = torch.argmax(pred_mask, dim=0).cpu()
        pred_mask = F.interpolate( # resizing predicted mask to original label size
            pred_mask.unsqueeze(0).unsqueeze(0).float(),
            size=(label_image.shape[0], label_image.shape[1]),
            mode='nearest'
        ).squeeze(0).squeeze(0).to(dtype=torch.uint8)
        label_tensor = torch.from_numpy(label_image).to(dtype=torch.uint8)
        gt_mask_class = torch.zeros_like(label_tensor, dtype=torch.uint8)
        for pixel_value, class_idx in pixel_to_class.items():
            gt_mask_class[label_tensor == pixel_value] = class_idx
        
        # Computing metrics
        iou_per_class, _ = compute_iou(pred_mask, gt_mask_class, classNum, exclude_classes=exludedClasses)
        _, mean_dice = compute_dice_coefficient(pred_mask, gt_mask_class, exclude_classes=exludedClasses)
        mean_dice = float(mean_dice)
        pixel_acc = compute_pixel_accuracy(pred_mask, gt_mask_class, exclude_classes=exludedClasses)

        for cls in range(classNum):
            if cls in exludedClasses:
                continue
            if not np.isnan(iou_per_class[cls]):
                all_intersection[cls] += torch.logical_and(pred_mask == cls, gt_mask_class == cls).sum().item()
                all_union[cls] += torch.logical_or(pred_mask == cls, gt_mask_class == cls).sum().item()

        all_dice.append(mean_dice)
        all_pixel_accs.append(pixel_acc)

    # Displaying overall scores
    final_iou_per_class = [
        all_intersection[cls] / all_union[cls] if all_union[cls] > 0 else float('nan')
        for cls in range(classNum) if cls not in exludedClasses
    ]
    overall_mean_iou = np.nanmean(final_iou_per_class)
    overall_mean_dice = np.nanmean(all_dice)
    overall_pixel_accuracy = np.mean(all_pixel_accs)

    print(f"IoU per class: {final_iou_per_class}")
    print(f"Overall Mean IoU: {overall_mean_iou:.4f}")
    print(f"Overall Mean Dice Coefficient: {overall_mean_dice:.4f}")
    print(f"Overall Pixel Accuracy: {overall_pixel_accuracy:.4f}")