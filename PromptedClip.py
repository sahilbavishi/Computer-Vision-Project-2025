# Authors: Sahil Mehul Bavishi (s2677266) and Matteo Spadaccia (s2748897)
# Subject: Computer Vision Coursework. Clip
# Date: 21.03.2025


""""0. Preliminary code"""

# Setting code-behaviour varaibles
extractFeatures = True  # if True, the CLIP fetures are extracted, otherwise the saved ones are used

preprocessedSize = (256, 256) # Desired size to read input images and masks in
imgChannels = 3
classesNum = 4

validSize = 0.2
epochsNum = 100
batchSize = 16
learningRate = 1e-3

random_seed = 42

# Importing useful libraries
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import open_clip

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

# Preparing train, valid and test sets
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
trainval_images = [os.path.join(output_folder_augmented_color, f) for f in os.listdir(output_folder_augmented_color) if f.endswith('.jpg')]
trainval_masks = [os.path.join(output_folder_augmented_label, f) for f in os.listdir(output_folder_augmented_label) if f.endswith('.png')]
train_images, val_images, train_masks, val_masks = train_test_split(trainval_images, trainval_masks, test_size=validSize, random_state=random_seed)
test_images = [os.path.join(input_folder_test, 'color', f) for f in os.listdir(os.path.join(input_folder_test, 'color')) if f.endswith('.jpg')]
test_masks = [os.path.join(input_folder_test, 'label', f) for f in os.listdir(os.path.join(input_folder_test, 'label')) if f.endswith('.png')]


print(f"Train set size: {len(train_images)} ({(1-validSize)*100}%)")
print(f"Valid set size: {len(val_images)} ({(validSize)*100}%)")
print(f" Test set size: {len(test_images)}")
print(f"\nInput dimension: {preprocessedSize + (imgChannels,)}")
print(f"Batches' size: {batchSize}")


""""1. CLIP features extraction"""

# Loading CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
model = model.to(device)
tokenizer = open_clip.get_tokenizer("ViT-B-32")

# Preparing features'saving directory
clip_features_dir = "Data/Output/Models/ResOnlyClipTrainFeatures3"
os.makedirs(clip_features_dir, exist_ok=True)

if extractFeatures:
    # Extract and save CLIP features
    def extract_clip_features(image_path):
        image = Image.open(image_path).convert("RGB")
        image_tensor = preprocess_val(image).unsqueeze(0).to(device)  # Use open_clip's preprocessing
        with torch.no_grad():
            features = model.encode_image(image_tensor)  # Extract image features
        features = features / features.norm(p=2, dim=-1, keepdim=True)  # Normalize
        return features.cpu()

    clip_features_trainval = {}
    print("Extracting Clip Features")
    for image_name in trainval_images:
        image_path = image_name
        clip_features = extract_clip_features(image_path)

        # Save each feature tensor separately
        feature_path = os.path.join(clip_features_dir, f"{image_name.split('/')[-1].split('.')[0]}.pt")
        torch.save(clip_features, feature_path)

        clip_features_trainval[image_name] = feature_path


""""2. CLIP-based segmentation"""

# Loading CLIP features
clip_features_trainval = {
    image_name: torch.load(os.path.join(clip_features_dir, f"{image_name.split('/')[-1].split('.')[0]}.pt"))
    for image_name in trainval_images
}
print(f"Loaded {len(clip_features_trainval)} CLIP feature tensors.")

print("Saving Model here: Data/Output/Models2/PROMPTED_CLIP_DICE2_epoch_")

class SegmentationDataset(Dataset):
    def __init__(self, image_files, label_files, clip_features_dict, transform=None):
        self.image_files = image_files
        self.label_files = label_files
        self.clip_features_dict = clip_features_dict
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label_path = img_path.replace('.jpg', '.png').replace('color', 'label')
        
        # Compute heatmap path: assuming heatmaps are stored in a folder "heatmaps" 
        # located in the same parent folder as "color"
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        color_dir = os.path.dirname(img_path)
        parent_dir = os.path.dirname(color_dir)
        heatmap_dir = os.path.join(parent_dir, "heatmaps")
        heatmap_path = os.path.join(heatmap_dir, base_name + "_heatmap.png")

        # Loading image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        # Loading heatmap and converting to 1-channel tensor
        heatmap = Image.open(heatmap_path).convert('L')
        heatmap = transforms.ToTensor()(heatmap)
        
        # Concatenating image and heatmap to form a 4-channel input
        image = torch.cat([image, heatmap], dim=0)

        # Loading CLIP feature tensor
        image_name = img_path
        clip_feature = self.clip_features_dict[image_name]  # Shape: (512,)

        # Loading and processing label
        label = Image.open(label_path).convert('L')
        label = label.resize(preprocessedSize, Image.NEAREST)
        label = torch.tensor(np.array(label), dtype=torch.long)

        # Converting label classes
        label = torch.where(label == 38, 1, label)
        label = torch.where(label == 75, 2, label)
        label = torch.where(label == 255, 3, label)
        label = torch.where(label == 0, 0, label)

        return image, clip_feature, label

# Defining Transformations
transform = transforms.Compose([
    transforms.Resize(preprocessedSize, interpolation=Image.NEAREST),
    transforms.ToTensor()
])

# Loading Dataset & DataLoader
segmentation_dataset = SegmentationDataset(
    image_files=train_images,
    label_files=train_masks,
    clip_features_dict=clip_features_trainval,
    transform=None
)
segmentation_dataloader = DataLoader(segmentation_dataset, batch_size=batchSize, shuffle=True)

val_segmentation_dataset = SegmentationDataset(
    image_files=val_images,
    label_files=val_masks,
    clip_features_dict=clip_features_trainval,
    transform=None
)

val_segmentation_dataloader = DataLoader(val_segmentation_dataset, batch_size=batchSize, shuffle=True)

class SegmentationDecoder(nn.Module):
    def __init__(self, clip_feature_dim=512, num_classes=classesNum):
        super(SegmentationDecoder, self).__init__()

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


def dice_loss(pred, target, smooth=1e-6):
    pred = torch.softmax(pred, dim=1)
    target = torch.nn.functional.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

# Initialize Model, Loss, Optimizer
segmentation_model = SegmentationDecoder().to(device)
# segmentation_criterion = nn.CrossEntropyLoss()
# segmentation_criterion = lambda outputs, masks: nn.CrossEntropyLoss()(outputs, masks) + dice_loss(outputs, masks)

segmentation_criterion = lambda outputs, masks: dice_loss(outputs, masks)

print("Loss: segmentation_criterion = lambda outputs, masks: nn.CrossEntropyLoss()(outputs, masks) + dice_loss(outputs, masks)")

segmentation_optimizer = optim.Adam(segmentation_model.parameters(), lr=learningRate)
scheduler = torch.optim.lr_scheduler.StepLR(segmentation_optimizer, step_size=5, gamma=0.5)
scaler = torch.cuda.amp.GradScaler()

# Training model
for epoch in tqdm(range(epochsNum), desc="Training Progress"):
    segmentation_model.train()
    running_loss = 0.0

    print(f"Epoch [{epoch+1}/{epochsNum}] Running:")
    
    for images, clip_features, labels in segmentation_dataloader:
        images, clip_features, labels = images.to(device), clip_features.to(device), labels.to(device)

        segmentation_optimizer.zero_grad()

        # Mixed precision training
        with torch.cuda.amp.autocast(): 
            outputs = segmentation_model(images, clip_features)
            loss = segmentation_criterion(outputs, labels)

        # Scaling loss and backpropagating
        scaler.scale(loss).backward()
        scaler.step(segmentation_optimizer)
        scaler.update()

        running_loss += loss.item()

    segmentation_model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, clip_features, labels in val_segmentation_dataloader:
            images, clip_features, labels = images.to(device), clip_features.to(device), labels.to(device)
            outputs = segmentation_model(images, clip_features)
            loss = segmentation_criterion(outputs, labels)
            val_loss += loss.item()

    print(f'Epoch [{epoch+1}/{epochsNum}], Validation Loss : {val_loss / len(val_segmentation_dataloader)}')

    scheduler.step()
    
    avg_loss = running_loss / len(segmentation_dataloader)
    print(f"Epoch [{epoch+1}/{epochsNum}], Training Loss: {avg_loss:.4f}")
    print(f"Saved here: Data/Output/Models_DICEONLY/PROMPTED_CLIP_DiceOnly_epoch_{epoch+1}.pth")
    torch.save(segmentation_model.state_dict(), f'Data/Output/Models_DICEONLY/PROMPTED_CLIP_DiceOnly_epoch_{epoch+1}.pth')

print("Training complete!")