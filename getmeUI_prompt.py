import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import open_clip
import numpy as np
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
import plotly.express as px
import plotly.graph_objs as go
from streamlit_image_coordinates import streamlit_image_coordinates

st.title("Image Segmentation App")
classesNum = 4
# Load CLIP model and segmentation model
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, _, preprocess_val = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    clip_model = clip_model.to(device)

    class ClipDecoder(nn.Module):
        def __init__(self, clip_feature_dim=512, num_classes=classesNum):
            super(ClipDecoder, self).__init__()

            # Project CLIP features
            self.clip_fc = nn.Linear(clip_feature_dim, 256)  

            # CNN Feature Extractor (Now 4 input channels: RGB + Heatmap)
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
            # Extract features from the CNN encoder
            cnn_features = self.encoder(image)
            
            # Pass the CLIP features through the linear layer and reshape
            clip_features = self.clip_fc(clip_features).view(clip_features.shape[0], 256, 1, 1)
            
            # Expand the CLIP features to match the spatial size of the CNN output
            clip_features = clip_features.expand(-1, -1, cnn_features.shape[2], cnn_features.shape[3])
            
            # Concatenate CNN features and CLIP features along the channel dimension
            fusion = torch.cat([cnn_features, clip_features], dim=1)
            
            # Pass the fused features through the decoder
            segmentation_output = self.decoder(fusion)
            
            return segmentation_output

    model_path = 'Data/Output/Models/PROMPTED_CLIP_CROSS_epoch_44.pth'
    segmentation_model = ClipDecoder().to(device)
    segmentation_model.load_state_dict(torch.load(model_path, map_location=device))
    segmentation_model.eval()
    return clip_model, preprocess_val, segmentation_model, device

clip_model, preprocess_val, model, device = load_models()

# Extract CLIP features
def extract_clip_features(image):
    image = preprocess_val(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = clip_model.encode_image(image)
    features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features.to(device)

# Segment the image
# def segment_image(image, heatmap):
#     # Resize the image first
#     transform = transforms.Compose([
#         transforms.Resize((256, 256)),  # Resize image first
#         transforms.ToTensor()
#     ])
    
#     input_tensor = transform(image).to(device)  # Resized image

#     # Generate heatmap **AFTER resizing the image**
#     heatmap_resized = generate_gaussian_heatmap((256, 256), (x, y), intensity=1.0, sigma=50)
    
#     # Convert heatmap to tensor
#     heatmap_resized = Image.fromarray((heatmap_resized * 255).astype(np.uint8))  # Convert to PIL
#     heatmap_resized = transforms.ToTensor()(heatmap_resized).to(device)  # Convert to tensor

#     # Concatenate along the channel dimension
#     input_tensor = torch.cat([input_tensor, heatmap_resized], dim=0)  # Shape (4, 256, 256)
#     input_tensor = input_tensor.unsqueeze(0)  # Add batch dim: (1, 4, 256, 256)

#     # Extract CLIP features
#     clip_features = extract_clip_features(image)

#     # Run segmentation model
#     with torch.no_grad():
#         output = model(input_tensor, clip_features)

#     output = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
#     return output

def segment_image(image, heatmap):
    clip_features = extract_clip_features(image)

    # Resize and normalize input image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    ])

    input_tensor = transform(image).to(device).to(torch.float32)
    heatmap = transforms.ToTensor()(heatmap).to(device).to(torch.float32)
    input_tensor = torch.cat([input_tensor, heatmap], dim=0)
    input_tensor = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor, clip_features)
    output = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
    return output


# Visualize segmented output as a colormap
def visualize_segmentation(segmentation_output):
    plt.figure(figsize=(5, 5))
    plt.imshow(segmentation_output, cmap='jet')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('/tmp/segmentation.png')
    return '/tmp/segmentation.png'

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

def generate_gaussian_heatmap(size, center, intensity=1.0, sigma=50):
    x0, y0 = center
    x = np.arange(size[0])
    y = np.arange(size[1])
    X, Y = np.meshgrid(x, y)
    heatmap = intensity * np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
    return heatmap

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((256, 256))  # Resize image to 256x256
    st.image(image, caption='Uploaded Image', use_column_width=True)

    #### Display the image using Plotly to enable clicks ####
    st.write("Click on a point in the image to select coordinates.")
    
    coords = streamlit_image_coordinates(image, key="click_image")

    if coords:
        x, y = coords["x"], coords["y"]
        st.write(f"Selected coordinates: x = {x}, y = {y}")
    else:
        x = 100
        y = 100
        st.write(f"Selected coordinates: x = {x}, y = {y}")
    
    # Generate heatmap with the resized image dimensions
    heatmap = generate_gaussian_heatmap((256, 256), (x, y), intensity=1.0, sigma=50)
    st.write("Heatmap generated!")

    # Display the heatmap
    st.image(heatmap, caption='Generated Gaussian Heatmap', use_column_width=True, clamp=True)

    st.write("Segmenting...")

    # Run segmentation with the heatmap as a parameter
    segmented_output = segment_image(image, heatmap)

    # Visualize and display segmented output
    segmented_image_path = visualize_segmentation(segmented_output)
    st.image(segmented_image_path, caption='Segmented Image', use_column_width=True)
    st.success("Segmentation Complete!")
