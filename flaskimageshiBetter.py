from flask import Flask, render_template, request, send_from_directory, jsonify
import os
from PIL import Image
import torch
import numpy as np
import torchvision.transforms as transforms
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import open_clip
import numpy as np
import matplotlib.pyplot as plt



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'
classesNum = 4
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

def generate_gaussian_heatmap(size, center, intensity=1.0, sigma=50):
    x0, y0 = center
    x = np.arange(size[0])
    y = np.arange(size[1])
    X, Y = np.meshgrid(x, y)
    heatmap = intensity * np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
    return heatmap

def segment_image(image, heatmap):
    clip_features = extract_clip_features(image)
    heatmap_resized = Image.fromarray((heatmap * 255).astype(np.uint8)).resize((256, 256))
    heatmap = np.array(heatmap_resized) / 255.0  # Convert back to float array
    print("Came here")
    # Resize and normalize input image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    input_tensor = transform(image).to(device).to(torch.float32)
    heatmap = torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0).to(device)
    input_tensor = torch.cat([input_tensor, heatmap], dim=0).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor, clip_features)
    output = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
    return output

def visualize_segmentation(segmented_output, original_size):
    # Define a color palette for 4 classes (0, 1, 2, 3)
    color_map = {
        0: [0, 0, 0],        # Black
        1: [0, 0, 255],      # Red
        2: [0, 255, 0],      # Green
        3: [255, 0, 0]       # Blue
    }

    # Create a color image based on the segmented output
    colored_output = np.zeros((segmented_output.shape[0], segmented_output.shape[1], 3), dtype=np.uint8)
    for label, color in color_map.items():
        colored_output[segmented_output == label] = color

    # Convert to a PIL image
    segmented_image = Image.fromarray(colored_output)
    segmented_image = segmented_image.resize(original_size, resample=Image.NEAREST)
    
    # Save the image
    segmented_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'segmented.png')
    segmented_image.save(segmented_image_path)
    
    return segmented_image_path

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        image = request.files['image']
        if image:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(image_path)
            return render_template('image_display.html', image_path=image_path)
        else:
            return "No image received"
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/process_click', methods=['POST'])
def process_click():
    try:
        data = request.json
        x, y = float(data['x']), float(data['y'])
        image_path = data['image_path']
        print(f"Clicked here {x}, {y}")
        image = Image.open(image_path).convert("RGB")
        image_size = image.size
        print(image_size)
        heatmap = generate_gaussian_heatmap(image_size, (x, y), intensity=1.0, sigma=50)
        heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], 'heatmap.png')

        plt.imsave(heatmap_path, heatmap, cmap='gray')

        # print("Came here")
        segmented_output = segment_image(image, heatmap)
        segmented_image_path = visualize_segmentation(segmented_output, image_size)
        print('Reached here')
        return jsonify({'heatmap': heatmap_path, 'segmented_image': segmented_image_path})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)