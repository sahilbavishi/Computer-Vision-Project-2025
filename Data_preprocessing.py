# Authors: Sahil Mehul Bavishi (s2677266) and Matteo Spadaccia (s2748897)
# Subject: Computer Vision Coursework. Pre-Processing 
# Date: 21.03.2025


""""0. Preliminary code"""

# Setting code-behaviour varaibles
doResize = True     # if True, the input images' resizing is run, otherwise the saved outcomes are used
doAugment = True    # if True, the resized images' augmentaion is run, otherwise the saved outcomes are used
genVISUALS = True   # set to False in order to avoid time-consuming visualizations' genaration (images are instead displayed as pre-saved in 'Output/Visuals' folder)
dpi = 500           # dpi for .pdf-saved images visualization (with genVISUALS = False)

# Importing useful libraries
import os
from pathlib import Path
from pdf2image import convert_from_path # (also install poppler-utils)
from IPython.display import display
import numpy as np
from PIL import Image
import seaborn as sns
import cv2
import albumentations as A
import matplotlib.pyplot as plt

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


"""1. Dataset exploration"""

# Gauging image sizes
if genVISUALS or doResize:
  widths = []
  heights = []
  for img_file in input_trainval:
      with Image.open(os.path.join(input_folder_trainval+'/color', img_file)) as img:
          width, height = img.size
          widths.append(width)
          heights.append(height)

# Visualizing histogram distribution of dimensions
if genVISUALS or doResize:
  plt.figure(figsize=(12, 5))
  plt.subplot(1, 2, 1)
  plt.hist(widths, bins=20, color='lightblue', edgecolor='black')
  plt.xlabel("Width")
  plt.ylabel("Frequency")
  plt.title("Widths distribution")
  plt.subplot(1, 2, 2)
  plt.hist(heights, bins=20, color='skyblue', edgecolor='black')
  plt.xlabel("Height")
  plt.ylabel("Frequency")
  plt.title("Heights distribution")
  plt.tight_layout()
  plt.savefig(Path('Data/Output/Visuals/dimensions_histogram.pdf'))
  plt.show()
else:
  for image in convert_from_path(Path('Data/Output/Visuals/dimensions_histogram.pdf'), dpi=dpi):
    display(image)

# Visualizing boxplot distribution of dimensions
if genVISUALS or doResize:
  plt.figure(figsize=(10, 6))
  data = {"Width": widths, "Height": heights}
  sns.boxplot(data=data, palette=['lightblue', 'skyblue'])
  plt.title("Boxplot of Widths and Heights")
  plt.ylabel("Pixels")
  plt.tight_layout()
  plt.savefig(Path('Data/Output/Visuals/dimensions_boxplot.pdf'))
  plt.show()
else:
  for image in convert_from_path(Path('Data/Output/Visuals/dimensions_boxplot.pdf'), dpi=dpi):
    display(image)

# Printing stats
print(f"Input set size: {len(input_trainval)}\n")
if doResize:
  min_width, min_height = np.min(widths), np.min(heights)
  median_width, median_height = np.median(widths), np.median(heights)
  mean_width, mean_height = np.mean(widths), np.mean(heights)
  mode_width, mode_height = max(set(widths), key=widths.count), max(set(heights), key=heights.count)
  q3_width, q3_height = np.percentile(widths, 75), np.percentile(heights, 75)
  iqr_width = np.percentile(widths, 75) - np.percentile(widths, 25)
  iqr_height = np.percentile(heights, 75) - np.percentile(heights, 25)
  outlier_count_width = np.sum(widths > (q3_width + 1.5 * iqr_width))
  outlier_count_height = np.sum(heights > (q3_height + 1.5 * iqr_height))
  print(f"Min Size: {min_width}x{min_height}")
  print(f"Median Size: {median_width}x{median_height}")
  print(f"Mean Size: {mean_width:.2f}x{mean_height:.2f}")
  print(f"Mode Size: {mode_width}x{mode_height}")
  print(f"Q3 Size: {q3_width}x{q3_height}")
  print(f"Outliers in width: {outlier_count_width}")
  print(f"Outliers in height: {outlier_count_height}")
else:
  print("Q3 width and height values (both 500 pixels) were chosen for resizing.")


"""2. Images' resizing"""

# Resizing images (to Q3 width, Q3 height)
if doResize:
  imgResize = (int(q3_width), int(q3_height))
  widthsNP = np.array(widths)
  heightsNP = np.array(heights)
  i = 0
  for img_file in input_trainval:
      with Image.open(os.path.join(input_folder_trainval+'/color', img_file)) as img:
          img_resized = img.resize(imgResize, Image.Resampling.LANCZOS)
          if img_resized.mode == "RGBA":
            img_resized = img_resized.convert("RGB")
          img_resized.save(os.path.join(output_folder_resized_color, img_file), format="JPEG")
          i += 1
  print(f"{i} images resized to {int(q3_width)}x{int(q3_height)} and saved in {output_folder_resized_color}.")

# Resizing labels
  imgResize = (int(q3_width), int(q3_height))
  widthsNP = np.array(widths)
  heightsNP = np.array(heights)
  i = 0
  for img_file in input_trainval_labels:
      with Image.open(os.path.join(input_folder_trainval+'/label', img_file)) as img:
          img_resized = img.resize(imgResize, Image.Resampling.LANCZOS)
          if img_resized.mode == "RGBA":
            img_resized = img_resized.convert("RGB")
          img_resized.save(os.path.join(output_folder_resized_label, img_file), format="PNG")
          i += 1
  print(f"{i} labels resized to {int(q3_width)}x{int(q3_height)} and saved in {output_folder_resized_label}.")

else:
  print("Using previously resized images and labels (500x500).")
  imgResize = (500, 500)


"""3. Augmenting dataset"""

def clean_mask(mask): #(uncomment in augmentation process if desired)
    '''Ensures the mask contains only values in {0, 38, 75}, replacing 255 (outline) and any unexpected values with 0.'''
    valid_values = {0, 38, 75}
    mask[mask == 255] = 0
    mask = np.where(np.isin(mask, list(valid_values)), mask, 0)
    return mask.astype(np.uint8)

# Defining augmentations
augmentation = A.Compose([
   
    # Geometric transformations (applied to both images and masks)
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Affine(scale=(0.9, 1.1), translate_percent=(0.1, 0.1), rotate=(-15, 15), shear=(-10, 10), p=0.5),
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
    A.GridDistortion(num_steps=5, distort_limit=0.3, interpolation=0, p=0.5), # NN-interpolation
    A.OpticalDistortion(distort_limit=0.3, shift_limit=0.3, interpolation=0, p=0.5), # Nearest Neighbor

    # Intensity augmentations (applied to images only)
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.MotionBlur(blur_limit=5, p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.3)
], additional_targets={'mask': 'mask'})

if doAugment:
    i = 0
    print('Augmenting the data...')
    for img_file in os.listdir(output_folder_resized_color):
        img_path = os.path.join(output_folder_resized_color, img_file)
        mask_filename = os.path.splitext(img_file)[0] + ".png"
        mask_path = os.path.join(output_folder_resized_label, mask_filename)

        # Loading image and mask(in grayscale)
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            print(f"Skipping {img_file} as corresponding image or mask is missing!")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Saving original image and mask
        orig_img_path = os.path.join(output_folder_augmented_color, f'orig_{img_file}')
        orig_mask_path = os.path.join(output_folder_augmented_label, f'orig_{mask_filename}')
        Image.fromarray(img).save(orig_img_path, "JPEG")
        Image.fromarray(mask).save(orig_mask_path, "PNG")

        # Applying augmentation
        augmented = augmentation(image=img, mask=mask)
        aug_img, aug_mask = augmented['image'], augmented['mask']
        aug_mask = np.round(aug_mask).astype(np.uint8) # Ensuring mask remains categorical

        # Cleaning mask (uncomment if desired)
        #aug_mask = clean_mask(aug_mask)

        # Saving augmented image and mask
        output_img_path = os.path.join(output_folder_augmented_color, f'aug_{i}_{img_file}')
        Image.fromarray(aug_img).save(output_img_path, "JPEG")
        output_mask_filename = f'aug_{i}_{mask_filename}'
        output_mask_path = os.path.join(output_folder_augmented_label, output_mask_filename)
        Image.fromarray(aug_mask).save(output_mask_path, "PNG")

        i += 1

    print(f"{i} images and masks augmented, including originals, output saved in {output_folder_augmented_color} & {output_folder_augmented_label}.")

    # Computing the number of images needed to balance cats and dogs
    target_cat_count = 2492
    existing_cat_count = 1188
    additional_cats_needed = target_cat_count - existing_cat_count

    def is_cat(filename):
        cat_breeds = ["abyssinian", "bengal", "birman", "bombay", "british_shorthair", 
                      "egyptian_mau", "maine_coon", "persian", "ragdoll", "russian_blue", 
                      "siamese", "sphynx"]
        filename_lower = filename.lower()
        return any(breed in filename_lower for breed in cat_breeds)

    # Calculating how many augmentations per cat image
    cat_images = [img for img in os.listdir(output_folder_augmented_color) if is_cat(img.lower())]
    augmentations_per_image = 1

    i = 0
    for img_file in cat_images:
        img_path = os.path.join(output_folder_augmented_color, img_file)
        mask_filename = os.path.splitext(img_file)[0] + ".png"
        mask_path = os.path.join(output_folder_augmented_label, mask_filename)

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            print(f"Skipping {img_file} as image or mask is missing!")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Applying augmentation multiple times to balance dataset
        for j in range(augmentations_per_image):
            augmented = augmentation(image=img, mask=mask)
            aug_img, aug_mask = augmented['image'], augmented['mask']

            # Ensuring mask remains categorical
            aug_mask = np.round(aug_mask).astype(np.uint8)
            unique_values = np.unique(mask)
            for value in np.unique(aug_mask):
                if value not in unique_values:
                    aug_mask[aug_mask == value] = 0  

            # Saving augmented image and mask
            output_img_path = os.path.join(output_folder_augmented_color, f'aug_{i}_{j}_{img_file}')
            Image.fromarray(aug_img).save(output_img_path, "JPEG")
            output_mask_filename = f'aug_{i}_{j}_{mask_filename}'
            output_mask_path = os.path.join(output_folder_augmented_label, output_mask_filename)
            Image.fromarray(aug_mask).save(output_mask_path, "PNG")

            i += 1
            if i >= additional_cats_needed:
                break

else:
    print("Using previously augmented data.")