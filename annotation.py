import cv2
import os
import pickle

# Define paths
image_dir = "/Users/sahilbavishi/Desktop/Edi/Semester Two/Computer Vision/Dataset/Repository/Computer-Vision-Project-2025/Data/Input/TrainVal/color"
annotations_dir = "/Users/sahilbavishi/Desktop/Edi/Semester Two/Computer Vision/Dataset/Repository/Computer-Vision-Project-2025/Data/Input/TrainVal/annotations"

# Ensure the annotations directory exists
os.makedirs(annotations_dir, exist_ok=True)

# Specify the range of images
start = 1  
end = 50   

# Generate the output file name for coordinates
output_file = os.path.join(annotations_dir, f"annotations_{start}_to_{end}.pkl")

# Get all images in the directory and sort them
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png"))])
image_files = image_files[start - 1:end]  # Adjusted to 0-based indexing

# Dictionary to store annotations
annotations = {}
clicked = False  # Flag to track click status
point = None  # Store clicked point

# Mouse click event callback function
def click_event(event, x, y, flags, param):
    global clicked, point
    if event == cv2.EVENT_LBUTTONDOWN:  
        point = (x, y)
        clicked = True  # Set flag

# Loop through images
for image_name in image_files:
    current_image = image_name
    image_path = os.path.join(image_dir, image_name)
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading {image_name}, skipping...")
        continue

    img = cv2.resize(img, (500, 500))
    clicked = False  
    point = None  

    # Display image and wait for a click
    cv2.imshow("Annotate Nose - Click on the nose", img)
    cv2.setMouseCallback("Annotate Nose - Click on the nose", click_event)

    while not clicked:
        cv2.waitKey(1)  

    cv2.destroyAllWindows()  

    if point:  
        annotations[current_image] = point  

        # Draw point on image
        cv2.circle(img, point, 5, (0, 0, 255), -1)  # Red dot

        # Save annotated image
        annotated_path = os.path.join(annotations_dir, f"annotated_{image_name}")
        cv2.imwrite(annotated_path, img)
        print(f"Saved annotated image: {annotated_path}")

# Save annotations to a .pkl file
with open(output_file, "wb") as f:
    pickle.dump(annotations, f)

print(f"Annotations saved to {output_file}")
cv2.destroyAllWindows()
