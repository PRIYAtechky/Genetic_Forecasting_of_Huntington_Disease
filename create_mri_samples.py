from PIL import Image
import os

# Folder to save images
folder = r"P:\HD_ViT_Project\data\mri_images"
os.makedirs(folder, exist_ok=True)

# Colors for each class
colors = {
    "normal": (0, 255, 0),        # green
    "intermediate": (255, 255, 0), # yellow
    "pathogenic": (255, 0, 0)     # red
}

# Create 3 images per class
for label, color in colors.items():
    for i in range(1, 4):
        img = Image.new('RGB', (224, 224), color)
        img.save(os.path.join(folder, f"{label}_{i}.jpg"))

print("Sample MRI images created!")
