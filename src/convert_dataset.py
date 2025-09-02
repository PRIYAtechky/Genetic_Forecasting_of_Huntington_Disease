import os
import shutil

# Paths
SRC_TRAIN = r"P:\Brain-Tumor-Classification-Data\Training"
SRC_TEST  = r"P:\Brain-Tumor-Classification-Data\Testing"
DEST_DIR  = r"P:\HD_ViT_Project\data\mri_images"

# Map original dataset classes to your 3-class setup
CLASS_MAPPING = {
    "glioma_tumor":      "intermediate",
    "meningioma_tumor":  "pathogenic",
    "pituitary_tumor":   "pathogenic",   # <— included now
    "no_tumor":          "normal"
}

# Create destination folders
for cls in set(CLASS_MAPPING.values()):
    os.makedirs(os.path.join(DEST_DIR, cls), exist_ok=True)

def copy_images(src_folder, split_name):
    for folder in os.listdir(src_folder):
        if folder in CLASS_MAPPING:
            new_class = CLASS_MAPPING[folder]
            src_path = os.path.join(src_folder, folder)
            dest_path = os.path.join(DEST_DIR, new_class)
            for file in os.listdir(src_path):
                if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    src_file = os.path.join(src_path, file)
                    # Avoid overwrite: prefix with split name
                    base, ext = os.path.splitext(file)
                    new_name = f"{split_name}_{base}{ext}"
                    shutil.copy(src_file, os.path.join(dest_path, new_name))

copy_images(SRC_TRAIN, "train")
copy_images(SRC_TEST,  "test")
print("✅ Dataset converted and copied to:", DEST_DIR)
