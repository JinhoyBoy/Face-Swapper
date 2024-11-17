import os
import shutil

def copy_pairs(image_dir, label_dir, output_image_dir, output_label_dir, num_pairs_to_copy):
    # Ensure output directories exist
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    
    # Get sorted lists of images and labels
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg'))])
    label_files = sorted([f for f in os.listdir(label_dir) if f.lower().endswith('.txt')])
    
    # Match only pairs that have both an image and a corresponding label
    matched_files = [(img, lbl) for img, lbl in zip(image_files, label_files) 
                     if os.path.splitext(img)[0] == os.path.splitext(lbl)[0]]
    
    # Limit to the specified number of pairs
    for i, (img_file, lbl_file) in enumerate(matched_files[:num_pairs_to_copy]):
        # Copy image and label files to their respective destinations
        shutil.copy(os.path.join(image_dir, img_file), os.path.join(output_image_dir, img_file))
        shutil.copy(os.path.join(label_dir, lbl_file), os.path.join(output_label_dir, lbl_file))
        print(f"Copied pair {i+1}: {img_file} and {lbl_file}")

    print(f"Completed copying {len(matched_files[:num_pairs_to_copy])} pairs of files.")

# Define the source and destination directories
image_dir = "dataset/images"
label_dir = "dataset/labels"
output_image_dir = "train/images"
output_label_dir = "train/labels"

# Define the number of pairs to copy
num_pairs_to_copy = 2000

# Call the function
copy_pairs(image_dir, label_dir, output_image_dir, output_label_dir, num_pairs_to_copy)