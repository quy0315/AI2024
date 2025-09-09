import os
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import hog

input_folder = 'new3/Goc'
resized_folder = 'resize'  # Folder to save resized images

# Create the folder if it doesn't exist
os.makedirs(resized_folder, exist_ok=True)

# Path to save the output CSV file
output_csv = 'Output/hog.csv'

# List to store HOG features
hog_features_list = []

# Labels for each group of 30 images
labels = ['1', '2', '3']  # Adjusted to match the requirement for 3 classes

def resize_image(image, target_size):
    # Get original size
    original_size = image.size
    target_width, target_height = target_size

    # Calculate resize ratio
    ratio = min(target_width / original_size[0], target_height / original_size[1])
    new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))

    # Resize image
    resized_image = image.resize(new_size)  # Removed Image.ANTIALIAS

    # Create a new blank (white or black) image with the target size
    new_image = Image.new("L", (target_width, target_height))

    # Calculate position to paste the resized image in the center
    paste_x = (target_width - new_size[0]) // 2
    paste_y = (target_height - new_size[1]) // 2

    # Paste the resized image onto the new blank image
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image

# Loop through all files in the folder
for idx, filename in enumerate(os.listdir(input_folder), start=1):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        try:
            # Read the binary image with PIL
            img_path = os.path.join(input_folder, filename)
            image = Image.open(img_path).convert('L')  # Convert to grayscale

            # Resize the image to a fixed size for HOG calculation
            resized_image = resize_image(image, (128, 128))  # Resize to 128x128
            resized_image_np = np.array(resized_image)

            # Save the resized image to the folder
            resized_image.save(os.path.join(resized_folder, filename))

            # Calculate HOG features
            hog_features = hog(resized_image_np, orientations=8, pixels_per_cell=(16, 16),
                               cells_per_block=(1, 1), visualize=False)

            # Flatten to 1D array for storage
            hog_features = hog_features.flatten()

            # Determine the label based on the image index
            label = labels[(idx - 1) // 30]  # Group images into sets of 30

            # Append the result to the list
            hog_features_list.append({
                'HOG': list(hog_features),  # Store the entire HOG feature as a list
                'Tên Ảnh': filename,        # Add the image name
                'Labels': label             # Add the label
            })

        except Exception as e:
            print(f"Error occurred while processing image {img_path}: {str(e)}")

# Convert HOG features list to DataFrame if not empty
if hog_features_list:
    df = pd.DataFrame(hog_features_list)

    # Separate each HOG value into individual columns
    hog_df = pd.DataFrame(df['HOG'].to_list(), columns=[f'HOG_{i + 1}' for i in range(len(df['HOG'][0]))])

    # Combine columns with image names and HOG features
    temp_df = pd.concat([df[['Tên Ảnh']], hog_df], axis=1)

    # Add the label column at the end
    final_df = pd.concat([temp_df, df[['Labels']]], axis=1)

    # Save the DataFrame to a CSV file
    final_df.to_csv(output_csv, index=False, encoding='utf-8-sig')

    print(f"Results have been saved to the CSV file: {output_csv}")
else:
    print("No HOG features were calculated.")
