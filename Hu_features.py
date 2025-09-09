import numpy as np
from PIL import Image
import os
import csv

# Function to calculate central moments
def central_moment(data, p, q, x_centroid, y_centroid):
    y, x = np.ogrid[:data.shape[0], :data.shape[1]]
    return np.sum(((x - x_centroid) ** p) * ((y - y_centroid) ** q) * data)

# Function to calculate central normalized moments
def central_normalized_moment(moments, p, q, m00):
    return moments[f'm_{p}{q}'] / (m00 ** (((p + q) / 2) + 1))

# Function to calculate Hu moments
def calculate_hu_moments(M):
    S = {}
    S[1] = M['M_20'] + M['M_02']
    S[2] = (M['M_20'] - M['M_02']) ** 2 + 4 * (M['M_11'] ** 2)
    S[3] = (M['M_30'] - 3 * M['M_12']) ** 2 + (3 * M['M_21'] - M['M_03']) ** 2
    S[4] = (M['M_30'] + M['M_12']) ** 2 + (M['M_03'] + M['M_21']) ** 2
    S[5] = (M['M_30'] - 3 * M['M_12']) * (M['M_30'] + M['M_12']) * (
            (M['M_30'] + M['M_12']) ** 2 - 3 * (M['M_03'] + M['M_21']) ** 2) + \
           (3 * M['M_21'] - M['M_03']) * (M['M_03'] + M['M_21']) * (
                   3 * (M['M_30'] + M['M_12']) ** 2 - (M['M_03'] + M['M_21']) ** 2)
    S[6] = (M['M_20'] - M['M_02']) * ((M['M_30'] + M['M_12']) ** 2 - (M['M_03'] + M['M_21']) ** 2) + \
           4 * M['M_11'] * (M['M_30'] + M['M_12']) * (M['M_03'] + M['M_21'])
    S[7] = (3 * M['M_21'] - M['M_03']) * (M['M_30'] + M['M_12']) * (
            (M['M_30'] + M['M_12']) ** 2 - 3 * (M['M_03'] + M['M_21']) ** 2) - \
           (M['M_30'] - 3 * M['M_12']) * (M['M_03'] + M['M_21']) * (
                   3 * (M['M_30'] + M['M_12']) ** 2 - (M['M_03'] + M['M_21']) ** 2)
    return S

# Function to process a single image and return Hu moments
def process_image(image_path):
    im = Image.open(image_path).convert("L")
    threshold = 128
    im_bin = im.point(lambda p: 255 if p > threshold else 0)
    data = np.array(im_bin)

    m00 = np.sum(data == 255).astype(np.float64) * 255

    # Tính toán tọa độ trọng tâm
    height, width = data.shape
    x_centroid = np.sum(np.arange(width) * np.sum(data, axis=0)) / m00
    y_centroid = np.sum(np.arange(height) * np.sum(data, axis=1)) / m00

    # Tính toán các mômen trung tâm đến bậc 3
    moments = {}
    for p in range(4):
        for q in range(4):
            if p + q <= 3:
                moments[f'm_{p}{q}'] = central_moment(data, p, q, x_centroid, y_centroid)

    # Tính toán các mômen chuẩn hóa trung tâm
    normalized_moments = {}
    for p in range(4):
        for q in range(4):
            if 0 < p + q <= 3:
                normalized_moments[f'M_{p}{q}'] = central_normalized_moment(moments, p, q, m00)

    # Tính toán các mômen Hu
    hu_moments = calculate_hu_moments(normalized_moments)

    return hu_moments

# Function to process a folder of images and save results to CSV
# Function to process a folder of images and save results to CSV
def process_images_in_folder(input_folder, output_csv):
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ['Image'] + [f"S{i}" for i in range(1, 8)] + ['Labels']
        writer.writerow(header)  # Write header to CSV

        # Iterate through the folder and process each image
        for filename in os.listdir(input_folder):
            if filename.endswith(".png") or filename.endswith(".jpg"):  # Filter for image files
                # Extract image number from filename (assuming a format like img1.png)
                image_num = int(''.join(filter(str.isdigit, filename)))

                # Determine class based on image number range
                if 1 <= image_num <= 30:
                    image_class = '1'
                elif 31 <= image_num <= 60:
                    image_class = '2'
                elif 61 <= image_num <= 90:
                    image_class = '3'
                else:
                    continue  # Skip if the image number is out of defined class ranges

                # Full path to image
                image_path = os.path.join(input_folder, filename)
                hu_moments = process_image(image_path)  # Calculate Hu moments

                # Prepare the row for CSV
                row = [filename] + [hu_moments[i] for i in range(1, 8)] + [image_class]
                writer.writerow(row)  # Write row to CSV

# Example usage
input_folder = "new3/NhiPhan"  # Input folder with images
output_csv = "Output/hu.csv"  # Output CSV file

process_images_in_folder(input_folder, output_csv)  # Process the folder and save results
