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


# Function to calculate Hu moments (already provided in your original code)
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


# Function to process a single image and return log Hu moments
def process_image(image_path):
    im = Image.open(image_path).convert("L")
    threshold = 128
    im_bin = im.point(lambda p: 255 if p > threshold else 0)
    data = np.array(im_bin)

    m00 = np.sum(data == 255).astype(np.float64) * 255

    # Calculate image centroid
    height, width = data.shape
    x_centroid = np.sum(np.arange(width) * np.sum(data, axis=0)) / m00
    y_centroid = np.sum(np.arange(height) * np.sum(data, axis=1)) / m00

    # Calculate central moments up to order 3
    moments = {}
    for p in range(4):
        for q in range(4):
            if p + q <= 3:
                moments[f'm_{p}{q}'] = central_moment(data, p, q, x_centroid, y_centroid)

    # Calculate central normalized moments
    normalized_moments = {}
    for p in range(4):
        for q in range(4):
            if 0 < p + q <= 3:
                normalized_moments[f'M_{p}{q}'] = central_normalized_moment(moments, p, q, m00)

    # Calculate Hu's moments
    hu_moments = calculate_hu_moments(normalized_moments)

    # Calculate log of absolute values of Hu's moments
    log_hu_moments = {f"log|S{i}|": -np.sign(m) * np.log10(abs(m)) for i, m in hu_moments.items()}

    return log_hu_moments


# Function to process a folder of images and save results to CSV
def process_images_in_folder(input_folder, output_csv):
    classes = ['1', '2', '3', '4', '5']  # Define your classes
    class_counter = [0] * len(classes)  # To keep track of the number of images per class
    train_limit = 40  # Number of training images per class
    test_limit = 10   # Number of test images per class

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ['Image'] + [f"S{i}" for i in range(1, 8)] + ['Labels', 'Set']  # Add Set to header
        writer.writerow(header)  # Write header to CSV

        # Iterate through the folder and process each image
        for filename in os.listdir(input_folder):
            if filename.endswith(".png") or filename.endswith(".jpg"):  # Filter for image files
                # Determine the class based on the count
                for idx, count in enumerate(class_counter):
                    if count < (train_limit + test_limit):  # If less than total images in the class
                        image_path = os.path.join(input_folder, filename)  # Full path to image
                        hu_moments = process_image(image_path)  # Calculate Hu moments

                        # Determine if it's a train or test image
                        if count < train_limit:
                            data_set = 'train'
                        else:
                            data_set = 'test'

                        row = [filename] + [hu_moments[f"log|S{i}|"] for i in range(1, 8)] + [classes[idx], data_set]

                        writer.writerow(row)  # Write row to CSV
                        class_counter[idx] += 1  # Increment the count for that class
                        break  # Exit loop after assigning the class


# Example usage
input_folder = "Anh/NhiPhan"  # Input folder with images
output_csv = "Output/loghu.csv"  # Output CSV file

process_images_in_folder(input_folder, output_csv)  # Process the folder and save results
