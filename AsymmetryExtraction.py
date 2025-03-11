import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float, color
import os
import csv
import pandas as pd

def calculate_asymmetry(binary_mask):
    def asymmetry_index(split1, split2):
        difference = np.sum(np.abs(split1.astype(float) - split2.astype(float)))
        total = np.sum(split1) + np.sum(split2)
        return difference / total if total > 0 else 0

    height, width = binary_mask.shape
    vertical_split = width // 2
    horizontal_split = height // 2

    left = binary_mask[:, :vertical_split]
    right = np.fliplr(binary_mask[:, vertical_split:])
    top = binary_mask[:horizontal_split, :]
    bottom = np.flipud(binary_mask[horizontal_split:, :])

    vertical_asymmetry = asymmetry_index(left, right[:, :left.shape[1]])
    horizontal_asymmetry = asymmetry_index(top, bottom[:top.shape[0], :])

    return vertical_asymmetry, horizontal_asymmetry

def process_single_image(image_path):
    image = img_as_float(io.imread(image_path))
    
    if image.shape[2] == 4:  # RGBA
        binary_mask = image[:,:,3] > 0  # Usar el canal alfa
    else:  # RGB
        binary_mask = color.rgb2gray(image[:,:,:3]) < 0.9  # Ajusta este umbral según sea necesario
    
    vertical_asymmetry, horizontal_asymmetry = calculate_asymmetry(binary_mask)
    
    return vertical_asymmetry, horizontal_asymmetry

def process_all_images(folder_path, csv_input_path, csv_output_path):
    df = pd.read_csv(csv_input_path)
    
    results = []
    
    for index, row in df.iterrows():
        image_name = row.iloc[1]
        image_name_with_ext = image_name + ".png"
        classification = row.iloc[2]
        
        image_path = os.path.join(folder_path, image_name_with_ext)
        
        if os.path.exists(image_path):
            vertical_asymmetry, horizontal_asymmetry = process_single_image(image_path)
            results.append([image_name, classification, vertical_asymmetry, horizontal_asymmetry])
        else:
            print(f"Imagen no encontrada: {image_path}")
    
    results_df = pd.DataFrame(results, columns=['Image ID', 'Classification', 'Vertical Asymmetry', 'Horizontal Asymmetry'])
    
    results_df.to_csv(csv_output_path, index=False)
    
    print(f"Resultados guardados en {csv_output_path}")

folder_path = "cropped_images"
csv_input_path = "HAM10000_metadata.csv"
csv_output_path = "assimetry_results.csv"

process_all_images(folder_path, csv_input_path, csv_output_path)