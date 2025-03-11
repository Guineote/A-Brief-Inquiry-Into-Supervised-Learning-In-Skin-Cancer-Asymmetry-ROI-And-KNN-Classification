import os
import imageio.v3 as iio
import numpy as np
import pandas as pd

def generate_histogram_vector(image):
    num_bins = 8
    histogram_vector = np.zeros(num_bins * 3)
    
    for channel_id in range(3):  # Para R, G y B
        histogram, bin_edges = np.histogram(
            image[:, :, channel_id], bins=num_bins, range=(0, 256)
        )
        histogram_vector[channel_id * num_bins:(channel_id + 1) * num_bins] = histogram
    
    return histogram_vector

def process_images_from_csv(csv_path, images_folder):
    # Leer el archivo CSV
    df = pd.read_csv(csv_path)
    
    image_ids = df.iloc[:, 1]
    
    histogram_vectors = []

    for image_id in image_ids:
        image_filename = f"{image_id}.png"
        image_path = os.path.join(images_folder, image_filename)
        if os.path.exists(image_path):
            image = iio.imread(image_path)
            histogram_vector = generate_histogram_vector(image)
            histogram_vectors.append(histogram_vector)
        else:
            print(f"Imagen {image_path} no encontrada.")
    
    return np.array(histogram_vectors)

csv_path = 'HAM10000_metadata.csv'
images_folder = 'cropped_images'

histogram_vectors = process_images_from_csv(csv_path, images_folder)

np.save('cropped_histogram_vectors.npy', histogram_vectors)
