import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import io, img_as_float, color, filters, morphology, restoration, draw
from skimage.measure import find_contours
import os
import pandas as pd

def remove_hair_skimage(image):
    # Convert the image to grayscale
    gray = color.rgb2gray(image)
    
    # Apply a blackhat filter to find hair
    selem = morphology.rectangle(9, 9)
    blackhat = morphology.black_tophat(gray, selem)
    
    # Apply a threshold to get a binary image
    binary = blackhat > 0.04  # You might need to adjust this threshold
    
    # Convert binary mask to uint8 for OpenCV inpaint function
    binary_uint8 = (binary * 255).astype(np.uint8)
    
    # Inpaint to remove hair using OpenCV
    image_uint8 = (image * 255).astype(np.uint8)
    inpainted = cv2.inpaint(image_uint8, binary_uint8, inpaintRadius=1, flags=cv2.INPAINT_TELEA)
    
    # Convert back to float
    inpainted = img_as_float(inpainted)
    
    return inpainted

def segment_lesion(image):
    # Convert to grayscale for segmentation
    gray = color.rgb2gray(image)
    
    # Apply threshold
    thresh = filters.threshold_otsu(gray)
    binary = gray > thresh

    # Find contours
    contours = find_contours(binary, 0.8)
    
    # Select the largest contour
    contour = max(contours, key=len)
    
    return contour

def segment_lesion2(image):
    # Convert to grayscale for segmentation
    gray = color.rgb2gray(image)
    
    # Get image dimensions
    height, width = gray.shape
    
    # Define a central region of interest (ROI)
    roi_size = min(height, width) // 2
    center_y, center_x = height // 2, width // 2
    roi = gray[center_y - roi_size//2:center_y + roi_size//2,
               center_x - roi_size//2:center_x + roi_size//2]
    
    # Apply threshold to ROI
    thresh = filters.threshold_otsu(roi)
    binary = roi > thresh
    
    # Apply morphological operations to clean up the binary image
    binary = morphology.remove_small_objects(binary, min_size=100)
    binary = morphology.closing(binary, morphology.disk(5))
    
    # Find contours in ROI
    contours = find_contours(binary, 0.5)
    
    # Select the largest contour
    if contours:
        contour = max(contours, key=len)
        
        # Adjust contour coordinates to full image space
        contour[:, 0] += center_y - roi_size//2
        contour[:, 1] += center_x - roi_size//2
    else:
        # If no contour found, return a default contour (full image)
        contour = np.array([[0, 0], [0, width-1], [height-1, width-1], [height-1, 0]])
    
    return contour

def detect_border(image):
    # Convert to grayscale
    gray = color.rgb2gray(image)
    
    # Parameters
    border_width = 20  # Width of border to check
    corner_size = 50   # Size of corner to check
    dark_threshold = 0.3  # Threshold for considering a pixel as dark
    required_dark_ratio = 0.5  # Ratio of dark pixels required to consider as border
    
    # Check corners
    corners = [
        gray[:corner_size, :corner_size],  # Top-left
        gray[:corner_size, -corner_size:],  # Top-right
        gray[-corner_size:, :corner_size],  # Bottom-left
        gray[-corner_size:, -corner_size:]  # Bottom-right
    ]
    
    corner_darkness = [np.mean(corner) < dark_threshold for corner in corners]
    if sum(corner_darkness) >= 2:  # If at least two corners are dark
        return True
    
    # Check borders
    borders = [
        gray[:border_width, :],  # Top
        gray[-border_width:, :],  # Bottom
        gray[:, :border_width],  # Left
        gray[:, -border_width:]  # Right
    ]
    
    for border in borders:
        dark_ratio = np.mean(border < dark_threshold)
        if dark_ratio > required_dark_ratio:
            return True
    
    return False

def smart_segment_lesion(image):
    if detect_border(image):
        return segment_lesion2(image)
    else:
        return segment_lesion(image)



def process_single_image(image_path):
    # Read and convert image
    image = img_as_float(io.imread(image_path))
    hairless = remove_hair_skimage(image)
    
    # Segment lesion
    contour = smart_segment_lesion(hairless)

    
    # Create a mask for the lesion
    mask = np.zeros(image.shape[:2], dtype=bool)
    rr, cc = draw.polygon(contour[:, 0], contour[:, 1], image.shape[:2])
    mask[rr, cc] = True
    
    # Create an RGBA image with transparent background
    rgba_image = np.zeros((image.shape[0], image.shape[1], 4))
    rgba_image[:,:,:3] = image
    rgba_image[:,:,3] = mask.astype(float)
    
    # Crop the image to the bounding box of the lesion
    rows, cols = np.where(mask)
    top, bottom, left, right = rows.min(), rows.max(), cols.min(), cols.max()
    cropped_image = rgba_image[top:bottom+1, left:right+1]
    
    # Create cropped_images directory if it doesn't exist
    if not os.path.exists('cropped_images'):
        os.makedirs('cropped_images')
    
    # Save the cropped image as PNG
    output_filename = os.path.join('cropped_images', os.path.basename(image_path).replace('.jpg', '.png'))
    plt.imsave(output_filename, cropped_image)
    
    """
    FOR VISUALIZATION

    # Visualize results
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2.imshow(image)
    ax2.plot(contour[:, 1], contour[:, 0], 'r-', linewidth=2)
    ax2.set_title('Segmented Lesion')
    ax2.axis('off')
    
    ax3.imshow(cropped_image)
    ax3.set_title('Cropped Image')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.show()
    """

# Suponiendo que tu CSV se llama 'image_data.csv' y está en el mismo directorio que tu script
csv_file = 'HAM10000_metadata.csv'

# Lee el CSV
df = pd.read_csv(csv_file)

# Comienza desde la fila 5000
start_row = 5000

# Extrae los nombres de las imágenes de la segunda columna (índice 1), comenzando desde la fila 5000
image_names = df.iloc[start_row:, 1].tolist()

# Directorio base donde se encuentran las imágenes
base_dir = 'HAM10000_images_part_1'

# Crea las rutas completas de las imágenes
image_paths = [os.path.join(base_dir, name + '.jpg') for name in image_names]

# Ahora puedes usar image_paths en tu bucle de procesamiento
for image_path in image_paths:
    features = process_single_image(image_path)
