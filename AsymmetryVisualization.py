import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float, color

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
    
    if image.shape[2] == 4:
        binary_mask = image[:,:,3] > 0.5
    else:  # RGB
        binary_mask = color.rgb2gray(image[:,:,:3]) < 0.9
    vertical_asymmetry, horizontal_asymmetry = calculate_asymmetry(binary_mask)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    ax1, ax2, ax3, ax4 = axes.flatten()
    
    ax1.imshow(image[:,:,:3])
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2.imshow(binary_mask, cmap='gray')
    ax2.set_title('Binary Mask')
    ax2.axis('off')
    
    height, width = binary_mask.shape
    ax3.imshow(np.hstack((binary_mask[:, :width//2], np.fliplr(binary_mask[:, width//2:]))), cmap='gray')
    ax3.axvline(x=width//2, color='r', linestyle='--')
    ax3.set_title('Vertical Asymmetry')
    ax3.axis('off')
    
    ax4.imshow(np.vstack((binary_mask[:height//2, :], np.flipud(binary_mask[height//2:, :]))), cmap='gray')
    ax4.axhline(y=height//2, color='r', linestyle='--')
    ax4.set_title('Horizontal Asymmetry')
    ax4.axis('off')
    
    fig.tight_layout()
    plt.show()
    
    return vertical_asymmetry, horizontal_asymmetry

image_path = r"cropped_images/ISIC_0034320.png"  # Asegúrate de que esta ruta sea correcta

vertical_asymmetry, horizontal_asymmetry = process_single_image(image_path)

print(f"Vertical Asymmetry: {vertical_asymmetry:.4f}")
print(f"Horizontal Asymmetry: {horizontal_asymmetry:.4f}")