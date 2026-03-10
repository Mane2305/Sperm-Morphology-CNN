import cv2
import numpy as np
from skimage import filters, morphology, exposure
import os

def preprocess_sperm_image(image_path, output_size=(224, 224)):
    """
    Preprocess sperm images with domain-specific techniques
    """
    # Read and convert to grayscale
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    # 1. Color normalization (stain normalization for microscope images)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # CLAHE on L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge back
    lab = cv2.merge([l, a, b])
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # 2. Background subtraction (useful for microscope images)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. Edge enhancement for sperm tails
    edges = cv2.Canny(gray, 50, 150)
    
    # 4. Combine original with enhanced features
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    enhanced = cv2.addWeighted(image, 0.7, edges_colored, 0.3, 0)
    
    # 5. Resize
    enhanced = cv2.resize(enhanced, output_size)
    
    return enhanced

def process_directory(input_dir, output_dir):
    """Process all images in directory"""
    os.makedirs(output_dir, exist_ok=True)
    
    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        if os.path.isdir(class_path):
            output_class_path = os.path.join(output_dir, class_name)
            os.makedirs(output_class_path, exist_ok=True)
            
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                processed = preprocess_sperm_image(img_path)
                
                if processed is not None:
                    output_path = os.path.join(output_class_path, img_name)
                    cv2.imwrite(output_path, processed)
    
    print(f"✅ Preprocessing complete. Output saved to {output_dir}")