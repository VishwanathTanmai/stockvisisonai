import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import streamlit as st
import io

def compare_images(image1, image2, title1="Before", title2="After"):
    """
    Compare two images side by side with visual indicators of differences.
    
    Args:
        image1: First PIL Image
        image2: Second PIL Image
        title1: Label for first image
        title2: Label for second image
        
    Returns:
        fig: Matplotlib figure with comparison
    """
    # Convert PIL images to numpy arrays
    img1_array = np.array(image1.convert('RGB'))
    img2_array = np.array(image2.convert('RGB'))
    
    # Resize images to the same dimensions if they differ
    if img1_array.shape != img2_array.shape:
        # Resize the second image to match the first
        image2 = image2.resize(image1.size)
        img2_array = np.array(image2.convert('RGB'))
    
    # Convert to grayscale for difference calculation
    img1_gray = cv2.cvtColor(img1_array, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2_array, cv2.COLOR_RGB2GRAY)
    
    # Calculate absolute difference between images
    diff = cv2.absdiff(img1_gray, img2_gray)
    
    # Threshold the difference image to highlight changes
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    # Create contours for differences
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on a copy of the second image
    diff_overlay = img2_array.copy()
    cv2.drawContours(diff_overlay, contours, -1, (0, 255, 0), 2)
    
    # Calculate similarity metrics
    similarity = 100 - (np.sum(thresh > 0) / thresh.size * 100)
    
    # Create a figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Display images side by side
    ax1.imshow(img1_array)
    ax1.set_title(title1)
    ax1.axis('off')
    
    ax2.imshow(img2_array)
    ax2.set_title(title2)
    ax2.axis('off')
    
    ax3.imshow(diff_overlay)
    ax3.set_title(f'Differences (Similarity: {similarity:.1f}%)')
    ax3.axis('off')
    
    plt.tight_layout()
    return fig

def create_comparison_image(image1, image2, title1="Before", title2="After"):
    """
    Create a combined image with both images side by side and labels.
    
    Args:
        image1: First PIL Image
        image2: Second PIL Image
        title1: Label for first image
        title2: Label for second image
        
    Returns:
        PIL Image: Combined image
    """
    # Resize images to the same height
    height = max(image1.height, image2.height)
    width1 = int(image1.width * (height / image1.height))
    width2 = int(image2.width * (height / image2.height))
    
    image1 = image1.resize((width1, height), Image.LANCZOS)
    image2 = image2.resize((width2, height), Image.LANCZOS)
    
    # Create a new blank image to place both images side by side
    total_width = width1 + width2 + 20  # Adding some padding
    combined_image = Image.new('RGB', (total_width, height + 40), color='white')
    
    # Paste the images
    combined_image.paste(image1, (0, 30))
    combined_image.paste(image2, (width1 + 20, 30))
    
    # Add labels
    draw = ImageDraw.Draw(combined_image)
    # Use a default font
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
        
    draw.text((width1//2 - 30, 5), title1, fill="black", font=font)
    draw.text((width1 + 20 + width2//2 - 30, 5), title2, fill="black", font=font)
    
    return combined_image

def highlight_roi(image, bbox=None, auto_detect=False):
    """
    Highlight a region of interest in an image.
    
    Args:
        image: PIL Image
        bbox: Bounding box coordinates (x1, y1, x2, y2) or None for auto-detection
        auto_detect: Whether to automatically detect skin lesions
        
    Returns:
        PIL Image: Image with ROI highlighted
    """
    img_array = np.array(image.convert('RGB'))
    
    if auto_detect:
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply GaussianBlur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Threshold the image
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # If contours found, use the largest one
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            bbox = (x, y, x+w, y+h)
    
    if bbox:
        # Draw bounding box
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add "ROI" label
        cv2.putText(img_array, "ROI", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Convert back to PIL Image
    result_image = Image.fromarray(img_array)
    return result_image