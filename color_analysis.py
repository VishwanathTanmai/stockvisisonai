import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.colors as mcolors

def analyze_colors(image, num_colors=5):
    """
    Analyze the color profile of a skin image.
    
    Args:
        image: PIL Image
        num_colors: Number of dominant colors to extract
        
    Returns:
        color_info: Dictionary of color analysis information
        visualization: Matplotlib figure with visualizations
    """
    # Convert PIL image to numpy array
    img_array = np.array(image.convert('RGB'))
    
    # Reshape the image to be a list of pixels
    pixels = img_array.reshape(-1, 3)
    
    # Perform KMeans clustering to find dominant colors
    kmeans = KMeans(n_clusters=num_colors, random_state=42)
    kmeans.fit(pixels)
    
    # Get the dominant colors
    colors = kmeans.cluster_centers_.astype(int)
    
    # Calculate percentages of each color
    labels = kmeans.labels_
    color_counts = np.bincount(labels)
    total_pixels = len(labels)
    color_percentages = color_counts / total_pixels * 100
    
    # Convert colors to hex for visualization
    hex_colors = ['#%02x%02x%02x' % (r, g, b) for r, g, b in colors]
    
    # Calculate color statistics in different color spaces
    # RGB
    avg_rgb = np.mean(pixels, axis=0).astype(int)
    std_rgb = np.std(pixels, axis=0).astype(int)
    
    # HSV for better skin tone analysis
    img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    pixels_hsv = img_hsv.reshape(-1, 3)
    avg_hsv = np.mean(pixels_hsv, axis=0).astype(int)
    std_hsv = np.std(pixels_hsv, axis=0).astype(int)
    
    # Calculate color uniformity (low std dev = more uniform)
    uniformity_score = 100 - (np.mean(std_rgb) / 255 * 100)
    
    # Create a dictionary with color analysis information
    color_info = {
        "dominant_colors": colors,
        "hex_colors": hex_colors,
        "color_percentages": color_percentages,
        "avg_rgb": avg_rgb,
        "std_rgb": std_rgb,
        "avg_hsv": avg_hsv,
        "std_hsv": std_hsv,
        "uniformity_score": uniformity_score
    }
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original image
    axes[0, 0].imshow(img_array)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Color distribution pie chart
    axes[0, 1].pie(color_percentages, colors=hex_colors, autopct='%1.1f%%', startangle=90)
    axes[0, 1].set_title('Dominant Color Distribution')
    
    # Color histogram (RGB)
    for i, color in enumerate(['Red', 'Green', 'Blue']):
        hist = cv2.calcHist([img_array], [i], None, [256], [0, 256])
        axes[1, 0].plot(hist, color=color.lower())
    axes[1, 0].set_title('RGB Histogram')
    axes[1, 0].set_xlim([0, 256])
    axes[1, 0].grid(alpha=0.3)
    
    # Dominant color swatches
    for i, (color, pct) in enumerate(zip(hex_colors, color_percentages)):
        axes[1, 1].add_patch(plt.Rectangle((0, i), 1, 1, color=color))
        axes[1, 1].text(1.1, i+0.5, f'{pct:.1f}%', va='center')
    axes[1, 1].set_xlim([0, 2])
    axes[1, 1].set_ylim([0, num_colors])
    axes[1, 1].set_title('Dominant Colors')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    return color_info, fig

def get_skin_tone_description(color_info):
    """
    Generate a skin tone description based on color analysis.
    
    Args:
        color_info: Dictionary of color analysis
        
    Returns:
        str: Description of skin tone
    """
    # Get average HSV values for analysis
    avg_hsv = color_info["avg_hsv"]
    h, s, v = avg_hsv
    
    # Normalize h to 0-360 range
    h = h * 2  # OpenCV uses 0-180 for hue
    
    # Define skin tone based on hue and saturation
    tone = ""
    undertone = ""
    
    # Determining basic skin tone based on value (brightness)
    if v < 85:
        tone = "Deep"
    elif v < 128:
        tone = "Dark"
    elif v < 170:
        tone = "Medium"
    elif v < 213:
        tone = "Light"
    else:
        tone = "Very Light"
    
    # Determining undertone based on hue
    if h < 20 or h > 340:
        undertone = "cool (pink/red)"
    elif 20 <= h <= 40:
        undertone = "warm (yellow/golden)"
    else:
        undertone = "neutral"
    
    # Get uniformity info
    uniformity = color_info["uniformity_score"]
    uniformity_text = ""
    if uniformity > 85:
        uniformity_text = "very uniform"
    elif uniformity > 70:
        uniformity_text = "uniform"
    elif uniformity > 50:
        uniformity_text = "moderately varying"
    else:
        uniformity_text = "highly varying"
    
    description = (
        f"The image shows a {tone.lower()} skin tone with {undertone} undertones. "
        f"The skin coloration appears {uniformity_text}. "
    )
    
    # Add skin condition hints based on color analysis
    dom_colors = color_info["dominant_colors"]
    has_red = any([c[0] > c[1] + 30 and c[0] > c[2] + 30 for c in dom_colors])
    has_dark_spots = any([np.mean(c) < 100 for c in dom_colors]) and uniformity < 70
    has_white_spots = any([np.mean(c) > 200 for c in dom_colors]) and uniformity < 70
    
    condition_hints = []
    if has_red:
        condition_hints.append("The presence of reddish coloration may indicate inflammation or irritation.")
    if has_dark_spots:
        condition_hints.append("Darker spots or areas may indicate hyperpigmentation or melanin concentration.")
    if has_white_spots:
        condition_hints.append("Lighter or white areas may indicate hypopigmentation or loss of melanin.")
    
    if condition_hints:
        description += " " + " ".join(condition_hints)
    
    return description

def create_color_map(image):
    """
    Create a color-coded map of an image highlighting different color regions.
    
    Args:
        image: PIL Image
        
    Returns:
        PIL Image: Color-coded map
    """
    # Convert to numpy array
    img_array = np.array(image.convert('RGB'))
    
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    
    # Define color ranges for skin conditions
    # Redness (inflammation, acne)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    
    # Dark areas (hyperpigmentation)
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 80])
    
    # Light areas (vitiligo, dry skin)
    lower_light = np.array([0, 0, 200])
    upper_light = np.array([180, 30, 255])
    
    # Create masks
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    
    mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)
    mask_light = cv2.inRange(hsv, lower_light, upper_light)
    
    # Create color-coded regions
    red_region = np.zeros_like(img_array)
    red_region[mask_red > 0] = [255, 0, 0]  # Red for inflammation
    
    dark_region = np.zeros_like(img_array)
    dark_region[mask_dark > 0] = [0, 0, 255]  # Blue for dark areas
    
    light_region = np.zeros_like(img_array)
    light_region[mask_light > 0] = [0, 255, 0]  # Green for light areas
    
    # Combine regions
    color_map = cv2.addWeighted(red_region, 1, dark_region, 1, 0)
    color_map = cv2.addWeighted(color_map, 1, light_region, 1, 0)
    
    # Create a legend
    legend = np.zeros((50, img_array.shape[1], 3), dtype=np.uint8)
    legend[:, :img_array.shape[1]//3, :] = [255, 0, 0]  # Red
    legend[:, img_array.shape[1]//3:2*img_array.shape[1]//3, :] = [0, 0, 255]  # Blue
    legend[:, 2*img_array.shape[1]//3:, :] = [0, 255, 0]  # Green
    
    # Add text to legend
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(legend, 'Inflammation', (10, 30), font, 0.5, (255, 255, 255), 1)
    cv2.putText(legend, 'Dark Areas', (img_array.shape[1]//3 + 10, 30), font, 0.5, (255, 255, 255), 1)
    cv2.putText(legend, 'Light Areas', (2*img_array.shape[1]//3 + 10, 30), font, 0.5, (255, 255, 255), 1)
    
    # Combine original image with color map and legend
    result = cv2.addWeighted(img_array, 0.7, color_map, 0.3, 0)
    result = np.vstack([result, legend])
    
    # Convert back to PIL image
    return Image.fromarray(result)