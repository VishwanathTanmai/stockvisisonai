import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

def analyze_texture(image):
    """
    Analyze texture features of a skin image.
    
    Args:
        image: PIL Image
        
    Returns:
        texture_features: Dictionary of texture features
        visualization: Matplotlib figure with visualizations
    """
    # Convert PIL image to numpy array
    img_array = np.array(image.convert('RGB'))
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Calculate GLCM (Gray-Level Co-occurrence Matrix)
    distances = [1, 3, 5]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(gray, distances=distances, angles=angles, 
                       levels=256, symmetric=True, normed=True)
    
    # Calculate texture properties
    contrast = graycoprops(glcm, 'contrast').mean()
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    ASM = graycoprops(glcm, 'ASM').mean()
    
    # Calculate Local Binary Pattern
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype('float') / sum(lbp_hist)
    
    # Store texture features
    texture_features = {
        "contrast": contrast,
        "dissimilarity": dissimilarity,
        "homogeneity": homogeneity,
        "energy": energy,
        "correlation": correlation,
        "ASM": ASM,
        "lbp_hist": lbp_hist
    }
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original image
    axes[0, 0].imshow(img_array)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # LBP visualization
    axes[0, 1].imshow(lbp, cmap='gray')
    axes[0, 1].set_title('Local Binary Pattern')
    axes[0, 1].axis('off')
    
    # Texture metrics
    metrics = [contrast, dissimilarity, homogeneity, energy, correlation, ASM]
    metric_names = ['Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation', 'ASM']
    colors = ['#2986cc', '#cc2929', '#29cc97', '#ccbe29', '#cc29cc', '#cc8829']
    
    axes[1, 0].bar(metric_names, metrics, color=colors)
    axes[1, 0].set_title('Texture Metrics')
    axes[1, 0].set_xticklabels(metric_names, rotation=45, ha='right')
    
    # LBP histogram
    axes[1, 1].bar(range(len(lbp_hist)), lbp_hist)
    axes[1, 1].set_title('LBP Histogram')
    axes[1, 1].set_xlabel('LBP Bins')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    
    return texture_features, fig

def get_texture_description(features):
    """
    Generate a human-readable description of texture features.
    
    Args:
        features: Dictionary of texture features
        
    Returns:
        str: Description text
    """
    contrast = features['contrast']
    homogeneity = features['homogeneity']
    energy = features['energy']
    correlation = features['correlation']
    
    descriptions = []
    
    # Contrast analysis
    if contrast > 10:
        descriptions.append("High contrast texture indicating significant variations in the skin surface.")
    elif contrast > 5:
        descriptions.append("Moderate contrast in the skin texture.")
    else:
        descriptions.append("Low contrast texture suggesting a relatively smooth skin surface.")
    
    # Homogeneity analysis
    if homogeneity > 0.9:
        descriptions.append("Very homogeneous texture pattern, suggesting uniform skin condition.")
    elif homogeneity > 0.7:
        descriptions.append("Moderately homogeneous texture with some variations.")
    else:
        descriptions.append("Heterogeneous texture indicating significant irregularities.")
    
    # Energy and uniformity
    if energy > 0.5:
        descriptions.append("High energy value indicating orderly texture patterns.")
    elif energy > 0.2:
        descriptions.append("Moderate energy value with some texture organization.")
    else:
        descriptions.append("Low energy value suggesting random or disordered texture patterns.")
    
    # Correlation analysis
    if correlation > 0.9:
        descriptions.append("Strong correlation between neighboring pixels indicating structured patterns.")
    elif correlation > 0.7:
        descriptions.append("Moderate pixel correlation in texture patterns.")
    else:
        descriptions.append("Low correlation suggesting more random texture distribution.")
    
    # LBP analysis (simplified)
    lbp_hist = features['lbp_hist']
    peak_bin = np.argmax(lbp_hist)
    if peak_bin < len(lbp_hist) // 3:
        descriptions.append("Dominant edge-like or corner patterns detected in texture.")
    elif peak_bin < 2 * len(lbp_hist) // 3:
        descriptions.append("Mixed texture patterns with balanced edge and flat regions.")
    else:
        descriptions.append("Predominantly flat or uniform texture regions detected.")
    
    return "\n".join(descriptions)

def segment_texture_regions(image):
    """
    Segment an image into regions with similar textures.
    
    Args:
        image: PIL Image
        
    Returns:
        PIL Image: Segmented image with color-coded texture regions
    """
    # Convert to numpy array
    img_array = np.array(image.convert('RGB'))
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Calculate LBP
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(blurred, n_points, radius, method='uniform')
    
    # Normalize LBP image for visualization
    lbp_normalized = (lbp * (255.0 / (n_points + 2))).astype(np.uint8)
    
    # Apply K-means clustering to segment the image based on LBP values
    lbp_flat = lbp_normalized.reshape(-1, 1)
    lbp_flat = np.float32(lbp_flat)
    
    # Define criteria and apply kmeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = 4  # Number of texture classes
    _, labels, centers = cv2.kmeans(lbp_flat, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert back to uint8 and reshape to original image size
    centers = np.uint8(centers)
    segmented_flat = centers[labels.flatten()]
    segmented = segmented_flat.reshape(gray.shape)
    
    # Create color map for visualization
    color_map = np.zeros((img_array.shape[0], img_array.shape[1], 3), dtype=np.uint8)
    colors = [
        [255, 0, 0],    # Red
        [0, 255, 0],    # Green
        [0, 0, 255],    # Blue
        [255, 255, 0]   # Yellow
    ]
    
    for i in range(k):
        mask = segmented == centers[i]
        color_map[mask] = colors[i % len(colors)]
    
    # Blend original image with the color map
    blended = cv2.addWeighted(img_array, 0.7, color_map, 0.3, 0)
    
    # Convert back to PIL image
    result = Image.fromarray(blended)
    
    return result