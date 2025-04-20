from PIL import Image
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
import io

def preprocess_image(image):
    """
    Preprocess an image for input to our skin disease model.
    
    Args:
        image: PIL Image object
        
    Returns:
        PIL Image: Resized image ready for model input
    """
    # Resize to 224x224 (standard size for many image models)
    image = image.resize((224, 224))
    
    # We return the PIL image directly now, since our model works with PIL images
    return image

def get_class_labels():
    """
    Return the list of class labels for the model.
    """
    return ["Acne", "Hyperpigmentation", "Nail Psoriasis", "SJS-TEN", "Vitiligo"]

def display_predictions(prediction, confidence):
    """
    Display the prediction results in a nicely formatted way.
    
    Args:
        prediction: The predicted class label
        confidence: The confidence score as a percentage
    """
    # Display prediction with confidence
    st.markdown(f"### Prediction: **{prediction}**")
    
    # Color code based on confidence
    if confidence >= 90:
        confidence_color = "green"
    elif confidence >= 70:
        confidence_color = "orange"
    else:
        confidence_color = "red"
    
    st.markdown(f"### Confidence: <span style='color:{confidence_color}'>{confidence:.2f}%</span>", unsafe_allow_html=True)
    
    # Display confidence meter
    st.progress(confidence / 100)
    
    # Display description of the predicted condition
    condition_descriptions = {
        "Acne": "An inflammatory skin condition characterized by pimples, blackheads, and clogged pores, typically occurring on the face, chest, and back.",
        "Hyperpigmentation": "A condition where patches of skin become darker than the surrounding areas due to excess melanin production.",
        "Nail Psoriasis": "A manifestation of psoriasis affecting the fingernails and toenails, causing pitting, discoloration, and separation from the nail bed.",
        "SJS-TEN": "Stevens-Johnson Syndrome / Toxic Epidermal Necrolysis is a severe skin reaction usually triggered by medications, causing skin peeling and painful blisters.",
        "Vitiligo": "A long-term condition where patches of skin lose their color due to the destruction of melanocytes, the cells responsible for skin pigmentation."
    }
    
    st.markdown(f"**About {prediction}**: {condition_descriptions[prediction]}")

def pil_image_to_byte_array(image):
    """
    Convert a PIL Image to a byte array for API transmission.
    
    Args:
        image: PIL Image object
        
    Returns:
        bytes: Image converted to byte array
    """
    img_byte_array = io.BytesIO()
    image.save(img_byte_array, format=image.format if image.format else 'JPEG')
    return img_byte_array.getvalue()

def create_heatmap(features, image_size=(224, 224)):
    """
    Create a heatmap from feature maps for visualization.
    
    Args:
        features: Feature activation maps from the model
        image_size: Size to resize the heatmap to
        
    Returns:
        numpy array: Heatmap as a numpy array
    """
    # Reduce dimensionality of features to create a heatmap
    feature_map = np.mean(np.abs(features), axis=0)
    
    # Normalize to [0, 1]
    feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
    
    # Resize to match image dimensions (simplified approach)
    from scipy.ndimage import zoom
    zoom_factors = (image_size[0] / feature_map.shape[0], image_size[1] / feature_map.shape[1])
    heatmap = zoom(feature_map, zoom_factors)
    
    return heatmap
