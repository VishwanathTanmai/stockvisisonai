import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
import io
import streamlit as st
from utils import get_class_labels
from PIL import Image

def plot_accuracy_curve():
    """
    Plot the accuracy curve for model using real-time accuracy data.
    
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Get real-time accuracy data from the model
    if 'model' not in st.session_state or not hasattr(st.session_state.model, 'accuracy_history') or len(st.session_state.model.accuracy_history) == 0:
        # If no predictions have been made yet, use an empty plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No prediction data available yet. Make some predictions to see accuracy metrics.',
                ha='center', va='center', fontsize=12)
        ax.set_xlabel('Predictions', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Real-time Model Accuracy', fontsize=14)
        return fig
    
    # Get accuracy history from the model
    accuracy_history = st.session_state.model.accuracy_history
    
    # Create indices (x-axis) for the predictions
    prediction_indices = range(1, len(accuracy_history) + 1)
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(prediction_indices, accuracy_history, label='Model Accuracy', marker='o', linestyle='-', color='#2986cc')
    
    # Add grid and labels
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel('Number of Predictions', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Real-time Model Accuracy', fontsize=14)
    
    # Add target accuracy threshold line
    ax.axhline(y=0.95, color='green', linestyle='-.', alpha=0.7, label='Target Accuracy (95%)')
    
    # Set y-axis limits
    ax.set_ylim([0.7, 1.0])
    
    # Add legend
    ax.legend(loc='lower right')
    
    # Annotate current accuracy if we have enough data points
    if len(accuracy_history) > 0:
        current_accuracy = accuracy_history[-1]
        ax.annotate(f'Current Accuracy: {current_accuracy:.2%}', 
                    xy=(prediction_indices[-1], current_accuracy), 
                    xytext=(max(1, prediction_indices[-1] - 3), current_accuracy - 0.05),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    plt.tight_layout()
    return fig

def plot_loss_curve():
    """
    Plot the loss curve for model using real-time loss data.
    
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Get real-time loss data from the model
    if 'model' not in st.session_state or not hasattr(st.session_state.model, 'loss_history') or len(st.session_state.model.loss_history) == 0:
        # If no predictions have been made yet, use an empty plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No prediction data available yet. Make some predictions to see loss metrics.',
                ha='center', va='center', fontsize=12)
        ax.set_xlabel('Predictions', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Real-time Model Loss', fontsize=14)
        return fig
    
    # Get loss history from the model
    loss_history = st.session_state.model.loss_history
    
    # Create indices (x-axis) for the predictions
    prediction_indices = range(1, len(loss_history) + 1)
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(prediction_indices, loss_history, label='Model Loss', marker='o', linestyle='-', color='#cc2929')
    
    # Add grid and labels
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel('Number of Predictions', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Real-time Model Loss', fontsize=14)
    
    # Set y-axis limits
    max_loss = max(loss_history) if loss_history else 0.5
    min_loss = min(loss_history) if loss_history else 0
    y_margin = (max_loss - min_loss) * 0.1 if max_loss > min_loss else 0.05
    ax.set_ylim([max(0, min_loss - y_margin), max_loss + y_margin])
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Annotate current loss if we have enough data points
    if len(loss_history) > 0:
        current_loss = loss_history[-1]
        ax.annotate(f'Current Loss: {current_loss:.3f}', 
                    xy=(prediction_indices[-1], current_loss), 
                    xytext=(max(1, prediction_indices[-1] - 3), current_loss + 0.05),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    plt.tight_layout()
    return fig

def plot_feature_extraction(features, prediction):
    """
    Visualize the extracted features using dimensionality reduction.
    
    Args:
        features: High-dimensional feature vector from the model
        prediction: The predicted class
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Ensure features are in the right shape for visualization
    if len(features.shape) > 2:
        # Reshape if needed - for example from [batch, seq_len, dim] to [batch*seq_len, dim]
        features_flat = features.reshape(-1, features.shape[-1])
    else:
        features_flat = features
        
    # If we have too many features, sample a subset for visualization
    if features_flat.shape[0] > 1000:
        indices = np.random.choice(features_flat.shape[0], 1000, replace=False)
        features_sample = features_flat[indices]
    else:
        features_sample = features_flat
    
    # Apply PCA for the first subplot
    if features_sample.shape[1] > 2:  # Only apply if dimension > 2
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features_sample)
        
        # Plot PCA
        ax1.scatter(features_pca[:, 0], features_pca[:, 1], alpha=0.7, color='#2986cc')
        ax1.set_title(f'PCA of Features for {prediction}')
        ax1.set_xlabel('Principal Component 1')
        ax1.set_ylabel('Principal Component 2')
        ax1.grid(True, linestyle='--', alpha=0.5)
    else:
        ax1.text(0.5, 0.5, 'PCA not applicable for these features', 
                 ha='center', va='center', transform=ax1.transAxes)
    
    # Create a heatmap visualization for feature activations
    # This is a simplified visualization of feature importance
    if len(features.shape) > 2:
        # For 3D features (like attention maps), take the mean across one dimension
        feature_importance = np.mean(np.abs(features), axis=1).squeeze()
    else:
        # For 2D features, just take the absolute values
        feature_importance = np.abs(features.squeeze())
    
    # If the feature map is too large, resize it for visualization
    if feature_importance.size > 1024:
        # Reshape to a square-ish grid
        side = int(np.sqrt(min(1024, feature_importance.size)))
        feature_importance = feature_importance.flatten()[:side*side].reshape(side, side)
    elif feature_importance.ndim == 1:
        # If it's a 1D vector, reshape to a square-ish grid
        side = int(np.sqrt(feature_importance.size))
        if side**2 < feature_importance.size:
            side += 1
        # Pad the array if needed
        padded = np.zeros(side*side)
        padded[:feature_importance.size] = feature_importance
        feature_importance = padded.reshape(side, side)
    
    # Create heatmap
    im = ax2.imshow(feature_importance, cmap='viridis')
    ax2.set_title(f'Feature Activation Heatmap for {prediction}')
    fig.colorbar(im, ax=ax2, label='Activation Strength')
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    plt.tight_layout()
    return fig

def overlay_heatmap(image, heatmap, alpha=0.5):
    """
    Overlay a heatmap on an image for attention visualization.
    
    Args:
        image: PIL Image object
        heatmap: Numpy array of the heatmap
        alpha: Transparency of the overlay
        
    Returns:
        PIL Image: The image with heatmap overlay
    """
    # Resize heatmap to match image dimensions
    heatmap_resized = Image.fromarray(
        (heatmap * 255).astype(np.uint8)
    ).resize(image.size, Image.LANCZOS)
    
    # Apply colormap to heatmap
    heatmap_colored = plt.cm.jet(np.array(heatmap_resized) / 255.0) * 255
    heatmap_colored = Image.fromarray(heatmap_colored.astype(np.uint8))
    
    # Blend the original image with the heatmap
    blended = Image.blend(image.convert('RGBA'), 
                          heatmap_colored.convert('RGBA'), 
                          alpha=alpha)
    
    return blended
