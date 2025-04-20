import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import random

def get_train_transforms(image_size=224):
    """
    Get transformations for training data with augmentation.
    
    Args:
        image_size: Target image size for the model
        
    Returns:
        torchvision.transforms: Composition of transforms
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_val_transforms(image_size=224):
    """
    Get transformations for validation/test data.
    
    Args:
        image_size: Target image size for the model
        
    Returns:
        torchvision.transforms: Composition of transforms
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def preprocess_image_for_training(image, transform=None):
    """
    Preprocess a single image for model training.
    
    Args:
        image: PIL Image object
        transform: Optional transformation to apply
        
    Returns:
        Tensor: Preprocessed image tensor
    """
    if transform is None:
        transform = get_train_transforms()
    
    return transform(image)

def apply_custom_augmentation(image):
    """
    Apply custom augmentations to an image for improved training.
    
    Args:
        image: PIL Image object
        
    Returns:
        PIL Image: Augmented image
    """
    # List of possible augmentations
    augmentations = [
        lambda img: ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2)),
        lambda img: ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2)),
        lambda img: ImageEnhance.Color(img).enhance(random.uniform(0.8, 1.2)),
        lambda img: ImageEnhance.Sharpness(img).enhance(random.uniform(0.8, 1.2)),
        lambda img: img.rotate(random.uniform(-15, 15)),
        lambda img: ImageOps.mirror(img) if random.random() > 0.5 else img,
        lambda img: ImageOps.flip(img) if random.random() > 0.5 else img,
    ]
    
    # Apply 2-4 random augmentations
    num_augmentations = random.randint(2, 4)
    selected_augmentations = random.sample(augmentations, num_augmentations)
    
    augmented_image = image
    for augment in selected_augmentations:
        augmented_image = augment(augmented_image)
    
    return augmented_image

def create_balanced_batch(images, labels, batch_size=32):
    """
    Create a balanced batch of images with equal representation of each class.
    
    Args:
        images: List of images
        labels: List of corresponding labels
        batch_size: Size of the batch to create
        
    Returns:
        batch_images, batch_labels: Balanced batch
    """
    # Group images by label
    grouped = {}
    for img, label in zip(images, labels):
        if label not in grouped:
            grouped[label] = []
        grouped[label].append(img)
    
    # Calculate how many samples per class
    num_classes = len(grouped)
    samples_per_class = batch_size // num_classes
    
    # Create balanced batch
    batch_images = []
    batch_labels = []
    
    for label, imgs in grouped.items():
        selected = random.sample(imgs, min(samples_per_class, len(imgs)))
        batch_images.extend(selected)
        batch_labels.extend([label] * len(selected))
    
    # Shuffle the batch
    combined = list(zip(batch_images, batch_labels))
    random.shuffle(combined)
    batch_images, batch_labels = zip(*combined)
    
    return list(batch_images), list(batch_labels)
