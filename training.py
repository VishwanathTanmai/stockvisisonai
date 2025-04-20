import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from torchvision.models import ViT_B_16_Weights
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
import time
from data_preprocessing import get_train_transforms, get_val_transforms

class SkinDiseaseDataset(Dataset):
    """
    Dataset class for skin disease images.
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = plt.imread(self.image_paths[idx])
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def train_vit_model(train_loader, val_loader, num_classes=5, num_epochs=30, learning_rate=0.001):
    """
    Train a Vision Transformer model for skin disease classification.
    
    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_classes: Number of disease classes
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        
    Returns:
        model: Trained PyTorch model
        history: Dictionary containing training history
    """
    # Initialize model
    model = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    
    # Modify the classifier head for our classes
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    # Train the model
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Print progress
        time_elapsed = time.time() - start_time
        print(f'Epoch {epoch+1}/{num_epochs} | Time: {time_elapsed:.2f}s')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
        print('-' * 50)
    
    return model, history

def evaluate_model(model, test_loader, class_names):
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained PyTorch model
        test_loader: DataLoader for test data
        class_names: List of class names
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Return evaluation results
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'class_names': class_names
    }
    
    return results

def prepare_for_fine_tuning(base_model, num_classes=5):
    """
    Prepare a pre-trained model for fine-tuning on skin disease data.
    
    Args:
        base_model: Pre-trained model
        num_classes: Number of target classes
        
    Returns:
        model: Model ready for fine-tuning
    """
    # Freeze all layers except the last few
    for param in base_model.parameters():
        param.requires_grad = False
    
    # Unfreeze the last few layers
    for param in base_model.encoder.layer[-2:].parameters():
        param.requires_grad = True
    
    # Replace the classification head
    in_features = base_model.heads.head.in_features
    base_model.heads.head = nn.Linear(in_features, num_classes)
    
    return base_model

def save_model(model, path='skin_disease_model.pth'):
    """
    Save a trained model to disk.
    
    Args:
        model: Trained PyTorch model
        path: File path to save the model
    """
    torch.save({
        'model_state_dict': model.state_dict(),
    }, path)
    print(f"Model saved to {path}")

def load_model(path='skin_disease_model.pth', num_classes=5):
    """
    Load a trained model from disk.
    
    Args:
        path: File path to load the model from
        num_classes: Number of target classes
        
    Returns:
        model: Loaded PyTorch model
    """
    # Initialize model architecture
    model = models.vit_b_16(weights=None)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    
    # Load weights
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model
