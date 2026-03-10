import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import json
from pathlib import Path

from src.dataset import SpermDataset
from src.transforms import get_val_test_transforms
from src.model import get_model


def plot_training_history(history_path, save_path=None):
    """
    Plot training and validation metrics over epochs
    """
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()


def visualize_predictions(model, dataset, device, num_samples=16, save_path=None):
    """
    Visualize model predictions on sample images
    """
    model.eval()
    
    class_names = dataset.classes
    
    # Select random samples
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    # Calculate grid size
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()
    
    with torch.no_grad():
        for idx, sample_idx in enumerate(indices):
            image, true_label = dataset[sample_idx]
            
            # Get prediction
            image_batch = image.unsqueeze(0).to(device)
            output = model(image_batch)
            probs = torch.softmax(output, dim=1)
            pred_label = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_label].item() # type: ignore
            
            # Denormalize image for display
            img_display = image.permute(1, 2, 0).cpu().numpy()
            img_display = img_display * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_display = np.clip(img_display, 0, 1)
            
            # Plot
            axes[idx].imshow(img_display)
            axes[idx].axis('off')
            
            # Title with prediction info
            true_class = class_names[true_label]
            pred_class = class_names[pred_label]
            
            color = 'green' if pred_label == true_label else 'red'
            title = f'True: {true_class}\nPred: {pred_class} ({confidence:.2%})'
            axes[idx].set_title(title, fontsize=10, color=color, fontweight='bold')
    
    # Hide empty subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction visualization saved to {save_path}")
    
    plt.show()


def visualize_feature_maps(model, image, device, save_path=None):
    """
    Visualize intermediate feature maps from the model
    """
    model.eval()
    
    # Hook to capture feature maps
    feature_maps = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            feature_maps[name] = output.detach()
        return hook
    
    # Register hooks (adjust based on your model architecture)
    if hasattr(model, 'backbone'):
        if hasattr(model.backbone, 'layer1'):
            model.backbone.layer1.register_forward_hook(hook_fn('layer1'))
        if hasattr(model.backbone, 'layer2'):
            model.backbone.layer2.register_forward_hook(hook_fn('layer2'))
        if hasattr(model.backbone, 'layer3'):
            model.backbone.layer3.register_forward_hook(hook_fn('layer3'))
        if hasattr(model.backbone, 'layer4'):
            model.backbone.layer4.register_forward_hook(hook_fn('layer4'))
    
    # Forward pass
    with torch.no_grad():
        image_batch = image.unsqueeze(0).to(device)
        _ = model(image_batch)
    
    # Visualize feature maps
    num_layers = len(feature_maps)
    if num_layers == 0:
        print("No feature maps captured. Model structure may be different.")
        return
    
    fig, axes = plt.subplots(num_layers, 8, figsize=(20, num_layers * 2.5))
    
    for layer_idx, (layer_name, features) in enumerate(feature_maps.items()):
        features = features[0]  # Remove batch dimension
        
        # Select first 8 channels
        num_channels = min(8, features.size(0))
        
        for ch_idx in range(num_channels):
            ax = axes[layer_idx, ch_idx] if num_layers > 1 else axes[ch_idx]
            
            feature_map = features[ch_idx].cpu().numpy()
            ax.imshow(feature_map, cmap='viridis')
            ax.axis('off')
            
            if ch_idx == 0:
                ax.set_title(f'{layer_name}', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature maps visualization saved to {save_path}")
    
    plt.show()


def analyze_misclassifications(model, dataset, device, save_path=None):
    """
    Analyze and visualize misclassified samples
    """
    model.eval()
    
    class_names = dataset.classes
    misclassified = []
    
    # Find misclassified samples
    with torch.no_grad():
        for idx in range(len(dataset)):
            image, true_label = dataset[idx]
            image_batch = image.unsqueeze(0).to(device)
            
            output = model(image_batch)
            probs = torch.softmax(output, dim=1)
            pred_label = torch.argmax(probs, dim=1).item()
            
            if pred_label != true_label:
                confidence = probs[0, pred_label].item() # type: ignore
                misclassified.append({
                    'idx': idx,
                    'true_label': true_label,
                    'pred_label': pred_label,
                    'confidence': confidence,
                    'image': image
                })
    
    if len(misclassified) == 0:
        print("No misclassifications found! Perfect accuracy!")
        return
    
    print(f"\nFound {len(misclassified)} misclassified samples")
    
    # Sort by confidence (most confident mistakes first)
    misclassified.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Visualize top misclassifications
    num_samples = min(16, len(misclassified))
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()
    
    for idx in range(num_samples):
        sample = misclassified[idx]
        
        # Denormalize image
        img_display = sample['image'].permute(1, 2, 0).cpu().numpy()
        img_display = img_display * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_display = np.clip(img_display, 0, 1)
        
        # Plot
        axes[idx].imshow(img_display)
        axes[idx].axis('off')
        
        true_class = class_names[sample['true_label']]
        pred_class = class_names[sample['pred_label']]
        confidence = sample['confidence']
        
        title = f'True: {true_class}\nPred: {pred_class}\nConf: {confidence:.2%}'
        axes[idx].set_title(title, fontsize=10, color='red', fontweight='bold')
    
    # Hide empty subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Most Confident Misclassifications', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Misclassification analysis saved to {save_path}")
    
    plt.show()


def main():
    """Main visualization function"""
    # Load config
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory
    viz_dir = Path(config['output']['model_dir']) / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Visualization Suite")
    print("="*60)
    
    # 1. Plot training history
    print("\n1. Plotting training history...")
    history_path = Path(config['output']['model_dir']) / 'training_history.json'
    if history_path.exists():
        plot_training_history(
            history_path,
            save_path=viz_dir / 'training_history.png'
        )
    else:
        print("   Training history not found. Train the model first.")
    
    # Load model and dataset for remaining visualizations
    model_type = config["training"].get("model_type", "resnet50")
    model = get_model(
        model_type=model_type,
        num_classes=config["training"]["num_classes"],
        pretrained=False
    ).to(device)
    
    # Load checkpoint
    checkpoint_path = Path(config['output']['model_dir']) / config['output']['best_model_name']
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        # Load test dataset
        test_dataset = SpermDataset(
            config["data"]["test_dir"],
            transform=get_val_test_transforms()
        )
        
        # 2. Visualize predictions
        print("\n2. Visualizing sample predictions...")
        visualize_predictions(
            model, test_dataset, device,
            num_samples=16,
            save_path=viz_dir / 'sample_predictions.png'
        )
        
        # 3. Analyze misclassifications
        print("\n3. Analyzing misclassifications...")
        analyze_misclassifications(
            model, test_dataset, device,
            save_path=viz_dir / 'misclassifications.png'
        )
        
        # 4. Visualize feature maps (on one sample)
        print("\n4. Visualizing feature maps...")
        sample_image, _ = test_dataset[0]
        visualize_feature_maps(
            model, sample_image, device,
            save_path=viz_dir / 'feature_maps.png'
        )
        
    else:
        print("\nModel checkpoint not found. Train the model first.")
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print(f"All plots saved to: {viz_dir}")
    print("="*60)


if __name__ == "__main__":
    main()