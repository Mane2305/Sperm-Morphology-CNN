import torch
import yaml
import numpy as np
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score,
    precision_recall_fscore_support
)
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import SpermDataset
from src.transforms import get_val_test_transforms, get_tta_transforms
from src.model import get_model


def evaluate_model(model, test_loader, device, use_tta=False, tta_transforms=None):
    """
    Evaluate model with optional Test-Time Augmentation
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            if use_tta and tta_transforms is not None:
                # Test-Time Augmentation
                tta_outputs = []
                
                for transform in tta_transforms:
                    # Apply transform to each image in batch
                    tta_images = []
                    for img_tensor in images:
                        # Convert tensor back to numpy for albumentations
                        img_np = img_tensor.permute(1, 2, 0).numpy()
                        img_np = (img_np * np.array([0.229, 0.224, 0.225]) + 
                                  np.array([0.485, 0.456, 0.406])) * 255
                        img_np = img_np.astype(np.uint8)
                        
                        # Apply transform
                        augmented = transform(image=img_np)['image']
                        tta_images.append(augmented)
                    
                    tta_batch = torch.stack(tta_images).to(device)
                    outputs = model(tta_batch)
                    tta_outputs.append(torch.softmax(outputs, dim=1))
                
                # Average predictions
                outputs = torch.stack(tta_outputs).mean(dim=0)
                probs = outputs
            else:
                # Standard evaluation
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
            
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Plot and optionally save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot heatmap
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_normalized_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Plot normalized confusion matrix (percentages)"""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(
        cm_normalized, 
        annot=True, 
        fmt='.2%', 
        cmap='RdYlGn',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Percentage'}
    )
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Normalized confusion matrix saved to {save_path}")
    
    plt.show()


def plot_per_class_metrics(y_true, y_pred, class_names, save_path=None):
    """Plot per-class precision, recall, and F1-score"""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Classes', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.set_ylim([0, 1.0]) # type: ignore
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Per-class metrics plot saved to {save_path}")
    
    plt.show()


def calculate_auc_scores(y_true, y_probs, class_names):
    """Calculate ROC AUC scores (one-vs-rest)"""
    try:
        # One-hot encode labels
        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
        
        # Calculate AUC for each class
        auc_scores = {}
        for i, class_name in enumerate(class_names):
            auc = roc_auc_score(y_true_bin[:, i], y_probs[:, i]) # type: ignore
            auc_scores[class_name] = auc
        
        # Macro average
        auc_scores['macro_avg'] = np.mean(list(auc_scores.values()))
        
        return auc_scores
    except Exception as e:
        print(f"Could not calculate AUC scores: {e}")
        return None


def main():
    # Load config
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Test dataset
    test_dataset = SpermDataset(
        config["data"]["test_dir"],
        transform=get_val_test_transforms()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=4
    )
    
    print(f"Test dataset size: {len(test_dataset)}\n")
    
    # Load model
    model_type = config["training"].get("model_type", "resnet50")
    model = get_model(
        model_type=model_type,
        num_classes=config["training"]["num_classes"],
        pretrained=False
    ).to(device)
    
    # Load checkpoint
    checkpoint_path = f"{config['output']['model_dir']}/{config['output']['best_model_name']}"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Validation accuracy: {checkpoint.get('val_acc', 'unknown'):.4f}\n")
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Class names
    class_names = ["normal", "abnormal", "non_sperm"]
    
    # Standard evaluation
    print("="*60)
    print("Standard Evaluation")
    print("="*60)
    
    y_true, y_pred, y_probs = evaluate_model(model, test_loader, device, use_tta=False)
    
    print("\n" + "="*60)
    print("Classification Report")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    # Calculate accuracy
    accuracy = (y_true == y_pred).mean()
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Calculate AUC scores
    print("\n" + "="*60)
    print("ROC AUC Scores (One-vs-Rest)")
    print("="*60)
    auc_scores = calculate_auc_scores(y_true, y_probs, class_names)
    if auc_scores:
        for class_name, auc in auc_scores.items():
            print(f"{class_name}: {auc:.4f}")
    
    # Visualizations
    print("\n" + "="*60)
    print("Generating Visualizations")
    print("="*60)
    
    # Create output directory for plots
    import os
    plot_dir = f"{config['output']['model_dir']}/evaluation_plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    # Confusion matrix
    plot_confusion_matrix(
        y_true, y_pred, class_names, 
        save_path=f"{plot_dir}/confusion_matrix.png"
    )
    
    # Normalized confusion matrix
    plot_normalized_confusion_matrix(
        y_true, y_pred, class_names,
        save_path=f"{plot_dir}/normalized_confusion_matrix.png"
    )
    
    # Per-class metrics
    plot_per_class_metrics(
        y_true, y_pred, class_names,
        save_path=f"{plot_dir}/per_class_metrics.png"
    )
    
    # Optional: Test-Time Augmentation
    use_tta = config.get("evaluation", {}).get("use_tta", False)
    
    if use_tta:
        print("\n" + "="*60)
        print("Test-Time Augmentation Evaluation")
        print("="*60)
        
        tta_transforms = get_tta_transforms()
        y_true_tta, y_pred_tta, y_probs_tta = evaluate_model(
            model, test_loader, device, use_tta=True, tta_transforms=tta_transforms
        )
        
        print("\n" + "="*60)
        print("Classification Report (with TTA)")
        print("="*60)
        print(classification_report(y_true_tta, y_pred_tta, target_names=class_names, digits=4))
        
        accuracy_tta = (y_true_tta == y_pred_tta).mean()
        print(f"\nOverall Accuracy (TTA): {accuracy_tta:.4f} ({accuracy_tta*100:.2f}%)")
        print(f"Improvement: {(accuracy_tta - accuracy)*100:.2f}%")
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)


if __name__ == "__main__":
    main()