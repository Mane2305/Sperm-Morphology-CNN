import torch
import yaml
import os
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
from collections import defaultdict

from src.dataset import SpermDataset
from src.transforms import get_train_transforms, get_val_test_transforms
from src.model import get_model


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label smoothing for regularization
    """
    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, outputs, targets):
        num_classes = outputs.size(-1)
        log_probs = nn.functional.log_softmax(outputs, dim=-1)
        targets_one_hot = torch.zeros_like(outputs).scatter_(1, targets.unsqueeze(1), 1)
        targets_smooth = (1 - self.epsilon) * targets_one_hot + self.epsilon / num_classes
        loss = (-targets_smooth * log_probs).sum(dim=-1).mean()
        return loss


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=7, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'max':
            if score < self.best_score + self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        else:  # mode == 'min'
            if score > self.best_score - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0


def train_epoch(model, train_loader, criterion, optimizer, device, scaler=None):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Per-class accuracy
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class metrics
            for label, pred in zip(labels, predicted):
                label_item = label.item()
                class_total[label_item] += 1
                if label == pred:
                    class_correct[label_item] += 1
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    # Print per-class accuracy
    print("\nPer-class accuracy:")
    class_names = ['normal', 'abnormal', 'non_sperm']
    for i in range(len(class_names)):
        if class_total[i] > 0:
            acc = 100. * class_correct[i] / class_total[i]
            print(f"  {class_names[i]}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})")
    
    return epoch_loss, epoch_acc


def main():
    # Load config
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    
    # Device
    device = torch.device(
        "cuda" if torch.cuda.is_available() and config["training"]["device"] == "cuda" else "cpu"
    )
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(config['output']['model_dir'], exist_ok=True)
    
    # Datasets
    train_dataset = SpermDataset(
        config["data"]["train_dir"],
        transform=get_train_transforms(advanced=True)
    )
    
    val_dataset = SpermDataset(
        config["data"]["val_dir"],
        transform=get_val_test_transforms()
    )
    
    print(f"\nDataset sizes:")
    print(f"  Training: {len(train_dataset)}")
    print(f"  Validation: {len(val_dataset)}")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Model
    model_type = config["training"].get("model_type", "resnet50")
    dropout = config["training"].get("dropout", 0.5)
    
    model = get_model(
        model_type=model_type,
        num_classes=config["training"]["num_classes"],
        pretrained=True,
        dropout=dropout
    ).to(device)
    
    print(f"\nModel: {model_type}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function
    loss_type = config["training"].get("loss_type", "focal")
    
    if loss_type == "focal":
        # Calculate class weights for focal loss
        class_weights = None
        if config["training"].get("use_class_weights", True):
            from collections import Counter
            labels = [label for _, label in train_dataset.images]
            class_counts = Counter(labels)
            total = sum(class_counts.values())
            weights = [total / (len(class_counts) * class_counts[i]) for i in range(len(class_counts))]
            class_weights = torch.FloatTensor(weights).to(device)
            print(f"\nClass weights: {weights}")
        
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    elif loss_type == "label_smoothing":
        criterion = LabelSmoothingCrossEntropy(epsilon=0.1)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 1e-4)
    )
    
    # Learning rate scheduler
    scheduler_type = config["training"].get("scheduler", "cosine")
    if scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config["training"]["num_epochs"],
            eta_min=1e-6
        )
    elif scheduler_type == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True # type: ignore
        )
    else:
        scheduler = None
    
    # Mixed precision training (for GPU)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config["training"].get("early_stopping_patience", 15),
        min_delta=0.001,
        mode='max'
    )
    
    # Training loop
    best_val_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    print(f"\n{'='*50}")
    print(f"Starting training for {config['training']['num_epochs']} epochs")
    print(f"{'='*50}\n")
    
    for epoch in range(config["training"]["num_epochs"]):
        print(f"\nEpoch [{epoch+1}/{config['training']['num_epochs']}]")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch summary
        print(f"\nEpoch Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Update learning rate
        if scheduler is not None:
            if scheduler_type == "plateau":
                scheduler.step(val_acc) # type: ignore
            else:
                scheduler.step() # type: ignore
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, f"{config['output']['model_dir']}/{config['output']['best_model_name']}")
            print(f"  ✅ Best model saved! (Val Acc: {val_acc:.4f})")
        
        # Early stopping
        early_stopping(val_acc)
        if early_stopping.early_stop:
            print(f"\n⚠️ Early stopping triggered at epoch {epoch+1}")
            break
    
    print(f"\n{'='*50}")
    print(f"Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"{'='*50}\n")
    
    # Save training history
    import json
    with open(f"{config['output']['model_dir']}/training_history.json", 'w') as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()