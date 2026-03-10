import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights, EfficientNet_B0_Weights


class SpermClassifier(nn.Module):
    """
    Modern sperm morphology classifier with improved architecture
    """
    def __init__(self, num_classes=3, backbone='resnet50', pretrained=True, dropout=0.5):
        super().__init__()
        
        self.backbone_name = backbone
        
        if backbone == 'resnet50':
            # Use ResNet50 for better feature extraction
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            num_features = self.backbone.fc.in_features
            
            # Replace classifier with improved head
            self.backbone.fc = nn.Sequential( # type: ignore
                nn.Dropout(dropout),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(dropout / 2),
                nn.Linear(512, num_classes)
            )
            
        elif backbone == 'efficientnet_b0':
            # EfficientNet is more parameter-efficient
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            num_features = self.backbone.classifier[1].in_features
            
            # Replace classifier
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(num_features, 512), # type: ignore
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(dropout / 2),
                nn.Linear(512, num_classes)
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x):
        return self.backbone(x)


class SpermClassifierWithAttention(nn.Module):
    """
    Advanced classifier with attention mechanism for focusing on sperm head/tail
    """
    def __init__(self, num_classes=3, dropout=0.5):
        super().__init__()
        
        # Use ResNet50 as feature extractor
        weights = ResNet50_Weights.IMAGENET1K_V2
        backbone = models.resnet50(weights=weights)
        
        # Remove the final fully connected layer
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        
        # Spatial attention
        self.attention = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # Extract features
        features = self.features(x)  # [B, 2048, H, W]
        
        # Apply attention
        attention_weights = self.attention(features)  # [B, 1, H, W]
        attended_features = features * attention_weights  # Element-wise multiplication
        
        # Global pooling
        pooled = self.global_pool(attended_features)  # [B, 2048, 1, 1]
        pooled = pooled.view(pooled.size(0), -1)  # [B, 2048]
        
        # Classification
        output = self.classifier(pooled)
        
        return output


def get_model(model_type='resnet50', num_classes=3, pretrained=True, dropout=0.5):
    """
    Factory function to get the appropriate model
    
    Args:
        model_type: 'resnet50', 'efficientnet_b0', or 'attention'
        num_classes: Number of output classes
        pretrained: Use pretrained weights
        dropout: Dropout rate
    """
    if model_type == 'attention':
        return SpermClassifierWithAttention(num_classes=num_classes, dropout=dropout)
    else:
        return SpermClassifier(
            num_classes=num_classes, 
            backbone=model_type, 
            pretrained=pretrained,
            dropout=dropout
        )