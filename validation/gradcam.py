# validation/gradcam.py
"""
Grad-CAM visualization for model interpretability
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for interpretability
    
    Shows which regions of the image the model focuses on for classification
    
    Example:
        >>> from validation.gradcam import GradCAM
        >>> 
        >>> gradcam = GradCAM(model, target_layer=model.layer4)
        >>> heatmap = gradcam.generate_cam(input_image, target_class=1)
        >>> overlay = GradCAM.overlay_heatmap(original_image, heatmap)
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Args:
            model: PyTorch model
            target_layer: Layer to visualize (e.g., model.layer4 for ResNet)
        """
        self.model = model
        self.target_layer = target_layer
        
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.forward_hook = target_layer.register_forward_hook(self._save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradient)
        
        logger.info(f"GradCAM initialized for layer: {target_layer.__class__.__name__}")
    
    def _save_activation(self, module, input, output):
        """Hook to save forward activation"""
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward gradient"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(
        self, 
        input_image: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_image: Input tensor (1, C, H, W) or (C, H, W)
            target_class: Target class index (None = predicted class)
        
        Returns:
            CAM heatmap (H, W) normalized to [0, 1]
        """
        # Ensure batch dimension
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        
        self.model.eval()
        
        # Forward pass
        output = self.model(input_image)
        
        # Get predicted class if not specified
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        class_score = output[0, target_class]
        class_score.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # type: ignore # (C, H, W)
        activations = self.activations[0]  # type: ignore # (C, H, W)
        
        # Calculate weights (global average pooling of gradients)
        weights = torch.mean(gradients, dim=(1, 2))  # (C,)
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)  # (H, W)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU (only positive influence)
        cam = torch.clamp(cam, min=0)
        
        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam - cam.min()
            cam = cam / cam.max()
        
        return cam.cpu().numpy()
    
    @staticmethod
    def overlay_heatmap(
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Overlay heatmap on original image
        
        Args:
            image: Original image (H, W, 3) in [0, 1] or [0, 255]
            heatmap: Grad-CAM heatmap (H, W) in [0, 1]
            alpha: Heatmap transparency (0=invisible, 1=opaque)
            colormap: OpenCV colormap (default: COLORMAP_JET)
        
        Returns:
            Overlayed image (H, W, 3) in [0, 1]
        """
        # Normalize image to [0, 1]
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        
        # Resize heatmap to match image
        if heatmap.shape != image.shape[:2]:
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Apply colormap to heatmap
        heatmap_colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8), 
            colormap
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        heatmap_colored = heatmap_colored.astype(np.float32) / 255.0
        
        # Overlay
        overlayed = (1 - alpha) * image + alpha * heatmap_colored
        overlayed = np.clip(overlayed, 0, 1)
        
        return overlayed
    
    @staticmethod
    def save_visualization(
        original_image: np.ndarray,
        heatmap: np.ndarray,
        save_path: str,
        prediction: str = None, # type: ignore
        confidence: float = None, # type: ignore
        ground_truth: str = None # type: ignore
    ):
        """
        Save Grad-CAM visualization with original image and overlay
        
        Args:
            original_image: Original image (H, W, 3)
            heatmap: Grad-CAM heatmap (H, W)
            save_path: Path to save visualization
            prediction: Predicted class name
            confidence: Prediction confidence
            ground_truth: True class name
        """
        # Create overlay
        overlay = GradCAM.overlay_heatmap(original_image, heatmap)
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Heatmap
        im = axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)
        
        # Overlay
        axes[2].imshow(overlay)
        title = 'Overlay'
        if prediction:
            title += f'\nPred: {prediction}'
            if confidence:
                title += f' ({confidence:.2%})'
        if ground_truth:
            title += f'\nTrue: {ground_truth}'
        axes[2].set_title(title, fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Grad-CAM visualization saved to: {save_path}")
    
    def cleanup(self):
        """Remove hooks"""
        self.forward_hook.remove()
        self.backward_hook.remove()
    
    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.cleanup()
        except:
            pass


def batch_gradcam_analysis(
    model: nn.Module,
    dataloader,
    target_layer: nn.Module,
    save_dir: str = 'gradcam_results',
    n_samples: int = 10,
    class_names: list = None # type: ignore
):
    """
    Generate Grad-CAM visualizations for multiple samples
    
    Args:
        model: PyTorch model
        dataloader: DataLoader with images
        target_layer: Layer to visualize
        save_dir: Directory to save results
        n_samples: Number of samples to visualize
        class_names: List of class names
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    if class_names is None:
        class_names = ['Class 0', 'Class 1', 'Class 2']
    
    gradcam = GradCAM(model, target_layer)
    device = next(model.parameters()).device
    
    count = 0
    for images, labels in dataloader:
        if count >= n_samples:
            break
        
        images = images.to(device)
        labels = labels.to(device)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            confidences = torch.max(probs, dim=1)[0]
        
        for i in range(images.size(0)):
            if count >= n_samples:
                break
            
            # Generate Grad-CAM
            heatmap = gradcam.generate_cam(images[i:i+1], target_class=preds[i].item()) # type: ignore
            
            # Convert image to numpy
            image_np = images[i].cpu().permute(1, 2, 0).numpy()
            
            # Denormalize if needed (assuming ImageNet normalization)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_np = image_np * std + mean
            image_np = np.clip(image_np, 0, 1)
            
            # Save visualization
            pred_name = class_names[preds[i].item()] # type: ignore
            true_name = class_names[labels[i].item()]
            correct = '✓' if preds[i] == labels[i] else '✗'
            
            filename = f'sample_{count:03d}_{correct}_pred_{pred_name}_true_{true_name}.png'
            save_path = os.path.join(save_dir, filename)
            
            GradCAM.save_visualization(
                image_np,
                heatmap,
                save_path,
                prediction=pred_name,
                confidence=confidences[i].item(),
                ground_truth=true_name
            )
            
            count += 1
    
    gradcam.cleanup()
    logger.info(f"✓ Generated {count} Grad-CAM visualizations in {save_dir}")


if __name__ == "__main__":
    # Example usage
    print("GradCAM module loaded successfully")
    print("\nExample usage:")
    print("""
    from validation.gradcam import GradCAM
    
    # Initialize
    gradcam = GradCAM(model, target_layer=model.layer4)
    
    # Generate heatmap
    heatmap = gradcam.generate_cam(input_image, target_class=1)
    
    # Create overlay
    overlay = GradCAM.overlay_heatmap(original_image, heatmap)
    
    # Save visualization
    GradCAM.save_visualization(
        original_image, heatmap, 
        save_path='gradcam_example.png',
        prediction='Abnormal', confidence=0.87
    )
    
    # Batch analysis
    from validation.gradcam import batch_gradcam_analysis
    batch_gradcam_analysis(model, test_loader, model.layer4, n_samples=20)
    """)