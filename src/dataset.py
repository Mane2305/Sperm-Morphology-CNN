import os
import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path


class SpermDataset(Dataset):
    """
    Dataset for sperm morphology classification
    
    Args:
        data_dir: Path to data directory with class subdirectories
        transform: Albumentations transform pipeline
        cache_images: Cache images in memory (for small datasets)
    """
    def __init__(self, data_dir, transform=None, cache_images=False):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.cache_images = cache_images
        
        # Get class names from subdirectories
        self.classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Build image list
        self.images = []
        for cls_name in self.classes:
            cls_idx = self.class_to_idx[cls_name]
            cls_path = self.data_dir / cls_name
            
            # Support multiple image formats
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            
            for img_path in cls_path.iterdir():
                if img_path.suffix.lower() in image_extensions:
                    self.images.append((str(img_path), cls_idx))
        
        # Optionally cache images
        self.image_cache = {}
        if cache_images:
            print(f"Caching {len(self.images)} images...")
            for idx, (img_path, _) in enumerate(self.images):
                image = cv2.imread(img_path)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    self.image_cache[idx] = image
            print(f"Cached {len(self.image_cache)} images")
        
        print(f"\nDataset initialized:")
        print(f"  Directory: {data_dir}")
        print(f"  Classes: {self.classes}")
        print(f"  Total images: {len(self.images)}")
        
        # Print class distribution
        from collections import Counter
        label_counts = Counter([label for _, label in self.images])
        print(f"  Class distribution:")
        for cls_name, cls_idx in self.class_to_idx.items():
            count = label_counts[cls_idx]
            print(f"    {cls_name}: {count}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        
        # Load from cache or disk
        if idx in self.image_cache:
            image = self.image_cache[idx].copy()
        else:
            image = cv2.imread(img_path)
            
            if image is None:
                raise ValueError(f"Failed to load image: {img_path}")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        
        return image, torch.tensor(label, dtype=torch.long)
    
    def get_class_weights(self):
        """Calculate class weights for handling imbalance"""
        from collections import Counter
        
        label_counts = Counter([label for _, label in self.images])
        total = sum(label_counts.values())
        
        weights = []
        for i in range(len(self.classes)):
            count = label_counts[i]
            weight = total / (len(self.classes) * count)
            weights.append(weight)
        
        return torch.FloatTensor(weights)


class BalancedBatchSampler(torch.utils.data.Sampler):
    """
    Sampler that ensures balanced batches during training
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Group indices by class
        self.class_indices = {}
        for idx, (_, label) in enumerate(dataset.images):
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)
        
        # Calculate number of batches
        self.num_batches = min(len(indices) for indices in self.class_indices.values()) // (batch_size // len(self.class_indices))
    
    def __iter__(self):
        # Shuffle indices for each class
        for label in self.class_indices:
            import random
            random.shuffle(self.class_indices[label])
        
        # Create balanced batches
        batch_indices = []
        samples_per_class = self.batch_size // len(self.class_indices)
        
        for batch_idx in range(self.num_batches):
            batch = []
            for label in self.class_indices:
                start_idx = batch_idx * samples_per_class
                end_idx = start_idx + samples_per_class
                batch.extend(self.class_indices[label][start_idx:end_idx])
            
            import random
            random.shuffle(batch)
            batch_indices.extend(batch)
        
        return iter(batch_indices)
    
    def __len__(self):
        return self.num_batches * self.batch_size