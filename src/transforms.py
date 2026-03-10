import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size=224, advanced=True):
    """
    Training transforms with aggressive augmentation for small datasets
    
    Args:
        image_size: Target image size
        advanced: Use advanced augmentations (recommended for medical images)
    """
    if advanced:
        return A.Compose([
            A.Resize(image_size, image_size),
            
            # Geometric transformations
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=30, p=0.7),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=30,
                p=0.5
            ),
            
            # Optical transformations (simulate microscopy variations)
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=7, p=1.0),
                A.MedianBlur(blur_limit=7, p=1.0),
            ], p=0.3),
            
            # Color/brightness (simulate staining variations)
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=15,
                sat_shift_limit=25,
                val_shift_limit=20,
                p=0.4
            ),
            
            # Additional augmentations for robustness
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0), # type: ignore
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            ], p=0.3),
            
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            
            # Coarse dropout (simulates occlusions)
            A.CoarseDropout(
                max_holes=8, # type: ignore
                max_height=20, # type: ignore
                max_width=20, # type: ignore
                min_holes=1, # type: ignore
                min_height=10, # type: ignore
                min_width=10, # type: ignore
                fill_value=0, # type: ignore
                p=0.3
            ),
            
            # Normalization (ImageNet stats)
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2()
        ])


def get_val_test_transforms(image_size=224):
    """
    Validation/Test transforms - only resize and normalize
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2()
    ])


def get_tta_transforms(image_size=224):
    """
    Test-Time Augmentation transforms for ensemble predictions
    Returns a list of different augmentations
    """
    base_transform = [
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2()
    ]
    
    return [
        # Original
        A.Compose(base_transform),
        
        # Horizontal flip
        A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ]),
        
        # Rotate 10 degrees
        A.Compose([
            A.Resize(image_size, image_size),
            A.Rotate(limit=10, p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ]),
        
        # Rotate -10 degrees
        A.Compose([
            A.Resize(image_size, image_size),
            A.Rotate(limit=(-10, -10), p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ]),
        
        # Brightness adjustment
        A.Compose([
            A.Resize(image_size, image_size),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    ]