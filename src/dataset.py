import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class FaceEmotionDataset(Dataset):
    """
    Custom PyTorch Dataset loading preprocessed face images from a split folder.
    
    Expected behavior:
    1. Scan the split directory (e.g., data/split/train) for class subfolders.
    2. Build a catalog of image file paths along with their class label index.
    3. Load images inside `__getitem__` on-the-fly to optimize RAM usage.
    4. Apply torchvision transforms (resizing, grayscale conversion, and augmentations).
    """
    def __init__(self, split_dir, classes, transform=None):
        """
        Args:
            split_dir (str): Path to split directory.
            classes (list): List of class names (e.g., ['happy', ...]) matching subfolder names.
            transform (callable, optional): Transforms to apply to images.
        """
        self.split_dir = split_dir
        self.classes = classes
        self.transform = transform
        self.samples = []  # List of tuples: (image_path, label_idx)
        
        # TODO: Populate self.samples by scanning the subdirectories.
        
    def __len__(self):
        # TODO: Return total number of items in dataset.
        return len(self.samples)
        
    def __getitem__(self, idx):
        """
        Args:
            idx (int): Sample index.
            
        Returns:
            tuple: (image_tensor, label_tensor) where image_tensor is model-ready.
        """
        # TODO: Load the image, apply self.transform, and return (image_tensor, label_idx).
        raise NotImplementedError("Implement __getitem__ in FaceEmotionDataset")


def get_dataloaders(config, logger):
    """
    Utility function to build PyTorch DataLoader instances for the train, validation, and test splits.
    
    Tasks to implement:
    1. Check config to see if data augmentation is enabled.
    2. Define transform pipelines:
       - Training transforms (with optional random crop, horizontal flip, rotation, etc.)
       - Validation/Testing transforms (only normalization/resizing, NO random augmentations)
    3. Initialize FaceEmotionDataset for train, val, and test splits.
    4. Wrap datasets in PyTorch DataLoader instances using config-defined batch_size and workers.
    
    Args:
        config (dict): Configuration dictionary.
        logger (logging.Logger): App logger.
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # TODO: Build transforms, instantiate FaceEmotionDataset objects, and setup DataLoaders.
    raise NotImplementedError("Implement get_dataloaders in src/dataset.py")
