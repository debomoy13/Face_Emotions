import os
import shutil
import random

def split_dataset(config, logger):
    """
    Splits the collected raw images from `data/raw/` into `data/split/train/`, `data/split/val/`, and `data/split/test/`.
    
    Steps to implement:
    1. Retrieve directories and split ratios from configuration.
    2. Empty the train, val, and test target folders to prevent accumulation from previous runs.
    3. Iterate through each emotion folder in the raw directory:
       - List all image file paths.
       - Shuffle the file list to ensure randomized distributions.
       - Calculate index splits (e.g., 70% train, 15% validation, 15% test).
       - Copy (or write) files to their destination split directories (maintaining subfolder class names).
    4. Log split counts for verification.
    
    Args:
        config (dict): Loaded application config.
        logger (logging.Logger): Centralized logging handle.
    """
    logger.info("Starting dataset splitting process...")
    # TODO: Implement file copying/moving, random shuffling, split boundary calculation.
    raise NotImplementedError("Implement split_dataset in src/dataset_prep.py")
