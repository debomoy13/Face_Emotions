import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from src.dataset import get_dataloaders
from src.model import EmotionCNN
from src.utils import get_device

def train_model(config, logger):
    """
    Coordinates the model training process.
    
    Tasks to implement:
    1. Get training and validation loaders from `src/dataset.py`.
    2. Instantiate the `EmotionCNN` model and transfer it to the target hardware device.
    3. Instantiate loss criterion (CrossEntropyLoss) and optimizer (e.g., Adam or SGD).
    4. Execute training epochs:
       - Set model to training mode. Run forward pass, compute loss, backpropagate gradients, optimize weights.
       - Track training batch losses and compute mean epoch loss.
       - Run validation epoch (eval mode):
         * Compute validation loss.
         * Compute validation accuracy.
       - Compare validation results. If it's the best seen so far, save the model weights (state_dict) to config-specified file path.
       - Print and log epoch summaries (Loss/Acc for Train/Val).
    5. Save training history curves (Loss and Accuracy) as a plot under the output directory.
    
    Args:
        config (dict): App config.
        logger (logging.Logger): App logger.
    """
    logger.info("Setting up for training...")
    # TODO: Implement full training loop, metrics computation, validation evaluation, 
    # checkpoint saving, and history plotting.
    raise NotImplementedError("Implement train_model in src/train.py")
