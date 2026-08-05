import torch
import torch.nn as nn

class EmotionCNN(nn.Module):
    """
    Custom CNN architecture for facial expression classification.
    
    This model builds its layers dynamically from configuration specifications:
    - Input channel count (1 for grayscale, 3 for RGB).
    - A list of convolutional layer specifications: `conv_layers`
    - A list of fully connected hidden dimensions: `fc_layers`
    - Number of target classes (e.g., 5 emotions).
    - Dropout rate.
    
    Example block design for each Conv step:
      Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d -> Dropout
    """
    def __init__(self, config):
        """
        Args:
            config (dict): Configuration dictionary containing model architecture details.
        """
        super(EmotionCNN, self).__init__()
        self.config = config
        
        # Read from config:
        # e.g., input_channels = config['dataset']['channels']
        # e.g., conv_configs = config['model']['conv_layers']
        # e.g., fc_configs = config['model']['fc_layers']
        # e.g., num_classes = len(config['dataset']['classes'])
        # e.g., dropout = config['model']['dropout_rate']
        
        self.features = nn.Sequential()
        # TODO: Dynamically build the convolutional feature extractor blocks
        
        # TODO: Compute flattening shape automatically with a dummy forward pass
        self.classifier = nn.Sequential()
        # TODO: Dynamically build fully connected layers mapping flat features to class logits
        
    def _get_conv_output_shape(self, input_shape):
        """
        Utility method to compute the output shape of the convolutional layers 
        by running a mock/dummy tensor through `self.features`.
        
        Args:
            input_shape (tuple): Shape representing (batch_size, channels, height, width).
            
        Returns:
            int: The flattened feature count size (e.g., Channels * H * W).
        """
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            dummy_output = self.features(dummy_input)
            return dummy_output.numel()

    def forward(self, x):
        """
        Executes the network's forward pass.
        
        Args:
            x (torch.Tensor): Image tensor of shape (Batch, Channels, Height, Width).
            
        Returns:
            torch.Tensor: Logits tensor of shape (Batch, NumClasses).
        """
        # TODO: Implement the forward logic passing inputs through features and classifier
        raise NotImplementedError("Implement forward in EmotionCNN")
