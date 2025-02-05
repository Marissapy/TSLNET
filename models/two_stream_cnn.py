import torch
import torch.nn as nn
import torchvision.models as models

class SpatialStream(nn.Module):
    """
    The spatial stream of the Two-Stream CNN.
    This stream captures the appearance information from individual frames.
    """
    def __init__(self, num_classes=10):
        super(SpatialStream, self).__init__()
        
        # Load a pre-trained ResNet50 model and modify it for our task
        self.resnet = models.resnet50(pretrained=True)
        self.num_features = self.resnet.fc.in_features
        
        # Replace the last fully connected layer to match our number of classes
        self.resnet.fc = nn.Linear(self.num_features, num_classes)

    def forward(self, x):
        """
        Forward pass through the spatial stream.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        return self.resnet(x)

class TemporalStream(nn.Module):
    """
    The temporal stream of the Two-Stream CNN.
    This stream captures the motion information from stacked optical flow images.
    """
    def __init__(self, num_classes=10):
        super(TemporalStream, self).__init__()
        
        # Load a pre-trained ResNet50 model and modify it for our task
        self.resnet = models.resnet50(pretrained=True)
        self.num_features = self.resnet.fc.in_features
        
        # Replace the last fully connected layer to match our number of classes
        self.resnet.fc = nn.Linear(self.num_features, num_classes)

    def forward(self, x):
        """
        Forward pass through the temporal stream.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        return self.resnet(x)

class TwoStreamCNN(nn.Module):
    """
    The Two-Stream CNN model that combines the spatial and temporal streams.
    """
    def __init__(self, num_classes=10):
        super(TwoStreamCNN, self).__init__()
        
        # Initialize the spatial and temporal streams
        self.spatial_stream = SpatialStream(num_classes)
        self.temporal_stream = TemporalStream(num_classes)
        
        # Final fusion layer
        self.fusion = nn.Linear(2 * num_classes, num_classes)

    def forward(self, spatial_input, temporal_input):
        """
        Forward pass through the two-stream CNN.
        
        Args:
            spatial_input (torch.Tensor): Input tensor for the spatial stream.
            temporal_input (torch.Tensor): Input tensor for the temporal stream.
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        # Pass through the spatial and temporal streams
        spatial_output = self.spatial_stream(spatial_input)
        temporal_output = self.temporal_stream(temporal_input)
        
        # Concatenate the outputs and pass through the fusion layer
        combined_output = torch.cat((spatial_output, temporal_output), dim=1)
        final_output = self.fusion(combined_output)
        
        return final_output
