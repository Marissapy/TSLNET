import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    The LSTM model for processing the time-series data from the Two-Stream CNN.
    This model captures long-term and short-term dependencies in the sequence of features.
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        
        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Define the fully connected layer for classification
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        Forward pass through the LSTM model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_size).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        # Pass through the LSTM layer
        out, _ = self.lstm(x)
        
        # Take the last output of the LSTM sequence
        out = out[:, -1, :]
        
        # Pass through the fully connected layer
        out = self.fc(out)
        
        return out
