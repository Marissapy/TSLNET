import torch
from two_stream_cnn import TwoStreamCNN
from lstm_model import LSTMModel

class ActionRecognizer:
    """
    Recognizes actions in a sequence of frames using a Two-Stream CNN and LSTM.
    """
    def __init__(self, spatial_model_path, temporal_model_path, lstm_model_path, num_classes=10):
        self.spatial_model = TwoStreamCNN(num_classes)
        self.temporal_model = TwoStreamCNN(num_classes)
        self.lstm_model = LSTMModel(input_size=num_classes, hidden_size=128, num_layers=2, num_classes=num_classes)
        
        self.spatial_model.load_state_dict(torch.load(spatial_model_path))
        self.temporal_model.load_state_dict(torch.load(temporal_model_path))
        self.lstm_model.load_state_dict(torch.load(lstm_model_path))
        
        self.spatial_model.eval()
        self.temporal_model.eval()
        self.lstm_model.eval()

    def recognize_action(self, spatial_frames, temporal_frames):
        """
        Recognizes the action in a sequence of frames.
        
        Args:
            spatial_frames (list of torch.Tensor): List of spatial frames.
            temporal_frames (list of torch.Tensor): List of temporal frames.
        
        Returns:
            int: Predicted class index.
        """
        with torch.no_grad():
            spatial_features = [self.spatial_model(frame.unsqueeze(0)) for frame in spatial_frames]
            temporal_features = [self.temporal_model(frame.unsqueeze(0)) for frame in temporal_frames]
            combined_features = [torch.cat((spatial, temporal), dim=1) for spatial, temporal in zip(spatial_features, temporal_features)]
            combined_features = torch.stack(combined_features)
            lstm_output = self.lstm_model(combined_features)
            _, predicted_class = torch.max(lstm_output, 1)
            return predicted_class.item()
