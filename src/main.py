import cv2
import os
from preprocessing.video_preprocessor import VideoPreprocessor
from tracking.pedestrian_tracker import PedestrianTracker
from recognition.action_recognizer import ActionRecognizer
from utils.visualization import draw_bounding_boxes, display_frame
from utils.config_loader import ConfigLoader

def main():
    # Load configuration
    config = ConfigLoader('config.json')
    video_path = config.get('video_path', 'input_video.mp4')
    output_dir = config.get('output_dir', 'output_frames')
    model_path = config.get('model_path', 'models/')
    spatial_model_path = os.path.join(model_path, 'spatial_model.pth')
    temporal_model_path = os.path.join(model_path, 'temporal_model.pth')
    lstm_model_path = os.path.join(model_path, 'lstm_model.pth')

    # Preprocess the video
    preprocessor = VideoPreprocessor(video_path, output_dir)
    preprocessor.extract_frames()
    preprocessor.compute_optical_flow()

    # Initialize the tracker and recognizer
    tracker = PedestrianTracker(model_path)
    recognizer = ActionRecognizer(spatial_model_path, temporal_model_path, lstm_model_path)

    # Process each frame
    frames = [os.path.join(output_dir, f) for f in sorted(os.listdir(output_dir)) if f.startswith('frame_')]
    for frame_path in frames:
        frame = cv2.imread(frame_path)
        bboxes = tracker.detect_pedestrians(frame)
        frame_with_bboxes = draw_bounding_boxes(frame, bboxes)
        display_frame(frame_with_bboxes)

        # For simplicity, we assume only one pedestrian per frame
        if bboxes:
            spatial_frame = torch.tensor(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            temporal_frame = torch.tensor(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            action = recognizer.recognize_action([spatial_frame], [temporal_frame])
            print(f'Action: {action}')

if __name__ == '__main__':
    main()
