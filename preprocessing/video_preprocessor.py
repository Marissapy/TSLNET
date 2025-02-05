import cv2
import os

class VideoPreprocessor:
    """
    Preprocesses a video by extracting frames and optionally computing optical flow.
    """
    def __init__(self, video_path, output_dir, frame_rate=30):
        self.video_path = video_path
        self.output_dir = output_dir
        self.frame_rate = frame_rate
        self.cap = cv2.VideoCapture(video_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def extract_frames(self):
        """
        Extracts frames from the video and saves them to the output directory.
        """
        frame_count = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame_filename = os.path.join(self.output_dir, f'frame_{frame_count:04d}.jpg')
            cv2.imwrite(frame_filename, frame)
            frame_count += 1
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, (frame_count + 1) * self.frame_rate)
        self.cap.release()

    def compute_optical_flow(self):
        """
        Computes optical flow for the extracted frames and saves the results.
        """
        frames = [os.path.join(self.output_dir, f) for f in sorted(os.listdir(self.output_dir)) if f.endswith('.jpg')]
        for i in range(len(frames) - 1):
            frame1 = cv2.imread(frames[i], cv2.IMREAD_GRAYSCALE)
            frame2 = cv2.imread(frames[i + 1], cv2.IMREAD_GRAYSCALE)
            flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow_filename = os.path.join(self.output_dir, f'flow_{i:04d}.npy')
            np.save(flow_filename, flow)
