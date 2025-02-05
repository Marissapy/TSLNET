import cv2

class PedestrianTracker:
    """
    Tracks pedestrians in a sequence of frames using a pre-trained model.
    """
    def __init__(self, model_path):
        self.model = cv2.dnn.readNetFromCaffe(model_path + 'deploy.prototxt', model_path + 'res10_300x300_ssd_iter_140000.caffemodel')

    def detect_pedestrians(self, frame):
        """
        Detects pedestrians in a single frame.
        
        Args:
            frame (numpy.ndarray): Input frame.
        
        Returns:
            list: List of bounding boxes [(x1, y1, x2, y2), ...].
        """
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.model.setInput(blob)
        detections = self.model.forward()
        boxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                boxes.append(box.astype(int))
        return boxes
