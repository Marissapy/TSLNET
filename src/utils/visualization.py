import cv2

def draw_bounding_boxes(frame, bboxes, color=(0, 255, 0), thickness=2):
    """
    Draws bounding boxes on a frame.
    
    Args:
        frame (numpy.ndarray): Input frame.
        bboxes (list): List of bounding boxes [(x1, y1, x2, y2), ...].
        color (tuple): Bounding box color.
        thickness (int): Bounding box thickness.
    
    Returns:
        numpy.ndarray: Frame with bounding boxes drawn.
    """
    for (x1, y1, x2, y2) in bboxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    return frame

def display_frame(frame, window_name='Frame'):
    """
    Displays a frame in a window.
    
    Args:
        frame (numpy.ndarray): Input frame.
        window_name (str): Window name.
    """
    cv2.imshow(window_name, frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
