import cv2
import numpy as np
from picamera2 import Picamera2


class Camera:
    def __init__(self):
        self.class_names = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", 
                            "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", 
                            "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

        self.net = cv2.dnn.readNetFromCaffe('/home/ws/ugv_rpi/shadowfax/models/deploy.prototxt', 
                                            '/home/ws/ugv_rpi/shadowfax/models/mobilenet_iter_73000.caffemodel')

        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_video_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
        self.picam2.start()
    
    def get_frame(self):
        frame = self.picam2.capture_array()  # read the camera frame
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame, img

    def annotate_rectangle(self, label, frame, startX, startY, endX, endY):
        """Annotate the object and confidence on the image"""
        cv2.rectangle(frame, (int(startX), int(startY)), (int(endX), int(endY)), (0, 255, 0), 2)
        y = int(startY) - 15 if int(startY) - 15 > 15 else int(startY) + 15
        cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

class Detection(Camera):
    """Object detection class inheriting from Camera

    Ensure future code instantiates Detection (or 
    another subclass implementing detect_object) 
    before calling detection helpers.
    """
    def __init__(self):
        super().__init__()

    def read_detections(self, img):
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()
        return detections

    def annotate_detection(self, label, frame, startX, startY, endX, endY):
        self.annotate_rectangle(label, frame, startX, startY, endX, endY)

    def detect_object(self, frame, img, object_class):
        object_height = None
        object_center_x = None
        object_center_y = None

        (frame_height, frame_width) = img.shape[:2]

        detections = self.read_detections(img=img)
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.4:
                idx = int(detections[0, 0, i, 1])
                if self.class_names[idx] == object_class:
                    box = detections[0, 0, i, 3:7] * np.array([frame_width, frame_height, frame_width, frame_height])
                    (startX, startY, endX, endY) = box.astype("int")
                    object_center_x = (startX + endX) // 2
                    object_center_y = (startY + endY) // 2
                    object_height = endY - startY
                    self.annotate_detection(f"C:{object_center_x}, H:{object_height}", frame, startX, startY, endX, endY)
        return object_height, object_center_x, object_center_y
