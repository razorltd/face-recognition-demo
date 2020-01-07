from face_recogniser_service import *
import cv2
import numpy as np
import imutils
from imutils.video import VideoStream
import time

threshold = 0.5
face_locations = []
face_encodings = []
face_names = []
frame_count = 0
process_this_frame = True

# load detection model from disk
modelPath = "/home/pi/opencv/data/haarcascades/haarcascade_frontalface_default.xml"
detection_model = cv2.CascadeClassifier(modelPath)
stored_employee_encodings = np.genfromtxt('employee_encodings.csv', dtype=str, delimiter=',')

video_capture = VideoStream(src=0).start()
time.sleep(1.0)

while True:
    # Grab a single frame of video
    frame = video_capture.read()
    h, w = frame.shape[:2]
    frame_small = imutils.resize(frame, width=int(w/4))
    
    if process_this_frame:
        face_locations = detect_faces_in_frame_cascade(detection_model, frame_small)
        
        rgb_frame = convert_to_rgb(frame_small)
        face_encodings = encode_faces_from_locations(rgb_frame, face_locations)
        face_names = [identify_face_from_encoding(encoding, threshold, stored_employee_encodings) for encoding in face_encodings]
    
    process_this_frame = not process_this_frame
    
    for (top, right, bottom, left), (recognised, name, certainty) in zip(face_locations, face_names):
        top = int(top * 4)
        bottom = int(bottom * 4)
        left = int(left * 4)
        right = int(right * 4)
        
        colour = (0, 255, 0)
        if not recognised:
            colour = (0, 0, 255)
            
        cv2.rectangle(frame, (left, top), (right, bottom - 10), colour, 2)
        cv2.rectangle(frame, (left, bottom + 10), (right, bottom - 10), colour, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom + 6), font, (right - left) / 250, (255, 255, 255), 1)
        
    cv2.imshow('Video', frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
video_capture.stop()
