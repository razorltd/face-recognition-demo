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
count = 0

# load detection model
prototxtPath = "./caffe/deploy.prototxt"
modelPath = "./caffe/res10_300x300_ssd_iter_140000.caffemodel"
detection_model = cv2.dnn.readNetFromCaffe(prototxtPath, modelPath)

#Load stored encodings
stored_employee_encodings = np.genfromtxt('employee_encodings.csv', dtype=str, delimiter=',')

video_capture = VideoStream(src=0).start()
time.sleep(1.0)

while True:
    # Grab a single frame of video
    frame = video_capture.read()
    resized_frame, blob, h, w = normalize_frame(frame)
    
    if count == 3:
        face_locations = detect_faces_in_frame(detection_model, blob)
        
        rgb_frame = convert_to_rgb(resized_frame)
        face_encodings = encode_faces_from_locations(rgb_frame, [(int(top * 300), int(right * 300), int(bottom * 300), int(left * 300))
                                                         for (left, top, right, bottom) in face_locations])
        face_names = [identify_face_from_encoding(encoding, threshold, stored_employee_encodings) for encoding in face_encodings]
    
        count = 0
    count+= 1
    
    for (left, top, right, bottom), (recognised, name, certainty) in zip(face_locations, face_names):
        top = int(top * h)
        bottom = int(bottom * h)
        left = int(left * w)
        right = int(right * w)
        
        # If we know who it is, draw in blue, else draw in red
        colour = (0, 255, 0)
        if not recognised:
            colour = (0, 0, 255)
            
        cv2.rectangle(frame, (left, top), (right, bottom), colour, 2)
        cv2.rectangle(frame, (left, bottom), (right, bottom - 30), colour, cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, (right - left) / 250, (255, 255, 255), 1)
        
    cv2.imshow('Video', frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
video_capture.stop()