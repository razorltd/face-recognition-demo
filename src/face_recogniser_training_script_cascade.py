from face_recogniser_service import *
import cv2
import numpy as np
import os
import imutils

# load detection model from disk
modelPath = "/home/pi/opencv/data/haarcascades/haarcascade_frontalface_default.xml"
detection_model = cv2.CascadeClassifier(modelPath)

employee_encodings = []
employee_directory_path = "../employee_directory"
employee_folders = os.listdir(employee_directory_path)
for employee_folder in employee_folders:
    employee_folder_path = os.path.join(employee_directory_path, employee_folder)
    images = os.listdir(employee_folder_path)
    for image_name in images:
        image_path = os.path.join(employee_folder_path, image_name)
        image = cv2.imread(image_path)
        
        h, w = image.shape[:2]
        image_small = imutils.resize(image, width=int(w/4))
    
        face_location = detect_faces_in_frame_cascade(detection_model, image_small)
        if face_location.shape[0] != 1:
            raise ValueError("Multiple/no faces identified in training image", image_path)
        
        rgb_frame = convert_to_rgb(image_small)
        face_encoding = encode_faces_from_locations(rgb_frame, face_location)[0]
        
        string_encoding = np.array([str(x) for x in face_encoding])
        row = np.insert(string_encoding, 0, employee_folder)
        employee_encodings.append(row)

np.savetxt("data/employee_encodings.csv", employee_encodings, '%s', ',')