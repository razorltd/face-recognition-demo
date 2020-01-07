from face_recogniser_service import *
import cv2
import numpy as np
import os

# load detection model from disk
prototxt_path = "./caffe/deploy.prototxt"
model_path = "./caffe/res10_300x300_ssd_iter_140000.caffemodel"
detection_model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

employee_encodings = []
employee_directory_path = "./employee_directory"
employee_folders = os.listdir(employee_directory_path)
for employee_folder in employee_folders:
    employee_folder_path = os.path.join(employee_directory_path, employee_folder)
    images = os.listdir(employee_folder_path)
    for image_name in images:
        image_path = os.path.join(employee_folder_path, image_name)
        image = cv2.imread(image_path)
        resized_frame, blob, h, w = normalize_frame(image)

        face_location = detect_faces_in_frame(detection_model, blob)
        if face_location.shape[0] != 1:
            raise ValueError("Multiple/no faces identified in training image", image_path)
        
        left, top, right, bottom = face_location[0]
        rgb_frame = convert_to_rgb(resized_frame)
        face_encoding = encode_faces_from_locations(rgb_frame, [(int(top * 300), int(right * 300), int(bottom * 300), int(left * 300))])[0]
        string_encoding = np.array([str(x) for x in face_encoding])
        row = np.insert(string_encoding, 0, employee_folder)
        employee_encodings.append(row)

np.savetxt("employee_encodings.csv", employee_encodings, '%s', ',')