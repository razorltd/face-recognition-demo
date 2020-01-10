import face_recognition
import cv2
import numpy as np

def normalize_frame(frame):
    h, w = frame.shape[:2]
    resized_frame = cv2.resize(frame, (300, 300))
    blob = cv2.dnn.blobFromImage(resized_frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    return resized_frame, blob, h, w

def convert_to_rgb(frame):
    rgb = frame[:, :, ::-1]
    return rgb

def detect_faces_in_frame(detection_model, blob):
    detection_model.setInput(blob)
    detections = detection_model.forward()
    face_locations = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.8:
            face_locations.append(detections[0, 0, i, 3:7])
    return np.array(face_locations)

def detect_faces_in_frame_cascade(detection_model, blob):
    frame_gray = cv2.cvtColor(blob, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    faces = detection_model.detectMultiScale(frame_gray)
    face_locations = [(y, x+w, y+h, x) for (x, y, w, h) in faces]
    return np.array(face_locations)

def encode_faces_from_locations(frame, locations):
    encodings = face_recognition.face_encodings(frame, locations)
    return encodings

def identify_face_from_encoding(encoding_to_compare, threshold, stored_employee_encodings):
    encodings = stored_employee_encodings[:, 1:].astype(float)
    face_distances = face_recognition.face_distance(encodings, encoding_to_compare)
    encoding_distances = np.c_[ stored_employee_encodings[:, 0], face_distances ]
    employees = np.unique(encoding_distances[:, 0])
    mean_distances = [np.mean(encoding_distances[np.where(encoding_distances[:, 0] == employee)][:, 1].astype(float))
                               for employee in employees]
    employees_avg = np.c_[ employees, mean_distances ]

    minimum = employees_avg[np.argmin(employees_avg[:, 1])]
    if float(minimum[1]) <= threshold:
        return True, minimum[0], minimum[1]
    return False, 'Unknown', minimum[1]
    