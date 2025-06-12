import face_recognition
import os
import pickle

def get_face_encodings(image_folder):
    encodings = []
    names = []

    for person_name in os.listdir(image_folder):
        person_folder = os.path.join(image_folder, person_name)
        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)
            if encoding:
                encodings.append(encoding[0])
                names.append(person_name)
    
    return encodings, names

dataset_path = 'dataset'
encodings, names = get_face_encodings(dataset_path)

# Save encodings to a file
with open('face_encodings.pkl', 'wb') as f:
    pickle.dump((encodings, names), f)

from sklearn import svm
import pickle

# Load face encodings
with open('face_encodings.pkl', 'rb') as f:
    encodings, names = pickle.load(f)


# Train the classifier
clf = svm.SVC(gamma='scale', probability=True)
clf.fit(encodings, names)    
# Save the trained model
with open('face_recognition_model.pkl', 'wb') as f:
    pickle.dump(clf, f)


import cv2
import face_recognition
import pickle
import numpy as np


# Load the trained model
with open('face_recognition_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the test image
test_image_path = 'testimage1.jpg'
test_image = face_recognition.load_image_file(test_image_path)
test_encodings = face_recognition.face_encodings(test_image)  

# Convert the test image to BGR format for OpenCV visualization (face_recognition loads images in RGB)
test_image_bgr = cv2.imread(test_image_path)

face_locations = face_recognition.face_locations(test_image)
test_encodings = face_recognition.face_encodings(test_image, face_locations)

# Threshold for unknown person detection
threshold = 0.6

for (top, right, bottom, left), encoding in zip(face_locations, test_encodings):
    # Get probabilities for each class (person)
    probabilities = model.predict_proba([encoding])[0]
    max_prob = max(probabilities)

    if max_prob < threshold:
        name = 'Unknown Person'
    else:
        name = model.classes_[np.argmax(probabilities)]

    print(f'Predicted name: {name} (confidence: {max_prob})')

    # Draw a rectangle around the face
    cv2.rectangle(test_image_bgr, (left, top), (right, bottom), (0, 255, 0), 2)

    # Draw a label with the name below the face
    cv2.rectangle(test_image_bgr, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
    cv2.putText(test_image_bgr, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    
