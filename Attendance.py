import cv2
import numpy as np
import face_recognition
import os
from tkinter import Tk, filedialog
from datetime import datetime, timedelta

# Path to the folder with student images
path = r'C:\Users\Lenovo\Downloads\Face-Recognition-Attendance-System-main\Students'
images = []
classNames = []

# Load the student images and their class names
myList = os.listdir(path)
print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    if curImg is None:
        print(f"Image {cl} not loaded correctly.")
        continue
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

print(classNames)

# Function to find encodings for the student images
def findEncodings(images):
    encodeList = []
    for img in images:
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
    return encodeList

# Function to mark attendance
def markAttendance(name):
    file_path = 'Attendance.csv'

    # Check if the file exists, if not, create it
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            f.write('Name,Time\n')  # Add header row for CSV

    # Append the new attendance record
    with open(file_path, 'a') as f:
        now = datetime.now()
        dtString = now.strftime('%H:%M:%S')
        f.write(f'{name},{dtString}\n')

# Get face encodings for known students
encodeKnown = findEncodings(images)
print('Encoding Complete')

# Function to upload an image and detect faces
def upload_image_for_detection():
    # Use tkinter to select an image file
    Tk().withdraw()  # Hide the main tkinter window
    file_path = filedialog.askopenfilename(title="Select an image file", filetypes=[("Image Files", "*.jpg *.jpeg *.png")])

    if not file_path:
        print("No file selected.")
        return

    # Load and process the uploaded image
    img = cv2.imread(file_path)
    if img is None:
        print(f"Image {file_path} not loaded correctly.")
        return

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Detect faces in the uploaded image
    facesInFrame = face_recognition.face_locations(imgS)
    encodesInFrame = face_recognition.face_encodings(imgS, facesInFrame)

    recognition_times = {}
    for encodeFace, faceLoc in zip(encodesInFrame, facesInFrame):
        matches = face_recognition.compare_faces(encodeKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex] and faceDis[matchIndex] < 0.5:
            name = classNames[matchIndex].upper()
        else:
            name = 'UNKNOWN'

        # Draw a rectangle and label the face
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        # Mark attendance if the person is recognized
        if name != 'UNKNOWN':
            now = datetime.now()
            if name in recognition_times:
                if now - recognition_times[name] >= timedelta(seconds=3):
                    markAttendance(name)
                    recognition_times[name] = now
            else:
                recognition_times[name] = now
                markAttendance(name)

    # Show the image with detected faces
    cv2.imshow('Uploaded Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Call the upload function to select and detect faces in an image
upload_image_for_detection()

