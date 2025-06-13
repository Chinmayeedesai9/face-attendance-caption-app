from flask import Flask, render_template, request, send_file
import os
import cv2
import numpy as np
import face_recognition
from datetime import datetime
import random
from object_caption import generate_caption_paragraph

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
DATASET_FOLDER = r"C:\Users\Admin\Desktop\Face-Recognition-Attendance-System-main\Dataset"
CSV_FILE = "Attendance.csv"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATASET_FOLDER, exist_ok=True)

images = []
classNames = []
encodeKnown = []
last_detected_names = []  # 🔁 GLOBAL variable to store recent detections

def loadStudentImages():
    global images, classNames
    images.clear()
    classNames.clear()
    for student_name in os.listdir(DATASET_FOLDER):
        student_folder = os.path.join(DATASET_FOLDER, student_name)
        if not os.path.isdir(student_folder):
            continue
        for img_name in os.listdir(student_folder):
            img_path = os.path.join(student_folder, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
                classNames.append(student_name)

def findEncodings(images):
    encodeList = []
    print(f"Encoding {len(images)} total images...")

    for i, img in enumerate(images):
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(img)
            if encodings:
                encodeList.append(encodings[0])
            else:
                print(f"⚠️ No encoding found for image {i}")
        except Exception as e:
            print(f"Encoding error at image {i}: {e}")
    
    print(f"✅ Encoded {len(encodeList)} faces successfully.")
    return encodeList


def markAttendance(name):
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w') as f:
            f.write('Name,Time\n')
    with open(CSV_FILE, 'a') as f:
        now = datetime.now()
        dtString = now.strftime('%H:%M:%S')
        f.write(f'{name},{dtString}\n')
        print(f"Attendance marked for {name} at {dtString}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global last_detected_names

    if 'image' not in request.files:
        return "No file part", 400
    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    img = cv2.imread(filepath)
    if img is None:
        return "Image loading failed", 500

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesInFrame = face_recognition.face_locations(imgS)
    encodesInFrame = face_recognition.face_encodings(imgS, facesInFrame)

    print("Faces detected:", len(facesInFrame))

    detected_names = []

    for encodeFace in encodesInFrame:
        matches = face_recognition.compare_faces(encodeKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex] and faceDis[matchIndex] < 0.5:
            name = classNames[matchIndex].upper()
            markAttendance(name)
            detected_names.append(name)
        else:
            print("Unknown face detected")

    last_detected_names = detected_names  # 🟡 Save to global variable
    return "Attendance marked successfully!"

@app.route('/add-student', methods=['POST'])
def add_student():
    if 'student_image' not in request.files or 'student_name' not in request.form:
        return "Missing data", 400

    file = request.files['student_image']
    student_name = request.form['student_name'].strip()

    if file.filename == '' or student_name == '':
        return "Invalid input", 400

    video_path = os.path.join(UPLOAD_FOLDER, f"{student_name}_temp.mp4")
    file.save(video_path)

    student_folder = os.path.join(DATASET_FOLDER, student_name)
    os.makedirs(student_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_indices = sorted(random.sample(range(total_frames), min(300, total_frames)))
    saved_count, frame_id = 0, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id in frame_indices:
            filename = f"img_frame_{saved_count:05d}.jpg"
            cv2.imwrite(os.path.join(student_folder, filename), frame)
            saved_count += 1
        frame_id += 1
        if saved_count >= 300:
            break

    cap.release()
    os.remove(video_path)

    loadStudentImages()
    global encodeKnown
    encodeKnown = findEncodings(images)

    return f"{saved_count} images saved for {student_name}"

@app.route('/download')
def download():
    if not os.path.exists(CSV_FILE):
        return "No attendance file found", 404
    return send_file(CSV_FILE, as_attachment=True)

@app.route('/reset')
def reset():
    for filename in os.listdir(UPLOAD_FOLDER):
        os.remove(os.path.join(UPLOAD_FOLDER, filename))

    if os.path.exists(CSV_FILE):
        os.remove(CSV_FILE)
    with open(CSV_FILE, 'w') as f:
        f.write("Name,Time\n")

    return "Reset complete!", 200

@app.route('/describe-image', methods=['POST'])
def describe_image():
    if 'image' not in request.files:
        return render_template("index.html", paragraph="No file uploaded")

    file = request.files['image']
    if file.filename == '':
        return render_template("index.html", paragraph="Empty file name")

    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)

    # 🟡 Run face recognition on this image
    img = cv2.imread(image_path)
    if img is None:
        return render_template("index.html", paragraph="Image loading failed")

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesInFrame = face_recognition.face_locations(imgS)
    encodesInFrame = face_recognition.face_encodings(imgS, facesInFrame)

    detected_names = []

    for encodeFace in encodesInFrame:
        matches = face_recognition.compare_faces(encodeKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex] and faceDis[matchIndex] < 0.5:
            name = classNames[matchIndex].upper()
            detected_names.append(name)
        else:
            print("Unknown face detected in describe-image")

    try:
        paragraph = generate_caption_paragraph(image_path, detected_names)
        return render_template("index.html", paragraph=paragraph)
    except Exception as e:
        return render_template("index.html", paragraph=f"Error: {str(e)}")


if __name__ == '__main__':
    loadStudentImages()
    encodeKnown = findEncodings(images)
    app.run(debug=True)
