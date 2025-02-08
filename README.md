**Title of the Project**: Face Recognition-Based Attendance System — Flask App — With GUI

**Abstract**:
This project aims to develop a robust and efficient face recognition-based attendance 
system implemented as a Flask web application with a graphical user interface (GUI). 
Traditional attendance systems often suffer from inaccuracies, buddy punching, and 
inefficiencies. Leveraging machine learning techniques, particularly facial recognition, 
this system automates the attendance marking process. It involves capturing images of 
individuals, identifying faces using deep learning algorithms, and recording attendance 
data in a database. The Flask framework facilitates the creation of a user-friendly 
interface for administrators and users, making attendance tracking seamless and 
convenient.

**Methodology**:
• Data Collection: Gather a diverse dataset of facial images representing individuals 
who will be using the attendance system.
• Preprocessing: Clean and preprocess the facial images to enhance feature 
extraction and improve model performance.
• Model Training: Employ deep learning architectures such as Convolutional Neural 
Networks (CNNs) to train a robust face recognition model.
• Integration with Flask: Develop a Flask web application to serve as the interface 
for the attendance system, allowing users to interact with the system through a 
GUI.
• Database Integration: Implement a database to store attendance records 
securely and eCiciently.
• Testing and Evaluation: Evaluate the performance of the system using metrics 
such as accuracy, precision, recall, and F1 score. Conduct user testing to ensure 
usability and reliability.
• Deployment: Deploy the Flask app on a suitable platform, ensuring scalability and 
accessibility.

**Technology**:
• Python: Programming language for implementing machine learning algorithms 
and web application development.
• OpenCV: Library for computer vision tasks such as face detection and image 
processing.
• TensorFlow or PyTorch: Deep learning frameworks for building and training neural 
networks.
• Flask: Micro web framework for developing web applications with Python.
• HTML/CSS/JavaScript: Front-end technologies for designing and developing the 
graphical user interface. 
• SQLite or MySQL: Database management systems for storing attendance data.

**Outcome**:
The outcome of this project will be a fully functional face recognition-based attendance 
system implemented as a Flask web application with a GUI. It will offer the following 
benefits:
• Automated attendance tracking: Eliminates the need for manual attendance 
marking, reducing errors and saving time.
• Enhanced security: Utilizes facial recognition technology for accurate 
identification, minimizing the risk of unauthorized access.
• User-friendly interface: The intuitive GUI allows administrators and users to 
interact with the system effortlessly.
• Scalability: The modular design enables easy integration with existing 
infrastructure and supports scalability for future expansion.
• Improved eCiciency: Streamlines the attendance management process, leading 
to increased productivity and cost savings for organizations.

**Flask Application Code (app.py)**:
from flask import Flask, render_template, request, redirect, url_for, flash
import face_recognition
import cv2
import numpy as np
import sqlite3
import os
import pickle
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'secret_key'

# Paths
ENCODING_PATH = "encodings/user_encodings.pkl"
UPLOAD_FOLDER = "uploads"
DB_PATH = "database.db"

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("encodings", exist_ok=True)

# Initialize encodings
if not os.path.exists(ENCODING_PATH):
    with open(ENCODING_PATH, "wb") as f:
        pickle.dump({}, f)

# Database initialization
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            timestamp DATETIME
        )
    """)
    conn.commit()
    conn.close()

init_db()

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        file = request.files['file']

        if name and file:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            # Encode the face
            image = face_recognition.load_image_file(file_path)
            face_encodings = face_recognition.face_encodings(image)

            if len(face_encodings) > 0:
                encoding = face_encodings[0]

                # Save encoding
                with open(ENCODING_PATH, "rb") as f:
                    encodings = pickle.load(f)

                encodings[name] = encoding

                with open(ENCODING_PATH, "wb") as f:
                    pickle.dump(encodings, f)

                flash("Registration successful!", "success")
            else:
                flash("No face detected. Try again with a different image.", "danger")
        else:
            flash("Please provide a name and an image.", "danger")

    return render_template('register.html')

@app.route('/mark', methods=['GET', 'POST'])
def mark():
    if request.method == 'POST':
        # Load face encodings
        with open(ENCODING_PATH, "rb") as f:
            known_encodings = pickle.load(f)

        known_names = list(known_encodings.keys())
        known_faces = list(known_encodings.values())

        # Open webcam
        cap = cv2.VideoCapture(0)
        attendance_marked = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_faces, face_encoding)
                face_distances = face_recognition.face_distance(known_faces, face_encoding)
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    name = known_names[best_match_index]

                    if name not in attendance_marked:
                        # Save attendance
                        conn = sqlite3.connect(DB_PATH)
                        cursor = conn.cursor()
                        cursor.execute("INSERT INTO attendance (name, timestamp) VALUES (?, ?)",
                                       (name, datetime.now()))
                        conn.commit()
                        conn.close()

                        attendance_marked.append(name)
                        flash(f"Attendance marked for {name}.", "success")

            cv2.imshow('Face Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    return render_template('mark.html')

@app.route('/report')
def report():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name, timestamp FROM attendance")
    records = cursor.fetchall()
    conn.close()

    return render_template('report.html', records=records)

if __name__ == '__main__':
    app.run(debug=True)

**index.html: Home Page**:
<!DOCTYPE html>
<html>
<head>
    <title>Face Recognition Attendance</title>
</head>
<body>
    <h1>Welcome to Face Recognition Attendance System</h1>
    <a href="/register">Register Face</a>
    <a href="/mark">Mark Attendance</a>
    <a href="/report">View Attendance Report</a>
</body>
</html>

**register.html: Register Page**:
<!DOCTYPE html>
<html>
<head>
    <title>Register Face</title>
</head>
<body>
    <h1>Register Face</h1>
    <form method="POST" enctype="multipart/form-data">
        <label>Name:</label>
        <input type="text" name="name" required>
        <label>Upload Image:</label>
        <input type="file" name="file" required>
        <button type="submit">Register</button>
    </form>
</body>
</html>

**mark.html: Mark Attendance**:
<!DOCTYPE html>
<html>
<head>
    <title>Mark Attendance</title>
</head>
<body>
    <h1>Mark Attendance</h1>
    <form method="POST">
        <button type="submit">Start Webcam</button>
    </form>
</body>
</html>

**report.html: Attendance Report**:
<!DOCTYPE html>
<html>
<head>
    <title>Attendance Report</title>
</head>
<body>
    <h1>Attendance Report</h1>
    <table border="1">
        <tr>
            <th>Name</th>
            <th>Timestamp</th>
        </tr>
        {% for name, timestamp in records %}
        <tr>
            <td>{{ name }}</td>
            <td>{{ timestamp }}</td>
        </tr>
        {% endfor %}
    </table>
</body>
</html>
