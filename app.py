import os
from flask import Flask, request, render_template
from datetime import date, datetime
import numpy as np
import cv2
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import joblib

app = Flask(__name__)

# Global variables
n_imgs = 10
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
attendance_file = f'Attendance/Attendance-{date.today().strftime("%m_%d_%y")}.csv'


def ensure_directories():
    directories = ['Attendance', 'static', 'static/faces']
    for directory in directories:
        if not os.path.isdir(directory):
            os.makedirs(directory)


def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except Exception as e:
        print(f"Error in face extraction: {e}")
        return []


def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


def train_model():
    faces = []
    labels = []
    user_list = os.listdir('static/faces')
    for user in user_list:
        for img_name in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{img_name}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')


def extract_attendance():
    if not os.path.isfile(attendance_file):
        with open(attendance_file, 'w') as f:
            f.write('Name,Roll,Time')
    df = pd.read_csv(attendance_file)
    return df['Name'], df['Roll'], df['Time'], len(df)


def add_attendance(name):
    username, userid = name.split('_')[0], name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    df = pd.read_csv(attendance_file)
    if int(userid) not in df['Roll'].values:
        with open(attendance_file, 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')


def get_all_users():
    user_list = os.listdir('static/faces')
    names = []
    rolls = []
    for user in user_list:
        name, roll = user.split('_')
        names.append(name)
        rolls.append(roll)
    return user_list, names, rolls, len(user_list)


@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=len(os.listdir('static/faces')),
                           datetoday2=date.today().strftime("%d-%B-%Y"))


@app.route('/start', methods=['GET'])
def start():
    names, rolls, times, l = extract_attendance()
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=len(os.listdir('static/faces')),
                               datetoday2=date.today().strftime("%d-%B-%Y"), mess='There is no trained model in the static folder. Please add a new face to continue.')
    ret = True
    cap = cv2.VideoCapture(0)
    while ret:
        ret, frame = cap.read()
        if len(extract_faces(frame)) > 0:
            (x, y, w, h) = extract_faces(frame)[0]
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            add_attendance(identified_person)
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=len(os.listdir('static/faces')),
                           datetoday2=date.today().strftime("%d-%B-%Y"))


@app.route('/add', methods=['GET', 'POST'])
def add():
    new_username = request.form['newusername']
    new_user_id = request.form['newuserid']
    user_image_folder = f'static/faces/{new_username}_{new_user_id}'
    if not os.path.isdir(user_image_folder):
        os.makedirs(user_image_folder)
    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    while 1:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            if j % 5 == 0:
                name = f'{new_username}_{i}.jpg'
                cv2.imwrite(f'{user_image_folder}/{name}', frame[y:y+h, x:x+w])
                i += 1
            j += 1
        if i == n_imgs:
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    train_model()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=len(os.listdir('static/faces')),
                           datetoday2=date.today().strftime("%d-%B-%Y"))


if __name__ == '__main__':
    ensure_directories()
    app.run(debug=True)
