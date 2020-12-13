from flask import Flask, render_template, request, redirect, url_for, session ,flash
from flask_mysqldb import MySQL
from functools import wraps
app = Flask(__name__)


app.secret_key = 'your secret key'
app.config['MYSQL_HOST'] ='localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'employ'

mysql=MySQL(app)





import re
import pickle
import numpy as np
import  sys
from PIL import  Image
import os
import cv2
import h5py, h5
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import shutil
import MySQLdb.cursors

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
model = tf.keras.models.load_model('mask_recognizer.h5')

dir = 'frames'
if os.path.exists(dir):
    shutil.rmtree(dir)
os.mkdir(dir)
frame_dir = "frames"

dir = 'frames_without_mask'
if os.path.exists(dir):
    shutil.rmtree(dir)
os.mkdir(dir)
withoutmask = "frames_without_mask"


def capture(type):
    video_capture = cv2.VideoCapture(type)
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
    count = 0
    if video_capture.isOpened()==False:
        print('Error')
    while video_capture.isOpened():

        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if ret :
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray,
                                                 scaleFactor=1.1,
                                                 minNeighbors=5,
                                                 minSize=(10, 10),
                                                 maxSize = (1000, 1000),
                                                 flags=cv2.CASCADE_SCALE_IMAGE)
            faces_list=[]
            preds=[]
            for (x, y, w, h) in faces:
                face_frame = frame[y:y+h,x:x+w]
                face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
                face_frame = cv2.resize(face_frame, (224, 224))
                face_frame = img_to_array(face_frame)
                face_frame = np.expand_dims(face_frame, axis=0)
                face_frame =  preprocess_input(face_frame)
                faces_list.append(face_frame)
                if len(faces_list)>0:
                    preds = model.predict(faces_list)
                for pred in preds:
                    (mask, withoutMask) = pred
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
                cv2.putText(frame, label, (x, y- 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

                cv2.rectangle(frame, (x, y), (x + w, y + h),color, 2)
                #folder = withmask if mask > withoutMask else withoutmask
                cv2.imwrite(frame_dir + '/frame%d.jpg' % count, frame)
                if count%30 == 0 :
                    new_img=frame[y:y+h,x:x+w]
                    cv2.imwrite(withoutmask + '/frame%d.jpg'%count,new_img)
                count = count + 1
                # Display the resulting frame
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    video_capture.release()

    cv2.destroyAllWindows()


def login_required(test):
    @wraps(test)
    def wrap(*args, **kwrags):
        if 'loggedin' in session:
            return test(*args, **kwrags)
        else:
            flash('You need to sign in first')
            return redirect(url_for('log'))
    return wrap



@app.route('/')
def signup():
    return render_template("signup.html")

@app.route('/login')
def login():
    return render_template("login.html")

@app.route('/faq')
@login_required
def faq():
    return render_template("faq.html")

@app.route('/home')
@login_required
def home():
    return render_template("home.html")




@app.route('/sign', methods=['POST','GET'])
def sign():
    msg = ''
    if request.method == 'POST' and 'name' in request.form and 'password' in request.form and 'email' in request.form and 'gender' in request.form:
        username = request.form['name']
        email = request.form['email']
        gender =request.form['gender']
        password = request.form['password']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM users WHERE email = % s', [email])
        account = cursor.fetchone()
        if account:
            msg = 'Account already exists !'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address !'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers !'
        elif not username or not password or not email:
            msg = 'Please fill out the form !'
        else:
            cursor.execute('INSERT INTO users VALUES (% s, % s, % s, % s)', (username, email,gender, password))
            mysql.connection.commit()

    if len(msg)!= 0:
        return render_template('signup.html', error=msg)

    return  render_template('login.html')


@app.route('/action',methods=['POST','GET'])
def log():
    error=None
    if request.method=='POST':
        if request.method == 'POST' and 'email' in request.form and 'password' in request.form:
            username = request.form['email']
            password = request.form['password']
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cursor.execute('SELECT * FROM users WHERE email = % s AND password = % s', (username, password,))
            account = cursor.fetchone()
            if account:
                session['loggedin'] = True
                session['email'] = account['email']
                session['username'] = account['name']
                return render_template('home.html')
            else:
                error="Wrong Credentials"

    return render_template("login.html", error=error)


@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('username', None)
    return redirect(url_for('login'))



@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        f=request.files['file']

        basepath=os.path.dirname(__file__)
        file_path=os.path.join(
            basepath,secure_filename(f.filename))
        f.save(file_path)
        X=capture(file_path)
    return redirect(url_for('home'))

@app.route('/image')
def image():
    return render_template('gallery.html')
if __name__ == '__main__':
    app.run(debug=True)
