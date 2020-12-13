#!/usr/bin/env python
# coding: utf-8

# In[15]:


import cv2
import os
import h5py, h5
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import shutil
import pickle



# In[16]:


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


def detect_face (imgPath):
    imgUMat = cv2.imread(imgPath)
    gray = cv2.cvtColor(imgUMat, cv2.COLOR_BGR2GRAY)
    face_cas = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    faces = face_cas.detectMultiScale (gray, scaleFactor=1.3, minNeighbors=4)
    if (len(faces) == 0):
        return 0
    return 1


# In[ ]:


type = input("Enter 0 for web cam, or enter name of a video file:")
if type=='0':
   type = int(type)
capture(type)


# In[ ]:



# In[ ]:




