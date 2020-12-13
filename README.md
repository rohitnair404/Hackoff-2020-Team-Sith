# Hackoff-2020-Team-Sith
Team Sith

The main aim of this project is to help various organisations in protecting their employees from getting infected by the COVID-19 virus and keeping their work environment safe. 

Here, we use the "haarcascade_frontalface_alt2.xml" model to detect faces, which is the implementation of the famous Voila-Jones algorithm for detecting faces. It is one of the best algorithms knows for this purpose. A video is input using the python library "OpenCV" and each frame is read. If a face is detected in the frames, the model "mask_recognizer.h5" is applied on the detected faces to predict if they are wearing mask or not. The detected faces are labelled accordingly with the accuracy that was given by the mask_recognizer using bounding boxes. The faces without masks are extracted and saved in the folder "framed_without_mask" that is created using the "os" library in python. 

This way, any video can be uploaded (such as the cctv footage inside the organisation premises) and the faces of those employees that were not wearing mask can be extracted. These faces are then displayed on the next page.

The website has a login page, after which the main home page opens. Here, a video can be uploaded, which will then redirect to the final page where the extracted faces are displayed. The home page also has a FAQ page for the employees.

