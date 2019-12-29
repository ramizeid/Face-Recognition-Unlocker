import cv2
import numpy as np
import pickle
import shutil
import os

face_cascade = cv2.CascadeClassifier('Cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('Cascades/data/haarcascade_eye.xml')
copied_face_number = 0

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {}
with open("labels.pickle", "rb") as f:
    original_labels = pickle.load(f)
    labels = {v:k for k, v in original_labels.items()}

capture = cv2.VideoCapture(0)  # Default camera

while True:
    # Capture frame-by-frame
    ret, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for (x, y, w, h) in faces:
        # print(x, y, w, h)
        roi_gray = gray[y:(y + h), x:(x + w)]
        roi_color = frame[y:(y + h), x:(x + w)]

        # Recognition
        id_, conf = recognizer.predict(roi_gray)
        if 50 <= conf <= 85:
            print(conf)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 1
            cv2.putText(frame, name, (x, y - 10), font, 1, color, stroke, cv2.LINE_AA)
        else:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = 'UNKNOWN'
            color = (255, 255, 255)
            stroke = 1
            cv2.putText(frame, name, (x, y - 10), font, 1, color, stroke, cv2.LINE_AA)

        img_item = f"copied-face{copied_face_number}.png"
        cv2.imwrite(img_item, roi_color)

        shutil.copy(f"copied-face{copied_face_number}.png", f"FACES LOCATION\\Faces\\{labels[id_]}")
        os.remove(f"copied-face{copied_face_number}.png")
        copied_face_number += 1

        if copied_face_number % 20 == 0:
            os.system('faces-training.py')

        color = (255, 0, 0)
        stroke = 2
        x_coord_end = x + w
        y_coord_end = y + h
        cv2.rectangle(frame, (x, y), (x_coord_end, y_coord_end), color, stroke)

        # Eyes Detection
        # eyes = eye_cascade.detectMultiScale(roi_gray)
        # for (ex, ey, ew, eh) in eyes:
        #     cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('frame', frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# Release the capture
capture.release()
cv2.destroyAllWindows()
