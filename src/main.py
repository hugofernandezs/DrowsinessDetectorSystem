import os
import cv2 as cv
import numpy as np
import tensorflow as tf
import ModelGenerator as mg

cap: cv.VideoCapture = cv.VideoCapture(0)

# Initializes the enviroment and controlls the flow.
def main() -> None:
    # Loads the OpenCV cascade files for face and eye detections.
    faceCascade = cv.CascadeClassifier(f"{cv.data.haarcascades}haarcascade_frontalface_alt2.xml")
    leyeCascade = cv.CascadeClassifier(f"{cv.data.haarcascades}haarcascade_lefteye_2splits.xml")
    reyeCascade = cv.CascadeClassifier(f"{cv.data.haarcascades}haarcascade_righteye_2splits.xml")
    # Trains or loads the model.
    model: tf.keras.models.Sequential = mg.create_model("datasets/eyes/train")
    # Keeps reading while the cap is opened.
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret is True:
            face = faceCascade.detectMultiScale(frame, minNeighbors=5, scaleFactor=1.1, minSize=(25,25))
            leftEye = leyeCascade.detectMultiScale(frame, minNeighbors=4, scaleFactor=1.1)
            rightEye =  reyeCascade.detectMultiScale(frame, minNeighbors=4, scaleFactor=1.1)
            for (x, y, w, h) in face:
                cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            for (x, y, w, h) in leftEye:
                cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            for (x, y, w, h) in rightEye:
                cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv.imshow("Frame", frame)
        if cv.waitKey(1) == ord('q'):
            cap.release()
            cv.destroyAllWindows()


# Checks we are running the app in the main function.
if __name__ == "__main__":
    main()
