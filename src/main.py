import os
import cv2 as cv
import numpy as np
import streamlit as st
import tensorflow as tf
import ModelGenerator as mg


# Function to stop the system.
def exit_sys():
    cap.release() 
    st.stop()

# Opens thw webcam for capturing images.
cap: cv.VideoCapture = cv.VideoCapture(0)

# Loads the OpenCV cascade files for face and eye detections.
faceCascade = cv.CascadeClassifier(f"{cv.data.haarcascades}haarcascade_frontalface_alt2.xml")
leftEyeCascade = cv.CascadeClassifier(f"{cv.data.haarcascades}haarcascade_lefteye_2splits.xml")
rightEyeCascade = cv.CascadeClassifier(f"{cv.data.haarcascades}haarcascade_righteye_2splits.xml")

# Prepare de streamlit widgets.
st.title("Drowsiness Detection System")
col1, col2 = st.columns([2, 1])
imagePlaceHolder = col1.empty()
exitButton: st.button = st.button(label="Exit", on_click=exit_sys)
st.sidebar.title("User Inputs")

# Initializes the enviroment and controlls the flow.
def main() -> None:
    # Trains or loads the model.
    model: tf.keras.models.Sequential = mg.create_model("datasets/eyes/train")
    label: str = "None"
    score: int = 0
    thicc: int = 0
    leftEye: None = None
    rightEye: None = None
    # Keeps reading while the cap is opened.
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret is True:
            # Detect the face and both eyes.
            frame: np.ndarray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faceShape: np.ndarray = faceCascade.detectMultiScale(frame, minNeighbors=5, scaleFactor=1.1, minSize=(25,25))
            leftEyeShape: np.ndarray = leftEyeCascade.detectMultiScale(frame, minNeighbors=4, scaleFactor=1.1)
            rightEyeShape: np.ndarray =  rightEyeCascade.detectMultiScale(frame, minNeighbors=4, scaleFactor=1.1)
            # Print rectangles on face and eyes for visualizing purposes.
            for (x, y, w, h) in faceShape:
                cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)

            for (x, y, w, h) in leftEyeShape:
                leftEye: np.ndarray = frame[y:y+h,x:x+w]
                leftEye: np.ndarray = cv.resize(leftEye, (mg.IMAGE_SIZE, mg.IMAGE_SIZE))
                leftEye: np.ndarray = leftEye / 255
                leftEye: np.ndarray = leftEye.reshape(mg.IMAGE_SIZE, mg.IMAGE_SIZE, -1)
                leftEye: np.ndarray = np.expand_dims(leftEye,axis=0)
                leftEye: np.ndarray = model.predict(leftEye, verbose=0)[0]
                cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)

            for (x, y, w, h) in rightEyeShape:
                rightEye: np.ndarray = frame[y:y+h,x:x+w]
                rightEye: np.ndarray = cv.resize(rightEye, (mg.IMAGE_SIZE, mg.IMAGE_SIZE))
                rightEye: np.ndarray = rightEye / 255
                rightEye: np.ndarray = rightEye.reshape(mg.IMAGE_SIZE, mg.IMAGE_SIZE, -1)
                rightEye: np.ndarray = np.expand_dims(rightEye,axis=0)
                rightEye: np.ndarray = model.predict(rightEye, verbose=0)[0]
                cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
            if leftEye[0] < 0.5 or rightEye[0] < 0.5:
                label: str = "Open"
                if score > 0:
                    score -= 1
                if thicc > 0:
                    thicc -= 2
            else:
                label: str = "Close"
                score += 1
                if score > 10 and thicc < 10:
                    thicc += 2
            cv.putText(frame, label, (10, frame.shape[0] - 20), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255),
                      1, cv.LINE_AA)
            cv.rectangle(frame, (0,0), (frame.shape[1]-1, frame.shape[0]-1), (0,0,255), thicc)
            cv.putText(frame, 'Score:'+str(score), (100, frame.shape[0]-20), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                      (255, 255, 255), 1, cv.LINE_AA)
            imagePlaceHolder.image(frame, caption='Video')

        if cv.waitKey(1) == ord('q'):
            cap.release()
            cv.destroyAllWindows()




# Checks we are running the app in the main function.
if __name__ == "__main__":
    main()
