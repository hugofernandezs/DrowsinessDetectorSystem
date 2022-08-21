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
    leftEyeCascade = cv.CascadeClassifier(f"{cv.data.haarcascades}haarcascade_lefteye_2splits.xml")
    rightEyeCascade = cv.CascadeClassifier(f"{cv.data.haarcascades}haarcascade_righteye_2splits.xml")
    # Trains or loads the model.
    model: tf.keras.models.Sequential = mg.create_model("datasets/eyes/train")
    score: int = 0
    thicc: int = 0
    leftEye: None = None
    rightEye: None = None
    # Keeps reading while the cap is opened.
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret is True:
            # Detect the face and both eyes.
            faceShape: np.ndarray = faceCascade.detectMultiScale(frame, minNeighbors=5, scaleFactor=1.1, minSize=(25,25))
            leftEyeShape: np.ndarray = leftEyeCascade.detectMultiScale(frame, minNeighbors=4, scaleFactor=1.1)
            rightEyeShape: np.ndarray =  rightEyeCascade.detectMultiScale(frame, minNeighbors=4, scaleFactor=1.1)

            # Print rectangles on face and eyes for visualizing purposes.
            for (x, y, w, h) in faceShape:
                cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)

            for (x, y, w, h) in leftEyeShape:
                leftEye: np.ndarray = frame[y:y+h,x:x+w]
                leftEye: np.ndarray = cv.resize(leftEye, (mg.IMAGE_SIZE, mg.IMAGE_SIZE))
                leftEye: np.ndarray = leftEye.reshape(mg.IMAGE_SIZE, mg.IMAGE_SIZE, -1)
                leftEye: np.ndarray = np.expand_dims(leftEye,axis=0)
                leftEye: np.ndarray = model.predict(leftEye)[0]
                leftEye: int = np.where(leftEye==np.amax(leftEye))[0]
                cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)

            for (x, y, w, h) in rightEyeShape:
                rightEye: np.ndarray = frame[y:y+h,x:x+w]
                rightEye: np.ndarray = cv.resize(rightEye, (mg.IMAGE_SIZE, mg.IMAGE_SIZE))
                rightEye: np.ndarray = rightEye.reshape(mg.IMAGE_SIZE, mg.IMAGE_SIZE, -1)
                rightEye: np.ndarray = np.expand_dims(rightEye,axis=0)
                rightEye: np.ndarray = model.predict(rightEye)[0]
                rightEye: int = np.where(rightEye==np.amax(rightEye))[0]
                cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)

            if leftEye == 1 or rightEye == 1:
                if score > 0:
                    score -= 1
                if thicc > 0:
                    thicc -= 2
            else:
                score += 1
                if score > 10 and thicc < 10:
                    thicc += 2
            cv.rectangle(frame, (0,0), (frame.shape[1]-1, frame.shape[0]-1), (0,0,255), thicc)
            cv.putText(frame, 'Score:'+str(score), (100, frame.shape[0]-20), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                      (255, 255, 255), 1, cv.LINE_AA)
            cv.imshow("Frame", frame)

        if cv.waitKey(1) == ord('q'):
            cap.release()
            cv.destroyAllWindows()


# Checks we are running the app in the main function.
if __name__ == "__main__":
    main()
