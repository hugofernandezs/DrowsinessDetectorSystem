import os
import cv2 as cv
import numpy as np
import pygame as pg
import streamlit as st
import tensorflow as tf
import ModelGenerator as mg


# Function to stop the system.
def exit_sys():
    cap.release() 
    st.stop()

# Inits the alarm system for using it wen needed.
pg.mixer.init()
sound = pg.mixer.Sound('alarm.wav')

# Open the camera before streamlit for not reopening it evry time the web reloads.
cap: cv.VideoCapture = cv.VideoCapture(0)

# Some variables
frameCount: int = 0
warningScore: int = 10

maxScore: int = 100
scoreIncrease: int = 1
scoreDecrease: int = 1

maxThicc: int = 10
thiccIncrease: int = 2
thiccDecrease: int = 2

# Loads the OpenCV cascade files for face and eye detections.
faceCascade = cv.CascadeClassifier(f"{cv.data.haarcascades}haarcascade_frontalface_alt2.xml")
leftEyeCascade = cv.CascadeClassifier(f"{cv.data.haarcascades}haarcascade_lefteye_2splits.xml")
rightEyeCascade = cv.CascadeClassifier(f"{cv.data.haarcascades}haarcascade_righteye_2splits.xml")
mouthCascade = cv.CascadeClassifier(f"haarcascade_mcs_mouth.xml")

# Prepare de streamlit widgets.
st.title("Drowsiness Detection System")
col1, _, col2 = st.columns([10, 2, 10])
imagePlaceHolder = col1.empty()
exitButton: st.button = st.button(label="Exit", on_click=exit_sys)
st.sidebar.title("User Inputs")

leftEyeTagText = col2.empty()
rightEyeTagText = col2.empty()
eyesTagText = col2.empty()
faceTagText = col2.empty()
labelText = col2.empty()
frameCountText = col1.empty()
spaceText = col2.empty()
warningScoreText = col2.empty()
maxScoreText = col2.empty()
scoreIncreaseText = col2.empty()
scoreDecreaseText = col2.empty()
maxThiccText = col2.empty()
thiccIncreaseText = col2.empty()
thiccDecreaseText = col2.empty()
frameDelayText = col2.empty()

framesSleepText = col1.empty()
framesAwakeText = col1.empty()

warningHolder = st.empty()

warningScoreSlider = st.sidebar.slider("Warning score:", min_value=0,
                                       max_value=100, value=10)
maxScoreSlider = st.sidebar.slider("Max score", min_value=0, max_value=100,
                                   value=100)
scoreIncreaseSlider = st.sidebar.slider("Score increase", min_value=0,
                                        max_value=10, value=1)
scoreDecreaseSlider = st.sidebar.slider("Score decrease", min_value=0,
                                        max_value=10, value=1)
maxThiccSlider = st.sidebar.slider("Max thiccness", min_value=0,
                                   max_value=100, value=10)
thiccIncreaseSlider = st.sidebar.slider("Thiccness increase", min_value=0, max_value=10, value=2)
thiccDecreaseSlider = st.sidebar.slider("Thiccness decrease",
                                        min_value=0, max_value=10, value=2)
frameDelaySlider = st.sidebar.slider("Frame delay", min_value=1,
                                     max_value=500, value=10)
col12, _, col22 = st.columns([10, 2, 10])
eyesFilename = col12.selectbox("Select eyes model", options=os.listdir("models\eyes_models"), index=0) 
mouthFilename = col22.selectbox("Select yawn model", options=os.listdir("models\mouth_models"), index=0)
eyesFile = col12.file_uploader("Upload an eyes model", type=['h5'])
mouthFile = col22.file_uploader("Upload a mouth model", type=['h5'])
if eyesFile is not None:
    with open(os.path.join("models\eyes_models",eyesFile.name),"wb") as f: 
        f.write(eyesFile.getbuffer()) 
if mouthFile is not None:
    with open(os.path.join("models\mouth_models",mouthFile.name),"wb") as f: 
        f.write(mouthFile.getbuffer()) 

# Initializes the enviroment and controlls the flow.
def main() -> None:
    # Trains or loads the model.
    eyesModel: tf.keras.models.Sequential = mg.create_model("datasets/eyes", exists=f"models\eyes_models\{eyesFilename}")
    mouthModel: tf.keras.models.Sequential = mg.create_model("datasets/yawn", exists=f"models\mouth_models\{mouthFilename}")

    # Some more variables.
    score: int = 0
    thicc: int = 0
    frameCount: int = 0
    framesSleep: int = 0
    framesAwake: int = 0
    
    # Keeps reading while the cap is opened.
    while(cap.isOpened()):
        label: str = ""
        eyesTag: str = ""
        leftEyeTag: str = ""
        rightEyeTag: str = ""
        yawnTag: str = ""
        warningHolder.text("")

        ret, frame = cap.read()

        # Get the user input widgets values.
        warningScore: int = warningScoreSlider

        maxScore: int = maxScoreSlider
        scoreIncrease: int = scoreIncreaseSlider
        scoreDecrease: int =scoreDecreaseSlider

        maxThicc: int = maxThiccSlider
        thiccIncrease: int = thiccIncreaseSlider
        thiccDecrease: int = thiccDecreaseSlider

        frameDelay: int = frameDelaySlider

        if ret is True:
            frameCount += 1

            # Reset the values of the arrays.
            leftEye: np.ndarray = np.ndarray([])
            rightEye: np.ndarray = np.ndarray([])
            face: np.ndarray = np.ndarray([])
            mouth: np.ndarray = np.ndarray([])

            # Detect the face and both eyes.
            frame: np.ndarray = cv.flip(frame, 1)
            grayFrame: np.ndarray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            faceShape: np.ndarray = faceCascade.detectMultiScale(grayFrame, minNeighbors=5, scaleFactor= 1.1, minSize=(25,25))[:1]
            leftEyeShape: np.ndarray = leftEyeCascade.detectMultiScale(grayFrame, minNeighbors=4, scaleFactor=1.1)[:1]
            rightEyeShape: np.ndarray =  rightEyeCascade.detectMultiScale(grayFrame, minNeighbors=4, scaleFactor=1.1)[:1]
            mouthShape: np.ndarray =  mouthCascade.detectMultiScale(grayFrame, minNeighbors=4, scaleFactor=1.1)[:1]
            
            # Print rectangles on face and eyes for visualizing purposes.
            # Makes the model predictions.
            for (x, y, w, h) in faceShape:
                face: np.ndarray = grayFrame[y:y+h,x:x+w]
                cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            for (x, y, w, h) in mouthShape:
                mouth: np.ndarray = grayFrame[y:y+h,x:x+w]
                mouth: np.ndarray = cv.resize(mouth, (mg.IMAGE_SIZE, mg.IMAGE_SIZE))
                mouth: np.ndarray = mouth / 255
                mouth: np.ndarray = mouth.reshape(mg.IMAGE_SIZE, mg.IMAGE_SIZE, -1)
                mouth: np.ndarray = np.expand_dims(mouth,axis=0)
                mouth: np.ndarray = mouthModel.predict(mouth, verbose=0)[0]
                cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)

            for (x, y, w, h) in leftEyeShape:
                leftEye: np.ndarray = grayFrame[y:y+h,x:x+w]
                leftEye: np.ndarray = cv.resize(leftEye, (mg.IMAGE_SIZE, mg.IMAGE_SIZE))
                leftEye: np.ndarray = leftEye / 255
                leftEye: np.ndarray = leftEye.reshape(mg.IMAGE_SIZE, mg.IMAGE_SIZE, -1)
                leftEye: np.ndarray = np.expand_dims(leftEye,axis=0)
                leftEye: np.ndarray = eyesModel.predict(leftEye, verbose=0)[0]
                cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
                if leftEye[0] < 0.5:
                    leftEyeTag: str = "Open"
                else:
                    leftEyeTag: str = "Close"

            for (x, y, w, h) in rightEyeShape:
                rightEye: np.ndarray = grayFrame[y:y+h,x:x+w]
                rightEye: np.ndarray = cv.resize(rightEye, (mg.IMAGE_SIZE, mg.IMAGE_SIZE))
                rightEye: np.ndarray = rightEye / 255
                rightEye: np.ndarray = rightEye.reshape(mg.IMAGE_SIZE, mg.IMAGE_SIZE, -1)
                rightEye: np.ndarray = np.expand_dims(rightEye,axis=0)
                rightEye: np.ndarray = eyesModel.predict(rightEye, verbose=0)[0]
                cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
                if rightEye[0] < 0.5:
                    rightEyeTag: str = "Open"
                else:
                    rightEyeTag: str = "Close"
            
            # Check if the was an error on the predictions.
            if rightEye is None or leftEye is None or face is None:
                warningHolder.warning("Error on processing the image!")
                label: str = "ERROR ON IMAGE"

            else:
                try:
                    # Analice the eyes.
                    if leftEyeTag == "Open" or rightEyeTag == "Open":
                        eyesTag: str = "Open"
                    else:
                        eyesTag: str = "Close"
                    
                    # Analice the face.
                    if mouth[1] < 0.5:
                        yawnTag: str = "No"
                    else:
                        yawnTag: str = "Yes"
                    
                    # Analice the results and increases or decreses the score and alarm system.
                    if yawnTag == "No" and eyesTag == "Open":
                        score -= scoreDecrease
                        thicc -= thiccDecrease
                        if score < 0:
                            score = 0
                        if thicc < 0:
                            thicc = 0
                        label: str = "Awake"
                        framesAwake += 1
                    else:
                        score += scoreIncrease
                        if score > warningScore:
                            thicc += thiccIncrease
                        if score > maxScore:
                            score = maxScore
                        if thicc > maxThicc:
                            thicc = maxThicc
                        label: str = "Sleep"
                        framesSleep += 1
                    
                    # Sounds the alarm if needed and controlls the volume.
                    sound.set_volume(thicc/maxThicc)
                    if label == "Sleep" and score > warningScore:
                        try:
                            sound.play()
                        except:
                            pass
                    if score < warningScore:
                        try:
                            sound.stop()
                        except:
                            pass
                
                except Exception as exc:
                    pass

                # Shows the results.
                cv.putText(frame, label, (10, frame.shape[0] - 20), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255),
                    1, cv.LINE_AA)
                cv.rectangle(frame, (0,0), (frame.shape[1]-1, frame.shape[0]-1), (0,0,255), thicc)
                cv.putText(frame, 'Score:'+str(score), (100, frame.shape[0]-20), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                    (255, 255, 255), 1, cv.LINE_AA)
                imagePlaceHolder.image(cv.cvtColor(frame, cv.COLOR_BGR2RGB), caption='Video')
                
                # Writes the results to the information widgets.
                leftEyeTagText.text(f"Left eye pred: {leftEyeTag}")
                rightEyeTagText.text(f"Right eye pred: {rightEyeTag}")
                eyesTagText.text(f"Eyes prediction: {eyesTag}")
                faceTagText.text(f"Yawn prediction: {yawnTag}")
                labelText.text(f"Drowsiness prediction: {label}")
                spaceText.text(f"")
                warningScoreText.text(f"Warning score: {warningScore}")
                maxScoreText.text(f"Max score: {maxScore}")
                scoreIncreaseText.text(f"Score increase: {scoreIncrease}")
                scoreDecreaseText.text(f"Score decrease: {scoreDecrease}")
                maxThiccText.text(f"Max thiccness: {maxThicc}")
                thiccIncreaseText.text(f"Thiccness increase: {thiccIncrease}")
                thiccDecreaseText.text(f"Thiccness decrease: {thiccDecrease}")
                framesSleepText.text(f"Frames sleep: {framesSleep}")
                framesAwakeText.text(f"Frames awake: {framesAwake}")
                frameDelayText.text(f"Frame delay: {frameDelay}")
        
        cv.waitKey(frameDelay)




# Checks we are running the app in the main function.
if __name__ == "__main__":
    main()
