import cv2 as cv
import numpy as np
import streamlit as st
import tensorflow as tf
import ModelGenerator as mg


# Function to stop the system.
def exit_sys():
    cap.release() 
    st.stop()

# Algunas constantes del programa
frameCount: int = 0
# warningScore: int = 10

# maxScore: int = 100
# scoreIncrease: int = 1

# maxThicc: int = 10
# thiccIncrease: int = 2

# Opens thw webcam for capturing images.
cap: cv.VideoCapture = cv.VideoCapture(0)

# Loads the OpenCV cascade files for face and eye detections.
faceCascade = cv.CascadeClassifier(f"{cv.data.haarcascades}haarcascade_frontalface_alt2.xml")
leftEyeCascade = cv.CascadeClassifier(f"{cv.data.haarcascades}haarcascade_lefteye_2splits.xml")
rightEyeCascade = cv.CascadeClassifier(f"{cv.data.haarcascades}haarcascade_righteye_2splits.xml")

# Prepare de streamlit widgets.
st.title("Drowsiness Detection System")
col1, col2 = st.columns([2, 2])
imagePlaceHolder = col1.empty()
exitButton: st.button = st.button(label="Exit", on_click=exit_sys)
st.sidebar.title("User Inputs")

eyesTagText = col2.empty()
faceTagText = col2.empty()
labelText = col2.empty()
frameCountText = col2.empty()

warningHolder = st.empty()

# Algunas constantes del programa
warningScore = st.sidebar.slider("Warning score:", min_value=0, max_value=100, value=10)

maxScore = st.sidebar.slider("Max score", min_value=0, max_value=100, value=100)
scoreIncrease = st.sidebar.slider("Score increase", min_value=0, max_value=10, value=1)
scoreDecrease = st.sidebar.slider("Score decrease", min_value=0, max_value=10, value=1)

maxThicc = st.sidebar.slider("Max thiccness", min_value=0, max_value=100, value=10)
thiccIncrease = st.sidebar.slider("Thiccness increase", min_value=0, max_value=10, value=2)
thiccDecrease = st.sidebar.slider("Thiccness decrease", min_value=0, max_value=10, value=2)

# Initializes the enviroment and controlls the flow.
def main() -> None:
    # Trains or loads the model.
    eyesModel: tf.keras.models.Sequential = mg.create_model("datasets/eyes")
    faceModel: tf.keras.models.Sequential = mg.create_model("datasets/yawn")
    score: int = 0
    thicc: int = 0
    frameCount: int = 0
    # Keeps reading while the cap is opened.

    print("\nStarting detection...\n")
    while(cap.isOpened()):
        label: str = ""
        eyesTag: str = ""
        yawnTag: str = ""

        ret, frame = cap.read()

        if ret is True:
            frameCount += 1
            leftEye: np.ndarray = np.ndarray([])
            rightEye: np.ndarray = np.ndarray([])
            face: np.ndarray = np.ndarray([])

            # Detect the face and both eyes.
            frame: np.ndarray = cv.flip(frame, 1)
            grayFrame: np.ndarray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faceShape: np.ndarray = faceCascade.detectMultiScale(grayFrame, minNeighbors=5, scaleFactor=1.1, minSize=(25,25))
            leftEyeShape: np.ndarray = leftEyeCascade.detectMultiScale(grayFrame, minNeighbors=4, scaleFactor=1.1)
            rightEyeShape: np.ndarray =  rightEyeCascade.detectMultiScale(grayFrame, minNeighbors=4, scaleFactor=1.1)
            
            # Print rectangles on face and eyes for visualizing purposes.
            for (x, y, w, h) in faceShape:
                face: np.ndarray = grayFrame[y:y+h,x:x+w]
                face: np.ndarray = cv.resize(face, (mg.IMAGE_SIZE, mg.IMAGE_SIZE))
                face: np.ndarray = face / 255
                face: np.ndarray = face.reshape(mg.IMAGE_SIZE, mg.IMAGE_SIZE, -1)
                face: np.ndarray = np.expand_dims(face,axis=0)
                face: np.ndarray = faceModel.predict(face, verbose=0)[0]
                cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            for (x, y, w, h) in leftEyeShape:
                leftEye: np.ndarray = grayFrame[y:y+h,x:x+w]
                leftEye: np.ndarray = cv.resize(leftEye, (mg.IMAGE_SIZE, mg.IMAGE_SIZE))
                leftEye: np.ndarray = leftEye / 255
                leftEye: np.ndarray = leftEye.reshape(mg.IMAGE_SIZE, mg.IMAGE_SIZE, -1)
                leftEye: np.ndarray = np.expand_dims(leftEye,axis=0)
                leftEye: np.ndarray = eyesModel.predict(leftEye, verbose=0)[0]
                # cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            for (x, y, w, h) in rightEyeShape:
                rightEye: np.ndarray = grayFrame[y:y+h,x:x+w]
                rightEye: np.ndarray = cv.resize(rightEye, (mg.IMAGE_SIZE, mg.IMAGE_SIZE))
                rightEye: np.ndarray = rightEye / 255
                rightEye: np.ndarray = rightEye.reshape(mg.IMAGE_SIZE, mg.IMAGE_SIZE, -1)
                rightEye: np.ndarray = np.expand_dims(rightEye,axis=0)
                rightEye: np.ndarray = eyesModel.predict(rightEye, verbose=0)[0]
                # cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            if rightEye is None or leftEye is None or face is None:
                warningHolder.warning("Error on processing the image!")
                label: str = "ERROR ON IMAGE!!"

            else:
                try:
                    # Analizamos los ojos.
                    if leftEye[0] < 0.5 or rightEye[0] < 0.5:
                        eyesTag: str = "Open"
                    else:
                        eyesTag: str = "Close"
                    
                    # Analizamos el rostro.
                    if face[0] < 0.5:
                        yawnTag: str = "No"
                    else:
                        yawnTag: str = "Yes"
                    
                    # Analizamos el conjunto.
                    if yawnTag == "No" and eyesTag == "Open":
                        if score > 0:
                            score -= scoreDecrease
                        if thicc > 0:
                            thicc -= thiccDecrease
                        label: str = "Awake"
                    else:
                        if score < maxScore:
                            score += scoreIncrease
                        if score > warningScore and thicc < maxThicc:
                            thicc += thiccIncrease
                        label: str = "Sleep"

                    # Mostramos los resultados.
                    cv.putText(frame, label, (10, frame.shape[0] - 20), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255),
                            1, cv.LINE_AA)
                    cv.rectangle(frame, (0,0), (frame.shape[1]-1, frame.shape[0]-1), (0,0,255), thicc)
                    cv.putText(frame, 'Score:'+str(score), (100, frame.shape[0]-20), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                            (255, 255, 255), 1, cv.LINE_AA)
                    imagePlaceHolder.image(cv.cvtColor(frame, cv.COLOR_BGR2RGB), caption='Video')
                
                except:
                    warningHolder.warning("Please, try better ilumination!")

                eyesTagText.text(f"Eyes prediction: {eyesTag}")
                faceTagText.text(f"Yawn prediction: {yawnTag}")
                labelText.text(f"Drowsiness prediction: {label}")
                frameCountText.text(f"Frames predicted: {frameCount}")
        
        if cv.waitKey(500) == ord('q'):
            cap.release()
            cv.destroyAllWindows()




# Checks we are running the app in the main function.
if __name__ == "__main__":
    main()
