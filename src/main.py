import os
import streamlit as st
import tensorflow as tf
import ModelGenerator as mg


# Initializes the enviroment and controlls the flow
def main() -> None:
    face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
    leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
    reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')
    model: tf.keras.models.Sequential = mg.create_model("datasets/eyes/train")


# Checks we are running the app in the main function.
if __name__ == "__main__":
    main()
