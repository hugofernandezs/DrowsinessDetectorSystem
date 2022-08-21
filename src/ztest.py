import os
import cv2 as cv
import numpy as np
import tensorflow as tf
import ModelGenerator as mg


model: tf.keras.models.Sequential = mg.create_model("datasets/eyes/train")

print(mg.test_model(model=model, dir="datasets/eyes/test/closed", result=1))
print("\n-------------------------------\n")
print("\n-------------------------------\n")
print("\n-------------------------------\n")
print(mg.test_model(model=model, dir="datasets/eyes/test/open", result=0))
