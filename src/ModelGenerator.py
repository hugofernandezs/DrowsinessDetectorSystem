import os
import cv2 as cv
import contextlib
import numpy as np
import tensorflow as tf

# Some constants.
COLOR_MODE: str = "rgb"
IMAGE_SIZE: int = 256
BATCH_SIZE: int = 32

# Remove Tensorflow annoying messages.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Creates the training and validation datasets from the passed dir.
def create_datasets(dir: str) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    trainingDataset: tf.data.Dataset = tf.keras.preprocessing.image_dataset_from_directory(
        dir,
        color_mode=COLOR_MODE,
        batch_size=BATCH_SIZE,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        seed=0,
        validation_split=0.2,
        subset="training"
    )
    validationDataset: tf.data.Dataset = tf.keras.preprocessing.image_dataset_from_directory(
        dir,
        color_mode=COLOR_MODE,
        batch_size=BATCH_SIZE,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        seed=0,
        validation_split=0.2,
        subset="validation"
    )
    trainingDataset: tf.data.Dataset = trainingDataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    validationDataset: tf.data.Dataset = validationDataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    return trainingDataset, validationDataset


# Creates and fits a convolutional neural network.
def create_model(dir: str, accuracy: int = 0.95, activation: str = "relu", optimizer: str = "adam",
                 epochs: int = 10) -> tf.keras.models.Sequential:
    modelName: str = str(f"{dir.split('/')[-2]}_{activation}_{optimizer}_CNN.h5")
    # If the model exists, it loads it and returns it.
    if os.path.exists(f"models/{modelName}"):
        print(f"{modelName} -> Loaded")
        return tf.keras.models.load_model(f"models/{modelName}")
    # If the model doesn't exists, it generates it.
    # Keras callback for setting an accuracy to the model fit proces.
    class Callback(tf.keras.callbacks.Callback):
        def __init__(self):
            super(Callback, self).__init__()

        def on_epoch_end(self, epoch, logs=None):
            if logs["val_accuracy"] >= accuracy and logs["accuracy"] >= accuracy:
                self.model.stop_training = True
    # Creates the model.
    model: tf.keras.models.Sequential = tf.keras.models.Sequential([
        tf.keras.layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
        tf.keras.layers.Rescaling(1./255, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        tf.keras.layers.Conv2D(16, 3, padding='same', input_shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3), activation=activation),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding='same', input_shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3), activation=activation),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', input_shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3), activation=activation),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=activation),
        tf.keras.layers.Dense(2)
    ])
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        trainingDataset, validationDataset = create_datasets(dir)
    model.fit(
        trainingDataset,
        validation_data=validationDataset,
        epochs=epochs,
        callbacks=[Callback()],
        shuffle = True
    )
    print(f"{modelName} -> Created")
    model.save(f"models/{modelName}")
    return model


def test_model(model: tf.keras.models.Sequential, dir: str, result: int) -> float:
    count: int = 0
    errors: int = 1
    for file in os.listdir(dir):
        count += 1
        img = cv.imread(f"{dir}/{file}", cv.IMREAD_GRAYSCALE)
        img: np.ndarray = cv.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img: np.ndarray = img.reshape(IMAGE_SIZE, IMAGE_SIZE, -1)
        img: np.ndarray = np.expand_dims(img,axis=0)
        pred = model.predict(img)[0]
        print(pred)
    return count / errors