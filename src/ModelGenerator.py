import os
import cv2 as cv
import numpy as np
import tensorflow as tf

# Some constants.
TARGET_SIZE: tuple[int, int] = (24, 24)
COLOR_MODE: str = "grayscale"
ACCURACY: float = 0.99
IMAGE_SIZE: int = 24
BATCH_SIZE: int = 32
EPOCHS: int = 99999

# Remove Tensorflow annoying messages.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def generator(dir, shuffle=True, batch_size=1, target_size=TARGET_SIZE,
              class_mode='categorical'):
    return tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)\
            .flow_from_directory(dir, batch_size=batch_size, shuffle=shuffle,
            color_mode=COLOR_MODE, class_mode=class_mode, target_size=target_size)


# Creates and fits a convolutional neural network.
def create_model(dir: str) -> tf.keras.models.Sequential:
    modelName: str = str(f"{dir.split('/')[-1]}CNN.h5")
    # If the model exists, it loads it and returns it.
    if os.path.exists(f"models/{modelName}"):
        return tf.keras.models.load_model(f"models/{modelName}")
    # If the model doesn't exists, it generates it.
    # Keras callback for setting an accuracy to the model fit proces.
    class Callback(tf.keras.callbacks.Callback):
        def __init__(self):
            super(Callback, self).__init__()

        def on_epoch_end(self, epoch, logs=None):
            if logs["val_accuracy"] >= 0.96 and logs["accuracy"] >= ACCURACY:
                self.model.stop_training = True
    # Creates the model.
    model: tf.keras.models.Sequential = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), input_shape=\
        (IMAGE_SIZE, IMAGE_SIZE, 1), activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=128, activation="relu"),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(units=2, activation="sigmoid")
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=["accuracy"]
    )

    trainBatch = generator(f"{dir}/train", shuffle=True, batch_size=32, target_size=(24,24))
    validBatch = generator(f"{dir}/test", shuffle=True, batch_size=32, target_size=(24,24))

    model.fit(
        trainBatch,
        validation_data=validBatch,
        epochs=EPOCHS,
        callbacks=[Callback()],
        shuffle = True
    )
    
    model.save(f"models/{modelName}")
    return model


def test_model(model: tf.keras.models.Sequential, dir: str, result: str) -> float:
    count: int = 0
    errors: int = 0
    for file in os.listdir(dir):
        count += 1
        img = cv.imread(f"{dir}/{file}", cv.IMREAD_GRAYSCALE)
        img: np.ndarray = cv.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img: np.ndarray = img.reshape(IMAGE_SIZE, IMAGE_SIZE, -1)
        img: np.ndarray = np.expand_dims(img,axis=0)
        pred = model.predict(img, verbose=0)[0]
        if result == "open" and pred[0] != 1:
            errors += 1
        elif result == "closed"and pred[1] != 1:
            errors += 1
    return errors, count