import os
import cv2 as cv
import contextlib
import numpy as np
import tensorflow as tf

# Some constants.
TARGET_SIZE: tuple[int, int] = (24, 24)
COLOR_MODE: str = "grayscale"
IMAGE_SIZE: int = 24
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

def generator(dir, gen=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255), shuffle=True,batch_size=1,target_size=(24,24),class_mode='categorical' ):

    return gen.flow_from_directory(dir,batch_size=batch_size,shuffle=shuffle,color_mode='grayscale',class_mode=class_mode,target_size=target_size)


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
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(1, 1)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(1, 1)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(1, 1)),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=128, activation="relu"),
        tf.keras.layers.Dropout(rate=0.5),
        # Usamos la activación sigmoid porque es la más correcta para salidas binarias.
        tf.keras.layers.Dense(units=2, activation="softmax")
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=["accuracy"]
    )
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        trainingDataset, validationDataset = create_datasets(dir)
    train_batch= generator('datasets/eyes/train',shuffle=True, batch_size=32,target_size=(24,24))
    valid_batch= generator('datasets/eyes/test',shuffle=True, batch_size=32,target_size=(24,24))

    model.fit(
        train_batch,
        validation_data=valid_batch,
        epochs=epochs,
        callbacks=[Callback()],
        shuffle = True
    )
    print(f"{modelName} -> Created")
    model.save(f"models/{modelName}")
    return model


def test_model(model: tf.keras.models.Sequential, dir: str, result: int) -> float:
    count: int = 0
    errors: int = 0
    for file in os.listdir(dir):
        count += 1
        img = cv.imread(f"{dir}/{file}", cv.IMREAD_GRAYSCALE)
        img: np.ndarray = cv.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img: np.ndarray = img.reshape(IMAGE_SIZE, IMAGE_SIZE, -1)
        img: np.ndarray = np.expand_dims(img,axis=0)
        pred = model.predict(img)[0]
        print(pred)
    return errors, count