import os
import tensorflow as tf

# Some constants
DATASET_TRAIN_DIR: str = "datasets/train"
DATASET_TEST_DIR: str = "datasets/test"
IMAGE_SIZE: int = 224
BATCH_SIZE: int = 32
EPOCHS: int = 10


# Creates the training and validation datasets
def create_datasets(dir: str) -> tuple[tf.data.Dataset, tf.data.Dataset]:
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
  trainingDataset: tf.data.Dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
  )
  validationDataset: tf.data.Dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
  )
  trainingDataset = trainingDataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
  validationDataset = validationDataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
  return trainingDataset, validationDataset
