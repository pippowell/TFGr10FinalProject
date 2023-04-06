import tensorflow_datasets as tfds
import tensorflow as tf
import splitfolders
from pathlib import Path
from model import phi_values
from train import version

# split the dataset into train, val, test dataset
# splitfolders.ratio(f"str(Path(__file__).parents[0])/images/", output=f"{directory}/train&val&test", seed=1337, ratio=(.8, 0.1,0.1)) # A seed makes splits reproducible.

directory = str(Path(__file__).parents[0]) + "/train&val&test/train"

# Define your data preprocessing pipeline
_, _, res, _ = phi_values[version]

image_size = (res, res)
batch_size = 32
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    directory,
    image_size=image_size,
    batch_size=batch_size
)

def preprocess(dataset):

    # convert data from uint8 to float32
    dataset = dataset.map(lambda img, target: (tf.cast(img, tf.float32), target))

    # input normalization, just bringing image values from range [0, 255] to [-1, 1]
    dataset = dataset.map(lambda img, target: ((img / 128.) - 1., target))

    # create one-hot targets with depth 2 since we need to distinguish grapes from non-grapes
    dataset = dataset.map(lambda img, target: (img, tf.one_hot(target, depth=2)))
    # cache
    dataset = dataset.cache()

    # shuffle, batch, prefetch
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # return preprocessed dataset # take 10 from 60000
    return dataset.take(10)


train_dataset = preprocess(dataset)
# val_dataset = preprocess(f"{directory}/train&val&test/val")
# test_dataset = preprocess(f"{directory}/train&val&test/test")


# checking
print(train_dataset)
