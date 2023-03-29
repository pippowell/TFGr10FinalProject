import tensorflow_datasets as tfds
import tensorflow as tf

batch_size = 64

# The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. 
(train_ds, test_ds), ds_info = tfds.load('cifar10', split =['train', 'test'], as_supervised = True, with_info = True)
NUM_CLASSES = ds_info.features["label"].num_classes

def preprocess(dataset):

    # convert data from uint8 to float32
    dataset = dataset.map(lambda img, target: (tf.cast(img, tf.float32), target))

    # input normalization, just bringing image values from range [0, 255] to [-1, 1]
    dataset = dataset.map(lambda img, target: ((img / 128.) - 1., target))

    # create one-hot targets with depth 10 since cifar 10 has 10 classes
    dataset = dataset.map(lambda img, target: (img, tf.one_hot(target, depth=10)))

    # cache
    dataset = dataset.cache()

    # shuffle, batch, prefetch
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # return preprocessed dataset # take 10 from 60000
    return dataset.take(10)

train_dataset = preprocess(train_ds)
test_dataset = preprocess(test_ds)

# checking
# print(train_dataset)
