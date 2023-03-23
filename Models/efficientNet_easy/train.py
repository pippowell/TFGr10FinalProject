from model import EffNet
from preprocess import train_dataset, test_dataset
import tensorflow as tf
import matplotlib.pyplot as plt

epochs = 2
lr = 1e-1

mymodel = EffNet()
mymodel.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(1e-4),
    metrics=["accuracy"]
)

history = mymodel.fit(
    train_dataset,
    epochs=epochs,
    validation_data=test_dataset
)

def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.savefig()
    plt.show()

plot_hist(history)
