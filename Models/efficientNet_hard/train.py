from model import EfficientNet, phi_values
from preprocess import train_dataset, test_dataset, NUM_CLASSES
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

epochs = 1
lr = 9e-1

version = "b0"
_, _, res, _ = phi_values[version]
mymodel = EfficientNet(version=version, num_classes=NUM_CLASSES)
mymodel.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(lr),
    metrics=["accuracy"]
)

history = mymodel.fit(
    train_dataset,
    epochs=epochs,
    validation_data=test_dataset
)

directory = str(Path(__file__).parents[0])
print(directory)

def plot_hist(hist):
    line1, = plt.plot(hist.history["accuracy"])
    line2, = plt.plot(hist.history["val_accuracy"])
    line3, = plt.plot(hist.history["loss"])
    line4, = plt.plot(hist.history["val_loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend([line1, line2, line3, line4], ["accuracy", "val_accuracy", "loss", "val_loss"], loc="upper left")
    plt.title(f"EfficientNet {version}")
    plt.ylabel("Accuracy/Loss")
    plt.xlabel("Epoch")
    plt.savefig(directory + '/' + f"plots/test_{epochs}epoch_{lr}lr")
    plt.show()

plot_hist(history)
