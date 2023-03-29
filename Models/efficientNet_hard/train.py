from model import EfficientNet, phi_values
from preprocess import train_dataset, test_dataset
import tensorflow as tf
import matplotlib.pyplot as plt

epochs = 10
lr = 1e-4

version = "b0"
_, _, res, _ = phi_values[version]
mymodel = EfficientNet(version=version, num_classes=10)
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

def plot_hist(hist):
    # plt.plot(hist.history["loss"])
    # plt.plot(hist.history["val_loss"])
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.savefig(f"efficientnet test epoch {epochs}")
    plt.show()

plot_hist(history)
