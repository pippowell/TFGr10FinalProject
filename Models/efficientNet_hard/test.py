import tensorflow as tf
from model import EfficientNet, phi_values

def test():
    version = "b0"
    _, _, res, _ = phi_values[version]
    network = EfficientNet(version=version, num_classes=2)
    x = tf.random.uniform(shape=[1, res, res, 3]) # batch x width x height x channel
    print(f"initial shape of x: {tf.shape(x)} and dtype: {x.dtype}")
    y = network(x)

    print(f"y.size: {tf.shape(y)}") # should be [1, num_classes]

test()
