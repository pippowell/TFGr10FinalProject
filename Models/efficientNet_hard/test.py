import tensorflow as tf
from efficientNet import EfficientNet, phi_values

def test():
    version = "b0"
    _, _, res, _ = phi_values[version]
    network = EfficientNet(version=version, num_classes=2)
    x = tf.random.uniform(shape=[1, res, res, 3]) # batch x width x height x channel
    print(f"initial shape of x: {tf.shape(x)} and dtype: {x.dtype}")
    y = network(x)

    print(f"y.size: {y.size()}")

test()
