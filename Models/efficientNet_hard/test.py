import tensorflow as tf
from efficientNet import EfficientNet, phi_values

def test():
    version = "b0"
    width_mul, depth_mul, res, dropout_rate = phi_values[version]
    net = EfficientNet(version=version, num_classes=2)
    x = tf.random.uniform(shape=[1, 3, res, res])
    y = net(x)

    print(f"y.size: {y.size()}")

test()
