from buildingBlocks import CNNBlock, SEBlock, InvertedResidualBlock
import tensorflow as tf

base_model = [
            # expand_ratio, channels, repeats, stride, kernel_size
            [1, 16, 1, 1, 3],
            [6, 24, 2, 2, 3],
            [6, 40, 2, 2, 5],
            [6, 80, 3, 2, 3],
            [6, 112, 3, 1, 5],
            [6, 192, 4, 2, 5],
            [6, 320, 1, 1, 3],
]

phi_values = {  
            # tuple of: (phi, resolution, drop_rate)
            "b0": (0, 224, 0.2),  # depth=alpha**phi, width=beta**phi
            "b1": (0.5, 240, 0.2),
            "b2": (1, 260, 0.3),
            "b3": (2, 300, 0.3),
            "b4": (3, 380, 0.4),
            "b5": (4, 456, 0.4),
            "b6": (5, 528, 0.5),
            "b7": (6, 600, 0.5),
}

class EfficientNet(tf.keras.Model):
    def __init__(self, version, num_classes):
        super(EfficientNet, self).__init__()
        width_factor, depth_factor, dropout_rate = self.calculate_factors(version)
        last_channels = tf.math.ceil(1280 * width_factor)

        self.features = self.create_features(width_factor, depth_factor, last_channels)
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.classifier = tf.keras.models.Sequential(
            tf.keras.layers.Dropout(rate=dropout_rate),
            tf.keras.layers.Dense(units=num_classes) # nn.Linear=(in_features=last_channels, out_features=num_classes),
        )

    def calculate_factors(self, version, alpha=1.2, beta=1.1):
        phi, res, drop_rate = phi_values[version]
        depth_factor = alpha**phi
        width_factor = beta**phi
        return width_factor, depth_factor, drop_rate

    def create_features(self, width_factor, depth_factor, last_channels):

        channels = int(32 * width_factor)
        features = tf.keras.models.Sequential([CNNBlock(3, channels, 3, stride=2, padding=1)])
        in_channels = channels

        for expand_ratio, channels, repeats, stride, kernel_size in base_model:
            out_channels = 4 *  tf.math.ceil(int(channels * width_factor) / 4)
            layers_repeats = tf.math.ceil(repeats * depth_factor)

            for layer in range(layers_repeats):
                features.append(
                    InvertedResidualBlock(
                        in_channels,
                        out_channels,
                        expand_ratio=expand_ratio,
                        stride=stride if layer == 0 else 1,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,  # if k=1:pad=0, k=3:pad=1, k=5:pad=2
                    )
                )
                in_channels = out_channels

        features.append(
            CNNBlock(in_channels, last_channels, kernel_size=1, stride=1, padding=0)
        )

        return features

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x.view(x.shape[0], -1)) 
        return x


