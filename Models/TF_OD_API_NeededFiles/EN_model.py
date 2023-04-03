from object_detection.models.EN_buildingBlocks import CNNBlock, InvertedResidualBlock
import tensorflow as tf


base_model = [
            # expand_ratio, channels, repeats, strides, kernel_size
            [1, 16, 1, 1, 3],
            [6, 24, 2, 2, 3],
            [6, 40, 2, 2, 5],
            [6, 80, 3, 2, 3],
            [6, 112, 3, 1, 5],
            [6, 192, 4, 2, 5],
            [6, 320, 1, 1, 3],
] 

phi_values = {  
            # version: (width multiplier, depth multiplier, resolution, drop_rate=survival_prob)
            "b0": (1.0, 1.0, 224, 0.2),  
            "b1": (1.0, 1.1, 240, 0.2),
            "b2": (1.1, 1.2, 260, 0.3),
            "b3": (1.2, 1.4, 300, 0.3),
            "b4": (1.4, 1.8, 380, 0.4),
            "b5": (1.6, 2.2, 456, 0.4),
            "b6": (1.8, 2.6, 528, 0.5),
            "b7": (2.0, 3.1, 600, 0.5),
}

class EfficientNet(tf.keras.Model):
    def __init__(self, version, num_classes):
        super(EfficientNet, self).__init__()
        width_factor, depth_factor, res, dropout_rate = phi_values[version]
        last_channels = int(tf.math.ceil(1280*width_factor))
        
        self.inputlayer = tf.keras.layers.InputLayer(input_shape=(res,res))
        self.layerlist = self.create_layers(width_factor, depth_factor, last_channels)
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self.lastlayer = tf.keras.layers.Dense(units=num_classes)

    def create_layers(self, width_factor, depth_factor, last_channels: int):
        channels = int(32*width_factor)
        sequential = tf.keras.Sequential()
        sequential.add(CNNBlock(filters=3, kernel_size=3, strides=2, padding="same")) # padding=1
        input_filters = channels

        for expand_ratio, channels, repeats, strides, kernel_size in base_model:
            output_filters = 4*int(tf.math.ceil(int(channels*width_factor)/4))
            layers_repeats = int(tf.math.ceil(repeats*depth_factor))

            for layer in range(layers_repeats):
                if kernel_size == 1:
                    pad = "valid"
                elif kernel_size == 3:
                    pad = "same"
                elif kernel_size == 5:
                    pad = "same"

                sequential.add(
                    InvertedResidualBlock(
                        input_filters=input_filters,
                        output_filters=output_filters,
                        expand_ratio=expand_ratio,
                        strides=strides if layer == 0 else 1,
                        kernel_size=kernel_size,
                        padding = pad
                    )
                )
                input_filters = output_filters

        sequential.add(CNNBlock(filters=last_channels, kernel_size=2, strides=1, padding="same")) #"valid"))

        return sequential

    def call(self, x, training=False):
        x = self.layerlist(x)
        x = self.pool(x)
        x = self.dropout(x, training=training) 
        x = self.lastlayer(x)

        return x
