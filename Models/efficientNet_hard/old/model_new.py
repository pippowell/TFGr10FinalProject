from buildingBlocks import CNNBlock, InvertedResidualBlock, CNNBlock2
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
            # tuple of: (width multiplier, depth multiplier, resolution, drop_rate=survival_prob)
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
        width_factor, depth_factor, _, dropout_rate = phi_values[version]
        last_channels = tf.math.ceil(1280*width_factor)

        self.layerlist = self.create_layers(width_factor, depth_factor, last_channels)
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self.lastlayer = tf.keras.layers.Dense(units=num_classes) # nn.Linear=(in_features=last_channels, out_features=num_classes),

    def create_layers(self, width_factor, depth_factor, last_channels):

        channels = int(32*width_factor)
        # features = tf.keras.models.Sequential([CNNBlock(filters=3, kernel_size=3, strides=2, padding=1)])
        features = [CNNBlock(filters=3, kernel_size=3, strides=2, padding="same")] # padding=1
        input_filters = channels
        
        kernels = [3, 3, 5, 3, 5, 5, 3]
        expansions = [1, 6, 6, 6, 6, 6, 6]
        num_channels = [16, 24, 40, 80, 112, 192, 320]
        num_layers = [1, 2, 2, 3, 3, 4, 1]
        strides =[1, 2, 2, 2, 1, 2, 1]
        
        # Scale channels and num_layers according to width and depth multipliers.
        scaled_num_channels = [4*tf.math.ceil(int(c*width_factor) / 4) for c in num_channels]
        scaled_num_layers = [int(d*depth_factor) for d in num_layers]
        
        for i in range(len(scaled_num_channels)):
            kernel_size = kernels[i]

            if kernel_size == 1:
                pad = "valid"
            elif kernel_size == 3:
                pad = "same"
            elif kernel_size == 5:
                pad = "same"
             
            features += [InvertedResidualBlock(input_filters if repeat==0 else scaled_num_channels[i], 
                               scaled_num_channels[i],
                               kernel_size = kernel_size,
                               strides = strides[i] if repeat==0 else 1, 
                               expand_ratio = expansions[i],
                               padding=pad
                              )
                       for repeat in range(scaled_num_layers[i])
                      ]
            input_filters = scaled_num_channels[i]

        features.append(
            CNNBlock2(filters=last_channels, kernel_size=1, strides=1, padding="valid")
        )
        return features

    def call(self, x, training=False):
        for (layer, i) in zip(self.layerlist, range(len(self.layerlist))):
            x = layer(x)        
            print(f"after iteration {i}: x is {tf.shape(x)} and dtype of {x.dtype}")
        print("done with layerlist")
        x = self.pool(x)
        x = self.dropout(x.view(x.shape[0], -1), training=training) 
        x = self.lastlayer(x)
        return x
