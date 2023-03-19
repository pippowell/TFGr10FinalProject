import tensorflow as tf

from preprocess import IMG_SIZE, NUM_CLASSES

height=IMG_SIZE
width=IMG_SIZE
num_of_channels=3
input_shape = (height, width, num_of_channels) 

class EffNet(tf.keras.Model): 
    '''
    Options: EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, ... up to  7
    Higher the number, the more complex the model is. and the larger resolutions it  can handle, but  the more GPU memory it will need# loading pretrained conv base model
    '''
    def __init__(self):
        super(EffNet, self).__init__()

        self.conv_base = tf.keras.applications.EfficientNetB0(weights=None, include_top=False, input_shape=input_shape) 
        # weights="imagenet" allows us to do transfer learning
        # include_top=False allows us to easily change the final layer to our custom dataset
        # conv_base.trainable = False # if we want to keep the weights from the pretrained model
        
        self.gpool = tf.keras.layers.GlobalAveragePooling2D()
        self.regularization = tf.keras.layers.Dense(units=512, activation = 'relu', bias_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.001))
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(rate=0.2)
        self.outputlayer = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax") # replacing the last layers with custom layers

    def call(self, input):
        # print(f"shape before anything: {input.shape}")  # (None, 32, 32, 3)
        x = self.conv_base(input)
        # print(f"shape after effnet: {x.shape}")  # (None, 1, 1, 1280)
        x = self.gpool(x)
        # print(f"shape after gpool: {x.shape}")  # (None, 1280)
        x = self.regularization(x)
        # print(f"shape after regularization: {x.shape}") # (None, 512)
        x = self.batchnorm(x)
        # print(f"shape after batchnorm: {x.shape}")  # (None, 512)
        x = self.dropout(x)
        # print(f"shape after dropout: {x.shape}")  # (None, 512)
        x = self.outputlayer(x)
        # print(f"shape after outputlayer: {x.shape}")  # (None, num_classes)
        
        return x