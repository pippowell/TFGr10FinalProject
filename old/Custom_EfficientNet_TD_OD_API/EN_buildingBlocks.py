import tensorflow as tf

class CNNBlock(tf.keras.Model):
    '''Simple CNN block'''
    def __init__(self, filters: int, kernel_size, strides, padding):
        super(CNNBlock, self).__init__()
        self.cnn = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.silu = tf.keras.layers.Activation(tf.nn.silu)

    def call(self, x):
        x = self.cnn(x)
        x = self.batchnorm(x)
        x = self.silu(x)

        return x
    
class SEBlock(tf.keras.Model):
    '''Squeeze and excitation block'''
    def __init__(self, initial_dim: int, reduce_dim: int):
        super(SEBlock, self).__init__()
        self.glob_avg_pool = tf.keras.layers.GlobalAveragePooling2D()  # height x width x channels(filters) -> 1 x 1 x channels(filters)
        self.reshape = tf.keras.layers.Reshape((1, 1, initial_dim))
        self.conv_squeeze = tf.keras.layers.Conv2D(filters=reduce_dim, kernel_size=1, strides=1, padding="valid", activation='silu') 
        self.conv_excite = tf.keras.layers.Conv2D(filters=initial_dim, kernel_size=1, strides=1, padding="valid", activation='sigmoid') 

    def call(self, input):
        '''
        Return:
            Element-wise multiplication between the input tensor inputs and the sigmoid output x results in a scaled version of the input tensor, 
            where each channel has been weighted by its corresponding channel-wise scaling factor from the sigmoid output.
            By weighting the input tensor in this way, the SE block can learn to emphasize the most informative channels of the input tensor 
            and suppress less informative channels, thereby improving the model's ability to capture important features and achieve better performance on a given task.
        '''
        x = self.glob_avg_pool(input)
        x = self.reshape(x)
        x = self.conv_squeeze(x)
        x = self.conv_excite(x)
        out = tf.math.multiply(input, x)

        return out           
        
class InvertedResidualBlock(tf.keras.Model):

    def __init__(self, input_filters: int, output_filters: int, kernel_size, strides, padding, expand_ratio, reduction=4, survival_prob=0.8):
        '''
        Args:
            input_filters:
            output_filters:
            kernel_size: 
            strides:
            padding:
            expand_ratio:
            reduction: for squeeze excitation
            survival_prob: for stochastic depth
        ''' 
        super(InvertedResidualBlock, self).__init__()
        self.survival_prob = survival_prob
        self.use_residual = input_filters == output_filters and strides == 1
        hidden_dim = int(input_filters*expand_ratio)
        self.expand = input_filters != hidden_dim
        reduced_dim = int(input_filters/reduction)

        if self.expand:
            self.expand_conv = tf.keras.layers.Conv2D(filters=hidden_dim, kernel_size=3, strides=1, padding="same")

        self.depthwise_conv = CNNBlock(filters=hidden_dim, kernel_size=kernel_size, strides=strides, padding=padding)
        self.seB = SEBlock(initial_dim=hidden_dim, reduce_dim=reduced_dim)
        self.pointwise_conv = CNNBlock(filters=int(output_filters), kernel_size=1, strides=strides, padding=padding)
        self.batchnorm = tf.keras.layers.BatchNormalization() 
        self.add = tf.keras.layers.Add()

    def stochastic_depth(self, x, training=False):
        '''
        Args:
            x (tensor): input
        Returns:


        Randomly drops out entire residual blocks during training with a certain probability.
        It forces the network to learn to operate even in the presence of missing blocks.
        '''
        if not training:
            return x

        else: 
            binary_tensor = tf.cast(tf.random.uniform([tf.shape(x)[0], 1, 1, 1]) < tf.cast(self.survival_prob, tf.float32), dtype=tf.float32)
            
            return tf.divide(x, tf.cast(self.survival_prob, tf.float32))*binary_tensor

    def call(self, inputs, training=False):
        x = self.expand_conv(inputs) if self.expand else inputs
        x = self.depthwise_conv(x)
        x = self.seB(x)
        x = self.pointwise_conv(x)
        x = self.batchnorm(x)
        
        if self.use_residual:
            x = self.stochastic_depth(x, training=training) 
            x += inputs

            return x
        
        else:
            return x
        