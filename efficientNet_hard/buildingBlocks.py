import tensorflow as tf
import tensorflow_addons as tfa

class CNNBlock(tf.keras.Model):
    '''Simple CNN block'''
    def __init__(self, filters, kernel_size, stride, padding):
        super(CNNBlock, self).__init__()
        self.cnn = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.silu = tf.keras.layers.Activation(tf.nn.silu)

    def call(self, x):
        x = self.cnn(x)
        x = self.batchnorm(x)
        x = self.silu(x)
        return x

class SEBlock(tf.keras.Model):
    '''Squeeze and excitation block'''
    def __init__(self, initial_dim, reduce_dim):
        super(SEBlock, self).__init__()
        self.glob_avg_pool = tf.keras.layers.GlobalAveragePooling2D()  # C x H x W -> C x 1 x 1
        self.conv_squeeze = tf.keras.layers.Conv2D(filters=reduce_dim, kernel_size=1, stride=1, padding=0, activation='silu') # size = reduce_dim x 
        self.conv_excite = tf.keras.layers.Conv2D(filters=initial_dim, kernel_size=1, stride=1, padding=0, activation='sigmoid') # size = initial_dim x
        

    def call(self, x):
        '''
        return:
        Element-wise multiplication between the input tensor inputs and the sigmoid output x results in a scaled version of the input tensor, 
        where each channel has been weighted by its corresponding channel-wise scaling factor from the sigmoid output.
        By weighting the input tensor in this way, the SE block can learn to emphasize the most informative channels of the input tensor 
        and suppress less informative channels, thereby improving the model's ability to capture important features and achieve better performance on a given task.
        '''
        x = self.glob_avg_pool(x)
        x = self.conv_squeeze(x)

        return x * self.conv_excite(x)            
        
class InvertedResidualBlock(tf.keras.Model):
    def __init__(
        self,
        input_filters, # Q. how do I get num of input filters from the input?
        output_filters,
        kernel_size,
        stride,
        padding,
        expand_ratio,
        reduction=4,  # for squeeze excitation
        survival_prob=0.8,  # for stochastic depth
    ):
        super(InvertedResidualBlock, self).__init__()
        self.survival_prob = survival_prob
        self.use_residual = input_filters == output_filters and stride == 1
        hidden_dim = input_filters * expand_ratio
        self.expand = input_filters != hidden_dim
        reduced_dim = int(input_filters / reduction)

        if self.expand:
            self.conv_expand = tf.keras.layers.Conv2D(filters=hidden_dim, kernel_size=3, stride=1, padding=1)

        self.convB = CNNBlock(output_filters, kernel_size, stride, padding)
        self.seB = SEBlock(initial_dim=hidden_dim, reduce_dim=reduced_dim)
        self.conv = tf.keras.layers.Conv2D(filters=output_filters, kernel_size=1, use_bias=False) # nn.Conv2d(in_channels, reduced_dim, kernel_size=1, bias=False)
        self.batchnorm = tf.keras.layers.BatchNormalization() # nn.BatchNorm2d(out_channels) 

    def stochastic_depth(self, x, training=False):
        '''
        randomly drops out entire residual blocks during training with a certain probability.
        It forces the network to learn to operate even in the presence of missing blocks.
        '''
        if not training:
            return x

        else: 
            binary_tensor = tf.random.uniform(shape=[x.shape[0], 1, 1, 1] < self.survival_prob)
            return tf.divide(x, self.survival_prob) * binary_tensor
        # return tfa.layers.StochasticDepth(survival_probability=self.survival_prob) # or just used built-in function

    def call(self, inputs, training=False):
        x = self.expand_conv(inputs) if self.expand else inputs

        x = self.convB(x)
        x = self.seB(x)
        x = self.conv(x)
        x = self.batchnorm(x)
        
        if self.use_residual:
            x = self.stochastic_depth(x, training=training) 
            x += inputs
            return x
        else:
            return x