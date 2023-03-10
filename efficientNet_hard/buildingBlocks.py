import tensorflow as tf
import tensorflow_addons as tfa

class CNNBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size, stride, padding, groups=1):
        super(CNNBlock, self).__init__()
        self.cnn = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.silu = tf.keras.layers.Activation(tf.nn.silu)

    def call(self, x):
        return self.silu(self.batchnorm(self.cnn(x)))

class SEBlock(tf.keras.Model):
    '''
    squeeze and excitation block
    '''
    def __init__(self, initial_dim, reduce_dim):
        super(SEBlock, self).__init__()
        self.glob_avg_pool = tf.keras.layers.GlobalAveragePooling2D()  # C x H x W -> C x 1 x 1
        self.conv_squeeze = tf.keras.layers.Conv2D(filters=reduce_dim, kernel_size=1, stride=1, padding=0, activation='silu') # size = reduce_dim x 
        # self.silu = tf.keras.layers.Activation(tf.nn.silu)
        self.conv_excite = tf.keras.layers.Conv2D(filters=initial_dim, kernel_size=1, stride=1, padding=0, activation = 'sigmoid') # size = initial_dim x
        # self.sigmoid = tf.keras.layers.Activation(tf.nn.sigmoid)
        

    def call(self, x):
        '''
        return:
        Element-wise multiplication between the input tensor inputs and the sigmoid output x results in a scaled version of the input tensor, 
        where each channel has been weighted by its corresponding channel-wise scaling factor from the sigmoid output.
        By weighting the input tensor in this way, the SE block can learn to emphasize the most informative channels of the input tensor 
        and suppress less informative channels, thereby improving the model's ability to capture important features and achieve better performance on a given task.
        '''
        return x * self.conv_excite(self.conv_squeeze(self.glob_avg_pool(x)))
        
class InvertedResidualBlock(tf.keras.Model):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        expand_ratio,
        reduction=4,  # squeeze excitation
        # survival_prob=0.8,  # for stochastic depth
    ):
        super(InvertedResidualBlock, self).__init__()
        self.survival_prob = 0.8
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim
        reduced_dim = int(in_channels / reduction)

        if self.expand:
            self.conv_expand = tf.keras.layers.Conv2D(filters=hidden_dim, kernel_size=3, stride=1, padding=1)

        self.convB = CNNBlock(
                hidden_dim,
                hidden_dim,
                kernel_size,
                stride,
                padding,
                groups=hidden_dim,
            )
        self.seB = SEBlock(hidden_dim, reduced_dim),
        self.conv = tf.keras.layers.Conv2D(filters=hidden_dim, out_channels, 1, bias=False)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def stochastic_depth(self, x, training=False):
        '''
        or just used built-in function
        tfa.layers.StochasticDepth(survival_probability=survival_prob)
        '''
        # if not training:
        #     return x

        # binary_tensor = (torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob)
        # return torch.div(x, self.survival_prob) * binary_tensor
        return tfa.layers.StochasticDepth(survival_probability=self.survival_prob)

    def call(self, inputs, training=False):
        x = self.expand_conv(inputs) if self.expand else inputs

        if self.use_residual:
            return self.stochastic_depth(self.conv(x), training=training) + inputs
        else:
            return self.conv(x)