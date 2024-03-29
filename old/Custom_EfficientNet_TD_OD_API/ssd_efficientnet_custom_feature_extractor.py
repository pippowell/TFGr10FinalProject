import tensorflow.compat.v1 as tf

from object_detection.meta_architectures import ssd_meta_arch
from object_detection.models import feature_map_generators
# from object_detection.models.keras_models import mobilenet_v2
from object_detection.utils import ops
from object_detection.utils import shape_utils
from object_detection.models.EN_model import EfficientNet, phi_values
# from .efficientnet import build_model_base_keras_model

# 1. copy the exactly from the class SSDMobileNetV2KerasFeatureExtractor
# 2. customize under efficientNet architecture
# 3. add an argument of efficientNet version in __init__ 

version = "b0"
_, _, res, _ = phi_values[version]

class EfficientNetFeatureExtractor(
    ssd_meta_arch.SSDKerasFeatureExtractor):
  def __init__(self, 
              is_training,
              depth_multiplier,
              min_depth,
              pad_to_multiple,
              conv_hyperparams,
              freeze_batchnorm,
              inplace_batchnorm_update,
              network_version='efficientnet-b0',
              min_feature_level=3,
              max_feature_level=7,
              additional_layer_depth=256,
              # reuse_weights=None,
              use_explicit_padding=None,
              use_depthwise=False,
              # use_antialias=False,
              override_base_feature_extractor_hyperparams=False,
              name=None,
              data_format="channels_last"):

            # (self,
#              is_training,
#              depth_multiplier,
#              min_depth,
#              pad_to_multiple,
#              conv_hyperparams,
#              freeze_batchnorm,
#              inplace_batchnorm_update,
#              use_explicit_padding=False,
#              use_depthwise=False,
#              num_layers=6,
#              override_base_feature_extractor_hyperparams=False,
#              name=None):
  
    """SSD Feature Extractor using EfficientNet features. # B0? should we define which version?
    Args:
      is_training: whether the network is in training mode.
      depth_multiplier: float depth multiplier for feature extractor.
      min_depth: minimum feature extractor depth.
      pad_to_multiple: the nearest multiple to zero pad the input height and width dimensions to.
      conv_hyperparams: `hyperparams_builder.KerasLayerHyperparams` object
        containing convolution hyperparameters for the layers added on top of
        the base feature extractor.
      freeze_batchnorm: Whether to freeze batch norm parameters during
        training or not. When training with a small batch size (e.g. 1), it is
        desirable to freeze batch norm update and use pretrained batch norm
        params.
      inplace_batchnorm_update: Whether to update batch norm moving average
        values inplace. When this is false train op must add a control
        dependency on tf.graphkeys.UPDATE_OPS collection in order to update
        batch norm statistics.
      use_explicit_padding: Whether to use explicit padding when extracting
        features. Default is False.
      use_depthwise: Whether to use depthwise convolutions. Default is False.
      num_layers: Number of SSD layers.
      override_base_feature_extractor_hyperparams: Whether to override
        hyperparameters of the base feature extractor with the one from
        `conv_hyperparams_fn`.
      name: A string name scope to assign to the model. If 'None', Keras
        will auto-generate one from the class name.
    """
    super(EfficientNetFeatureExtractor, self).__init__(
        is_training=is_training,
        depth_multiplier=depth_multiplier,
        min_depth=min_depth,
        pad_to_multiple=pad_to_multiple,
        conv_hyperparams=conv_hyperparams,
        freeze_batchnorm=freeze_batchnorm,
        inplace_batchnorm_update=inplace_batchnorm_update,
        use_explicit_padding=use_explicit_padding,
        use_depthwise=use_depthwise,
        override_base_feature_extractor_hyperparams=override_base_feature_extractor_hyperparams,
        name=name)
    self._data_format = data_format
    self._min_feature_level = min_feature_level
    self._max_feature_level = max_feature_level
    self._additional_layer_depth = additional_layer_depth   
    self._network_name = network_version
    if network_version == "efficientnet-b0":
        default_nodes = ['sequential', 'global_average_pooling2d_16', 'dropout']
        # default_nodes = ["block_4", "block_10", "block_15", "", "", "", ""] 
        self._backbone_layers = 15
    elif network_version == "efficientnet-b1":
        default_nodes = ["block_7", "block_15", "block_22", "", "", "", ""]
        self._backbone_layers = 22
    elif network_version == 'efficientnet-b2':
        default_nodes = ["block_7", "block_15", "block_22", "", "", "", ""]
        self._backbone_layers = 22
    elif network_version == "efficientnet-b3":
        default_nodes = ["block_7", "block_17", "block_25", "", "", "", ""]
        self._backbone_layers = 25
    elif network_version == "efficientnet-b4":
        default_nodes = ["block_9", "block_21", "block_31", "", "", "", ""]
        self._backbone_layers = 31
    elif network_version == "efficientnet-b5":
        default_nodes = ["block_12", "block_26", "block_38", "", "", "", ""]
        self._backbone_layers = 38
    elif network_version == "efficientnet-b6":
        default_nodes = ["block_14", "block_30", "block_44", "", "", "", ""]
        self._backbone_layers = 44
    elif network_version == "efficientnet-b7":
        default_nodes = ["block_17", "block_37", "block_54", "", "", "", ""]
        self._backbone_layers = 54
    else:
        raise ValueError("Unknown efficientnet name: {}".format(network_version))
    # default_nodes = ["reduction_3", "reduction_4", "reduction_5", "", "", ""]

    default_nodes_depth = [-1, -1, -1, 512, 256, 256, 128]
    self._used_nodes = default_nodes[min_feature_level-3:max_feature_level-2]
    self._used_nodes_depth = default_nodes_depth[min_feature_level-3:max_feature_level-2]
    self._feature_map_layout = {
        'from_layer': self._used_nodes,
        'layer_depth': self._used_nodes_depth,
        'use_depthwise': self._use_depthwise,
        'use_explicit_padding': self._use_explicit_padding
    }
    self._feature_map_generator = None 
    self._coarse_feature_layers = []
    self.net = None 

  def build(self, input_shape=(res, res)):    
    model = EfficientNet(version="b0", num_classes=2)

    # inputs = tf.keras.layers.Input(input_shape[1:])
    # print(f"input: {inputs}")
    inputs = model.inputs
    # _, endpoints = build_model_base(inputs, self._network_name, self._is_training)
    # outputs = [endpoints[x] for x in self._used_nodes if x]
    outputs = [model.get_layer(x).output for x in self._used_nodes if x]
    print(f"output: {outputs}")

    self.net = tf.keras.Model(inputs=inputs, outputs=outputs)

    # feature map generator
    self.feature_map_generator = feature_map_generators.KerasMultiResolutionFeatureMaps(
        feature_map_layout=self._feature_map_layout,
        depth_multiplier=self._depth_multiplier,
        min_depth=self._min_depth,
        insert_1x1_conv=True, 
        is_training=self._is_training,
        conv_hyperparams=self._conv_hyperparams,
        freeze_batchnorm=self._freeze_batchnorm,
        name=None
    )
    self.built = True 

    # full_mobilenet_v2 = mobilenet_v2.mobilenet_v2(
    #     batchnorm_training=(self._is_training and not self._freeze_batchnorm),
    #     conv_hyperparams=(self._conv_hyperparams
    #                       if self._override_base_feature_extractor_hyperparams
    #                       else None),
    #     weights=None,
    #     use_explicit_padding=self._use_explicit_padding,
    #     alpha=self._depth_multiplier,
    #     min_depth=self._min_depth,
    #     include_top=False)
    # conv2d_11_pointwise = full_mobilenet_v2.get_layer(
    #     name='block_13_expand_relu').output
    # conv2d_13_pointwise = full_mobilenet_v2.get_layer(name='out_relu').output
    # self.classification_backbone = tf.keras.Model(
    #     inputs=full_mobilenet_v2.inputs,
    #     outputs=[conv2d_11_pointwise, conv2d_13_pointwise])
    # self.feature_map_generator = (
    #     feature_map_generators.KerasMultiResolutionFeatureMaps(
    #         feature_map_layout=self._feature_map_layout,
    #         depth_multiplier=self._depth_multiplier,
    #         min_depth=self._min_depth,
    #         insert_1x1_conv=True,
    #         is_training=self._is_training,
    #         conv_hyperparams=self._conv_hyperparams,
    #         freeze_batchnorm=self._freeze_batchnorm,
    #         name='FeatureMaps'))
    # self.built = True

  def preprocess(self, resized_inputs):
    """SSD preprocessing.
    Maps pixel values to the range [-1, 1].
    Args:
      resized_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.
    Returns:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.
    """
    return (2.0 / 255.0) * resized_inputs - 1.0

  def _extract_features(self, preprocessed_inputs):
    """Extract features from preprocessed inputs.
    Args:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.
    Returns:
      feature_maps: a list of tensors where the ith tensor has shape
        [batch, height_i, width_i, depth_i]
    """
    # preprocessed_inputs = shape_utils.check_min_image_dim(
    #     33, preprocessed_inputs)

    # image_features = self.classification_backbone(
    #     ops.pad_to_multiple(preprocessed_inputs, self._pad_to_multiple))

    # feature_maps = self.feature_map_generator({
    #     'layer_15/expansion_output': image_features[0],
    #     'layer_19': image_features[1]})
    # return list(feature_maps.values())
    
    """Extract features from preprocessed inputs"""        
    preprocessed_inputs = shape_utils.check_min_image_dim(33, preprocessed_inputs)
    image_features = self.net(ops.pad_to_multiple(preprocessed_inputs, self._pad_to_multiple))
    layouts = {self._used_nodes[i]: image_features[i] for i, x in enumerate(self._used_nodes) if x}
    feature_maps = self._feature_map_generator(layouts)
    if self._additional_layer_depth:
        final_feature_map = []
        for idx, feature in enumerate(feature_maps.values()):
            feature = tf.keras.layers.Conv2D(filters=self._additional_layer_depth,
                                            kernel_size=1,
                                            strides=[1, 1],
                                            use_bias=True,
                                            data_format=self._data_format,
                                            name='conv1x1_'+str(idx))(feature)
            feature = tf.keras.layers.BatchNormalization()(feature, training=self._is_training)
            feature = tf.keras.layers.ReLU(max_value=6)(feature)
            final_feature_map.append(feature)
        return final_feature_map
    else:
        return list(feature_maps.values())

