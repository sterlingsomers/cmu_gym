import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
from common.modules import multihead_attention, feedforward, build_Relation, normalize

class FullyConvPolicy:
    """
    FullyConv network structure from https://arxiv.org/pdf/1708.04782.pdf
    Some implementation ideas are borrowed from https://github.com/xhujoy/pysc2-agents
    """

    def __init__(self,
            agent,
            trainable: bool = True,
            classic = True,
            params={}):

        # type agent: ActorCriticAgent
        self.params=params
        self.placeholders = agent.placeholders
        self.trainable = trainable
        self.unittype_emb_dim = agent.unit_type_emb_dim
        self.num_actions = agent.num_actions
        self.classic=classic
        self.use_batchnorm = tf.placeholder_with_default(False, (), "use_batchnorm")  #BOB for batch normalization
        print("policy.py FullyConvPolicy self.istrain tensor {}".format(self.use_batchnorm))


    def _build_convs(self, inputs, name):

        conv1 = layers.conv2d(
            inputs=inputs,
            data_format="NHWC",
            num_outputs= 32,
            kernel_size= 8, #8
            stride= 4,#4
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv1" % name,
            trainable=self.trainable
        )

        conv2 = layers.conv2d(
            inputs=conv1,
            data_format="NHWC",
            num_outputs=64,
            kernel_size=4,
            stride=1,#2,#
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv2" % name,
            trainable=self.trainable
        )


        if self.trainable:
            layers.summarize_activation(conv1)
            layers.summarize_activation(conv2)


        return conv2

    def build(self):
        # units_embedded = layers.embed_sequence(
        #     self.placeholders.screen_unit_type,
        #     vocab_size=SCREEN_FEATURES.unit_type.scale, # 1850
        #     embed_dim=self.unittype_emb_dim, # 5
        #     scope="unit_type_emb",
        #     trainable=self.trainable
        # )
        #
        # # Let's not one-hot zero which is background
        # player_relative_screen_one_hot = layers.one_hot_encoding(
        #     self.placeholders.player_relative_screen,
        #     num_classes=SCREEN_FEATURES.player_relative.scale
        # )[:, :, :, 1:]
        # player_relative_minimap_one_hot = layers.one_hot_encoding(
        #     self.placeholders.player_relative_minimap,
        #     num_classes=MINIMAP_FEATURES.player_relative.scale
        # )[:, :, :, 1:]
        #
        #channel_axis = 2
        # alt0_all = tf.concat(
        #     [self.placeholders.alt0_grass, self.placeholders.alt0_bush, self.placeholders.alt0_drone, self.placeholders.alt0_hiker],
        #     axis=channel_axis
        # )
        # alt1_all = tf.concat(
        #     [self.placeholders.alt1_pine, self.placeholders.alt1_pines, self.placeholders.alt1_drone],
        #     axis=channel_axis
        # )
        # alt2_all = tf.concat(
        #     [self.placeholders.alt2_drone],
        #     axis=channel_axis
        # )
        # alt3_all = tf.concat(
        #     [self.placeholders.alt3_drone],
        #     axis=channel_axis
        # )

        # VOLUMETRIC APPROACH
        # alt_all = tf.concat(
        #     [self.placeholders.alt0_grass, self.placeholders.alt0_bush, self.placeholders.alt0_drone, self.placeholders.alt0_hiker,
        #      self.placeholders.alt1_pine, self.placeholders.alt1_pines, self.placeholders.alt1_drone, self.placeholders.alt2_drone,
        #      self.placeholders.alt3_drone],
        #     axis=channel_axis
        # )
        # self.spatial_action_logits = layers.conv2d(
        #     alt_all,
        #     data_format="NHWC",
        #     num_outputs=1,
        #     kernel_size=1,
        #     stride=1,
        #     activation_fn=None,
        #     scope='spatial_action',
        #     trainable=self.trainable
        # )
        # self.screen_output = self._build_convs(screen_numeric_all, "screen_network")
        # self.minimap_output = self._build_convs(minimap_numeric_all, "minimap_network")
        screen_px = tf.cast(self.placeholders.rgb_screen, tf.float32) / 255. # rgb_screen are integers (0-255) and here we convert to float and normalize
        alt_px = tf.cast(self.placeholders.alt_view, tf.float32) / 255.
        self.screen_output = self._build_convs(screen_px, "screen_network")

        print("policy.py FullyConvPolicy.build use_egocentric {}".format(self.params['use_egocentric']))
        if self.params['use_egocentric']:
            self.alt_output = self._build_convs(alt_px, "alt_network")

        #minimap_px = tf.cast(self.placeholders.rgb_screen, tf.float32) / 255.
        # self.alt0_output = self._build_convs(alt0_all, "alt0_network")
        # self.alt1_output = self._build_convs(alt1_all, "alt1_network")
        # self.alt2_output = self._build_convs(alt2_all, "alt2_network")
        # self.alt3_output = self._build_convs(alt3_all, "alt3_network")

        # VOLUMETRIC APPROACH
        # self.alt0_output = self._build_convs(self.spatial_action_logits, "alt0_network")

        """(MINE) As described in the paper, the state representation is then formed by the concatenation
        of the screen and minimap outputs as well as the broadcast vector stats, along the channer dimension"""
        # State representation (last layer before separation as described in the paper)
        #self.map_output = tf.concat([self.alt0_output, self.alt1_output, self.alt2_output, self.alt3_output], axis=2)
        #self.map_output = tf.concat([self.alt0_output, self.alt1_output], axis=2)

        if self.params['use_egocentric']:
            self.map_output = tf.concat([self.screen_output, self.alt_output], axis=2)
        else:
            self.map_output = self.screen_output

        #self.map_output = self.screen_output
        # The output layer (conv) of the spatial action policy with one ouput. So this means that there is a 1-1 mapping
        # (no filter that convolvues here) between layer and output. So eventually for every position of the layer you get
        # one value. Then you flatten it and you pass it into a softmax to get probs.
        # self.spatial_action_logits = layers.conv2d(
        #     self.map_output,
        #     data_format="NHWC",
        #     num_outputs=1,
        #     kernel_size=1,
        #     stride=1,
        #     activation_fn=None,
        #     scope='spatial_action',
        #     trainable=self.trainable
        # )
        #
        # spatial_action_probs = tf.nn.softmax(layers.flatten(self.spatial_action_logits))

        map_output_flat = layers.flatten(self.map_output)

        if self.params['use_additional_fully_connected']:
            # (MINE) This is the last layer (fully connected -fc) for the non-spatial (categorical) actions
            self.fc1 = layers.fully_connected(
                map_output_flat,
                num_outputs=256,
                activation_fn=tf.nn.relu,
                scope="fc1",
                trainable=self.trainable
            )
        else:
            self.fc1 = map_output_flat

        # Add layer normalization for better stability
        self.fc1 = layers.layer_norm(self.fc1,trainable=self.trainable)

        # fc1 = normalize(fc1, train=False) # wont work cauz PPO compares global variables with trainable variables so no matter True/False assertion will give error
        # (MINE) From the previous layer you extract action_id_probs (non spatial - categorical - actions) and value
        # estimate
        action_id_probs = layers.fully_connected(
            self.fc1,
            num_outputs=self.num_actions,#len(actions.FUNCTIONS),
            activation_fn=tf.nn.softmax,
            scope="action_id",
            trainable=self.trainable
        )
        value_estimate = tf.squeeze(layers.fully_connected( # squeeze removes a dimension of 1 elements. e.g.: [n_batches,1,value_est_dim]--->[n_batches,value_est_dim]
            self.fc1,
            num_outputs=1,
            activation_fn=None,
            scope='value',
            trainable=self.trainable
        ), axis=1)

        # disregard non-allowed actions by setting zero prob and re-normalizing to 1 ((MINE) THE MASK)
        # action_id_probs *= self.placeholders.available_action_ids
        # action_id_probs /= tf.reduce_sum(action_id_probs, axis=1, keepdims=True)

        def logclip(x):
            return tf.log(tf.clip_by_value(x, 1e-12, 1.0))

        # spatial_action_log_probs = (
        #     logclip(spatial_action_probs)
        #     * tf.expand_dims(self.placeholders.is_spatial_action_available, axis=1)
        # )

        # non-available actions get log(1e-10) value but that's ok because it's never used
        action_id_log_probs = logclip(action_id_probs)

        self.value_estimate = value_estimate
        self.action_id_probs = action_id_probs
        #self.spatial_action_probs = spatial_action_probs
        self.action_id_log_probs = action_id_log_probs
       # self.spatial_action_log_probs = spatial_action_log_probs
        return self


class DeepFullyConvPolicy(FullyConvPolicy):

    def _build_convs(self, inputs, name):

        # 100x100 input image ( 20x5 x 20x5 )

        conv1 = layers.conv2d(
            inputs=inputs,
            data_format="NHWC",
            num_outputs=64, #BOB was 32,
            kernel_size=5, #BOB was 8, #8
            stride=5, #BOB was 4,#4
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv1" % name,
            trainable=self.trainable
        )

        # Stride 5 above should reduce this to 20x20 image

        conv2 = layers.conv2d(
            inputs=conv1,
            data_format="NHWC",
            num_outputs=64,
            kernel_size=3, #4
            stride=1,#2,#
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv2" % name,
            trainable=self.trainable
        )

        conv3 = layers.conv2d(
            inputs=conv2,
            data_format="NHWC",
            num_outputs=128,
            kernel_size=3,
            stride=1,
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv3" % name,
            trainable=self.trainable
        )

        pool1 = tf.layers.max_pooling2d(conv3, 2, 2)

        # Pooling should half this - now 10x10

        conv4 = layers.conv2d(
            inputs=pool1,
            data_format="NHWC",
            num_outputs=128,
            kernel_size=3,
            stride=1,
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv4" % name,
            trainable=self.trainable
        )

        conv5 = layers.conv2d(
            inputs=conv4,
            data_format="NHWC",
            num_outputs=192,
            kernel_size=3,
            stride=1,
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv5" % name,
            trainable=self.trainable
        )

        conv6 = layers.conv2d(
            inputs=conv5,
            data_format="NHWC",
            num_outputs=192,
            kernel_size=3,
            stride=1,
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv6" % name,
            trainable=self.trainable
        )

        pool2 = tf.layers.max_pooling2d(conv6, 2, 2)

        # Now pooling will give us roughly ~ 5x5

        conv7 = layers.conv2d(
            inputs=conv2,
            data_format="NHWC",
            num_outputs=400,
            kernel_size=3,
            stride=1,
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv7" % name,
            trainable=self.trainable
        )

        return conv7


class DeepDensePolicy(FullyConvPolicy):

    """This model has 10 conv layers (instead of DeepFullyConvPolicy's 7).
       An architecture inspired by simplenet v2 https://arxiv.org/pdf/1608.06037.pdf
       however we do not implement dropout and currently batch_norm is turned off until
       we move it to the other side of the Relu - see

        https://towardsdatascience.com/how-to-use-batch-normalization-with-tensorflow-and-tf-keras-to-train-deep-neural-networks-faster-60ba4d054b73

       and incorporate 'dense' style connections so that the last layer can see multiple scales.
       These are implemented by concatenating layers 2c,7 and 10.


       100x100x3       input                   RGB Channels
                         |
       20x20x64        conv1                   (stride 5 does collapse)
                         |
                       conv2a
                       conv2b
       20x20x128       conv2c----------+
                         |             |
                       pool1           |       pool 2x2 halfs size
                         |             |
                       conv5a          |
                       conv5b          |
       10x10x128       conv7-----+     |
                         |       |     |
                       pool2     |     |       pool 2x2 halfs size
                         |       |     |
                       conv8a    |     |
                       conv8b    |     |
       5x5x128         conv10    |     |
                         |       |     |
       3x1x128         pool    pool   pool     complete pooling over spatial dimensions
                         |       |     |
       1x384           multi_scale_concat

       """

    def _build_convs(self, inputs, name):

        # 100x100x3 RGB input image ( 20x5 x 20x5 )

        print("---------------------------------------------------------------------")
        print("DeepDensePolicy builder")
        print("---------------------------------------------------------------------")
        print("policy.py DeepDensePolicy.init use_batchnorm flag ".format(self.use_batchnorm))
        print("policy.py DeepDensePolicy.init input shape {}".format(inputs.shape))



        inputs_norm = tf.layers.batch_normalization(inputs, training=self.use_batchnorm, renorm=True)


        # -----------------------------------------------------------------------
        # group 1
        # -----------------------------------------------------------------------

        conv1 = layers.conv2d(
            inputs=inputs_norm,
            data_format="NHWC",
            num_outputs=64, #BOB was 32,
            kernel_size=5, #BOB was 8, #8
            stride=5, #BOB was 4,#4
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv1" % name,
            trainable=self.trainable
        )

        conv1_norm = tf.layers.batch_normalization(conv1, training=self.params['use_batchnorm'], renorm=True)

        # Stride 5 reduces to 20x20x64 image
        print("policy.py DeepDensePolicy.init after conv1 stride 5 shape {}".format(conv1.shape))


        conv2a = layers.conv2d(
            inputs=conv1_norm,
            data_format="NHWC",
            num_outputs=128,
            kernel_size=3, #4
            stride=1,#2,#
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv2a" % name,
            trainable=self.trainable
        )

        conv2a_norm = tf.layers.batch_normalization(conv2a, training=self.params['use_batchnorm'], renorm=True)


        conv2b = layers.conv2d(
            inputs=conv2a_norm,
            data_format="NHWC",
            num_outputs=128,
            kernel_size=3,
            stride=1,
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv2b" % name,
            trainable=self.trainable
        )


        conv2b_norm = tf.layers.batch_normalization(conv2b, training=self.use_batchnorm, renorm=True)


        conv2c = layers.conv2d(
            inputs=conv2b_norm,
            data_format="NHWC",
            num_outputs=128,
            kernel_size=3,
            stride=1,
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv2c" % name,
            trainable=self.trainable
        )


        conv2c_norm = tf.layers.batch_normalization(conv2c, training=self.use_batchnorm, renorm=True)


        #pool1 = tf.layers.max_pooling2d(conv2c_norm, 2, 2, scope="%s/pool1" % name)
        pool1 = tf.contrib.layers.max_pool2d(conv2c_norm, 2, 2, scope="%s/pool1" % name)

        print("policy.py DeepDensePolicy.init after pool1 shape {}".format(pool1.shape))


        pool1_norm = tf.layers.batch_normalization(pool1, training=self.use_batchnorm, renorm=True)


        # -----------------------------------------------------------------------
        # group 2
        # -----------------------------------------------------------------------

        # Pooling spatial dimension is now 10x10

        conv5a = layers.conv2d(
            inputs=pool1_norm,
            data_format="NHWC",
            num_outputs=128,
            kernel_size=3,
            stride=1,
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv5a" % name,
            trainable=self.trainable
        )

        conv5a_norm = tf.layers.batch_normalization(conv5a, training=self.use_batchnorm, renorm=True)

        conv5b = layers.conv2d(
            inputs=conv5a_norm,
            data_format="NHWC",
            num_outputs=128,
            kernel_size=3,
            stride=1,
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv5b" % name,
            trainable=self.trainable
        )

        conv5b_norm = tf.layers.batch_normalization(conv5b, training=self.use_batchnorm, renorm=True)

        conv7 = layers.conv2d(
            inputs=conv5b_norm,
            data_format="NHWC",
            num_outputs=128,
            kernel_size=3,
            stride=1,
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv7" % name,
            trainable=self.trainable
        )

        conv7_norm = tf.layers.batch_normalization(conv7, training=self.use_batchnorm, renorm=True)

        #pool2 = tf.layers.max_pooling2d(conv5b, 2, 2, scope="%s/pool2" % name)
        pool2 = tf.contrib.layers.max_pool2d(conv7_norm, 2, 2, scope="%s/pool2" % name)
        print("policy.py DeepDensePolicy.init after pool2 shape {}".format(pool2.shape))

        pool2_norm = tf.layers.batch_normalization(pool2, training=self.use_batchnorm, renorm=True)

        # Now pooling will give us roughly ~ 5x5

        conv8a = layers.conv2d(
            inputs=pool2_norm,
            data_format="NHWC",
            num_outputs=128,
            kernel_size=3,
            stride=1,
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv8a" % name,
            trainable=self.trainable
        )

        conv8a_norm = tf.layers.batch_normalization(conv8a, training=self.use_batchnorm, renorm=True)

        conv8b = layers.conv2d(
            inputs=conv8a_norm,
            data_format="NHWC",
            num_outputs=128,
            kernel_size=3,
            stride=1,
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv8b" % name,
            trainable=self.trainable
        )

        conv8b_norm = tf.layers.batch_normalization(conv8b, training=self.use_batchnorm, renorm=True)

        conv10 = layers.conv2d(
            inputs=conv8b_norm,
            data_format="NHWC",
            num_outputs=128,
            kernel_size=3,
            stride=1,
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv10" % name,
            trainable=self.trainable
        )

        conv10_norm = tf.layers.batch_normalization(conv10, training=self.use_batchnorm, renorm=True)

        print("policy.py DeepDensePolicy.init after conv10 shape {}".format(conv10.shape))


        # -----------------------------------------------------------------------
        # Pool and flatten different scales
        # -----------------------------------------------------------------------


        conv2c_all_pooled = tf.contrib.layers.max_pool2d(conv2c_norm, 20, 20, scope="%s/conv2c_all_pooled" % name)

        conv2c_flat = layers.flatten( conv2c_all_pooled, scope="%s/conv2c_flat" % name )
        print("policy.py DeepDensePolicy.init conv2c_flat shape {}".format(conv2c_flat.shape))


        conv7_all_pooled = tf.contrib.layers.max_pool2d(conv7_norm, 10, 10, scope="%s/conv7_all_pooled" % name)

        conv7_flat = layers.flatten( conv7_all_pooled,scope="%s/conv7_flat" % name )
        print("policy.py DeepDensePolicy.init conv7_flat shape {}".format(conv7_flat.shape))


        conv10_all_pooled = tf.contrib.layers.max_pool2d(conv10_norm, 5, 5, scope="%s/conv10_all_pooled" % name)

        conv10_flat = layers.flatten( conv10_all_pooled,scope="%s/conv10_flat" % name )
        print("policy.py DeepDensePolicy.init conv10_flat shape {}".format(conv10_flat.shape))


        # -----------------------------------------------------------------------
        # Concatenate different scales
        # -----------------------------------------------------------------------

        #multi_scale = tf.concat([ conv2c_flat, conv7_flat, conv10_flat, conv13_flat ], axis=1)
        multi_scale = tf.concat([ conv2c_flat, conv7_flat, conv10_flat], axis=1)

        print("policy.py DeepDensePolicy.init after multi_scale shape {}".format(multi_scale.shape))

        # Downstream code expects a 3D tensor of shape batch x X x Y so we just arbitrarily add a dimension here

        multi_scale_3d = tf.reshape(multi_scale, [-1, 1, multi_scale.shape[1]])

        print("policy.py DeepDensePolicy.init after multi_scale_3d shape {}".format(multi_scale_3d.shape))

        return multi_scale_3d



class DeepDensePolicy2(FullyConvPolicy):

    """This model has 10 conv layers (instead of DeepFullyConvPolicy's 7).
       An architecture inspired by simplenet v2 https://arxiv.org/pdf/1608.06037.pdf
       however we do not implement dropout and currently batch_norm is turned off until
       we move it to the other side of the Relu - see

        https://towardsdatascience.com/how-to-use-batch-normalization-with-tensorflow-and-tf-keras-to-train-deep-neural-networks-faster-60ba4d054b73

       and incorporate 'dense' style connections so that the last layer can see multiple scales.
       These are implemented by concatenating layers 2c,7 and 10.


       100x100x3       input                   RGB Channels
                         |
       20x20x64        conv1                   (stride 5 does collapse)
                         |
                       conv2a
                       conv2b
       20x20x128       conv2c----------+
                         |             |
                       pool1           |       pool 2x2 halfs size
                         |             |
                       conv5a          |
                       conv5b          |
       10x10x128       conv7-----+     |
                         |       |     |
                       pool2     |     |       pool 2x2 halfs size
                         |       |     |
                       conv8a    |     |
                       conv8b    |     |
       5x5x128         conv10    |     |
                         |       |     |
                       pool3     |     |       pool 2x2 halfs size
                         |       |     |
                       conv11    |     |
                       conv12    |     |
       5x5x128         conv13    |     |
                         |       |     |
       3x1x128         pool    pool   pool     complete pooling over spatial dimensions
                         |       |     |
       1x384           multi_scale_concat

       """

    def _build_convs(self, inputs, name):

        # 100x100x3 RGB input image ( 20x5 x 20x5 )

        print("---------------------------------------------------------------------")
        print("DeepDensePolicy2 builder")
        print("---------------------------------------------------------------------")
        print("policy.py DeepDensePolicy.init use_batchnorm flag ".format(self.use_batchnorm))
        print("policy.py DeepDensePolicy.init input shape {}".format(inputs.shape))



        inputs_norm = tf.layers.batch_normalization(inputs, training=self.use_batchnorm, renorm=True)


        # -----------------------------------------------------------------------
        # group 1
        # -----------------------------------------------------------------------

        conv1 = layers.conv2d(
            inputs=inputs_norm,
            data_format="NHWC",
            num_outputs=64, #BOB was 32,
            kernel_size=5, #BOB was 8, #8
            stride=5, #BOB was 4,#4
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv1" % name,
            trainable=self.trainable
        )

        conv1_norm = tf.layers.batch_normalization(conv1, training=self.params['use_batchnorm'], renorm=True)

        # Stride 5 reduces to 20x20x64 image
        print("policy.py DeepDensePolicy.init after conv1 stride 5 shape {}".format(conv1.shape))


        conv2a = layers.conv2d(
            inputs=conv1_norm,
            data_format="NHWC",
            num_outputs=128,
            kernel_size=3, #4
            stride=1,#2,#
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv2a" % name,
            trainable=self.trainable
        )

        conv2a_norm = tf.layers.batch_normalization(conv2a, training=self.params['use_batchnorm'], renorm=True)


        conv2b = layers.conv2d(
            inputs=conv2a_norm,
            data_format="NHWC",
            num_outputs=128,
            kernel_size=3,
            stride=1,
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv2b" % name,
            trainable=self.trainable
        )


        conv2b_norm = tf.layers.batch_normalization(conv2b, training=self.use_batchnorm, renorm=True)


        conv2c = layers.conv2d(
            inputs=conv2b_norm,
            data_format="NHWC",
            num_outputs=128,
            kernel_size=3,
            stride=1,
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv2c" % name,
            trainable=self.trainable
        )


        conv2c_norm = tf.layers.batch_normalization(conv2c, training=self.use_batchnorm, renorm=True)


        #pool1 = tf.layers.max_pooling2d(conv2c_norm, 2, 2, scope="%s/pool1" % name)
        pool1 = tf.contrib.layers.max_pool2d(conv2c_norm, 2, 2, scope="%s/pool1" % name)

        print("policy.py DeepDensePolicy.init after pool1 shape {}".format(pool1.shape))


        pool1_norm = tf.layers.batch_normalization(pool1, training=self.use_batchnorm, renorm=True)


        # -----------------------------------------------------------------------
        # group 2
        # -----------------------------------------------------------------------

        # Pooling spatial dimension is now 10x10

        conv5a = layers.conv2d(
            inputs=pool1_norm,
            data_format="NHWC",
            num_outputs=128,
            kernel_size=3,
            stride=1,
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv5a" % name,
            trainable=self.trainable
        )

        conv5a_norm = tf.layers.batch_normalization(conv5a, training=self.use_batchnorm, renorm=True)

        conv5b = layers.conv2d(
            inputs=conv5a_norm,
            data_format="NHWC",
            num_outputs=128,
            kernel_size=3,
            stride=1,
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv5b" % name,
            trainable=self.trainable
        )

        conv5b_norm = tf.layers.batch_normalization(conv5b, training=self.use_batchnorm, renorm=True)

        conv7 = layers.conv2d(
            inputs=conv5b_norm,
            data_format="NHWC",
            num_outputs=128,
            kernel_size=3,
            stride=1,
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv7" % name,
            trainable=self.trainable
        )

        conv7_norm = tf.layers.batch_normalization(conv7, training=self.use_batchnorm, renorm=True)

        #pool2 = tf.layers.max_pooling2d(conv5b, 2, 2, scope="%s/pool2" % name)
        pool2 = tf.contrib.layers.max_pool2d(conv7_norm, 2, 2, scope="%s/pool2" % name)
        print("policy.py DeepDensePolicy.init after pool2 shape {}".format(pool2.shape))

        pool2_norm = tf.layers.batch_normalization(pool2, training=self.use_batchnorm, renorm=True)

        # Now pooling will give us roughly ~ 5x5

        conv8a = layers.conv2d(
            inputs=pool2_norm,
            data_format="NHWC",
            num_outputs=128,
            kernel_size=3,
            stride=1,
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv8a" % name,
            trainable=self.trainable
        )

        conv8a_norm = tf.layers.batch_normalization(conv8a, training=self.use_batchnorm, renorm=True)

        conv8b = layers.conv2d(
            inputs=conv8a_norm,
            data_format="NHWC",
            num_outputs=128,
            kernel_size=3,
            stride=1,
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv8b" % name,
            trainable=self.trainable
        )

        conv8b_norm = tf.layers.batch_normalization(conv8b, training=self.use_batchnorm, renorm=True)

        conv10 = layers.conv2d(
            inputs=conv8b_norm,
            data_format="NHWC",
            num_outputs=128,
            kernel_size=3,
            stride=1,
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv10" % name,
            trainable=self.trainable
        )

        conv10_norm = tf.layers.batch_normalization(conv10, training=self.use_batchnorm, renorm=True)

        print("policy.py DeepDensePolicy.init after conv10 shape {}".format(conv10.shape))




        pool3 = tf.contrib.layers.max_pool2d(conv10_norm, 2, 2, scope="%s/pool3" % name)
        print("policy.py DeepDensePolicy.init after pool3 shape {}".format(pool3.shape))


        # -----------------------------------------------------------------------
        # group 3
        # -----------------------------------------------------------------------

        conv11 = layers.conv2d(
            inputs=pool3,
            data_format="NHWC",
            num_outputs=128,
            kernel_size=3,
            stride=1,
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv11" % name,
            trainable=self.trainable
        )

        conv12 = layers.conv2d(
            inputs=conv11,
            data_format="NHWC",
            num_outputs=128,
            kernel_size=3,
            stride=1,
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv12" % name,
            trainable=self.trainable
        )

        conv13 = layers.conv2d(
            inputs=conv12,
            data_format="NHWC",
            num_outputs=128,
            kernel_size=3,
            stride=1,
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv13" % name,
            trainable=self.trainable
        )




        # -----------------------------------------------------------------------
        # Pool and flatten different scales
        # -----------------------------------------------------------------------


        conv2c_all_pooled = tf.contrib.layers.max_pool2d(conv2c_norm, 20, 20, scope="%s/conv2c_all_pooled" % name)

        conv2c_flat = layers.flatten( conv2c_all_pooled, scope="%s/conv2c_flat" % name )
        print("policy.py DeepDensePolicy.init conv2c_flat shape {}".format(conv2c_flat.shape))


        conv7_all_pooled = tf.contrib.layers.max_pool2d(conv7_norm, 10, 10, scope="%s/conv7_all_pooled" % name)

        conv7_flat = layers.flatten( conv7_all_pooled,scope="%s/conv7_flat" % name )
        print("policy.py DeepDensePolicy.init conv7_flat shape {}".format(conv7_flat.shape))


        conv10_all_pooled = tf.contrib.layers.max_pool2d(conv10_norm, 5, 5, scope="%s/conv10_all_pooled" % name)

        conv10_flat = layers.flatten( conv10_all_pooled,scope="%s/conv10_flat" % name )
        print("policy.py DeepDensePolicy.init conv10_flat shape {}".format(conv10_flat.shape))


        conv13_all_pooled = tf.contrib.layers.max_pool2d(conv13, 2, 2, scope="%s/conv13_all_pooled" % name)

        conv13_flat = layers.flatten( conv13_all_pooled,scope="%s/conv13_flat" % name )
        print("policy.py DeepDensePolicy.init conv13_flat shape {}".format(conv13_flat.shape))


        # -----------------------------------------------------------------------
        # Concatenate different scales
        # -----------------------------------------------------------------------

        #multi_scale = tf.concat([ conv2c_flat, conv7_flat, conv10_flat, conv13_flat ], axis=1)
        multi_scale = tf.concat([ conv2c_flat, conv7_flat, conv10_flat, conv13_flat], axis=1)

        print("policy.py DeepDensePolicy.init after multi_scale shape {}".format(multi_scale.shape))

        # Downstream code expects a 3D tensor of shape batch x X x Y so we just arbitrarily add a dimension here

        multi_scale_3d = tf.reshape(multi_scale, [-1, 1, multi_scale.shape[1]])

        print("policy.py DeepDensePolicy.init after multi_scale_3d shape {}".format(multi_scale_3d.shape))

        return multi_scale_3d


class MetaPolicy:
    """
    Meta Policy with recurrency on observations, actions and rewards
    """

    def __init__(self,
                 agent,
                 trainable: bool = True
                 ):
        # type agent: ActorCriticAgent
        self.placeholders = agent.placeholders
        self.trainable = trainable
        self.unittype_emb_dim = agent.unit_type_emb_dim
        self.num_actions = agent.num_actions

    def _build_convs(self, inputs, name):
        conv1 = layers.conv2d(
            inputs=inputs,
            data_format="NHWC",
            num_outputs=32,
            kernel_size=8,  # 8
            stride=4,  # 4
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv1" % name,
            trainable=self.trainable
        )
        conv2 = layers.conv2d(
            inputs=conv1,
            data_format="NHWC",
            num_outputs=64,
            kernel_size=4,  # 4
            stride=1,  # 2,#
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv2" % name,
            trainable=self.trainable
        )
        # conv3 = layers.conv2d(
        #     inputs=conv2,
        #     data_format="NHWC",
        #     num_outputs=64,
        #     kernel_size=3,
        #     stride=1,
        #     padding='SAME',
        #     activation_fn=tf.nn.relu,
        #     scope="%s/conv3" % name,
        #     trainable=self.trainable
        # )

        if self.trainable:
            layers.summarize_activation(conv1)
            layers.summarize_activation(conv2)
            # layers.summarize_activation(conv3)

        return conv2
        # return conv3

    def build(self):
        screen_px = tf.cast(self.placeholders.rgb_screen,
                            tf.float32) / 255.  # rgb_screen are integers (0-255) and here we convert to float and normalize
        alt_px = tf.cast(self.placeholders.alt_view, tf.float32) / 255.
        self.screen_output = self._build_convs(screen_px, "screen_network")
        self.alt_output = self._build_convs(alt_px, "alt_network")
        self.map_output = tf.concat([self.screen_output, self.alt_output], axis=2) # should be 3

        self.prev_rewards = tf.placeholder(shape=[None, 1], dtype=tf.float32) # num_envs x num_steps (it was [None,1])
        self.prev_actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.prev_actions_onehot = tf.one_hot(self.prev_actions, self.num_actions, dtype=tf.float32)
        # self.prev_actions_onehot = tf.squeeze(self.prev_actions_onehot,[1])
        # self.prev_actions_onehot = layers.embed_sequence(
        #                                 self.prev_actions,
        #                                 vocab_size=a_size,  # 1850
        #                                 embed_dim=5,  # 5
        #                                 scope="unit_type_emb",
        #                                 trainable=self.training
        # )

        hidden = tf.concat([layers.flatten(self.map_output), self.prev_actions_onehot, self.prev_rewards], 1)
        # hidden = layers.flatten(self.map_output)
        # Below, feed the batch_size!
        # self.batch_size = tf.placeholder(shape=[], dtype=tf.int32)#.shape(self.placeholders.rgb_screen)
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)
        # lstm_cell.trainable = self.trainable

        # Initialization: you create an initial state which will be fed as self.state in the unfolded net. So you have to define the self.state_init AND the self.state
        # or maybe have only the self.state defined and assined?
        # init_vars = lstm_cell.zero_state(2, tf.float32)
        # init_c = tf.Variable(init_vars.c, trainable=self.trainable)
        # init_h = tf.Variable(init_vars.h, trainable=self.trainable)
        # self.state_init = tf.contrib.rnn.LSTMStateTuple(init_c, init_h)
        #
        # state_vars = lstm_cell.zero_state(2, tf.float32)
        # state_c = tf.Variable(state_vars.c, trainable=self.trainable)
        # state_h = tf.Variable(state_vars.h, trainable=self.trainable)
        # self.state = tf.contrib.rnn.LSTMStateTuple(state_c, state_h)
        # self.state = (state_c, state_h)


        c_init = np.zeros((2, lstm_cell.state_size.c), np.float32)
        h_init = np.zeros((2, lstm_cell.state_size.h), np.float32)
        # # (or bring the batch_size from out) The following should be defined in the runner and you need a self before the lstm_cell. Because you get a numpy array below you need the batch size
        self.state_init = [c_init, h_init]#lstm_cell.zero_state(2, dtype=tf.float32)# Its already a tensor with a numpy array full of zeros
        self.c_in = tf.placeholder(tf.float32, [2, lstm_cell.state_size.c])
        self.h_in = tf.placeholder(tf.float32, [2, lstm_cell.state_size.h])
        self.state_in = (self.c_in, self.h_in) # You need this so from outside you can feed the two placeholders
        rnn_in = tf.reshape(hidden,[-1,1,80017])#tf.expand_dims(hidden, [0]) # 1 is the timestep, if you have more you might need -1 also there
        # step_size = tf.shape(self.prev_rewards)[:1]
        state_in = tf.contrib.rnn.LSTMStateTuple(self.c_in, self.h_in)
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm_cell, rnn_in, initial_state=state_in, time_major=False) #sequence_length=step_size,
        # # lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
        # #     lstm_cell, rnn_in, initial_state=self.state_init,time_major=False)
        lstm_c, lstm_h = lstm_state
        self.state_out = (lstm_c, lstm_h)#(lstm_c[:1, :], lstm_h[:1, :])
        # self.state_out = lstm_state

        # layer = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.7)
        # lstm_outputs, self.new_state = tf.nn.dynamic_rnn(lstm_cell, rnn_in, initial_state=self.state, dtype=tf.float32)
        #
        # self.trained_state_c = tf.assign(self.state[0], self.new_state[0])
        # trained_state_h = tf.assign(self.state[1], self.new_state[1])
        # self.state_out = tf.contrib.rnn.LSTMStateTuple(self.trained_state_c, trained_state_h) # the new state will be get in the net as self.state

        rnn_out = tf.reshape(lstm_outputs, [-1, 256])

        # Add layer normalization
        # fc1_ = layers.layer_norm(rnn_out,trainable=self.trainable)

        # map_output_flat = layers.flatten(self.map_output)
        ''' COMBOS '''
        #TODO: Omit the fc layer and go straight to the action and value layer:
        # Just use the rnn_out as input to these layers
        #TODO: Use the layer normalization:
        # (MINE) This is the last layer (fully connected -fc) for the non-spatial (categorical) actions
        fc1 = layers.fully_connected(
            rnn_out,
            num_outputs=256,
            activation_fn=tf.nn.relu,
            scope="fc1",
            trainable=self.trainable
        )

        # Add layer normalization
        # fc1_ = layers.layer_norm(fc1,trainable=self.trainable)

        # (MINE) From the previous layer you extract action_id_probs (non spatial - categorical - actions) and value
        # estimate
        action_id_probs = layers.fully_connected(
            fc1, #rnn_out
            num_outputs=self.num_actions,  # len(actions.FUNCTIONS),
            activation_fn=tf.nn.softmax,
            scope="action_id",
            trainable=self.trainable
        )
        value_estimate = tf.squeeze(layers.fully_connected(
            # squeeze removes a dimension of 1 elements. e.g.: [n_batches,1,value_est_dim]--->[n_batches,value_est_dim]
            fc1, #rnn_out
            num_outputs=1,
            activation_fn=None,
            scope='value',
            trainable=self.trainable
        ), axis=1)

        def logclip(x):
            return tf.log(tf.clip_by_value(x, 1e-12, 1.0))

        action_id_log_probs = logclip(action_id_probs)

        self.value_estimate = value_estimate
        self.action_id_probs = action_id_probs
        self.action_id_log_probs = action_id_log_probs
        return self

class RelationalPolicy:
    """
    Relational RL from Attention is All you need and Relational RL for SCII and BoxWorld
    """

    def __init__(self,
                 agent,
                 trainable: bool = True
                 ):
        # type agent: ActorCriticAgent
        self.placeholders = agent.placeholders
        self.trainable = trainable
        self.unittype_emb_dim = agent.unit_type_emb_dim
        self.num_actions = agent.num_actions
        self.MHDPA_blocks = 2

    def _build_convs(self, inputs, name):
        conv1 = layers.conv2d(
            inputs=inputs,
            data_format="NHWC",
            num_outputs=32,#32,#12,
            kernel_size=8,#2
            stride=4,#1
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv1" % name,
            trainable=self.trainable
        )
        conv2 = layers.conv2d(
            inputs=conv1,
            data_format="NHWC",
            num_outputs=64,#64, #24
            kernel_size=4, #2
            stride=1,#1
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv2" % name,
            trainable=self.trainable
        )

        if self.trainable:
            layers.summarize_activation(conv1)
            layers.summarize_activation(conv2)
            # layers.summarize_activation(conv3)

        return conv2
        # return conv3

    def build(self):
        screen_px = tf.cast(self.placeholders.rgb_screen, tf.float32) / 255. # rgb_screen are integers (0-255) and here we convert to float and normalize
        alt_px = tf.cast(self.placeholders.alt_view, tf.float32) / 255.
        self.screen_output = self._build_convs(screen_px, "screen_network")
        self.alt_output = self._build_convs(alt_px, "alt_network")

        self.cnn_outputs = tf.concat([self.screen_output, self.alt_output], axis=3) # if you use 2 then you calculate relations between the ego and the allo

        # self.cnn_outputs = tf.layers.max_pooling2d(
        #     self.cnn_outputs,
        #     3,
        #     2,
        #     padding='valid',
        #     data_format='channels_last',
        #     name='max_pool_for_inputs'
        # ) #for 3,2 then out is 12,12,128
        # with tf.device("/cpu:0"):
        #     self.relation = build_Relation(self.cnn_outputs)

        shape = self.cnn_outputs.get_shape().as_list()
        channels = shape[3]
        dim = shape[1]
        self.relation = tf.reshape(self.cnn_outputs, [-1, shape[1] * shape[2], shape[3]])
        # Stacked MHDPA Blocks with shared weights
        for i in range(self.MHDPA_blocks):
            with tf.variable_scope("num_blocks_{}".format(i)):
                self.relation, self.attention_w = multihead_attention(queries=self.relation,
                                                 keys=self.relation,
                                                 num_units=64,  # how many dims you want the keys to be, None gives you the dims of ur entity. Should be 528/8=64+2
                                                 num_heads=2,
                                                 trainable = self.trainable,
                                                 channels = channels
                                                 # dropout_rate=hp.dropout_rate,
                                                 # is_training=is_training,
                                                 # causality=False
                                                 ) # NUM UNITS YOU DONT NEED πολλαπλασια του conv output but πολλαπλασια του num_heads!!! Deepmind use 64* (2-4 heads)

        # self.cnn1d = feedforward(self.MHDPA, num_units=[4 * 66, 66])  # You can use MLP instead of conv1d
        # The max pooling which converts a nxnxk to a k vector
        self.relation = tf.reshape(self.relation, [-1, dim, dim, channels]) # [-1, 13, 13, 66] [-1, 25, 25, 130]
        self.max_pool = tf.layers.max_pooling2d(self.relation, dim, dim)
        map_output_flat = layers.flatten(self.max_pool)
        # map_output_flat = layers.flatten(self.spatial_softmax)

        fc1 = layers.fully_connected(
            map_output_flat,
            num_outputs=256,
            activation_fn=tf.nn.relu,
            scope="fc1",
            trainable=self.trainable
        )
        fc1 = layers.layer_norm(fc1, trainable=self.trainable)
        # fc2 = layers.fully_connected(
        #     fc1,
        #     num_outputs=256,
        #     activation_fn=tf.nn.relu,
        #     scope="fc2",
        #     trainable=self.trainable
        # )
        # fc3 = layers.fully_connected(
        #     fc2,
        #     num_outputs=256,
        #     activation_fn=tf.nn.relu,
        #     scope="fc3",
        #     trainable=self.trainable
        # )
        # fc4 = layers.fully_connected(
        #     fc3,
        #     num_outputs=256,
        #     activation_fn=tf.nn.relu,
        #     scope="fc4",
        #     trainable=self.trainable
        # )

        # Policy
        action_id_probs = layers.fully_connected(
            fc1,
            num_outputs=self.num_actions,  # actions are from 0 to num_actions-1 || len(actions.FUNCTIONS),
            activation_fn=tf.nn.softmax,
            scope="action_id",
            trainable=self.trainable
        )
        value_estimate = tf.squeeze(layers.fully_connected(
            # squeeze removes a dimension of 1 elements. e.g.: [n_batches,1,value_est_dim]--->[n_batches,value_est_dim]
            fc1,
            num_outputs=1,
            activation_fn=None,
            scope='value',
            trainable=self.trainable
        ), axis=1)

        def logclip(x):
            return tf.log(tf.clip_by_value(x, 1e-12, 1.0))

        # non-available actions get log(1e-10) value but that's ok because it's never used
        action_id_log_probs = logclip(action_id_probs)

        self.value_estimate = value_estimate
        self.action_id_probs = action_id_probs
        self.action_id_log_probs = action_id_log_probs
        return self

