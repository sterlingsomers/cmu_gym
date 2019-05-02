import tensorflow as tf
import numpy as np
from pysc2.lib import actions
from pysc2.lib.features import SCREEN_FEATURES, MINIMAP_FEATURES
from tensorflow.contrib import layers


class FullyConvPolicy:
    """
    FullyConv network structure from https://arxiv.org/pdf/1708.04782.pdf
    Some implementation ideas are borrowed from https://github.com/xhujoy/pysc2-agents
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
            kernel_size=8, #8
            stride=4,#4
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv1" % name,
            trainable=self.trainable
        )
        conv2 = layers.conv2d(
            inputs=conv1,
            data_format="NHWC",
            num_outputs=64,
            kernel_size=4, #4
            stride=1,#2,#
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

        screen_px = tf.cast(self.placeholders.rgb_screen, tf.float32) / 255. # rgb_screen are integers (0-255) and here we convert to float and normalize
        alt_px = tf.cast(self.placeholders.alt_view, tf.float32) / 255.
        self.screen_output = self._build_convs(screen_px, "screen_network")
        self.alt_output = self._build_convs(alt_px, "alt_network")
        
        self.map_output = tf.concat([self.screen_output, self.alt_output], axis=3)

        # BUILD CONVLSTM
        self.rnn_in = tf.reshape(self.map_output, [1, -1, 25, 25, 128])
        self.cell = tf.contrib.rnn.Conv2DLSTMCell(input_shape=[25, 25, 1],  # input dims
                                                  kernel_shape=[3, 3],  # for a 3 by 3 conv
                                                  output_channels=128)  # number of feature maps
        c_init = np.zeros((1, 25, 25, 128), np.float32)
        h_init = np.zeros((1, 25, 25, 128), np.float32)
        self.state_init = [c_init, h_init]
        step_size = tf.shape(self.map_output)[:1]  # Get step_size from input dimensions
        c_in = tf.placeholder(tf.float32, [None, 25, 25, 128])
        h_in = tf.placeholder(tf.float32, [None, 25, 25, 128])
        self.state_in = (c_in, h_in)
        state_in = tf.nn.rnn_cell.LSTMStateTuple(c_in, h_in)
        self.step_size = tf.placeholder(tf.float32, [1])
        (self.outputs, self.state) = tf.nn.dynamic_rnn(self.cell, self.rnn_in, initial_state=state_in,
                                                       sequence_length=step_size, time_major=False,
                                                       dtype=tf.float32)
        lstm_c, lstm_h = self.state
        self.state_out = (lstm_c[:1, :], lstm_h[:1, :])

        rnn_out = tf.reshape(self.outputs, [-1, 25, 25, 128])

        map_output_flat = tf.reshape(self.outputs, [-1, 80000])# 25*25*128 OR FLATTEN THE rnn_out like u were doing with the concated volume
        # (MINE) This is the last layer (fully connected -fc) for the non-spatial (categorical) actions
        fc1 = layers.fully_connected(
            map_output_flat,
            num_outputs=256,
            activation_fn=tf.nn.relu,
            scope="fc1",
            trainable=self.trainable
        )
        # (MINE) From the previous layer you extract action_id_probs (non spatial - categorical - actions) and value
        # estimate
        action_id_probs = layers.fully_connected(
            fc1,
            num_outputs=self.num_actions,#len(actions.FUNCTIONS),
            activation_fn=tf.nn.softmax,
            scope="action_id",
            trainable=self.trainable
        )
        value_estimate = tf.squeeze(layers.fully_connected( # squeeze removes a dimension of 1 elements. e.g.: [n_batches,1,value_est_dim]--->[n_batches,value_est_dim]
            fc1,
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
