# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains some basic model components"""

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell


class RNNEncoder(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.

    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code uses a bidirectional GRU, but you could experiment with other types of RNN.
    """

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks, variable_scope='RNNEncoder'):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope(variable_scope):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(
                self.rnn_cell_fw,
                self.rnn_cell_bw,
                inputs,
                input_lens,
                dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out


class SimpleSoftmaxLayer(object):
    """
    Module to take set of hidden states, (e.g. one for each context location),
    and return probability distribution over those states.
    """

    def __init__(self):
        pass

    def build_graph(self, inputs, masks, scope=None, reuse=None):
        """
        Applies one linear downprojection layer, then softmax.

        Inputs:
          inputs: Tensor shape (batch_size, seq_len, hidden_size)
          masks: Tensor shape (batch_size, seq_len)
            Has 1s where there is real input, 0s where there's padding.

        Outputs:
          logits: Tensor shape (batch_size, seq_len)
            logits is the result of the downprojection layer, but it has -1e30
            (i.e. very large negative number) in the padded locations
          prob_dist: Tensor shape (batch_size, seq_len)
            The result of taking softmax over logits.
            This should have 0 in the padded locations, and the rest should sum to 1.
        """
        with vs.variable_scope("SimpleSoftmaxLayer"):

            # Linear downprojection layer
            logits = tf.contrib.layers.fully_connected(
                inputs, num_outputs=1, activation_fn=None, reuse=reuse, scope=scope) # shape (batch_size, seq_len, 1)
            logits = tf.squeeze(logits, axis=[2]) # shape (batch_size, seq_len)

            # Take softmax over sequence
            masked_logits, prob_dist = masked_softmax(logits, masks, 1)

            return masked_logits, prob_dist


class BidafAttn(object):
    """Model for attention described in Bidaf(bi-directional attention flow).
    """
    def __init__(self, keep_prob, num_keys, value_vec_size):
        """
        Note: The key_vec_size has to be the same as the value_vec_size.

        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          num_keys: int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.num_keys = num_keys
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys, keys_mask):
        """Keys attend to values.
        For each key, return an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, key_vec_size)
          keys_mask: Tensor shape (batch_size, num_keys).
            1s where there's real input, 0s where there's padding

        Outputs:
          output: attention output. Tensor shape (batch_size, num_keys, hidden_size*4).
        """
        with tf.variable_scope('BidafAttn'):
            keys_mask_3d = tf.cast(tf.expand_dims(keys_mask, 2), tf.float32)  # (batch_size, num_keys, 1)
            masked_keys = tf.multiply(keys, keys_mask_3d, name='masked_keys')  # (batch_size, num_keys, value_vec_size)
            values_mask_3d = tf.cast(tf.expand_dims(values_mask, 2), tf.float32)  # (batch_size, num_values, 1)
            masked_values = tf.multiply(values, values_mask_3d, name='masked_values')  # (batch_size, num_values, value_vec_size)

            with tf.variable_scope('SimilarityMatrix'):
                W_sim_cq = tf.get_variable('W_sim_cq', shape=(1, 1, self.value_vec_size))
                W_sim_c = tf.get_variable('W_sim_c', shape=(1, 1, self.value_vec_size))
                W_sim_q = tf.get_variable('W_sim_q', shape=(1, 1, self.value_vec_size))

                similarity_matrix_cq = tf.matmul(W_sim_cq * masked_keys, tf.transpose(masked_values, perm=[0, 2, 1]))  # (batch_size, num_keys, num_values)
                similarity_matrix_c = tf.reduce_sum(masked_keys * W_sim_c, -1, keep_dims=True, name='similarity_matrix_c')  # (batch_size, num_keys, 1)
                similarity_matrix_q = tf.expand_dims(tf.reduce_sum(masked_values * W_sim_q, -1), 1, name='similarity_matrix_q')  # (batch_size, 1, num_values)
                similarity_matrix = similarity_matrix_cq + similarity_matrix_c + similarity_matrix_q  # (batch_size, num_keys, num_values)

            # c2q
            with tf.variable_scope('C2QAttn'):
                c2q_attn_weights = tf.nn.softmax(similarity_matrix, 2, name='c2q_attn_weights')  # (batch_size, num_keys, num_values)
                c2q_attn = tf.matmul(c2q_attn_weights, masked_values) # (batch_size, num_keys, value_vec_size)

            # q2c
            with tf.variable_scope('Q2CAttn'):
                q2c_attn_max = tf.reduce_max(similarity_matrix, axis=2, name='q2c_attn_max')  # (batch_size, num_keys)
                q2c_attn_softmax = tf.expand_dims(tf.nn.softmax(q2c_attn_max), 1, name='q2c_attn_softmax')  # (batch_size, 1, num_keys)
                q2c_attn = tf.matmul(q2c_attn_softmax, masked_keys, name='q2c_attn')  # (batch_size, 1, value_vec_size)

            with tf.variable_scope('SelfAttn'):
                mul_attn = MulAttn(self.keep_prob, self.value_vec_size, self.value_vec_size)
                self_attn = mul_attn.build_graph(keys, keys_mask, keys, keys_mask)  # (batch_size, num_values, value_vec_size)


            blended_reps = tf.concat([
                masked_keys,
                c2q_attn,
                masked_keys * c2q_attn,
                masked_keys * q2c_attn,
                self_attn,
            ], 2)  # (batch_size, num_keys, value_vec_size*5)
            tf.assert_equal(
                blended_reps.get_shape().as_list()[1:],
                [self.num_keys, self.value_vec_size * 5])

            blended_reps = tf.nn.dropout(blended_reps, self.keep_prob)

            return blended_reps


class SelfAttn(object):
    """Self matching attention.
    """
    def __init__(self, keep_prob, value_vec_size):
        self.keep_prob = keep_prob
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask):
        """
        Inputs:
          values: (batch_size, num_values, value_vec_size).
          values_mask: (batch_size, num_values).

        Outputs:
          Returns concat([values, self_attn_values], 2)
        """
        with tf.variable_scope('SelfAttn'):
            self_attn = MulAttn(self.keep_prob, self.value_vec_size, self.value_vec_size)
            self_output = self_attn.build_graph(values, values_mask, values, values_mask)  # (batch_size, num_values, value_vec_size)
            return tf.concat([values, self_output], 2)  # (batch_size, context_len, value_vec_size * 2)


class BasicAttn(object):
    """Module for basic attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("BasicAttn"):

            # Calculate attention distribution
            values_t = tf.transpose(values, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values)
            attn_logits = tf.matmul(keys, values_t) # shape (batch_size, num_keys, num_values)
            attn_logits_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)
            _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # shape (batch_size, num_keys, num_values). take softmax over values

            # Use attention distribution to take weighted sum of values
            output = tf.matmul(attn_dist, values) # shape (batch_size, num_keys, value_vec_size)

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return attn_dist, output


class MulAttn(object):
    """Module for multiplicative attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys, keys_mask):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, key_vec_size)

        Outputs:
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("MulAttn"):
            # Use attention distribution to take weighted sum of values
            W = tf.get_variable('W_attn', shape=(self.key_vec_size, self.value_vec_size))
            keys_mask_3d = tf.cast(tf.expand_dims(keys_mask, 2), tf.float32)
            values_mask_3d = tf.cast(tf.expand_dims(values_mask, 2), tf.float32)
            masked_keys = keys * keys_mask_3d  # (batch_size, num_keys, key_vec_size)
            masked_vals = values * values_mask_3d  # (batch_size, num_values, value_vec_size)

            attn_logits = tf.tensordot(masked_keys, W, [[2], [0]])  # (batch_size, num_keys, self.value_vec_size)
            attn_logits = tf.matmul(attn_logits, tf.transpose(masked_vals, perm=[0, 2, 1]))  # (batch_size, num_keys, num_values)
            # Logits divided by sqrt(dim) proposed from "Attention is all you need".
            # http://ruder.io/deep-learning-nlp-best-practices/index.html#attention
            attn_dist = tf.nn.softmax(attn_logits / tf.sqrt(tf.cast(self.key_vec_size, tf.float32)), 2)

            output = tf.matmul(attn_dist, masked_vals) # (batch_size, num_keys, value_vec_size)

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return output


def masked_softmax(logits, mask, dim):
    """
    Takes masked softmax over given dimension of logits.

    Inputs:
      logits: Numpy array. We want to take softmax over dimension dim.
      mask: Numpy array of same shape as logits.
        Has 1s where there's real data in logits, 0 where there's padding
      dim: int. dimension over which to take softmax

    Returns:
      masked_logits: Numpy array same shape as logits.
        This is the same as logits, but with 1e30 subtracted
        (i.e. very large negative number) in the padding locations.
      prob_dist: Numpy array same shape as logits.
        The result of taking softmax over masked_logits in given dimension.
        Should be 0 in padding locations.
        Should sum to 1 over given dimension.
    """
    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30) # -large where there's padding, 0 elsewhere
    masked_logits = tf.add(logits, exp_mask) # where there's padding, set logits to -large
    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist
