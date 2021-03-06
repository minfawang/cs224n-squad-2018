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

"""This file defines the top-level model"""

from __future__ import absolute_import
from __future__ import division

import time
import logging
import os
import sys
import heapq

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import embedding_ops

from evaluate import exact_match_score, f1_score
from data_batcher import get_batch_generator
from pretty_print import print_example
from modules import RNNEncoder, SimpleSoftmaxLayer, BasicAttn, MulAttn, BidafAttn, CoAttnLite

logging.basicConfig(level=logging.INFO)


class QAModel(object):
    """Top-level Question Answering module"""

    def __init__(self, FLAGS, id2word, word2id, emb_matrix, char2id ,id2char):
        """
        Initializes the QA model.

        Inputs:
          FLAGS: the flags passed in from main.py
          id2word: dictionary mapping word idx (int) to word (string)
          word2id: dictionary mapping word (string) to word idx (int)
          emb_matrix: numpy array shape (400002, embedding_size) containing pre-traing GloVe embeddings
        """
        print "Initializing the QAModel..."
        self.FLAGS = FLAGS
        self.id2word = id2word
        self.word2id = word2id
        self.char2id = char2id
        self.id2char = id2char

        # Add all parts of the graph
        with tf.variable_scope("QAModel", initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, uniform=True)):
            self.add_placeholders()
            self.add_embedding_layer(emb_matrix)
            if self.FLAGS.enable_cnn:
              self.add_char_embedding_layer()
            self.build_graph()
            self.add_loss()

        # Define trainable parameters, gradient, gradient norm, and clip by gradient norm
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        self.gradient_norm = tf.global_norm(gradients)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.max_gradient_norm)
        self.param_norm = tf.global_norm(params)

        # Define optimizer and updates
        # (updates is what you need to fetch in session.run to do a gradient update)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate) # you can try other optimizers
        self.updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

        # Define savers (for checkpointing) and summaries (for tensorboard)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.keep)
        self.bestmodel_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        self.summaries = tf.summary.merge_all()


    def add_placeholders(self):
        """
        Add placeholders to the graph. Placeholders are used to feed in inputs.
        """
        # Add placeholders for inputs.
        # These are all batch-first: the None corresponds to batch_size and
        # allows you to run the same model with variable batch_size
        self.context_ids = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len])
        self.context_mask = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len])
        self.qn_ids = tf.placeholder(tf.int32, shape=[None, self.FLAGS.question_len])
        self.qn_mask = tf.placeholder(tf.int32, shape=[None, self.FLAGS.question_len])
        self.ans_span = tf.placeholder(tf.int32, shape=[None, 2])

        ## placeholder for character level embeddings.
        ## shape (batch_size, context_len, word_len)
        if self.FLAGS.enable_cnn:
            self.context_char_ids = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len, self.FLAGS.word_len])
            self.qn_char_ids = tf.placeholder(tf.int32, shape=[None, self.FLAGS.question_len, self.FLAGS.word_len])

        # Add a placeholder to feed in the keep probability (for dropout).
        # This is necessary so that we can instruct the model to use dropout when training, but not when testing
        self.keep_prob = tf.placeholder_with_default(1.0, shape=())


    def add_embedding_layer(self, emb_matrix):
        """
        Adds word embedding layer to the graph.

        Inputs:
          emb_matrix: shape (400002, embedding_size).
            The GloVe vectors, plus vectors for PAD and UNK.
        """
        with vs.variable_scope("embeddings"):

            # Note: the embedding matrix is a tf.constant which means it's not a trainable parameter
            embedding_matrix = tf.constant(emb_matrix, dtype=tf.float32, name="emb_matrix") # shape (400002, embedding_size)

            # Get the word embeddings for the context and question,
            # using the placeholders self.context_ids and self.qn_ids
            self.context_embs = embedding_ops.embedding_lookup(embedding_matrix, self.context_ids) # shape (batch_size, context_len, embedding_size)
            self.qn_embs = embedding_ops.embedding_lookup(embedding_matrix, self.qn_ids) # shape (batch_size, question_len, embedding_size)


    def add_char_embedding_layer(self):
        """
        Adds character embedding layer to the graph. This consists of embedding
        layer and a CNN layer.
        """

        # randomly generate char embeddings.
        with vs.variable_scope("embeddings"):
            vocab_size = len(self.char2id)
            char_emb_matrix = tf.Variable(
                tf.random_uniform([vocab_size, self.FLAGS.char_embedding_size], -1.0, 1.0),
                name="char_emb_matrix")

            context_char_embs = embedding_ops.embedding_lookup(char_emb_matrix, self.context_char_ids) # shape (batch_size, context_len, word_len, embedding_size)
            context_drop = tf.layers.dropout(tf.reshape(context_char_embs,
                                                        [-1, self.FLAGS.word_len, self.FLAGS.char_embedding_size]), rate=1-self.keep_prob)
            context_conv = tf.layers.conv1d(context_drop,
                                            self.FLAGS.cnn_filters,
                                            self.FLAGS.cnn_kernel_size,
                                            padding='valid',
                                            reuse=False,
                                            name='cnn',
                                            activation=tf.nn.relu) # shape (batch_size*context_len, word_len-kernel+1, filter)
            context_char_cnn = tf.reduce_max(context_conv, axis=1) # shape (batch_size*context_len, filter)
            self.context_char_embs = tf.reshape(context_char_cnn,
                                                [-1, self.FLAGS.context_len, self.FLAGS.cnn_filters]) # shape (batch_size, context_len, filter)

            qn_char_embs = embedding_ops.embedding_lookup(char_emb_matrix, self.qn_char_ids) # shape (batch_size, question_len, word_len, embedding_size)
            qn_drop = tf.layers.dropout(tf.reshape(qn_char_embs,
                                                   [-1, self.FLAGS.word_len, self.FLAGS.char_embedding_size]), rate=1-self.keep_prob)
            qn_conv = tf.layers.conv1d(qn_drop,
                                       self.FLAGS.cnn_filters,
                                       self.FLAGS.cnn_kernel_size,
                                       padding='valid',
                                       reuse=True,
                                       name='cnn',
                                       activation=tf.nn.relu) # shape (batch_size*question_len, word_len-kernel+1, filter)
            qn_char_cnn = tf.reduce_max(qn_conv, axis=1) # shape (batch_size*question_len, filter)
            self.qn_char_embs = tf.reshape(qn_char_cnn,
                                           [-1, self.FLAGS.question_len, self.FLAGS.cnn_filters]) # shape (batch_size, question_len, filter)


    def build_graph(self):
        """Builds the main part of the graph for the model, starting from the input embeddings to the final distributions for the answer span.

        Defines:
          self.logits_start, self.logits_end: Both tensors shape (batch_size, context_len).
            These are the logits (i.e. values that are fed into the softmax function) for the start and end distribution.
            Important: these are -large in the pad locations. Necessary for when we feed into the cross entropy function.
          self.probdist_start, self.probdist_end: Both shape (batch_size, context_len). Each row sums to 1.
            These are the result of taking (masked) softmax of logits_start and logits_end.
        """

        # Use a RNN to get hidden states for the context and the question
        # Note: here the RNNEncoder is shared (i.e. the weights are the same)
        # between the context and the question.
        encoder = RNNEncoder(self.FLAGS.hidden_size, self.keep_prob)

        context_embs = self.context_embs
        qn_embs = self.qn_embs
        if self.FLAGS.enable_cnn:
            context_embs =  tf.concat([self.context_embs, self.context_char_embs], axis=2)
            qn_embs = tf.concat([self.qn_embs, self.qn_char_embs], axis=2)

        context_hiddens = encoder.build_graph(context_embs, self.context_mask) # (batch_size, context_len, hidden_size*2)
        question_hiddens = encoder.build_graph(qn_embs, self.qn_mask) # (batch_size, question_len, hidden_size*2)

        # Encode query-aware representations of the context words
        bidaf_attn_layer = BidafAttn(self.keep_prob, self.FLAGS.context_len, self.FLAGS.hidden_size * 2)
        bidaf_out = bidaf_attn_layer.build_graph(question_hiddens, self.qn_mask, context_hiddens, self.context_mask)  # (batch_size, context_len, hidden_size*8)

        # Condense the information: hidden_size*8 --> hidden_size*2
        bidaf_out = tf.contrib.layers.fully_connected(
            bidaf_out,
            num_outputs=self.FLAGS.hidden_size*2,
            normalizer_fn=tf.contrib.layers.batch_norm
        )  # (batch_size, context_len, hidden_size*2)

        # Co-attention
        co_attn_layer = CoAttnLite(self.keep_prob, self.FLAGS.hidden_size, self.FLAGS.hidden_size * 2)
        co_out = co_attn_layer.build_graph(question_hiddens, self.qn_mask, context_hiddens, self.context_mask)  # (batch_size, context_len, hidden_size*2)

        bico_out = tf.concat([bidaf_out, co_out], 2)  # (batch_size, context_len, hidden_size*4)

        # Capture interactions among context words conditioned on the query.
        gru_layer1 = RNNEncoder(self.FLAGS.hidden_size, self.keep_prob)  # params: (hidden_size*4 + hidden_size) * hidden_size * 2 * 3
        model_reps1 = gru_layer1.build_graph(bico_out, self.context_mask, variable_scope='ModelGRU1')  # (batch_size, context_len, hidden_size*2)

        gru_layer2 = RNNEncoder(self.FLAGS.hidden_size, self.keep_prob)  # params: (2*hidden_size + hidden_size) * hidden_size * 2 * 3
        model_reps2 = gru_layer2.build_graph(model_reps1, self.context_mask, variable_scope='ModelGRU2')  # (batch_size, context_len, hidden_size*2)

        # Self Attention & GRU layer parallel to GRU layer2.
        with tf.variable_scope('SelfAttnGRU'):
            self_attn_layer = MulAttn(self.keep_prob, self.FLAGS.hidden_size * 2, self.FLAGS.hidden_size * 2)
            se_attn = self_attn_layer.build_graph(model_reps1, self.context_mask, model_reps1, self.context_mask)  # (batch_size, context_len, hidden_size*2)
            se_gru_layer = RNNEncoder(self.FLAGS.hidden_size, self.keep_prob)  # params: (2*hidden_size + hidden_size) * hidden_size * 2 * 3
            se_out = se_gru_layer.build_graph(se_attn, self.context_mask, variable_scope='SelfGRU')  # (batch_size, context_len, hidden_size*2)

        model_reps = tf.concat([model_reps2, se_out], 2)  # (batch_size, context_len, hidden_size*4)

        # Create gate for model reps
        gate = tf.contrib.layers.fully_connected(model_reps1, 1, activation_fn=tf.sigmoid)  # (batch_size, context_len, 1)
        model_reps = gate * model_reps  # (batch_size, context_len, hidden_size*4)

        # Use softmax layer to compute probability distribution for start location
        # Note this produces self.logits_start and self.probdist_start, both of which have shape (batch_size, context_len)
        with vs.variable_scope("StartDist"):
            start_reps = tf.concat([bico_out, model_reps], 2)  # (batch_size, context_len, hidden_size*10)
            softmax_layer_start = SimpleSoftmaxLayer()
            self.logits_start, self.probdist_start = softmax_layer_start.build_graph(start_reps, self.context_mask)

        # Use softmax layer to compute probability distribution for end location
        # Note this produces self.logits_end and self.probdist_end, both of which have shape (batch_size, context_len)
        with vs.variable_scope("EndDist"):
            gru_end_layer = RNNEncoder(self.FLAGS.hidden_size, self.keep_prob)
            model_end_reps = gru_end_layer.build_graph(model_reps, self.context_mask, variable_scope='EndGRU')  # (batch_size, context_len, hidden_size*2)
            end_reps = tf.concat([bico_out, model_end_reps], 2)  # (batch_size, context_len, hidden_size*10)

            softmax_layer_end = SimpleSoftmaxLayer()
            self.logits_end, self.probdist_end = softmax_layer_end.build_graph(end_reps, self.context_mask)

        for variable in tf.trainable_variables():
            tf.summary.histogram(variable.name.replace(':', '/'), variable)

    def add_loss(self):
        """
        Add loss computation to the graph.

        Uses:
          self.logits_start: shape (batch_size, context_len)
            IMPORTANT: Assumes that self.logits_start is masked (i.e. has -large in masked locations).
            That's because the tf.nn.sparse_softmax_cross_entropy_with_logits
            function applies softmax and then computes cross-entropy loss.
            So you need to apply masking to the logits (by subtracting large
            number in the padding location) BEFORE you pass to the
            sparse_softmax_cross_entropy_with_logits function.

          self.ans_span: shape (batch_size, 2)
            Contains the gold start and end locations

        Defines:
          self.loss_start, self.loss_end, self.loss: all scalar tensors
        """
        with vs.variable_scope("loss"):

            # Calculate loss for prediction of start position
            loss_start = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_start, labels=self.ans_span[:, 0]) # loss_start has shape (batch_size)
            self.loss_start = tf.reduce_mean(loss_start) # scalar. avg across batch
            tf.summary.scalar('loss_start', self.loss_start) # log to tensorboard

            # Calculate loss for prediction of end position
            loss_end = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_end, labels=self.ans_span[:, 1])
            self.loss_end = tf.reduce_mean(loss_end)
            tf.summary.scalar('loss_end', self.loss_end)

            # Add the two losses
            self.loss = self.loss_start + self.loss_end
            tf.summary.scalar('loss', self.loss)


    def run_train_iter(self, session, batch, summary_writer):
        """
        This performs a single training iteration (forward pass, loss computation, backprop, parameter update)

        Inputs:
          session: TensorFlow session
          batch: a Batch object
          summary_writer: for Tensorboard

        Returns:
          loss: The loss (averaged across the batch) for this batch.
          global_step: The current number of training iterations we've done
          param_norm: Global norm of the parameters
          gradient_norm: Global norm of the gradients
        """
        # Match up our input data with the placeholders
        input_feed = {}
        input_feed[self.context_ids] = batch.context_ids
        input_feed[self.context_mask] = batch.context_mask
        input_feed[self.qn_ids] = batch.qn_ids
        input_feed[self.qn_mask] = batch.qn_mask
        input_feed[self.ans_span] = batch.ans_span
        input_feed[self.keep_prob] = 1.0 - self.FLAGS.dropout # apply dropout

        # Feed for character level CNN
        if self.FLAGS.enable_cnn:
            input_feed[self.context_char_ids] = batch.context_char_ids
            input_feed[self.qn_char_ids] = batch.qn_char_ids

        # output_feed contains the things we want to fetch.
        output_feed = [self.updates, self.summaries, self.loss, self.global_step, self.param_norm, self.gradient_norm]

        # Run the model
        [_, summaries, loss, global_step, param_norm, gradient_norm] = session.run(output_feed, input_feed)

        # All summaries in the graph are added to Tensorboard
        summary_writer.add_summary(summaries, global_step)

        return loss, global_step, param_norm, gradient_norm


    def get_loss(self, session, batch):
        """
        Run forward-pass only; get loss.

        Inputs:
          session: TensorFlow session
          batch: a Batch object

        Returns:
          loss: The loss (averaged across the batch) for this batch
        """

        input_feed = {}
        input_feed[self.context_ids] = batch.context_ids
        input_feed[self.context_mask] = batch.context_mask
        input_feed[self.qn_ids] = batch.qn_ids
        input_feed[self.qn_mask] = batch.qn_mask
        input_feed[self.ans_span] = batch.ans_span

        if self.FLAGS.enable_cnn:
            input_feed[self.context_char_ids] = batch.context_char_ids
            input_feed[self.qn_char_ids] = batch.qn_char_ids

        # note you don't supply keep_prob here, so it will default to 1 i.e. no dropout

        output_feed = [self.loss]

        [loss] = session.run(output_feed, input_feed)

        return loss


    def get_prob_dists(self, session, batch):
        """
        Run forward-pass only; get probability distributions for start and end positions.

        Inputs:
          session: TensorFlow session
          batch: Batch object

        Returns:
          probdist_start and probdist_end: both shape (batch_size, context_len)
        """
        input_feed = {}
        input_feed[self.context_ids] = batch.context_ids
        input_feed[self.context_mask] = batch.context_mask
        input_feed[self.qn_ids] = batch.qn_ids
        input_feed[self.qn_mask] = batch.qn_mask

        if self.FLAGS.enable_cnn:
            input_feed[self.context_char_ids] = batch.context_char_ids
            input_feed[self.qn_char_ids] = batch.qn_char_ids

        # note you don't supply keep_prob here, so it will default to 1 i.e. no dropout

        output_feed = [self.probdist_start, self.probdist_end]
        [probdist_start, probdist_end] = session.run(output_feed, input_feed)
        return probdist_start, probdist_end


    def get_start_end_pos(self, session, batch):
        """
        Run forward-pass only; get the most likely answer span.
        Inputs:
          session: TensorFlow session
          batch: Batch object
        Returns:
          start_pos, end_pos: both numpy arrays shape (batch_size).
            The most likely start and end positions for each example in the batch.
        """
        # Get start_dist and end_dist, both shape (batch_size, context_len)
        start_dist, end_dist = self.get_prob_dists(session, batch)

        assert start_dist.shape == (batch.batch_size, self.FLAGS.context_len)
        assert end_dist.shape == (batch.batch_size, self.FLAGS.context_len)
        top_n = self.FLAGS.beam_search_size

        def nlargest(start_dist_example):
            return heapq.nlargest(top_n, enumerate(start_dist_example), lambda (i, prob): (prob, i))

        def beam_search(top_start_idx_probs, top_end_idx_probs):
            """Beam search on final start and end indices.
            Find the (start_i, end_i) pair fulfills:
              start_i + 15 >= end_i >= start_i and
              maximize(start_prob * end_prob).
            If no such pair is found in the beam search range, then
            i = i_start = i_end = argmax_i(start_prob, end_prob)
            Reference:
            https://nlp.stanford.edu/pubs/chen2017reading.pdf
            """
            max_prob, max_pair = 0.0, None
            for start_i, start_prob in top_start_idx_probs:
                for end_i, end_prob in top_end_idx_probs:
                    # Skip invalid range.
                    if (end_i < start_i) or (end_i >= start_i + 15):
                        continue
                    cur_prob = start_prob * end_prob
                    cur_pair = (start_i, end_i)
                    if cur_prob > max_prob:
                        max_prob, max_pair = cur_prob, cur_pair

            if max_pair is None:
                i = start_i if start_prob > end_prob else end_i
                return (i, i)

            return max_pair

        start_idx_prob_pairs = map(nlargest, start_dist)
        end_idx_prob_pairs = map(nlargest, end_dist)
        start_end_pos_pairs = [
            beam_search(start_prob_idxs, end_prob_idxs)
            for start_prob_idxs, end_prob_idxs
            in zip(start_idx_prob_pairs, end_idx_prob_pairs)]
        start_pos, end_pos = np.array(zip(*start_end_pos_pairs))

        return start_pos, end_pos


    def get_dev_loss(self, session, dev_context_path, dev_qn_path, dev_ans_path):
        """
        Get loss for entire dev set.

        Inputs:
          session: TensorFlow session
          dev_qn_path, dev_context_path, dev_ans_path: paths to the dev.{context/question/answer} data files

        Outputs:
          dev_loss: float. Average loss across the dev set.
        """
        logging.info("Calculating dev loss...")
        tic = time.time()
        loss_per_batch, batch_lengths = [], []

        # Iterate over dev set batches
        # Note: here we set discard_long=True, meaning we discard any examples
        # which are longer than our context_len or question_len.
        # We need to do this because if, for example, the true answer is cut
        # off the context, then the loss function is undefined.
        for batch in get_batch_generator(self.word2id, self.char2id, dev_context_path,
                                         dev_qn_path, dev_ans_path, self.FLAGS.batch_size,
                                         context_len=self.FLAGS.context_len,
                                         question_len=self.FLAGS.question_len,
                                         word_len=self.FLAGS.word_len,
                                         discard_long=True):

            # Get loss for this batch
            loss = self.get_loss(session, batch)
            curr_batch_size = batch.batch_size
            loss_per_batch.append(loss * curr_batch_size)
            batch_lengths.append(curr_batch_size)

        # Calculate average loss
        total_num_examples = sum(batch_lengths)
        toc = time.time()
        print "Computed dev loss over %i examples in %.2f seconds" % (total_num_examples, toc-tic)

        # Overall loss is total loss divided by total number of examples
        dev_loss = sum(loss_per_batch) / float(total_num_examples)

        return dev_loss


    def check_f1_em(self, session, context_path, qn_path, ans_path, dataset, num_samples=100, print_to_screen=False):
        """
        Sample from the provided (train/dev) set.
        For each sample, calculate F1 and EM score.
        Return average F1 and EM score for all samples.
        Optionally pretty-print examples.

        Note: This function is not quite the same as the F1/EM numbers you get from "official_eval" mode.
        This function uses the pre-processed version of the e.g. dev set for speed,
        whereas "official_eval" mode uses the original JSON. Therefore:
          1. official_eval takes your max F1/EM score w.r.t. the three reference answers,
            whereas this function compares to just the first answer (which is what's saved in the preprocessed data)
          2. Our preprocessed version of the dev set is missing some examples
            due to tokenization issues (see squad_preprocess.py).
            "official_eval" includes all examples.

        Inputs:
          session: TensorFlow session
          qn_path, context_path, ans_path: paths to {dev/train}.{question/context/answer} data files.
          dataset: string. Either "train" or "dev". Just for logging purposes.
          num_samples: int. How many samples to use. If num_samples=0 then do whole dataset.
          print_to_screen: if True, pretty-prints each example to screen

        Returns:
          F1 and EM: Scalars. The average across the sampled examples.
        """
        logging.info("Calculating F1/EM for %s examples in %s set..." % (str(num_samples) if num_samples != 0 else "all", dataset))

        f1_total = 0.
        em_total = 0.
        example_num = 0

        tic = time.time()

        # Note here we select discard_long=False because we want to sample from the entire dataset
        # That means we're truncating, rather than discarding, examples with too-long context or questions
        for batch in get_batch_generator(self.word2id, self.char2id,
                                         context_path, qn_path, ans_path,
                                         self.FLAGS.batch_size,
                                         context_len=self.FLAGS.context_len,
                                         question_len=self.FLAGS.question_len,
                                         word_len=self.FLAGS.word_len,
                                         discard_long=False):

            pred_start_pos, pred_end_pos = self.get_start_end_pos(session, batch)

            # Convert the start and end positions to lists length batch_size
            pred_start_pos = pred_start_pos.tolist() # list length batch_size
            pred_end_pos = pred_end_pos.tolist() # list length batch_size

            for ex_idx, (pred_ans_start, pred_ans_end, true_ans_tokens) in enumerate(zip(pred_start_pos, pred_end_pos, batch.ans_tokens)):
                example_num += 1

                # Get the predicted answer
                # Important: batch.context_tokens contains the original words (no UNKs)
                # You need to use the original no-UNK version when measuring F1/EM
                pred_ans_tokens = batch.context_tokens[ex_idx][pred_ans_start : pred_ans_end + 1]
                pred_answer = " ".join(pred_ans_tokens)

                # Get true answer (no UNKs)
                true_answer = " ".join(true_ans_tokens)

                # Calc F1/EM
                f1 = f1_score(pred_answer, true_answer)
                em = exact_match_score(pred_answer, true_answer)
                f1_total += f1
                em_total += em

                # Optionally pretty-print
                if print_to_screen:
                    print_example(self.word2id, batch.context_tokens[ex_idx], batch.qn_tokens[ex_idx], batch.ans_span[ex_idx, 0], batch.ans_span[ex_idx, 1], pred_ans_start, pred_ans_end, true_answer, pred_answer, f1, em)

                if num_samples != 0 and example_num >= num_samples:
                    break

            if num_samples != 0 and example_num >= num_samples:
                break

        f1_total /= example_num
        em_total /= example_num

        toc = time.time()
        logging.info("Calculating F1/EM for %i examples in %s set took %.2f seconds" % (example_num, dataset, toc-tic))

        return f1_total, em_total


    def train(self, session, train_context_path, train_qn_path, train_ans_path, dev_qn_path, dev_context_path, dev_ans_path):
        """
        Main training loop.

        Inputs:
          session: TensorFlow session
          {train/dev}_{qn/context/ans}_path: paths to {train/dev}.{context/question/answer} data files
        """

        # Print number of model parameters
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retrieval took %f secs)" % (num_params, toc - tic))

        # We will keep track of exponentially-smoothed loss
        exp_loss = None

        # Checkpoint management.
        # We keep one latest checkpoint, and one best checkpoint (early stopping)
        checkpoint_path = os.path.join(self.FLAGS.train_dir, "qa.ckpt")
        bestmodel_dir = os.path.join(self.FLAGS.train_dir, "best_checkpoint")
        bestmodel_ckpt_path = os.path.join(bestmodel_dir, "qa_best.ckpt")
        best_dev_f1 = None
        best_dev_em = None

        # for TensorBoard
        summary_writer = tf.summary.FileWriter(self.FLAGS.train_dir, session.graph)

        epoch = 0

        logging.info("Beginning training loop...")
        while self.FLAGS.num_epochs == 0 or epoch < self.FLAGS.num_epochs:
            epoch += 1
            epoch_tic = time.time()

            # Loop over batches
            for batch in get_batch_generator(self.word2id, self.char2id,
                                             train_context_path, train_qn_path,
                                             train_ans_path, self.FLAGS.batch_size,
                                             context_len=self.FLAGS.context_len,
                                             question_len=self.FLAGS.question_len,
                                             word_len=self.FLAGS.word_len,
                                             discard_long=True):

                # Run training iteration
                iter_tic = time.time()
                loss, global_step, param_norm, grad_norm = self.run_train_iter(session, batch, summary_writer)
                iter_toc = time.time()
                iter_time = iter_toc - iter_tic

                # Update exponentially-smoothed loss
                if not exp_loss: # first iter
                    exp_loss = loss
                else:
                    exp_loss = 0.99 * exp_loss + 0.01 * loss

                # Sometimes print info to screen
                if global_step % self.FLAGS.print_every == 0:
                    logging.info(
                        'epoch %d, iter %d, loss %.5f, smoothed loss %.5f, grad norm %.5f, param norm %.5f, batch time %.3f' %
                        (epoch, global_step, loss, exp_loss, grad_norm, param_norm, iter_time))

                # Sometimes save model
                if global_step % self.FLAGS.save_every == 0:
                    logging.info("Saving to %s..." % checkpoint_path)
                    self.saver.save(session, checkpoint_path, global_step=global_step)

                # Sometimes evaluate model on dev loss, train F1/EM and dev F1/EM
                if global_step % self.FLAGS.eval_every == 0:

                    # Get loss for entire dev set and log to tensorboard
                    dev_loss = self.get_dev_loss(session, dev_context_path, dev_qn_path, dev_ans_path)
                    logging.info("Epoch %d, Iter %d, dev loss: %f" % (epoch, global_step, dev_loss))
                    write_summary(dev_loss, "dev/loss", summary_writer, global_step)


                    # Get F1/EM on train set and log to tensorboard
                    train_f1, train_em = self.check_f1_em(session, train_context_path, train_qn_path, train_ans_path, "train", num_samples=1000)
                    logging.info("Epoch %d, Iter %d, Train F1 score: %f, Train EM score: %f" % (epoch, global_step, train_f1, train_em))
                    write_summary(train_f1, "train/F1", summary_writer, global_step)
                    write_summary(train_em, "train/EM", summary_writer, global_step)


                    # Get F1/EM on dev set and log to tensorboard
                    dev_f1, dev_em = self.check_f1_em(session, dev_context_path, dev_qn_path, dev_ans_path, "dev", num_samples=0)
                    logging.info("Epoch %d, Iter %d, Dev F1 score: %f, Dev EM score: %f" % (epoch, global_step, dev_f1, dev_em))
                    write_summary(dev_f1, "dev/F1", summary_writer, global_step)
                    write_summary(dev_em, "dev/EM", summary_writer, global_step)


                    # Early stopping based on dev EM. You could switch this to use F1 instead.
                    if best_dev_em is None or dev_em > best_dev_em:
                        best_dev_em = dev_em
                        logging.info("Saving to %s..." % bestmodel_ckpt_path)
                        self.bestmodel_saver.save(session, bestmodel_ckpt_path, global_step=global_step)


            epoch_toc = time.time()
            logging.info("End of epoch %i. Time for epoch: %f" % (epoch, epoch_toc-epoch_tic))

        sys.stdout.flush()



def write_summary(value, tag, summary_writer, global_step):
    """Write a single summary value to tensorboard"""
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=value)
    summary_writer.add_summary(summary, global_step)
