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

"""This file contains the entrypoint to the rest of the code"""

from __future__ import absolute_import
from __future__ import division

import copy
import os
import io
import json
import sys
import logging
import heapq
import importlib

import numpy as np
import tensorflow as tf

# from cnn_qa_model import QAModel
from vocab import get_glove, get_char_mapping
from official_eval_helper import get_json_data, generate_answers, generate_answers_with_start_end


logging.basicConfig(level=logging.INFO)

MAIN_DIR = os.path.relpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # relative path of the main directory
DEFAULT_DATA_DIR = os.path.join(MAIN_DIR, "data") # relative path of data dir
EXPERIMENTS_DIR = os.path.join(MAIN_DIR, "experiments") # relative path of experiments dir


# High-level options
tf.app.flags.DEFINE_integer("gpu", 0, "Which GPU to use, if you have multiple.")
tf.app.flags.DEFINE_string("mode", "train", "Available modes: train / show_examples / official_eval")
tf.app.flags.DEFINE_string("experiment_name", "", "Unique name for your experiment. This will create a directory by this name in the experiments/ directory, which will hold all data related to this experiment")
tf.app.flags.DEFINE_integer("num_epochs", 0, "Number of epochs to train. 0 means train indefinitely")
tf.app.flags.DEFINE_string("model_name", "", "Name of the model to train. If name is 'qa', then file 'qa_model.py' will be imported.")

# Hyperparameters
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 100, "Batch size to use")
tf.app.flags.DEFINE_integer("hidden_size", 200, "Size of the hidden states")
tf.app.flags.DEFINE_integer("context_len", 600, "The maximum context length of your model") # preferred 400
tf.app.flags.DEFINE_integer("question_len", 30, "The maximum question length of your model") # preferred 27
tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained word vectors. This needs to be one of the available GloVe dimensions: 50/100/200/300")

# For CNN
tf.app.flags.DEFINE_bool("enable_cnn", False, "Flag to control CNN.")
tf.app.flags.DEFINE_integer("char_embedding_size", 20, "Size of the character embeddings.") # suggested by handout.
tf.app.flags.DEFINE_integer("word_len", 18, "the maximum word length.") # 18 filters 0.05% of the tokens, 17:0.079%, 16:0.12%, 15:0.19%
tf.app.flags.DEFINE_integer("cnn_filters", 100, "the number of filters for char CNN.")
tf.app.flags.DEFINE_integer("cnn_kernel_size", 5, "the kernel size for char CNN.")

# How often to print, save, eval
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("save_every", 500, "How many iterations to do per save.")
tf.app.flags.DEFINE_integer("eval_every", 500, "How many iterations to do per calculating loss/f1/em on dev set. Warning: this is fairly time-consuming so don't do it too often.")
tf.app.flags.DEFINE_integer("keep", 1, "How many checkpoints to keep. 0 indicates keep all (you shouldn't need to do keep all though - it's very storage intensive).")

# Reading and saving data
tf.app.flags.DEFINE_string("train_dir", "", "Training directory to save the model parameters and other info. Defaults to experiments/{experiment_name}")
tf.app.flags.DEFINE_string("glove_path", "", "Path to glove .txt file. Defaults to data/glove.6B.{embedding_size}d.txt")
tf.app.flags.DEFINE_string("data_dir", DEFAULT_DATA_DIR, "Where to find preprocessed SQuAD data for training. Defaults to data/")
tf.app.flags.DEFINE_string("ckpt_load_dir", "", "For official_eval mode, which directory to load the checkpoint fron. You need to specify this for official_eval mode.")
tf.app.flags.DEFINE_string("json_in_path", "", "For official_eval mode, path to JSON input file. You need to specify this for official_eval_mode.")
tf.app.flags.DEFINE_string("json_out_path", "predictions.json", "Output path for official_eval mode. Defaults to predictions.json")

# Flags for ensemble model
tf.app.flags.DEFINE_bool("is_codalab_eval", False, "This flag will change how to read the experiment best checkpoints.")
tf.app.flags.DEFINE_bool("enable_ensemble_model", False, "Flag to control whether to ensemble multiple models or not.")
tf.app.flags.DEFINE_string("ensemble_model_names", "", "A list of model names to ensemble separated by ';'. 'all' is a special value.")
tf.app.flags.DEFINE_string('ensemble_schema', "sum", "Schema used to ensemble models.")
tf.app.flags.DEFINE_string("sum_weights", "", "If use sum schema, can optionally pass weights here. A list of weights separated by ;. If not set, assume all weights are 1.0. 'default' is a special value.")
tf.app.flags.DEFINE_bool("enable_beam_search", False, "Use beam search in prediction.")
tf.app.flags.DEFINE_integer("beam_search_size", 5, "Size of the beam search.")
tf.app.flags.DEFINE_integer("max_ans_len", 15, "Max answer length allowed to make predictions.")


FLAGS = tf.app.flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)


def initialize_model(session, model, train_dir, expect_exists):
    """
    Initializes model from train_dir.

    Inputs:
      session: TensorFlow session
      model: QAModel
      train_dir: path to directory where we'll look for checkpoint
      expect_exists: If True, throw an error if no checkpoint is found.
        If False, initialize fresh model if no checkpoint is found.
    """
    print "Initializing model at %s" % train_dir
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        print "Reading model parameters from %s" % ckpt.model_checkpoint_path
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        if expect_exists:
            raise Exception("There is no saved checkpoint at %s" % train_dir)
        else:
            print "There is no saved checkpoint at %s. Creating model with fresh parameters." % train_dir
            session.run(tf.global_variables_initializer())
            print 'Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables())


def resolve_ensemble_model_preds(ensemble_model_pred, ensemble_schema=FLAGS.ensemble_schema, sum_weights=FLAGS.sum_weights,
                                 enable_beam_search=FLAGS.enable_beam_search, beam_search_size=FLAGS.beam_search_size):
    """ Resolve the prediction of multiple models according to ensemble_schema

    Input:
      ensemble_model_pred: list of list of list of map,
      [
        model_1_pred [
          batch1 {
            'start': 2d array of size batch_size * context_len
            'end': 2d array of size batch_size * context_len
          },
          batch2,
          ...
        ],
        model_2_pred,
        ...
      ]

      ensemble_schema:
        'sum' means averaging over all pred,
        'max' means keeping the max over all pred,
        (NotSupported)'conf' means keeping the highest confidence score.

    Return:
      pred_start_batches, pred_end_batches: list of list, size is model_flags.batch_size

    """
    def nlargest(start_dist_example):
        return heapq.nlargest(beam_search_size, enumerate(start_dist_example), lambda (i, prob): (prob, i))

    def beam_search(top_start_idx_probs, top_end_idx_probs):
        """Find the (start_i, end_i) pair that max_ans_len >= end_i >= start_i and start_prob * end_prob is max.
        """
        max_prob, max_pair = 0.0, None
        for start_i, start_prob in top_start_idx_probs:
            for end_i, end_prob in top_end_idx_probs:
                if (end_i < start_i) or (end_i - start_i >= FLAGS.max_ans_len):
                    continue
                cur_prob = start_prob * end_prob
                cur_pair = (start_i, end_i)
                if cur_prob > max_prob:
                    max_prob, max_pair = cur_prob, cur_pair

        if max_pair is None:
            i = start_i if start_prob > end_prob else end_i
            return (i, i)

        return max_pair

    # validate flag value
    sum_weight = np.ones(len(ensemble_model_pred))
    if ensemble_schema == "sum" and len(sum_weights)>0:
        weights = np.array([float(w) for w in sum_weights.split(';')])
        assert len(ensemble_model_pred) == len(weights)
        sum_weight = weights

    pred_start_batches = []
    pred_end_batches = []

    num_batch = len(ensemble_model_pred[0])
    context_len = len(ensemble_model_pred[0][0]['start'][0])

    for i in range(0, num_batch):
        # For each batch
        batch_size = len(ensemble_model_pred[0][i]['start'])
        start_batch = np.zeros((batch_size, context_len))
        end_batch = np.zeros((batch_size, context_len))


        for m in range(0, len(ensemble_model_pred)):
            model = ensemble_model_pred[m]
            assert model[i]['start'].shape == start_batch.shape
            assert model[i]['end'].shape == end_batch.shape

            # For each model
            if ensemble_schema == 'sum':
                start_batch += sum_weight[m] * model[i]['start']
                end_batch += sum_weight[m] * model[i]['end']
            elif ensemble_schema == 'max':
                start_batch = np.maximum(start_batch, model[i]['start'])
                end_batch = np.maximum(start_batch, model[i]['end'])
            else:
                print "ERROR: ensemble schema not supported: " %ensemble_schema

        if enable_beam_search:
            # Make final predictions using beam search
            start_idx_prob_pairs = map(nlargest, start_batch)
            end_idx_prob_pairs = map(nlargest, end_batch)
            start_end_pos_pairs = [
                beam_search(start_prob_idxs, end_prob_idxs)
                for start_prob_idxs, end_prob_idxs
                in zip(start_idx_prob_pairs, end_idx_prob_pairs)]
            start_pos, end_pos = np.array(zip(*start_end_pos_pairs))
            pred_start_batches.append(start_pos)
            pred_end_batches.append(end_pos)
        else:
            pred_start_batches.append(np.argmax(start_batch, axis=1))
            pred_end_batches.append(np.argmax(end_batch, axis=1))
    return pred_start_batches, pred_end_batches


def main(unused_argv):
    # Print an error message if you've entered flags incorrectly
    if len(unused_argv) != 1:
        raise Exception("There is a problem with how you entered flags: %s" % unused_argv)

    # Check for Python 2
    if sys.version_info[0] != 2:
        raise Exception("ERROR: You must use Python 2 but you are running Python %i" % sys.version_info[0])

    # Check for ensemble model param setting
    if FLAGS.enable_ensemble_model and (FLAGS.mode != "official_eval" or not FLAGS.ensemble_model_names):
        raise Exception("ERROR: model ensemble is only supported in official_eval mode, you must specify ensemble_model_names")

    # Print out Tensorflow version
    print "This code was developed and tested on TensorFlow 1.4.1. Your TensorFlow version: %s" % tf.__version__

    # Define train_dir
    if (not FLAGS.enable_ensemble_model and not FLAGS.experiment_name and not FLAGS.train_dir and FLAGS.mode != "official_eval") or (FLAGS.enable_ensemble_model and not FLAGS.ensemble_model_names):
        raise Exception("You need to specify either --experiment_name or --train_dir, or ensemble_model_names if ensemble is enabled.")
    FLAGS.train_dir = FLAGS.train_dir or os.path.join(EXPERIMENTS_DIR, FLAGS.experiment_name)

    # Initialize bestmodel directory
    bestmodel_dir = os.path.join(FLAGS.train_dir, "best_checkpoint")

    # Define path for glove vecs
    FLAGS.glove_path = FLAGS.glove_path or os.path.join(DEFAULT_DATA_DIR, "glove.6B.{}d.txt".format(FLAGS.embedding_size))

    # Load embedding matrix and vocab mappings
    emb_matrix, word2id, id2word = get_glove(FLAGS.glove_path, FLAGS.embedding_size)

    # Build character level vocab mappings
    char2id, id2char = get_char_mapping()

    # Get filepaths to train/dev datafiles for tokenized queries, contexts and answers
    train_context_path = os.path.join(FLAGS.data_dir, "train.context")
    train_qn_path = os.path.join(FLAGS.data_dir, "train.question")
    train_ans_path = os.path.join(FLAGS.data_dir, "train.span")
    dev_context_path = os.path.join(FLAGS.data_dir, "dev.context")
    dev_qn_path = os.path.join(FLAGS.data_dir, "dev.question")
    dev_ans_path = os.path.join(FLAGS.data_dir, "dev.span")


    if not FLAGS.enable_ensemble_model:
        # Initialize model only when ensemble model is disabled.
        qa_model_name = FLAGS.model_name + '_model'
        QAModel = importlib.import_module(qa_model_name).QAModel
        print('model loaded from: %s' % qa_model_name)
        qa_model = QAModel(FLAGS, id2word, word2id, emb_matrix, char2id ,id2char)

    # Some GPU settings
    config=tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Split by mode
    if FLAGS.mode == "train":

        # Setup train dir and logfile
        if not os.path.exists(FLAGS.train_dir):
            os.makedirs(FLAGS.train_dir)
        file_handler = logging.FileHandler(os.path.join(FLAGS.train_dir, "log.txt"))
        logging.getLogger().addHandler(file_handler)

        # Save a record of flags as a .json file in train_dir
        with open(os.path.join(FLAGS.train_dir, "flags.json"), 'w') as fout:
            json.dump(FLAGS.__flags, fout)

        # Make bestmodel dir if necessary
        if not os.path.exists(bestmodel_dir):
            os.makedirs(bestmodel_dir)

        with tf.Session(config=config) as sess:

            # Load most recent model
            initialize_model(sess, qa_model, FLAGS.train_dir, expect_exists=False)

            # Train
            qa_model.train(sess, train_context_path, train_qn_path, train_ans_path, dev_qn_path, dev_context_path, dev_ans_path)


    elif FLAGS.mode == "show_examples":
        with tf.Session(config=config) as sess:

            # Load best model
            initialize_model(sess, qa_model, bestmodel_dir, expect_exists=True)

            # Show examples with F1/EM scores
            _, _ = qa_model.check_f1_em(sess, dev_context_path, dev_qn_path, dev_ans_path, "dev", num_samples=10, print_to_screen=True)


    elif FLAGS.mode == "official_eval":
        if FLAGS.json_in_path == "":
            raise Exception("For official_eval mode, you need to specify --json_in_path")
        if not FLAGS.enable_ensemble_model and FLAGS.ckpt_load_dir == "":
            raise Exception("For official_eval mode, you need to specify --ckpt_load_dir or use ensemble_model_names")

        # Read the JSON data from file
        qn_uuid_data, context_token_data, qn_token_data = get_json_data(FLAGS.json_in_path)

        if FLAGS.enable_ensemble_model:
            print('FLAGS.ensemble_model_names: %s' % FLAGS.ensemble_model_names)
            print('FLAGS.sum_weights: %s' % FLAGS.sum_weights)
            # KV is 'label': ('model_file', 'exp_name', 'codalab_bundle_name', 'has_cnn', weight),
            ensemble_label_to_model_meta = {
                'binco_legacy': ['binco_legacy_model', 'binco_30b15_hidden=100_lr=0.001_batch=100_context=400_qn=27', 'binco_30b15', False, 0.6692],  # 0.6900  (74.0, 63.5)
                'chsebinco_real': ['chsebinco_model', 'chsebinco_real_1c999_hidden=100_lr=0.001_batch=100_context=400_qn=27', 'chsebinco_real_1c999', True, 0.6733],  # 0.6958, (74.7, 64.0)
                'chsebinco_legacy': ['chsebinco_legacy_model', 'chsebinco_4a81a_hidden=100_lr=0.001_batch=100_context=400_qn=27', 'chsebinco_4a81a', True, 0.6507],  # 0.6954, (?, ?)
                'chgasebinco': ['chgasebinco_model', 'chgasebinco_1c999_hidden=100_lr=0.001_batch=100_context=400_qn=27', 'chgasebinco_1c999', True, 0.7045],  # 0.7101  (76.6, 66.4)
                'chgasebinco_91ca1': ['chgasebinco_model', 'chgasebinco_91ca1_hidden=100_lr=0.001_batch=100_context=400_qn=27', 'chgasebinco_91ca1', True, 0.69],  # 0.68  (? ?)
                'chgasebinco_888ca': ['chgasebinco_model', 'chgasebinco_888ca_hidden=100_lr=0.001_batch=100_context=400_qn=27', 'chgasebinco_888ca', True, 0.69],  # 0.67  (? ?)
                'chgasebinco_888ca_run2': ['chgasebinco_model', 'chgasebinco_888ca_run2_hidden=100_lr=0.001_batch=100_context=400_qn=27', 'chgasebinco_888ca_run2', True, 0.69],  # 0.6911  (? ?)
            }
            model_labels = FLAGS.ensemble_model_names.split(';')
            if len(model_labels) == 1 and model_labels[0].lower() == 'all':
                model_labels = ensemble_label_to_model_meta.keys()
            else:
                for label in model_labels:
                    assert label in ensemble_label_to_model_meta

            # A list to store the output of all predictions
            # each entry is a map, storing the start and end dist for that batch.
            # len(ensemble_model_pred) is len(model_labels)
            # len(ensemble_model_pred[0]) is number of batches
            # len(ensemble_model_pred[0]['start']) is batch_size
            # len(ensemble_model_pred[0]['end']) is batch_size
            ensemble_model_pred = []
            sum_weights_list = []
            for label in model_labels:
                tf.reset_default_graph()
                model_name, model_exp_name, cl_bundle_name, has_cnn, weight = ensemble_label_to_model_meta[label]
                sum_weights_list += str(weight),
                print "Loading model: %s" % model_name
                # TODO(binbinx): change this to appropriate models
                QAModel = importlib.import_module(model_name).QAModel
                qa_model = (
                    QAModel(FLAGS, id2word, word2id, emb_matrix, char2id ,id2char)
                    if has_cnn
                    else QAModel(FLAGS, id2word, word2id, emb_matrix))

                with tf.Session(config=config) as sess:
                    # Initialize bestmodel directory
                    ckpt_load_dir = (
                        os.path.join(EXPERIMENTS_DIR, model_exp_name, "best_checkpoint")
                        if not FLAGS.is_codalab_eval
                        else cl_bundle_name)

                    # Load model from ckpt_load_dir
                    initialize_model(sess, qa_model, ckpt_load_dir, expect_exists=True)

                    # Get a predicted answer for each example in the data
                    # Return a mapping answers_dict from uuid to answer
                    # WE MUST USE A DEEPCOPY HERE!!
                    qn_uuid_data_ = copy.deepcopy(qn_uuid_data)
                    context_token_data_ = copy.deepcopy(context_token_data)
                    qn_token_data_ = copy.deepcopy(qn_token_data)
                    answers_dict = generate_answers(sess, qa_model, word2id, char2id, qn_uuid_data_,
                                                    context_token_data_, qn_token_data_, ensemble_model_pred)

            sum_weights = ';'.join(sum_weights_list) if FLAGS.sum_weights.lower() == 'default' else FLAGS.sum_weights
            pred_start_batches, pred_end_batches = resolve_ensemble_model_preds(ensemble_model_pred, sum_weights=sum_weights)

            final_ans_dict = generate_answers_with_start_end(FLAGS, word2id, char2id, qn_uuid_data,
                 context_token_data, qn_token_data, pred_start_batches, pred_end_batches)

            # Write the uuid->answer mapping a to json file in root dir
            print "Writing predictions to %s..." % FLAGS.json_out_path
            with io.open(FLAGS.json_out_path, 'w', encoding='utf-8') as f:
                f.write(unicode(json.dumps(final_ans_dict, ensure_ascii=False)))
                print "Wrote predictions to %s" % FLAGS.json_out_path
        else:
            with tf.Session(config=config) as sess:
                # Load model from ckpt_load_dir
                initialize_model(sess, qa_model, FLAGS.ckpt_load_dir, expect_exists=True)

                # Get a predicted answer for each example in the data
                # Return a mapping answers_dict from uuid to answer
                answers_dict = generate_answers(sess, qa_model, word2id, char2id, qn_uuid_data, context_token_data, qn_token_data)

                # Write the uuid->answer mapping a to json file in root dir
                print "Writing predictions to %s..." % FLAGS.json_out_path
                with io.open(FLAGS.json_out_path, 'w', encoding='utf-8') as f:
                    f.write(unicode(json.dumps(answers_dict, ensure_ascii=False)))
                    print "Wrote predictions to %s" % FLAGS.json_out_path


    else:
        raise Exception("Unexpected value of FLAGS.mode: %s" % FLAGS.mode)

if __name__ == "__main__":
    tf.app.run()
