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

"""This file contains a function to read the GloVe vectors from file,
and return them as an embedding matrix"""

from __future__ import absolute_import
from __future__ import division

from tqdm import tqdm
import numpy as np

_PAD = b"<pad>"
_UNK = b"<unk>"
_START_VOCAB = [_PAD, _UNK]
PAD_ID = 0
UNK_ID = 1

# Hard code this dictionary here.
_CHAR_PAD = '\x00'
_CHAR_UNK = '\x11'
CHAR_PAD_ID = 0
CHAR_UNK_ID = 1

# Threshold, imp > 1000. Only lose 0.1% imp, removed 40 char.
CHAR_DICT = ['\xc4', '\xd8', '[', '\xd0', 'o', '\xe8', 'k', 'g', '\xe0', 'c', 
             'w', 's', '\x8c', '\x88', '\x84', '\x80', '\x9c', '\x94', '\x90', 
             '\xac', '/', '\xa8', '+', '\xa4', "'", '\xa0', '\xbc', '?', 
             '\xb8', ';', '\xb4', '7', '\xb0', '3', '\xcf', '\xcb', '\xc3', 
             '\xd7', 'l', 'h', 'd', '\xe7', '\xe3', 'x', 't', 'p', '\x87', 
             '\x83', '\x9f', '\x9b', '\x97', '\x93', ',', '\xaf', '(', '\xab', 
             '$', '\xa7', '\xa3', '\xbf', '8', '\xbb', '4', '\xb7', '0', 
             '\xb3', '\xce', '\xca', '\xc2', ']', 'm', 'i', 'e', '\xe6', 'a', 
             '\xe2', 'y', 'u', 'q', '\x8e', '\x8a', '\x82', '\x92', '-', 
             '\xae', ')', '\xaa', '%', '\xa6', '!', '\xa2', '\xbe', '9', 
             '\xba', '5', '\xb6', '1', '\xb2', '\xc9', '\xc5', '\xd9', '\xd1',
             'n', 'j', '\xe5', 'f', '\xe1', 'b', 'z', 'v', 'r', '\x8d', '\x89', 
             '\x85', '\x81', '\x9d', '\x99', '\x91', '\xad', '.', '\xa9',
             '\xa5', '&', '\xa1', '"', '\xbd', '\xb9', ':', '\xb5', '6', '\xb1',
             '2']

def get_char_mapping():
  """Return hard coded char2id and id2char mapping for en/US
  """
  char2id = {}
  id2char = {}
  
  char2id[_CHAR_PAD] = CHAR_PAD_ID
  char2id[_CHAR_UNK] = CHAR_UNK_ID
  id2char[CHAR_PAD_ID] = _CHAR_PAD
  id2char[CHAR_UNK_ID] = _CHAR_UNK
  
  for i in range(0, len(CHAR_DICT)):
    idx = i+2
    char2id[CHAR_DICT[i]] = idx
    id2char[idx] = CHAR_DICT[i]
  
  return char2id, id2char

def get_glove(glove_path, glove_dim):
    """Reads from original GloVe .txt file and returns embedding matrix and
    mappings from words to word ids.

    Input:
      glove_path: path to glove.6B.{glove_dim}d.txt
      glove_dim: integer; needs to match the dimension in glove_path

    Returns:
      emb_matrix: Numpy array shape (400002, glove_dim) containing glove embeddings
        (plus PAD and UNK embeddings in first two rows).
        The rows of emb_matrix correspond to the word ids given in word2id and id2word
      word2id: dictionary mapping word (string) to word id (int)
      id2word: dictionary mapping word id (int) to word (string)
    """

    print "Loading GLoVE vectors from file: %s" % glove_path
    vocab_size = int(4e5) # this is the vocab size of the corpus we've downloaded

    emb_matrix = np.zeros((vocab_size + len(_START_VOCAB), glove_dim))
    word2id = {}
    id2word = {}

    random_init = True
    # randomly initialize the special tokens
    if random_init:
        emb_matrix[:len(_START_VOCAB), :] = np.random.randn(len(_START_VOCAB), glove_dim)

    # put start tokens in the dictionaries
    idx = 0
    for word in _START_VOCAB:
        word2id[word] = idx
        id2word[idx] = word
        idx += 1

    # go through glove vecs
    with open(glove_path, 'r') as fh:
        for line in tqdm(fh, total=vocab_size):
            line = line.lstrip().rstrip().split(" ")
            word = line[0]
            vector = list(map(float, line[1:]))
            if glove_dim != len(vector):
                raise Exception("You set --glove_path=%s but --embedding_size=%i. If you set --glove_path yourself then make sure that --embedding_size matches!" % (glove_path, glove_dim))
            emb_matrix[idx, :] = vector
            word2id[word] = idx
            id2word[idx] = word
            idx += 1

    final_vocab_size = vocab_size + len(_START_VOCAB)
    assert len(word2id) == final_vocab_size
    assert len(id2word) == final_vocab_size
    assert idx == final_vocab_size

    return emb_matrix, word2id, id2word

