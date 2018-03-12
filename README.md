# cs224n-win18-squad
Code for the Default Final Project (SQuAD) for [CS224n](http://web.stanford.edu/class/cs224n/), Winter 2018

Note: this code is adapted in part from the [Neural Language Correction](https://github.com/stanfordmlgroup/nlc/) code by the Stanford Machine Learning Group.

## Model name and explanations.

`qa_model.py`: the baseline model.

`edsa_model.py`: shortened from end-dep-start-add model. It borrows the idea from handout section 5.3. After making the prediction of the start position. Create a start attention vector using the sum of blended position representation weighted by the probability distribution of that position being the start position. Then it adds that attention vector to the blended vector for every postion to make the end prediction.

`edsc_model.py`: shortened from end-dep-start-concat model. Similar to `edsa_model.py`, except it concatenates the attention vector to the blended vector.

`edsc2_model.py`: Similar to 'edsc_model.py', except it uses different weights in final softmax layer for start and end predictions.

`bidaf_model.py`: In generally followed the logic of BiDAF paper but with several modifications:
  * Used `GRU` instead of `LSTM` as RNN cells. This reduces the number of parameters.
 * Added a fully connected layer with batch norm before 2 layers of `GRU`s to reduce the data dimensionality and again reduce the parameters count.

`bidaf_self_model.py`: Modified based on `bidaf_model.py`.
  * Added a self-attention layer (with multiplicative attention applied internally) after the bi-directional attention. This will allow each context word representation to also have awareness of the overall passsage representations.
  * Explicit dependency of start predictions. Produce attention vector from based on start prediction distributions.

## Generate Mini Data

Run the command from terminal below will create mini data in "mini_data/" directory.

```
python gen_mini.py
```

To use mini data in training, run a command like below:

```
python code/main.py --mode=train --data_dir=mini_data --train_dir="mini_experiments/bidaf_edsc3"
```

## Notes

### Hyperparameters

#### context_len & question_len

TODO: Image of Histogram.

The advantage of choosing a smaller `context_len` and/or `question_len` is that the training speed will improve drastically. The negative consequence is that we would need to "declare defeat" for the examples who needs to access context at the tail of very long passages, and/or the core of the question appear at the tail of very long questions.

When context_len = `400`, we are only giving up 0.2% of examples.

###
