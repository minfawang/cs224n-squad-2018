import os
import numpy as np

TRAIN_SAMPLE_SIZE = 1000  # Total 86318
DEV_SAMPLE_SIZE = 1000 # Total 10391

MAIN_DIR = os.path.relpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # relative path of the main directory
DATA_DIR = os.path.join(MAIN_DIR, "data") # relative path of data dir
EXPERIMENTS_DIR = os.path.join(MAIN_DIR, "experiments") # relative path of experiments dir
MINI_DATA_DIR = os.path.join(MAIN_DIR, "mini_data") # relative path of data dir
MINI_EXPERIMENTS_DIR = os.path.join(MAIN_DIR, "mini_experiments") # relative path of experiments dir

train_files = [
    'train.question',
    'train.context',
    'train.answer',
    'train.span',
]

dev_files = [
    'dev.question',
    'dev.context',
    'dev.answer',
    'dev.span',
]

def get_data_tuples(data_files):
  """
  Returns a list of (question, context, answer, span) tuples.
  """
  data = []
  count = None
  for data_file in data_files:
    data_path = os.path.join(DATA_DIR, data_file)
    print('Reading from file: {}'.format(data_path))
    with open(data_path, 'r') as fin:
      column = [item.strip() for item in fin]
      if count is None:
        count = len(column)
      else:
        assert count == len(column), 'Inconsistent length in data.'
      data.append(column)
  return list(zip(*data))


def write_mini_data_files(data_files, mini_data):
  mini_data_t = zip(*mini_data)
  for column, data_file in zip(mini_data_t, data_files):
    data_path = os.path.join(MINI_DATA_DIR, data_file)
    print('Writing to file: {}'.format(data_path))
    with open(data_path, 'w') as fout:
      fout.write('\n'.join(column))

if __name__ == '__main__':
  if not os.path.exists(MINI_DATA_DIR):
    os.makedirs(MINI_DATA_DIR)
  if not os.path.exists(MINI_EXPERIMENTS_DIR):
    os.makedirs(MINI_EXPERIMENTS_DIR)

  train_data = get_data_tuples(train_files)
  np.random.shuffle(train_data)
  mini_train_data = train_data[:TRAIN_SAMPLE_SIZE]
  write_mini_data_files(train_files, mini_train_data)

  dev_data = get_data_tuples(dev_files)
  np.random.shuffle(dev_data)
  mini_dev_data = dev_data[:DEV_SAMPLE_SIZE]
  write_mini_data_files(dev_files, mini_dev_data)
