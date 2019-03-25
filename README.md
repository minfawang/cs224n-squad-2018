# Disclaimer

This repository holds the code used in the CS224N final project in Winter 2018. The code is not longer being actively developed/maintained. Use at your own risk.

For future CS224N students, please read the [Honor Code](https://communitystandards.stanford.edu/policies-and-guidance/honor-code) and make sure there is no violations of policies before proceding with reading/using this code base.

# cs224n-win18-squad
Code for the Default Final Project (SQuAD) for [CS224n](http://web.stanford.edu/class/cs224n/), Winter 2018

Note: this code is adapted in part from the [Neural Language Correction](https://github.com/stanfordmlgroup/nlc/) code by the Stanford Machine Learning Group.

1. Run ensemble models:
```python code/main.py --enable_cnn=1 --mode=official_eval --enable_ensemble_model=1 --ensemble_model_names="binbin-cnn;binbin-cnn;binbin-cnn"  --json_in_path=data/tiny-dev.json --enable_beam_search=1 --context_len=400 --question_len=27```

*Also note the default param of ```--beam_search_size=5 --ensemble_schema='sum' --sum_weights=''```
