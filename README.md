# gci-grammar-grader
This repository was made for Google Code-in task "Automatic Text Scoring" by [CCExtractor Development](https://ccextractor.org/).

## Requirements
Requires python with version 3.6. The rest can be satisfied with command:
```
pip install -r requirements.txt
```

For Keras I use Tensorflow backend.

## File description
### Data
+ `training_set_rel3.tsv` - raw data from [Kaggle competition](https://www.kaggle.com/c/asap-aes)
+ `preprocessed_train.csv` - preprocessed data
### Data preprocess
+ `data_investigation.ipynb` - notebook with preprocessing data. Outputs `preprocessed_train.csv`.
+ `preprocess_for_training.py` - prepares data for training. Outputs `processed_data.pickle` for training neural network.
### Neural networks
+ `test_lstm.py` - file for testing LSTM. If you run this, it will provide a verbose logging of training. Outputs `lstm_test.pickle`.
+ `prod_lstm.py` - file that trains LSTM on entire dataset for production. It runs way slower than `test_lstm.py`, but produce more accurate results. Outputs `lstm_prod.pickle`.
+ `lstm_prod.pickle` - pickled tuple of prediction function and Keras model. Example of using this is in `sample.py` file.

## Resourses
+ [Automatic Text Scoring Using Neural Networks](https://www.researchgate.net/publication/306093850_Automatic_Text_Scoring_Using_Neural_Networks) tells about SSWE. Algorithm released at [Github repository](https://github.com/dimalik/ats/), but I still don't understand how to use it, so I decided to use BLSTM + Word2Vec.
+ [Kaggle kernel](https://www.kaggle.com/guichristmann/lstm-classification-model-with-word2vec) for help with building model with BLSTM and Word2Vec

## TODO
+ Create another dataset from corpus that considered correct by corrupting grammar and decreasing score.