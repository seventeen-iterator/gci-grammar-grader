import re

from keras.preprocessing.sequence import pad_sequences


TOKEN_REGEX = r"(\w+|\d+|[^\w\d\s]+)"
MAX_SEQUENCE_LENGTH = 800


class EssayIter():
    '''Iterates over tokens in text, i.e. words, numbers, punctuation'''
    def __init__(self, corpus):
        self.corpus = corpus

    def __iter__(self):
        for sentence in self.corpus:
            yield [token for token in re.findall(TOKEN_REGEX, sentence)]


def word2token(word, model):
    '''Returns number that stands for this word in word2vec vocabulary'''
    try:
        return model.wv.vocab[word].index + 1
    except KeyError:
        return 0


class SequenceIter:
    '''Iterator that yields tokenized data and scaled marks'''
    def __init__(self, ds, maxlen, model):
        self.X = ds.essay
        self.y = ds.domain1_score / 5
        self.maxlen = maxlen
        self.model = model

    def __iter__(self):
        for essay, mark in zip(self.X, self.y):
            yield (
                [word2token(token[0], self.model) for token in re.findall(
                    TOKEN_REGEX, essay)[:self.maxlen] if token[0] != ''],
                mark
            )


class Predictor:
    '''Creates a predict function'''
    def __init__(self, nets, model, preprocess_texts):
        self.nets = nets
        self.model = model
        self.preprocess_texts = preprocess_texts

    def predict(self, texts):
        '''Accepts array of strings, returns array of scores'''
        return (self.nets.predict(self.preprocess_texts(texts,
                                                        self.model)) * 5)


def preprocess_texts(texts, model):
    '''Preprocess text for prediction'''
    return pad_sequences([
        [word2token(token[0], model)
         for token in re.findall(TOKEN_REGEX, text)[:MAX_SEQUENCE_LENGTH]
         if token[0] != '']
        for text in texts
    ], padding='pre', value=0, maxlen=MAX_SEQUENCE_LENGTH)
