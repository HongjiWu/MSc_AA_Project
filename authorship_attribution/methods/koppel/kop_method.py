import random
from typing import Optional
import warnings

import numpy as np
from sklearn.neighbors import NearestCentroid
from scipy.special import softmax

from authorship_attribution.methods.base_aa_method import BaseAAMethod

from authorship_attribution.feature_extractors import NGramKoppelExtractor


def warn(*args, **kwargs):
    pass


warnings.warn = warn


# pylint: disable=invalid-name
class KopMethod(BaseAAMethod):
    def __init__(self,
                 low_memory: bool = True,
                 max_vocab_size: Optional[int] = None,
                 batch_size: int = 100,
                 k1: int = 50,
                 k2: float = 0.3,
                 thresh: float = 0.6
                 ):
        super().__init__()
        self.low_memory: bool = low_memory
        self.max_vocab_size = max_vocab_size  # 100000
        self.batch_size = batch_size

        self.train_features = None
        self.train_labels = None

        self.k1 = k1
        self.k2 = k2
        self.thresh = thresh

    def fit_extractor(self, texts):
        self.extractor = NGramKoppelExtractor(vocab_size=self.max_vocab_size)
        self.extractor.fit(texts)

    def data_processing(self, train, test):
        self.fit_extractor(list(train.text))

        train_features = None
        train_labels = []
        train_ids = []

        if self.low_memory:
            for label in train.dummy.unique():
                tmp = train[train.dummy == label]
                feature_vec = np.mean(self.extractor.transform(list(tmp.text)), axis=0)
                train_features = np.vstack((train_features, feature_vec)) if train_features is not None else feature_vec
                train_ids.extend(tmp.id.values)
                train_labels.append(label)

            # test features will be batch computed at inference time
            test_features = list(test.text)

        else:
            train_features = self.extractor.transform(list(train.text))
            train_labels = np.array(train.dummy)
            train_ids = np.array(train.id)
            del train
            test_features = self.extractor.transform(list(test.text))

        test_labels = np.array(test.dummy)
        test_ids = np.array(test.id)
        del test

        return train_features, np.array(train_labels), np.array(train_ids), test_features, test_labels, test_ids

    def fit_model(self, train_features, train_labels):
        """Kept for API consistency"""
        self.train_features = train_features
        self.train_labels = train_labels

    def infer(self, model, test_features) -> (np.array, np.array):

        num_features = self.train_features.shape[1]

        predictions_per_model = []
        for _ in range(self.k1):
            sampled_features_index = random.sample(list(np.arange(num_features)), int(self.k2*num_features))
            train_features = self.train_features[:, sampled_features_index]
            model = NearestCentroid(metric="cosine").fit(train_features, self.train_labels)

            if isinstance(test_features, list):
                predictions = []
                for i in range(0, len(test_features), self.batch_size):
                    predictions.extend(model.predict(
                        self.extractor.transform(test_features[i:i + self.batch_size])[:, sampled_features_index]))
            else:
                sampled_test_features = test_features[:, sampled_features_index]
                predictions = model.predict(sampled_test_features)

            predictions_per_model.append(predictions)

        predictions_per_model = np.array(predictions_per_model)

        stacked_confidence = np.zeros((predictions_per_model.shape[1], len(np.unique(self.train_labels))), dtype=int)
        for i in range(stacked_confidence.shape[1]):
            stacked_confidence[:, i] = np.sum(predictions_per_model == i, axis=0)

        assert stacked_confidence.sum() == self.k1 * len(test_features)

        ordered_scores = softmax(np.flip(np.sort(stacked_confidence, axis=1), axis=1), axis=1)
        return np.flip(np.argsort(stacked_confidence, axis=1), axis=1), ordered_scores
