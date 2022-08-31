import warnings
from typing import Optional
import numpy as np

from sklearn.neighbors import NearestCentroid
from sklearn.metrics.pairwise import pairwise_distances
from scipy.special import softmax

from authorship_attribution.methods.base_aa_method import BaseAAMethod

from authorship_attribution.feature_extractors import NGramKoppelExtractor


def warn(*args, **kwargs):
    pass


warnings.warn = warn


class NaiveKopMethod(BaseAAMethod):
    def __init__(self,
                 low_memory: bool = True,
                 max_vocab_size: Optional[int] = None,
                 batch_size: int = 100,
                 ):
        super().__init__()
        self.low_memory: bool = low_memory
        self.max_vocab_size = max_vocab_size  # 100000
        self.batch_size = batch_size

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
        clf = NearestCentroid(metric="cosine").fit(train_features, train_labels)
        return clf

    def infer(self, model, test_features) -> (np.array, np.array):

        if isinstance(test_features, list):
            predictions = []
            scores = []
            for i in range(0, len(test_features), self.batch_size):
                tmp_preds, tmp_scores = self.predict_top_n(model, self.extractor.transform(
                    test_features[i:i + self.batch_size]))
                predictions.extend(tmp_preds)
                scores.extend(tmp_scores)
        else:
            predictions, scores = self.predict_top_n(model, test_features)

        # Compute 1/distance to match ordering of classification methods
        return np.array(predictions), softmax(1 / (np.array(scores) + 0.0001), axis=1)

    @staticmethod
    def predict_top_n(model, features):
        distances = pairwise_distances(features, model.centroids_, metric=model.metric)
        return model.classes_[np.argsort(distances, axis=1)], np.sort(distances, axis=1)
