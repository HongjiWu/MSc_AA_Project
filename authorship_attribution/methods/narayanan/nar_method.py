from typing import Optional
import numpy as np
from sklearn.linear_model import RidgeClassifier
from scipy.special import softmax

from authorship_attribution.methods.base_aa_method import BaseAAMethod

from authorship_attribution.feature_extractors.writeprints_extractor import WriteprintsExtractor
from authorship_attribution.feature_extractors import NarayananExtractor
from authorship_attribution.helpers.normalization import Normalizer


class NarMethod(BaseAAMethod):
    def __init__(self, class_weight: Optional[str] = "balanced", stanford_parser: bool = False):
        super().__init__()
        self.class_weight = class_weight
        self.stanford_parser = stanford_parser
        if stanford_parser:
            self.name = "Stanford" + self.name

    def fit_extractor(self, _):
        if self.stanford_parser:
            self.extractor = NarayananExtractor()
        else:
            self.extractor = WriteprintsExtractor()

    @staticmethod
    def normalize_features(features):
        norm_1 = Normalizer(features, "feature-mean")   # nonzero
        norm_2 = Normalizer(norm_1.normalize(features), "row-norm")
        return lambda x: norm_2.normalize(norm_1.normalize(x))

    def data_processing(self, train, test):
        self.fit_extractor(list(train.text))
        train_features = self.extractor.transform(list(train.text))
        normalizer = self.normalize_features(train_features)
        train_features = normalizer(train_features)
        train_labels = np.array(train.dummy)
        train_ids = np.array(train.id)

        del train

        test_features = normalizer(self.extractor.transform(list(test.text)))
        test_labels = np.array(test.dummy)
        test_ids = np.array(test.id)

        del test
        del normalizer

        return train_features, train_labels, train_ids, test_features, test_labels, test_ids

    def fit_model(self, train_features, train_labels):
        # One vs all approach when given n-classes - class-weight=balanced helps minority class
        clf = RidgeClassifier(class_weight=self.class_weight).fit(train_features, train_labels)
        return clf

    def infer(self, model, test_features) -> (np.array, np.array):

        scores = model.decision_function(test_features)
        if len(scores.shape) == 1:
            scores = np.array([- scores, scores]).transpose()
        predictions = model.classes_[np.flip(np.argsort(scores, axis=1), axis=1)]
        ordered_scores = softmax(np.flip(np.sort(scores, axis=1), axis=1), axis=1)

        return predictions, ordered_scores
