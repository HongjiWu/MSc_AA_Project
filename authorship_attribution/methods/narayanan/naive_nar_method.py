import numpy as np

from sklearn.neighbors import KNeighborsClassifier

from authorship_attribution.helpers.normalization import Normalizer
from authorship_attribution.methods.base_aa_method import BaseAAMethod
from authorship_attribution.feature_extractors import WriteprintsExtractor


class NaiveNarMethod(BaseAAMethod):

    def fit_extractor(self, _):
        self.extractor = WriteprintsExtractor()

    @staticmethod
    def normalize_features(features):
        norm_1 = Normalizer(features, "feature-mean")
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
        clf = KNeighborsClassifier(n_neighbors=1).fit(train_features, train_labels)
        return clf

    def infer(self, model, test_features) -> np.array:
        predictions = model.predict(test_features)
        return predictions
