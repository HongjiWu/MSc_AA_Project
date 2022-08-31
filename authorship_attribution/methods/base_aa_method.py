class BaseAAMethod:
    """
    Parent Class for all AA methods.
    """
    def __init__(self):
        self.name = self.__class__.__name__
        self.extractor = None

    def fit_extractor(self, texts):
        """
        For methods requiring a fitted extractor

        :param texts: Training texts
        """
        return None

    def fit_model(self, train_features, train_labels):
        """
        Fit the method to the training data

        :param train_features: Training samples
        :param train_labels: Author classes for each training sample
        :return: model
        """
        raise NotImplementedError

    def infer(self, model, test_features):
        """
        Given the method and test samples, yield predictions and confidence scores.

        :param model: The trained method
        :param test_features: Test samples
        :return:  predicted_labels, confidence_scores
        """
        raise NotImplementedError

    def data_processing(self, train, test):
        """
        Convert the reader train and test Pandas dataframe to a processed format with features, labels and ids.

        :param train: Source Pandas DataFrame.
        :param test: Target Pandas DataFrame.
        :return: train_features, train_labels, train_ids, test_features, test_labels, test_ids
        """
        raise NotImplementedError
