import numpy as np
from torch.optim.adam import Adam

import torch

from authorship_attribution.methods.base_aa_method import BaseAAMethod

from authorship_attribution.feature_extractors import NGramKoppelExtractor
from authorship_attribution.methods.shrestha.cnn_ngram_model import CNNNGramNet
from authorship_attribution.methods.sari.model_training import Trainer, TrainingArgs


class ShresthaMethod(BaseAAMethod):
    def __init__(self,
                 embedding_dim: int = 300,
                 pad_length: int = 2048,
                 vocab_size=20000,
                 n: int = 2,
                 learning_rate = 0.0005,
                 training_args: TrainingArgs = TrainingArgs()):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.pad_length = pad_length
        self.training_args = training_args
        self.vocab_size = vocab_size
        self.n = n
        self.test_dataset = None
        self.name = self.name + f"_{self.training_args.batch_size}" + f"_{self.training_args.train_epochs}"
        self.lr = learning_rate

    def fit_extractor(self, texts):
        self.extractor = NGramKoppelExtractor(n=self.n, vocab_size=self.vocab_size, split_words=False)
        self.extractor.fit(texts)

    @staticmethod
    def dirty_hack(encodings, labels, ids, split=5):
        thresh = int(len(encodings[0])/split)
        encodings = [encod[i * thresh:(i + 1) * thresh] for i in range(split) for encod in encodings]
        labels = [label for _ in range(split) for label in labels]
        ids = [id_ for _ in range(split) for id_ in ids]
        return encodings, labels, ids

    def data_processing(self, train, test):
        self.fit_extractor(list(train.text))
        train_encodings = [self.extractor.tokenize(text, pad_length=self.pad_length) for text in list(train.text)]
        train_labels = np.array(train.dummy)
        train_ids = np.array(train.id)

        del train
        test_encodings = [self.extractor.tokenize(text, pad_length=self.pad_length) for text in list(test.text)]
        test_labels = np.array(test.dummy)
        test_ids = np.array(test.id)

        self.test_dataset = SariDataset(test_encodings, test_labels)

        return train_encodings, train_labels, train_ids, test_encodings, test_labels, test_ids

    def fit_model(self, train_encodings, train_labels):
        train_dataset = SariDataset(train_encodings, train_labels)

        model = CNNNGramNet(embedding_dim=self.embedding_dim,
                            vocab_size=self.extractor.vocab_size,
                            nb_class=len(set(train_labels)))

        trainer = Trainer(model,
                          train_dataset,
                          self.test_dataset,
                          Adam(model.parameters(), lr=self.lr),
                          training_args=self.training_args)
        trainer.train()
        return model

    def infer(self, model, test_encodings) -> (np.array, np.array):
        trainer = Trainer(model,
                          None,
                          None,
                          self.training_args)

        test_dataset = SariDataset(test_encodings, [0] * (len(test_encodings)))
        predictions, scores = trainer.infer(test_dataset=test_dataset)
        return predictions, scores


class SariDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        # self.ids = ids

    def __getitem__(self, idx):
        item = {"input_ids": torch.tensor(self.encodings[idx]), "labels": torch.tensor(self.labels[idx])}
        # item['ids'] = self.ids[idx]
        return item

    def __len__(self):
        return len(self.labels)
