import numpy as np
from torch.utils.data import DataLoader
from torch.optim.adam import Adam


import torch
from transformers import AutoTokenizer

from scipy.special import softmax
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestCentroid


from authorship_attribution.helpers.tokenizer_dataset import TokenizerDataset
from authorship_attribution.methods.saedi_triplet.saedi_network import SaediNetwork

from authorship_attribution.methods.base_aa_method import BaseAAMethod
from authorship_attribution.methods.saedi_triplet.model_training_triplet import TrainingArgs, Trainer


class TripletSaediMethod(BaseAAMethod):
    def __init__(self,
                 embedding_dim: int = 300,
                 pad_length: int = 300,
                 tokenizer_name: str = "bert-base-cased",
                 learning_rate = 0.0005,
                 training_args: TrainingArgs = TrainingArgs()):
        super().__init__()

        self.extractor = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.embedding_dim = embedding_dim
        self.pad_length = pad_length
        self.training_args = training_args
        self.test_dataset = None
        self.triplet_model = None
        self.lr = learning_rate
        self.name = self.name + f"_{self.training_args.batch_size}" + f"_{self.training_args.train_epochs}"

    def data_processing(self, train, test):
        train_encodings = self.extractor(list(train.text), truncation=True, padding="max_length",
                                         max_length=self.pad_length)
        train_labels = np.array(train.dummy)
        train_ids = np.array(train.id)

        del train
        test_encodings = self.extractor(list(test.text), truncation=True, padding="max_length",
                                        max_length=self.pad_length)
        test_labels = np.array(test.dummy)
        test_ids = np.array(test.id)

        self.test_dataset = TokenizerDataset(test_encodings, test_labels)
        return train_encodings, train_labels, train_ids, test_encodings, test_labels, test_ids

    def fit_model(self, train_features, train_labels):
        train_dataset = TokenizerDataset(train_features, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=self.training_args.batch_size, shuffle=False)

        self.triplet_model = SaediNetwork(embedding_dim=self.embedding_dim,
                                          vocab_size=self.extractor.vocab_size,
                                          output_dim=400)

        trainer = Trainer(self.triplet_model,
                          train_dataset,
                          self.test_dataset,
                          optimizer=Adam(self.triplet_model.parameters(), lr=self.lr),
                          loss_function = torch.nn.CosineEmbeddingLoss(),
                          training_args=self.training_args)
        trainer.train()

        self.triplet_model.eval()
        train_results = []
        with torch.no_grad():
            for batch in train_loader:
                imgs = batch['input_ids'].to(self.training_args.device)
                train_results.append(self.triplet_model(imgs).cpu().numpy())

        train_results = np.concatenate(train_results)

        clf = NearestCentroid(metric="cosine").fit(train_results, train_labels)

        return clf

    def infer(self, model, test_encodings) -> (np.array, np.array):

        test_dataset = TokenizerDataset(test_encodings, [0] * (len(test_encodings.input_ids)))

        trainer = Trainer(self.triplet_model,
                          None,
                          None,
                          self.training_args)

        test_features = trainer.infer(test_dataset=test_dataset)
        del test_encodings
        del test_dataset

        if isinstance(test_features, list):
            predictions = []
            scores = []
            for i in range(0, len(test_features), self.training_args.batch_size):
                tmp_preds, tmp_scores = self.predict_top_n(model, self.extractor.transform(
                    test_features[i:i + self.training_args.batch_size]))
                predictions.extend(tmp_preds)
                scores.extend(tmp_scores)
        else:
            predictions, scores = self.predict_top_n(model, test_features)

        # Compute 1/distance to match ordering of classification methods
        return np.array(predictions), softmax(1/(np.array(scores) + 0.0001), axis=1)

    @staticmethod
    def predict_top_n(model, features):
        distances = pairwise_distances(features, model.centroids_, metric=model.metric)
        return model.classes_[np.argsort(distances, axis=1)], np.sort(distances, axis=1)
