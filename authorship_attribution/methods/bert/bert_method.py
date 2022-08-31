import numpy as np
from torch.optim.adam import Adam
from transformers import AutoTokenizer

from authorship_attribution.helpers.tokenizer_dataset import TokenizerDataset
from authorship_attribution.methods.bert.bert_model import BertNet

from authorship_attribution.methods.base_aa_method import BaseAAMethod
from authorship_attribution.methods.bert.model_training import Trainer, TrainingArgs


class BertMethod(BaseAAMethod):
    def __init__(self,
                 embedding_dim: int = 768,
                 pad_length: int = 500,
                 tokenizer_name: str = "bert-base-cased",
                 learning_rate = 0.01,
                 training_args: TrainingArgs = TrainingArgs()):
        super().__init__()

        self.extractor = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.embedding_dim = embedding_dim
        self.pad_length = pad_length
        self.training_args = training_args
        self.test_dataset = None
        self.lr = learning_rate

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

        model = BertNet(nb_class=len(set(train_labels)),
                        embedding_dim=self.embedding_dim,
                        freeze_bert=True)

        trainer = Trainer(model,
                          train_dataset,
                          self.test_dataset,
                          Adam(model.parameters(), lr=self.lr),
                          training_args=self.training_args)
        trainer.train()

        # acc = float(np.mean(trainer.infer()))
        return model

    def infer(self, model, test_features) -> (np.array, np.array):
        trainer = Trainer(model,
                          None,
                          None,
                          self.training_args)

        test_dataset = TokenizerDataset(test_features, [0] * (len(test_features.encodings)))
        predictions, scores = trainer.infer(test_dataset=test_dataset)
        return predictions, scores
