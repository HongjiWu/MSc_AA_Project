import numpy as np
from torch.optim.adam import Adam

from transformers import AutoTokenizer

from authorship_attribution.helpers.tokenizer_dataset import TokenizerDataset
from authorship_attribution.methods.sari.ngram_embedding_model import ContinuousNGramNet

from authorship_attribution.methods.base_aa_method import BaseAAMethod
from authorship_attribution.methods.sari.model_training import Trainer, TrainingArgs




class SariWordpieceMethod(BaseAAMethod):
    def __init__(self,
                 embedding_dim: int = 518,
                 pad_length: int = 1000,
                 tokenizer_name: str = "bert-base-cased",
                 training_args: TrainingArgs = TrainingArgs(),
                 num_linear=1,
                 learning_rate = 0.001
                 ):
        super().__init__()

        self.extractor = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.embedding_dim = embedding_dim
        self.pad_length = pad_length
        self.training_args = training_args
        self.num_linear = num_linear
        self.learning_rate = learning_rate
        self.name = self.name + f"_{self.training_args.batch_size}" + f"_{self.training_args.train_epochs}"
        if self.num_linear != 1:
            self.name = self.name + f"_lin_{self.num_linear}"

        self.test_dataset = None

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

        model = ContinuousNGramNet(embedding_dim=self.embedding_dim,
                                   vocab_size=self.extractor.vocab_size,
                                   nb_class=len(set(train_labels)),
                                   num_linear=self.num_linear)

        trainer = Trainer(model,
                          train_dataset,
                          self.test_dataset,
                          Adam(model.parameters(), lr=self.learning_rate),
                          training_args=self.training_args)
        trainer.train()

        # acc = float(np.mean(trainer.infer()))
        return model

    def infer(self, model, test_encodings) -> (np.array, np.array):
        trainer = Trainer(model,
                          None,
                          None,
                          self.training_args)

        test_dataset = TokenizerDataset(test_encodings, [0] * (len(test_encodings.input_ids)))
        predictions, scores = trainer.infer(test_dataset=test_dataset)
        return predictions, scores
