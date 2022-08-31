import os
from typing import Union
from tqdm import tqdm

import numpy as np
import torch
import logging
from torch.utils.data import DataLoader

from authorship_attribution.methods.saedi_triplet.saedi_network import SaediNetwork, ContinuousNGramEmbeddingNet
from authorship_attribution.methods.saedi_triplet.triplet_loss import OnlineTripletLoss
from authorship_attribution.methods.sari.model_training import TrainingArgs


class Trainer:
    def __init__(self,
                 model: Union[ContinuousNGramEmbeddingNet, SaediNetwork],
                 train_dataset,
                 test_dataset,
                 optimizer=None,
                 loss_function=None,
                 training_args: TrainingArgs = TrainingArgs()):
        self.model = model.to(training_args.device)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.criterion = loss_function if loss_function is not None else OnlineTripletLoss(margin=1.0,
                                                                                           device=training_args.device,
                                                                                           batch_hard=True)

        self.training_args = training_args
        self.optimizer = optimizer

        if self.training_args.from_pretrained and os.path.isfile(self.training_args.from_pretrained):
            self.model.from_pretrained(self.training_args.from_pretrained)

    def train(self):

        train_loader = DataLoader(self.train_dataset, batch_size=self.training_args.batch_size, shuffle=True)

        self.model.train()
        #assert isinstance(self.criterion, OnlineTripletLoss)
        for epoch in tqdm(range(self.training_args.train_epochs), desc="Epochs"):
            running_loss = []
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.training_args.device)
                labels = batch['labels'].to(self.training_args.device).long()
                # self.optimizer.zero_grad()
                self.model.zero_grad()
                embeddings = self.model(input_ids).to(self.training_args.device)
                if isinstance(self.criterion, torch.nn.CosineEmbeddingLoss):
                    logging.info("cosine")
                    logging.info(str(labels.shape))
                    loss = self.criterion(labels, embeddings, torch.Tensor( labels.shape[0]).cuda().fill_(1.0))
                else:
                    loss = self.criterion(labels, embeddings)
                loss.backward()
                self.optimizer.step()
                running_loss.append(loss.cpu().detach().numpy())
            if self.training_args.print_progress:
                print("Epoch: {}/{} - Loss: {:.4f}".format(epoch + 1, self.training_args.train_epochs,
                                                           np.mean(running_loss)))

        if self.training_args.embedding_output_path:
            self.model.save_pretrained(self.training_args.embedding_output_path)

    def infer(self, test_dataset=None):
        self.model.eval()
        data_loader = DataLoader(test_dataset if test_dataset else self.test_dataset,
                                 batch_size=self.training_args.batch_size,
                                 shuffle=False)

        results = []
        for batch in data_loader:
            input_ids = batch['input_ids'].to(self.training_args.device)
            with torch.no_grad():
                output = self.model(input_ids)
                results.append(output.cpu().numpy())

        results = np.concatenate(results)
        return np.array(results)
