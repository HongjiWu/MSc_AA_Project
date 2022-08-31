# pylint: disable=too-many-arguments
from dataclasses import dataclass
from typing import Optional
import os
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from authorship_attribution.methods.bert.bert_model import BertNet


@dataclass
class TrainingArgs:
    train_epochs: int = 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 50
    print_progress: bool = False
    embedding_output_path: Optional[str] = None
    from_pretrained: Optional[str] = None


class Trainer:
    def __init__(self,
                 model: BertNet,
                 train_dataset,
                 test_dataset,
                 optimizer,
                 loss_function=nn.CrossEntropyLoss(),
                 training_args: TrainingArgs = TrainingArgs()):
        self.model = model.to(training_args.device)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.training_args = training_args
        self.writer = SummaryWriter()
        self.train_epochs = min(self.training_args.train_epochs, 20)
        self.batch_size = max(self.training_args.batch_size, 10)  # Upper bound

        if self.training_args.from_pretrained and os.path.isfile(self.training_args.from_pretrained):
            self.model.from_pretrained(self.training_args.from_pretrained)

    def train(self):

        # self.infer(epoch=-1)
        self.model.train()
        data_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in tqdm(range(self.train_epochs)):
            total_loss = 0
            for batch in data_loader:
                # inputs = batch['input_ids'].to(self.training_args.device)
                labels = batch['labels'].to(self.training_args.device).long()
                self.model.zero_grad()
                output = self.model(input_ids=batch["input_ids"].to(self.training_args.device),
                                    token_type_ids=batch["token_type_ids"].to(self.training_args.device),
                                    attention_mask=batch["attention_mask"].to(self.training_args.device))
                if len(output.size()) == 1:
                    output = output.unsqueeze(0)
                loss = self.loss_function(output, labels)
                loss.backward()
                self.optimizer.step()

                # Get the Python number from a 1-element Tensor by calling tensor.item()
                total_loss += loss.item()
            self.writer.add_scalar("Loss/train", total_loss, epoch)
            # self.infer(epoch=epoch)
            if self.training_args.print_progress:
                print(f"Epoch {epoch} --> {total_loss}")

        if self.training_args.embedding_output_path:
            self.model.save_pretrained(self.training_args.embedding_output_path)

    def infer(self, test_dataset=None):
        self.model.eval()
        data_loader = DataLoader(test_dataset if test_dataset else self.test_dataset,
                                 batch_size=self.batch_size,
                                 shuffle=False)

        results = []
        scores = []
        for batch in data_loader:
            with torch.no_grad():
                output = self.model(input_ids=batch["input_ids"].to(self.training_args.device),
                                    token_type_ids=batch["token_type_ids"].to(self.training_args.device),
                                    attention_mask=batch["attention_mask"].to(self.training_args.device))

                if len(output.size()) == 1:
                    output = output.unsqueeze(0)

                values, indices = torch.sort(output, descending=True)
                results.extend(indices.cpu().numpy())
                scores.extend(torch.softmax(values, dim=1).cpu().numpy())

        return np.array(results), np.array(scores)
