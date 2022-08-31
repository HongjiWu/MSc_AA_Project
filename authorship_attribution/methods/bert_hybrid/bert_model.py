import torch
from transformers import AutoModel


class BertNet(torch.nn.Module):
    def __init__(self, nb_class: int, model_name="bert-base-cased", embedding_dim: int = 518, freeze_bert=True):
        super().__init__()
        self.embedding = AutoModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(p=0.5, inplace=False)
        self.pooling = torch.nn.AdaptiveAvgPool2d(output_size=(1, None))
        self.linear = torch.nn.Linear(in_features=embedding_dim, out_features=nb_class)
        self.freeze_bert = freeze_bert

        if self.freeze_bert:
            for param in self.embedding.parameters():
                param.requires_grad = False

    def forward(self, **kwargs):
        x = self.embedding(**kwargs, return_dict=True)
        x = self.dropout(x.last_hidden_state)
        x = self.pooling(x).squeeze()
        x = self.linear(x)
        return x

    def save_pretrained(self, embedding_path):
        torch.save(self.embedding, embedding_path)

    def from_pretrained(self, embedding_path):
        self.embedding = torch.load(embedding_path)
        if self.freeze_bert:
            for param in self.embedding.parameters():
                param.requires_grad = False
