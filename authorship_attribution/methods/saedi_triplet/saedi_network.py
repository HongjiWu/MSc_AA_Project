import torch
from torch import nn


class SaediNetwork(nn.Module):
    def __init__(self, embedding_dim, vocab_size, output_dim=400):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size,
                                            embedding_dim=embedding_dim)

        self.pooling = torch.nn.AdaptiveMaxPool2d(output_size=(None, 1))
        self.dropout = nn.Dropout(p=0.3, inplace=False)

        self.conv = nn.Sequential(
            nn.Conv1d(embedding_dim, 350, 1),
            nn.ReLU(),
            nn.Conv1d(350, 300, 2),
            nn.ReLU(),
            nn.Conv1d(300, 250, 3),
            nn.ReLU(),
            nn.Conv1d(250, 250, 3),
        )

        self.linear_seq = nn.Sequential(
            nn.Linear(250, output_dim)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.conv(x.transpose(1, 2))
        x = self.pooling(x)
        x = x.view(-1, 250)
        # x = self.dropout(x)
        x = self.linear_seq(x)
        x = nn.functional.normalize(x)
        return x

    def save_pretrained(self, model_path):
        torch.save(self.state_dict(), model_path)

    def from_pretrained(self, model_path):
        state_dict = torch.load(model_path)
        self.load_state_dict(state_dict, strict=False)


class ContinuousNGramEmbeddingNet(nn.Module):
    def __init__(self, embedding_dim: int, vocab_size: int, output_dim: int = 128):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size,
                                            embedding_dim=embedding_dim)
        self.dropout = nn.Dropout(p=0.3, inplace=False)
        self.pooling = nn.AdaptiveAvgPool2d(output_size=(1, None))
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=embedding_dim, out_features=output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x = self.pooling(x).squeeze()
        x = self.linear2(x)
        x = nn.functional.normalize(x)
        return x

    def save_pretrained(self, model_path):
        torch.save(self.state_dict(), model_path)

    def from_pretrained(self, model_path):
        state_dict = torch.load(model_path)
        self.load_state_dict(state_dict, strict=False)
