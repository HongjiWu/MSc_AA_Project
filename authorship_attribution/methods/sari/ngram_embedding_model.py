import torch


class ContinuousNGramNet(torch.nn.Module):
    def __init__(self, embedding_dim: int, vocab_size: int, nb_class: int, num_linear: int = 1, drop_out = 0.75):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size,
                                            embedding_dim=embedding_dim)
        self.dropout = torch.nn.Dropout(p=drop_out, inplace=False)
        # self.pooling = torch.nn.AvgPool2d(kernel_size=(pad_length, 1))
        self.pooling = torch.nn.AdaptiveAvgPool2d(output_size=(1, None))

        assert isinstance(num_linear, int)
        assert num_linear > 0

        if num_linear == 1:
            self.linear = torch.nn.Linear(in_features=embedding_dim, out_features=nb_class)
        else:
            linear = []
            for _ in range(num_linear - 1):
                linear.append(torch.nn.Linear(embedding_dim, embedding_dim))
                linear.append(torch.nn.ReLU())
            linear.append(torch.nn.Linear(embedding_dim, nb_class))
            self.linear = torch.nn.Sequential(*linear)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x = self.pooling(x).squeeze()
        x = self.linear(x)
        return x

    def save_pretrained(self, embedding_path):
        torch.save(self.embedding, embedding_path)

    def from_pretrained(self, embedding_path):
        self.embedding = torch.load(embedding_path)
