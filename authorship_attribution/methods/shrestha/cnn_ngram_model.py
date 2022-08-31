import torch


class CNNNGramNet(torch.nn.Module):
    def __init__(self, embedding_dim: int, vocab_size: int, nb_class: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        channels = 500
        self.dropout = torch.nn.Dropout(p=0.25, inplace=False)
        self.cnn_3 = torch.nn.Conv1d(in_channels=embedding_dim, out_channels=channels, kernel_size=3)
        self.cnn_4 = torch.nn.Conv1d(in_channels=embedding_dim, out_channels=channels, kernel_size=4)
        self.cnn_5 = torch.nn.Conv1d(in_channels=embedding_dim, out_channels=channels, kernel_size=5)

        self.pooling3 = torch.nn.AdaptiveMaxPool2d(output_size=(None, 1))
        self.pooling4 = torch.nn.AdaptiveMaxPool2d(output_size=(None, 1))
        self.pooling5 = torch.nn.AdaptiveMaxPool2d(output_size=(None, 1))

        self.linear = torch.nn.Linear(in_features=3*channels, out_features=nb_class)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)

        x3 = self.pooling3(torch.sigmoid(self.cnn_3(x))).squeeze()
        x4 = self.pooling4(torch.sigmoid(self.cnn_4(x))).squeeze()
        x5 = self.pooling5(torch.sigmoid(self.cnn_5(x))).squeeze()

        x = torch.cat((x3, x4, x5), 1)
        x = self.linear(x)
        return x

    def save_pretrained(self, model_path):
        torch.save(self.state_dict(), model_path)

    def from_pretrained(self, model_path):
        state_dict = torch.load(model_path)
        del state_dict['linear.bias']
        del state_dict['linear.weight']
        self.load_state_dict(state_dict, strict=False)
