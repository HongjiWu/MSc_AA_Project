import random
from torch.utils.data import Dataset


class TripletDataset(Dataset):
    def __init__(self, examples_array, users=None, training=True):
        """
        :param examples_array: Should be an array with Array[List[str]] with each row corresponding to an author
        :param training: Make the process deterministic or stochastic
        """
        self.data = examples_array
        self.users = users

        if self.users:
            assert len(self.data) == len(self.users)

        # self.data = examples_array
        self.training = training
        self.n_labels = len(self.data)

    def __len__(self):
        """Make epochs longer"""
        return self.n_labels * 100

    def make_sample(self, idx):
        example = self.data[idx]

        anchor_label = self.users[idx] if self.users else None

        if self.training:
            # random choice for question and for answer from within the author pool
            sampled_pair = random.sample(example, k=2)
            anchor_sample = sampled_pair[0]
            positive_sample = sampled_pair[1]
            negative_idx = random.choice([i for i in range(self.n_labels) if i != idx])
            negative_sample = random.choice(self.data[negative_idx])
        else:
            # deterministic
            anchor_sample = example[0]
            positive_sample = example[-1]
            negative_idx = (idx + 1) % self.n_labels
            negative_sample = self.data[negative_idx][0]

        return anchor_sample, positive_sample, negative_sample, anchor_label

    def __getitem__(self, idx):
        return self.make_sample(idx % self.n_labels)
