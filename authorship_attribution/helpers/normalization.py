from typing import Optional
import numpy as np


class Normalizer:
    def __init__(self, features, method: str = "row-norm"):
        self.param1 = None
        self.param2 = None

        assert method in ["row-norm", "feature-mean", "feature-mean-nonzero"]
        self.method = method

        self.features = features
        self.compute_params()

    def compute_params(self):
        if self.method == "feature-mean":
            self.param2 = np.clip(self.features.std(axis=0), 0.01, None)
            self.param1 = self.features.mean(axis=0)
        if self.method == "feature-mean-nonzero":
            self.features[:, self.features.sum(axis=0) == 0] = 1
            self.param2 = self.features.sum(axis=0)/np.count_nonzero(self.features, axis=0, keepdims=True)

    def normalize(self, features: Optional[np.array]) -> np.array:
        if features is None:
            features = self.features

        if self.method == "row-norm":
            return features / np.linalg.norm(features, axis=1, keepdims=True)
        if self.method == "feature-mean":
            return (features - self.param1) / self.param2
        if self.method == "feature-mean-nonzero":
            return features / self.param2
        return features
