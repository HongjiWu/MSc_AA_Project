from typing import List


class BaseExtractor:
    """Base Class for feature extraction"""
    def fit(self, texts: List[str]):
        pass

    def transform(self, texts: List[str]):
        raise NotImplementedError
