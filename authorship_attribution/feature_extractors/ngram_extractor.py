from typing import List, Optional
from collections import Counter
import logging
import string

import numpy as np
import logging
from authorship_attribution.feature_extractors.base_extractor import BaseExtractor


class NGramKoppelExtractor(BaseExtractor):
    def __init__(self, n=4, vocab_size=None, split_words=True, keep_rare_chars=True):
        """
        :param n: n-gram size
        :param vocab_size: default: None (unlimited vocab size). An int value keeps the (vocab_size -
        len(default_tokens)) most frequent n-grams.
        """
        self.mapping = None
        self.split_words = split_words
        self.vocab_size = vocab_size
        self.default_tokens = ["_pad_", "_unk_"]
        self.keep_rare_chars = keep_rare_chars
        self.common_chars = ' '+ ':;?.!,"' + string.ascii_uppercase + string.ascii_lowercase + string.digits
        self.n = n

    def fit(self, texts: List[str]):
        logging.info(f"Building vocabulary for {len(texts)} samples.")
        vocab = set()

        if not self.vocab_size:
            counter = Counter()
            for text in texts:
                counter.update(self.get_ngrams(text))
            vocab = [key for key, value in counter.most_common(int(len(counter)))]
        else:
            counter = Counter()
            for text in texts:
                counter.update(self.get_ngrams(text))
            vocab = [key for key, value in counter.most_common(self.vocab_size - len(self.default_tokens))]

        self.vocab_size = len(vocab) + len(self.default_tokens)
        self.mapping = dict(zip(self.default_tokens + list(vocab), range(self.vocab_size)))

        logging.info(f"Vocabulary size: {self.vocab_size}")

    def get_ngrams(self, text: str):
        """Note: Case sensitive"""
        if not self.keep_rare_chars:
            text = "".join(list(filter(lambda x: x in self.common_chars, text)))

        if not self.split_words:
            return [text[i:i + self.n] for i in range(max(len(text) - self.n + 1, 1))]

        ngrams = []
        for word in text.split():
            ngrams.extend([word[i:i + self.n] for i in range(max(len(word) - self.n + 1, 1))])
        return ngrams

    def tokenize(self, text: str, pad_length: Optional[int] = None):
        """Note: All Ngrams should be in the vocab !"""

        if self.vocab_size:
            unk_value = self.mapping["_unk_"]
            encoding = [self.mapping.get(ngram, unk_value) for ngram in self.get_ngrams(text)]
        else:
            # Faster than dict.get()
            encoding = [self.mapping[ngram] for ngram in self.get_ngrams(text)]

        if pad_length:
            if len(encoding) > pad_length:
                return encoding[:pad_length]
            return encoding + [self.mapping["_pad_"]]*(pad_length - len(encoding))
        return encoding

    def transform_sample(self, text):
        unk_value = self.mapping["_unk_"]
        vector = np.zeros((self.vocab_size,), dtype=int)
        for ngram in self.get_ngrams(text):
            try:
                vector[self.mapping[ngram]] += 1
            except KeyError:
                vector[unk_value] += 1
        return vector

    def transform(self, texts: List[str]):
        transformed = np.array(list(map(self.transform_sample, texts)))
        return transformed/transformed.sum(axis=1, keepdims=True)
