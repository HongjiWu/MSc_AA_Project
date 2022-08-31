# pylint: disable=duplicate-code
import random
import logging
from typing import List, Optional

import pandas as pd


class DataReader:
    def __init__(self, data_path: str,
                 min_samples_per_author: int = 10,
                 max_samples_per_author: Optional[int] = None,
                 compensate_test_split=False):
        """
        :param data_path: path to the source dataframe (formatted csv file)
        :param min_samples_per_author: minimum number of samples for authors in the experiments
        :param max_samples_per_author: maximum number of samples for authors in the experiments
        :param compensate_test_split: In single-domain cases, setting this to True guarantees min_samples_per_author
            samples in the training set. Otherwise, the train/test split is done from the min_samples_per_author samples.

        Data should ideally already processed and has a length of 500 words.
        """

        self.data = pd.read_csv(data_path, usecols=["user", "text", "id"],
                                dtype={"user": "str", "text": "str", "id": "str"}).dropna().drop_duplicates()

        self.extra_samples = max(2, int(0.20*min_samples_per_author)) if compensate_test_split else 0

        self.users = self.data.user.value_counts(ascending=False)
        self.min_samples_per_author = min_samples_per_author
        self.max_samples_per_author = max_samples_per_author if max_samples_per_author else self.users[0]

        self.users = self.users[
            (self.users >= self.min_samples_per_author) & (self.users < self.max_samples_per_author)]

        self.users = sorted(list(self.users.index))

        logging.info(
            f"{len(self.users)} users selected in the range of {self.min_samples_per_author}"
            f"-{self.max_samples_per_author} texts.")

        # Can be commented out for speed (disables open-world)
        # self.data = self.data[self.data.user.apply(lambda x: x in self.users)]

    # pylint: disable=unused-argument
    def subsample_split_df(self,
                           num_authors=10,
                           random_seed=None,
                           authors: List[str] = None,
                           domain=None,
                           open_world=False):

        """
        :param num_authors: Number of authors in the output df
        :param random_seed: Set seed for deterministic experiments
        :param domain: None - Here for compatibility
        :param authors: Specify author names
        :param open_world: Open world scenario - Not supported for Cross-domain
        :return: train_df, test_df
        """

        random.seed(random_seed)
        users = self.users

        if authors is None:
            subset = random.sample(list(users), k=min(num_authors, len(users)))
        else:
            subset = random.sample(authors,  k=min(num_authors, len(authors)))

        intermediate_df = self.data.copy()[self.data.user.apply(lambda x: x in subset)]
        intermediate_df["dummy"] = intermediate_df.user.astype("category").cat.codes

        train_df, test_df = None, None

        for user in subset:
            tmp = intermediate_df[intermediate_df.user == user]
            tmp = tmp.sample(min(self.min_samples_per_author + self.extra_samples, len(tmp)), random_state=random_seed)

            train_df = pd.concat([train_df, tmp.iloc[:self.min_samples_per_author]])
            test_df = pd.concat([test_df, tmp.iloc[self.min_samples_per_author:]])

        if open_world:
            unknown_df = self.data.copy()[self.data.user.apply(lambda x: x not in subset)]
            unknown_df = unknown_df.sample(len(test_df), random_state=random_seed)
            unknown_df["dummy"] = -1
            test_df = pd.concat([test_df, unknown_df])

        return train_df, test_df, subset
