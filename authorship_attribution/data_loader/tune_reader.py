# pylint: disable=duplicate-code
import random
import logging
from typing import List, Optional

import pandas as pd


class DataReader:
    def __init__(self, data_path: str,
                 subreddit_data_path: str,
                 min_samples_per_author: int = 10,
                 max_samples_per_author: Optional[int] = None,
                 compensate_test_split=False,
                 train_sample_ratio : float = 0.9):

        """
        :param data_path: path to the source dataframe (formatted csv file)
        :param min_samples_per_author: minimum number of samples for authors in the experiments
        :param max_samples_per_author: maximum number of samples for authors in the experiments
        :param compensate_test_split: In single-domain cases, setting this to True guarantees min_samples_per_author
            samples in the training set. Otherwise, the train/test split is done from the min_samples_per_author samples.

        Data should ideally already processed and has a length of 500 words.
        """
        self.train_sample_ratio = train_sample_ratio
        self.data = pd.read_csv(data_path, usecols=["user", "text", "id"],
                                dtype={"user": "str", "text": "str", "id": "str"}).dropna().drop_duplicates()

        self.extra_samples = max(2, int(0.20*min_samples_per_author)) if compensate_test_split else 0

        self.users = self.data.user.value_counts(ascending=False)
        self.min_samples_per_author = min_samples_per_author
        self.max_samples_per_author = max_samples_per_author if max_samples_per_author else self.users[0]

        self.users = self.users[
            (self.users >= self.min_samples_per_author) & (self.users < self.max_samples_per_author)]

        self.users = sorted(list(self.users.index))

        self.aux_data = pd.read_csv(subreddit_data_path, usecols=["user", "text", "subreddit", "id"],
                                                dtype={"user": "str", "text": "str", "subreddit": "str",
                                                        "id": "str"}).dropna().drop_duplicates()

        self.aux_users = self.aux_data.user.value_counts(ascending=False)
        self.users = sorted(list(set(self.users).intersection(set(self.aux_users.index))))


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
        
        user2cat = dict(zip(list(subset), range(len(subset))))
        
        #intermediate_df["dummy"] = intermediate_df.user.astype("category").cat.codes
        intermediate_df["dummy"] = intermediate_df.user.apply(lambda x: user2cat[x])

        train_df, test_df = None, None

        for user in subset:
            tmp = intermediate_df[intermediate_df.user == user]
            tmp = tmp.sample(min(self.min_samples_per_author + self.extra_samples, len(tmp)), random_state=random_seed)
            train_len = int(tmp.shape[0] * self.train_sample_ratio)
            train_df = pd.concat([train_df, tmp.iloc[:train_len]])
            test_df = pd.concat([test_df, tmp.iloc[train_len:]])

        intermediate_aux = self.aux_data.copy()[self.aux_data.user.apply(lambda x: x in subset)]
        train_df.dummy = train_df.user.apply(lambda x: user2cat[x])
        intermediate_aux["dummy"] = intermediate_aux.user.apply(lambda x: user2cat[x])
        
        test_aux_df = None

        for user in subset:
                tmp = intermediate_aux[intermediate_aux.user == user]
                tmp = tmp.sample(min(self.min_samples_per_author, len(tmp)), random_state=random_seed)
                test_aux_df = pd.concat([test_aux_df, tmp])

        print(test_aux_df.columns)
        print(test_aux_df.shape)

        
        
        '''
        if open_world:
            unknown_df = self.data.copy()[self.data.user.apply(lambda x: x not in subset)]
            unknown_df = unknown_df.sample(len(test_df), random_state=random_seed)
            unknown_df["dummy"] = -1
            test_df = pd.concat([test_df, unknown_df])
        '''
        print(train_df.shape)
        print(test_df.shape)
        return train_df, test_df, test_aux_df, subset
