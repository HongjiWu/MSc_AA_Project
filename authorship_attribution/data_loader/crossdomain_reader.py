# pylint: disable=duplicate-code

import random
from typing import Optional, List
import logging

import pandas as pd


class CrossDomainReader:
    def __init__(self,
                 source_data_path: str,
                 target_data_path: str,
                 min_samples_per_author: int = 10,
                 max_samples_per_author: Optional[int] = None,
                 force_single_topic: bool = False):
        """
        :param source_data_path: path to the source dataframe (formatted csv file)
        :param target_data_path: path to the target dataframe (formatted csv file)
        :param min_samples_per_author: minimum number of samples for authors in the experiments
        :param max_samples_per_author: maximum number of samples for authors in the experiments
        :param force_single_topic: This parameter is useful for the domain heterogeneity experiments. Setting it to true
            guarantees every author in the data has data from a unique subreddit.
        """

        self.min_samples_per_author = min_samples_per_author
        self.force_single_topic = force_single_topic

        # TODO: For the moment, data needs a subreddit column for it to work - change or explain in readme

        self.source_data = pd.read_csv(source_data_path, usecols=["user", "text", "id"],
                                       dtype={"user": "str", "text": "str",
                                              "subreddit": "str"}).dropna().drop_duplicates()

        self.target_data = pd.read_csv(target_data_path, usecols=["user", "text", "id", "subreddit"],
                                       dtype={"user": "str", "text": "str", "subreddit": "str"}).dropna().drop_duplicates()

        source_users = self.source_data.user.value_counts(ascending=False)
        target_users = self.target_data.user.value_counts(ascending=False)

        self.max_samples_per_author = max_samples_per_author if max_samples_per_author else max(source_users[0],
                                                                                                target_users[0])

        source_users = source_users[
            (source_users >= self.min_samples_per_author) & (source_users < self.max_samples_per_author)]

        target_users = target_users[
            (target_users >= self.min_samples_per_author) & (target_users < self.max_samples_per_author)]

        self.users = sorted(list(set(source_users.index).intersection(set(target_users.index)))) if not self.force_single_topic else sorted(list(target_users.index))

        logging.info(
            f"{len(self.users)} users selected in the range of {self.min_samples_per_author}"
            f"-{self.max_samples_per_author} texts.")

        self.source_data = self.source_data[self.source_data.user.apply(lambda x: x in self.users)]
        self.target_data = self.target_data[self.target_data.user.apply(lambda x: x in self.users)]

    def subsample_split_df(self, num_authors=10, random_seed=None, domain="rt", authors: List[str] = None,
                           open_world=False):

        """

        :param num_authors: Number of authors in the output df
        :param random_seed: Set seed for deterministic experiments
        :param domain: Parameter to choose what the source and target data is: Choose between ["rt", "tr", "tt", "rr"]
        :param authors: Specify author names
        :param open_world: Open world scenario - Not supported for Cross-domain
        :return: train_df, test_df
        """

        assert domain in ["rt", "tr", "tt", "rr"]
        logging.info(f"CrossDomain type: {domain}")
        if open_world:
            logging.info("Open world not supported yet for CrossDomain data")

        random.seed(random_seed)

        if authors is None:
            subset = random.sample(list(self.users), k=min(num_authors, len(self.users)))
        else:
            subset = random.sample(authors, k=min(num_authors, len(authors)))

        # mapping user hashes to id
        if domain in ["rt", "tr"]:

            user2cat = dict(zip(list(subset), range(len(subset))))

            intermediate_target_df = self.target_data.copy()[self.target_data.user.apply(lambda x: x in subset)]
            intermediate_target_df["dummy"] = intermediate_target_df.user.apply(lambda x: user2cat[x])

            intermediate_source_df = self.source_data.copy()[self.source_data.user.apply(lambda x: x in subset)]
            intermediate_source_df["dummy"] = intermediate_source_df.user.apply(lambda x: user2cat[x])

            source_df, target_df = None, None
            for user in subset:
                tmp_source = intermediate_source_df[intermediate_source_df.user == user].sample(
                    self.min_samples_per_author, random_state=random_seed)
                tmp_target = intermediate_target_df[intermediate_target_df.user == user].sample(
                    self.min_samples_per_author, random_state=random_seed)

                source_df = pd.concat([source_df, tmp_source])
                target_df = pd.concat([target_df, tmp_target])

            if domain == "tr":
                return target_df, source_df
            return source_df, target_df

        if domain == "rr":
            return self._subsample_split_intra(self.source_data, random_seed=random_seed, split=0.8, authors=subset)

        if domain == "tt":
            return self._subsample_split_intra(self.target_data, random_seed=random_seed, split=0.8, authors=subset)

        raise ValueError

    def _subsample_split_intra(self, data, split=0.8, random_seed=None, authors: List[str] = None):
        random.seed(random_seed)
        subset = authors.copy()
        intermediate_df = data.copy()[data.user.apply(lambda x: x in subset)]

        # Here add the script for single-topic forcing
        if self.force_single_topic and ("subreddit" in data.columns):
            grouped = data.groupby(["subreddit", "user"]).count().reset_index()
            sampled_subs = grouped[grouped.id >= (self.min_samples_per_author)].sample(
                frac=1, random_state=random_seed).drop_duplicates(subset="subreddit").drop_duplicates(
                subset="user")

            assert len(sampled_subs.subreddit.unique()) == len(sampled_subs)
            assert len(sampled_subs.user.unique()) == len(sampled_subs)

            combs = [tuple(pair) for pair in sampled_subs.iloc[:, 0:2].values]

            intermediate_df = data[data.apply(lambda x: (x['subreddit'], x["user"]) in combs, axis=1)]
            subset = random.sample(list(intermediate_df.user.unique()), k=min(len(intermediate_df), len(authors)))
            intermediate_df = intermediate_df[intermediate_df.user.apply(lambda x: x in subset)]

        intermediate_df["dummy"] = intermediate_df.user.astype("category").cat.codes
        logging.info("Splitting train/test from intradomain dataset")

        train_df, test_df = None, None
        for user in subset:
            tmp = intermediate_df[intermediate_df.user == user].sample(self.min_samples_per_author,
                                                                       random_state=random_seed)

            split_ = int(len(tmp) * split)
            train_df = pd.concat([train_df, tmp.iloc[:split_]])
            test_df = pd.concat([test_df, tmp.iloc[split_:]])

        return train_df, test_df
