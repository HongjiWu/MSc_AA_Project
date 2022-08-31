# pylint: disable=duplicate-code
import random
from typing import List, Optional
import logging

import pandas as pd

from authorship_attribution.data_loader.file_reader import DataReader


class CrossRedditReader:
    def __init__(self,
                 eli5_data_path: str,
                 subreddit_data_path: str,
                 min_samples_per_author: int = 10,
                 max_samples_per_author: Optional[int] = None,
                 force_single_topic: bool = False
                 ):
        """

        :param eli5_data_path: path to the source dataframe (formatted csv file)
        :param subreddit_data_path: path to the target dataframe (formatted csv file)
        :param min_samples_per_author: minimum number of samples for authors in the experiments
        :param max_samples_per_author: maximum number of samples for authors in the experiments
        :param force_single_topic: This parameter is useful for the domain heterogeneity experiments. Setting it to true
            guarantees every author in the data has data from a unique subreddit.

        We assume min samples is defined for the training set (reddit eli5) in this case.
        Use the CrossDomain Reader to enforce a min samples per author in the target data.
        """

        self.eli5_reader = DataReader(eli5_data_path, min_samples_per_author, max_samples_per_author)
        self.min_samples_per_author = min_samples_per_author
        self.force_single_topic = force_single_topic

        self.data = pd.read_csv(subreddit_data_path, usecols=["user", "text", "subreddit", "id"],
                                dtype={"user": "str", "text": "str", "subreddit": "str",
                                       "id": "str"}).dropna().drop_duplicates()

        self.users = self.data.user.value_counts(ascending=False)
        self.users = sorted(list(set(self.eli5_reader.users).intersection(set(self.users.index))))

        logging.info(f"{len(self.users)} users selected in the external subreddit test set.")

        # Can be commented out for speed
        # self.data = self.data[self.data.user.apply(lambda x: x in self.users)]

    # pylint: disable=unused-argument
    def subsample_split_df(self,
                           num_authors=10,
                           split=1,
                           random_seed=None,
                           authors: List[str] = None,
                           domain=None,
                           # add_eli5=False,
                           open_world=False):

        """

        :param num_authors: Number of authors in the output df
        :param split: Split fraction for train/test splits
        :param random_seed: Set seed for deterministic experiments
        :param domain: None - Here for API uniformity
        :param authors: Specify author names
        :param open_world: Open world scenario - Not supported for Cross-domain
        :return: train_df, test_df
        """
        random.seed(random_seed)

        data = self.data
        users = self.users

        if self.force_single_topic:
            tmp = self.data.groupby(["subreddit", "user"]).user.unique().groupby("subreddit").count()
            # if add_eli5:
            #    topic = random.choice(list(tmp[tmp > num_authors].index) + ["explainlikeimfive"])
            #else:
            topic = random.choice(list(tmp[tmp > num_authors].index))

            if topic != "explainlikeimfive":
                data = self.data[self.data.subreddit == topic]
                users = sorted(list(set(data.user)))

            logging.info(f"{topic} selected, with {len(users)} users.")
        logging.info("1")
        if authors is None:
            subset = random.sample(list(users), k=min(num_authors, len(users)))
        else:
            subset = random.sample(authors, k=min(num_authors, len(authors)))

        logging.info("2")
        train_df, test_df, author = self.eli5_reader.subsample_split_df(num_authors,
                                                                authors=subset,
                                                                random_seed=random_seed,
                                                                open_world=False)

        intermediate_df = data.copy()[data.user.apply(lambda x: x in subset)]
        logging.info("2")
        # Add ELI5 data to the test set for intra-subreddit baseline
        # if add_eli5:
        #    test_df["subreddit"] = "explainlikeimfive"
        #    if self.force_single_topic and topic == "explainlikeimfive":
        #         intermediate_df = test_df.drop("dummy", axis=1)
        #     else:
        #         intermediate_df = pd.concat([intermediate_df, test_df.drop("dummy", axis=1)])

        user2cat = dict(zip(list(subset), range(len(subset))))
        train_df.dummy = train_df.user.apply(lambda x: user2cat[x])
        intermediate_df["dummy"] = intermediate_df.user.apply(lambda x: user2cat[x])

        test_df = None
        for user in subset:
            tmp = intermediate_df[intermediate_df.user == user]
            tmp = tmp.sample(min(self.min_samples_per_author, len(tmp)), random_state=random_seed)
            test_df = pd.concat([test_df, tmp])

        if open_world:
            unknown_df = self.data.copy()[self.data.user.apply(lambda x: x not in subset)]
            unknown_df = unknown_df.sample(len(test_df), random_state=random_seed)

            unknown_df["dummy"] = -1
            test_df = pd.concat([test_df, unknown_df])

        return train_df, test_df, subset
