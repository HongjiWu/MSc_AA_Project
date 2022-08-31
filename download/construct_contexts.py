import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()


class ContextConstructor:
    def __init__(self, source_data_path: str, target_data_path):
        self.source_data_path = source_data_path
        self.target_data_path = target_data_path
        self.df_source = pd.read_csv(source_data_path, index_col=0)
        self.df_target = pd.read_csv(target_data_path, index_col=0)

    def split_by_time(self):
        """Splits source and target data similarly as what is done in the experiments"""
        year = 3.154e7
        baseline = max(self.df_target.timestamp_max) - 2 * year

        df_target_cut = self.df_target[self.df_target.timestamp_min.apply(lambda x: x > baseline)]

        user_whitelist = set(df_target_cut.user.unique())
        df = self.df_source[self.df_source.user.apply(lambda x: x in user_whitelist)]

        df_vhigh = df[df.timestamp_min.apply(lambda x: x > (baseline - 8 * year)) & df.timestamp_max.apply(
            lambda x: x < (baseline - 6 * year))]
        df_high = df[df.timestamp_min.apply(lambda x: x > (baseline - 6 * year)) & df.timestamp_max.apply(
            lambda x: x < (baseline - 4 * year))]
        df_med = df[df.timestamp_min.apply(lambda x: x > (baseline - 4 * year)) & df.timestamp_max.apply(
            lambda x: x < (baseline - 2 * year))]
        df_low = df[df.timestamp_min.apply(lambda x: x > (baseline - 2 * year)) & df.timestamp_max.apply(
            lambda x: x < (baseline - 0 * year))]

        df_vhigh.to_csv(self.source_data_path[:-4] + "_time_vhigh.csv")
        df_high.to_csv(self.source_data_path[:-4] + "_time_high.csv")
        df_med.to_csv(self.source_data_path[:-4] + "_time_med.csv")
        df_low.to_csv(self.source_data_path[:-4] + "_time_low.csv")
        df_target_cut.to_csv(self.target_data_path[:-4] + "_time_cut.csv")

    def split_by_similarity(self, sim_scores_path):
        subs = self.df_target.groupby(["subreddit", "user"]).count().reset_index().groupby(
            "subreddit").count().sort_values(
            by="user", ascending=False)

        bin_class = pd.read_csv(sim_scores_path, usecols=["subreddit", "softmax"], index_col=0)

        subs_merged = subs.merge(bin_class, left_index=True, right_index=True).sort_values("softmax", ascending=False)[
            ["user", "softmax"]]

        subs_merged.columns = ["Number of authors", "Similarity score"]

        scores = subs_merged.sort_values("Number of authors", ascending=False).head(500).sort_values("Similarity score",
                                                                                                     ascending=False)[
            "Similarity score"].values
        thresh_1 = scores[int(len(scores) / 3)]
        thresh_2 = scores[int(len(scores) * 2 / 3)]

        high_sim = list(bin_class[(bin_class.softmax > thresh_1)].index)
        med_sim = list(bin_class[(bin_class.softmax > thresh_2) & (bin_class.softmax <= thresh_1)].index)
        low_sim = list(bin_class[(bin_class.softmax <= thresh_2)].index)

        df_high = self.df_target[self.df_target.subreddit.apply(lambda x: x in high_sim)]
        df_med = self.df_target[self.df_target.subreddit.apply(lambda x: x in med_sim)]
        df_low = self.df_target[self.df_target.subreddit.apply(lambda x: x in low_sim)]

        df_high.to_csv(self.target_data_path[:-4] + "_highsim.csv")
        df_med.to_csv(self.target_data_path[:-4] + "_medsim.csv")
        df_low.to_csv(self.target_data_path[:-4] + "_lowsim.csv")
