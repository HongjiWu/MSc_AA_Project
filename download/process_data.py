import re
import logging
from uuid import uuid4

import numpy as np
import pandas as pd

from tqdm.auto import tqdm
tqdm.pandas()


class DataProcessor:
    def __init__(self, data_path: str, output_path: str, is_eli5: bool, sl = 250):
        self.data_path = data_path
        self.output_path = output_path
        self.is_eli5 = is_eli5
        self.sl = sl

    def process(self):
        logging.info(f"Reading data from {self.data_path}")

        if self.is_eli5:
            df_og = pd.read_csv(self.data_path, usecols=["id", "user", "text", "timestamp"],
                                dtype={"id": "str", "user": "str", "text": "str"}).dropna().drop_duplicates()
        else:
            df_og = pd.read_csv(self.data_path, usecols=["id", "user", "text", "subreddit", "timestamp"],
                                dtype={"id": "str", "user": "str", "text": "str",
                                       "subreddit": "str"}, lineterminator = '\n').dropna().drop_duplicates()

            df_og = df_og[df_og.subreddit != "explainlikeimfive"]

            df_og = df_og[df_og.subreddit != 'LifeProTips']


        df_lpt = pd.read_csv('data/LifeProTips_20_comments_subs.csv', usecols=["id", "user", "text", "timestamp"],
                                dtype={"id": "str", "user": "str", "text": "str"}, lineterminator = '\n').dropna().drop_duplicates()
        lpt_aux_ids = set(df_lpt['id'])
        
        df_og = df_og[~df_og['id'].isin(lpt_aux_ids)]

        df_og = df_og[df_og.user != "AutoModerator"]
    
        url_regex = re.compile(
            r"(http(s)?:\/\/.)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)")
        hashtag_at_regex = re.compile(r"(#|@|r\/|u\/)\w+")

        logging.info("Filtering for URLs and social media handles or usernames")
        df_og.text = df_og.text.progress_apply(lambda x: re.sub(url_regex, "", x))
        df_og.text = df_og.text.progress_apply(lambda x: re.sub(hashtag_at_regex, "", x))

        logging.info("Grouping texts by samples of 250 words")

        if self.is_eli5:
            merged = df_og.groupby('user').apply(self.group_)
        else:
            df_og["words"] = df_og["text"].apply(lambda x: len(x.split()))
            # speed-dup
            df = df_og.groupby(["subreddit", "user"]).filter(lambda x: x['words'].sum() >= self.sl)
            merged = df.groupby(['user', 'subreddit']).apply(self.group_)

        df, _ = self.expand(merged)
        df = df.reset_index().drop(["index"], axis=1)
        df["id"] = [str(uuid4()) for _ in range(len(df))]
        logging.info(f"Saving processed data to {self.output_path}")

        df.to_csv(self.output_path)

    def group_(self, x):
        tmp_dic = {}
        tmp_dic['user'] = x['user'].iloc[0]

        tmp_dic['text'], tmp_dic['timestamp_min'], tmp_dic['timestamp_max'], tmp_dic["id"] = self.group_texts(
            zip(x["text"], x["timestamp"], x["id"]), n_words= self.sl)

        if self.is_eli5:
            return pd.Series(tmp_dic, index=['id', 'user', 'text', 'timestamp_min', 'timestamp_max'])

        tmp_dic['subreddit'] = x['subreddit'].iloc[0]
        return pd.Series(tmp_dic, index=['id', 'user', 'text', 'subreddit', 'timestamp_min', 'timestamp_max'])

    def expand(self, df):
        ids = []
        tmp = None
        for _, row in df.iterrows():
            ids.extend(row.id)
            texts = row.text
            timestamps_min = row.timestamp_min
            timestamps_max = row.timestamp_max

            if self.is_eli5:
                tmp = pd.concat([tmp,
                                 pd.DataFrame(list(zip([row.user] * len(texts), texts, timestamps_min, timestamps_max)),
                                              columns=['user', 'text', "timestamp_min", "timestamp_max"])])
            else:
                tmp = pd.concat([tmp, pd.DataFrame(list(
                    zip([row.user] * len(row.text), row.text, [row.subreddit] * len(row.text), row.timestamp_min,
                        row.timestamp_max)), columns=['user', 'text', 'subreddit', 'timestamp_min', 'timestamp_max'])])
        return tmp, ids

    @staticmethod
    def group_texts(x, n_words = 250):
        grouped_texts = []
        min_timestamps = []
        max_timestamps = []
        sample_ids = []

        tmp = ""
        rolling_time = []
        rolling_ids = []
        for text, time, id_ in x:
            rolling_ids.append(id_)
            tmp += text + " "
            rolling_time.append(time)

            if len(tmp.split()) >= n_words:
                grouped_texts.append(" ".join(tmp.split()[:n_words]))
                min_timestamps.append(np.min(rolling_time))
                max_timestamps.append(np.max(rolling_time))
                sample_ids.append(rolling_ids)

                tmp = ""
                rolling_time = []
                rolling_ids = []
        return grouped_texts, min_timestamps, max_timestamps, sample_ids


if __name__ == "__main__":
    DataProcessor(data_path="comments_eli5.csv", output_path="processed_eli5.csv", is_eli5=True).process()
    DataProcessor(data_path="comments_subs.csv", output_path="processed_subs.csv", is_eli5=False).process()
