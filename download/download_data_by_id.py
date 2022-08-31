from typing import List, Dict, Optional
import os
import json
import requests
from tqdm import tqdm
import pandas as pd


class IdDownloader:
    """
    First version of data downloader
    """
    def __init__(self, input_path: str, output_path: str):
        self.max_tries: int = 4
        self.batch_size = 25
        self.output_path = output_path
        self.processed = set()

        if os.path.isfile(self.output_path):
            self.processed = set(pd.read_csv(self.output_path, usecols=["id"])["id"].unique())
        else:
            with open(self.output_path, "w") as file:
                file.write(",id,user,timestamp,score,text,subreddit\n")

        ids = pd.read_csv(input_path).id
        self.download_comments(ids)

    def download_comments(self, id_list: List[str]):
        """Download and store user list as csv file
        Note: A lot of write access to the output file but allows to have a low RAM need and the overhead
        is small compared to the API request time."""

        id_list = sorted(list(set(id_list) - set(self.processed)), reverse=True)
        for index in tqdm(range(0, len(id_list), self.batch_size)):
            batched_ids = id_list[index: index + self.batch_size]
            comments = self.request_batch(batched_ids)
            if comments is not None:
                pd.DataFrame(self.filter_responses(comments["data"])).to_csv(self.output_path,
                                                                             mode='a',
                                                                             header=False)

    def request_batch(self, batched_comment_ids: List) -> Optional[Dict]:
        request_url = f"https://api.pushshift.io/reddit/comment/search?ids={','.join(batched_comment_ids)}"

        for _ in range(self.max_tries):
            try:
                response = requests.get(request_url)
                if str(response)[10:15] == "[200]":
                    return json.loads(response.content)
            except Exception as exception:
                print(exception)
        return None

    @staticmethod
    def filter_responses(responses: List[Dict]) -> List[Dict]:
        filtered = []

        for comment in responses:
            filtered.append(
                {"id": comment["id"],
                 "user": comment["author"],
                 "timestamp": comment["created_utc"],
                 "score": comment["score"],
                 "text": comment["body"],
                 "subreddit": comment["subreddit"]
                 }
            )

        return filtered


if __name__ == "__main__":
    IdDownloader("ids_eli5.csv", output_path="comments_eli5.csv")
    IdDownloader("ids_subs.csv", output_path="comments_subs.csv")
