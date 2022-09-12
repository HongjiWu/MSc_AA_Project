from typing import List, Dict, Optional
import os
import json
import requests
from tqdm import tqdm
import pandas as pd


import logging
from datetime import datetime

class UserDownloader:
    """
    First version of data downloader
    """
    def __init__(self, input_path: str, output_path: str, timestamp_min: int, timestamp_max : int, subreddit : str):
        self.max_tries: int = 4
        self.max_per_user = 1000
        self.batch_size = 25
        self.output_path = output_path
        self.processed = set()
        self.min_score = 2
        self.min_body_length = 5
        self.subreddit = subreddit
        self.timestamp_min = timestamp_min
        self.timestamp_max = timestamp_max
        self.subreddit_blacklist = [self.subreddit]

        
        logging.basicConfig(level = logging.INFO)

        if os.path.isfile(self.output_path):
            self.processed = set(pd.read_csv(self.output_path, usecols=["user"], lineterminator='\n')["user"].unique())
        else:
            with open(self.output_path, "w") as file:
                file.write(",id,user,timestamp,score,text,subreddit\n")

        user_list = list(set(pd.read_csv(input_path).user))
        self.download_users(user_list)

    def download_users(self, user_list: List[str]):
        """Download and store user list as csv file
        Note: A lot of write access to the output file but allows to have a low RAM need and the overhead
        is small compared to the API request time."""

        user_list = sorted(list(set(user_list) - set(self.processed)), reverse=True)

        for user in tqdm(user_list):
            logging.info(user)
            logging.info(datetime.now())
            user_comments = self.request_user(user, self.timestamp_min)
            while user_comments is not None and len(user_comments['data']) != 0:
                logging.info(len(user_comments['data']))
                logging.info(user_comments['data'][0]['created_utc'])
                logging.info(user_comments['data'][-1]['created_utc'])
                cur_max = user_comments['data'][-1]['created_utc']
                pd.DataFrame(self.filter_responses(user_comments["data"])).to_csv(self.output_path,
                                                                                  mode='a',
                                                                                  header=False)
                user_comments = self.request_user(user, cur_max)
            
            #'''
    def request_user(self, author_name: str, cur_max) -> Optional[Dict]:

        request_url = f"https://api.pushshift.io/reddit/search/comment/?author={author_name}&sort=asc&after={cur_max}&before={self.timestamp_max}&size={500}"


        for _ in range(self.max_tries):
            try:
                response = requests.get(request_url)
                if str(response)[10:15] == "[200]":
                    return json.loads(response.content)
            except Exception as exception:
                print(exception)
        return None

    def filter_responses(self, responses: List[Dict]) -> List[Dict]:
        filtered = []

        for comment in responses:
            # Filter by subreddit
            if comment["subreddit"] in self.subreddit_blacklist:
                continue
            # Filter by score
            #if comment["score"] < self.min_score:
                #continue
            # Filter by body length
            if len(comment["body"].split()) < self.min_body_length:
                continue

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
    UserDownloader("user_list.csv", output_path="comments_from_list.csv")

