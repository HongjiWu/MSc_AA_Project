import os
from typing import Optional
import json
import random
import logging
from tqdm.auto import tqdm


import pandas as pd
from convokit import download
from dotenv import load_dotenv


class ConvokitDownloader:
    def __init__(self,
                 output_file: str,
                 path: Optional[str] = None,
                 min_samples: int = 10,
                 max_users: Optional[int] = None,
                 subreddit : str = "eli5"
                 ):
        logging.basicConfig(level=logging.INFO)
        self.user_whitelist = None
        self.path = path if path else os.path.join(download('subreddit-explainlikeimfive'), "utterances.jsonl")
        self.output_file = output_file
        self.min_samples = min_samples
        self.max_users = max_users
        self.subreddit = subreddit
        assert os.path.isfile(self.path)

    def _filter_line(self, line, min_score: int = 3, min_length: int = 5):
        data = json.loads(line)

        if self.user_whitelist:
            if data["user"] not in self.user_whitelist:
                return None
        
        if (data["id"] == data["root"]) or (data["timestamp"] > 1519557042):
            return None
        if len(data.keys()) > 7 or len(data["meta"].keys()) > 9:
            return None

        if data["meta"]["score"] < min_score or len(data["text"].split()) < min_length:
            return None
        for key in ['user', 'root', 'text']:
            if data[key].lower() in ['[removed]', '[deleted]']:
                return None
        data["score"] = data["meta"]["score"]
        data["is_top_level"] = data["meta"]["top_level_comment"] == data["id"]

        del data["meta"]
        del data["reply_to"]

        return data

    def get_best_users(self):
        user_counts = {}
        with open(self.path, "r") as reader: 
            for line in tqdm(reader):
                res = self._filter_line(line)
                if res is not None:
                    user_counts[res["user"]] = user_counts.get(res["user"], 0) + 1
        return user_counts

    def convert_to_df(self):
        with open(self.path, "r") as reader:
            total_lines = []

            for line in tqdm(reader):
                res = self._filter_line(line)
                if res:
                    total_lines.append(res)

            return pd.json_normalize(total_lines).dropna()

    def construct_df(self):
        
        
        logging.info(f"Getting users with more than {self.min_samples} samples") 
        users = self.get_best_users() 
        users = pd.DataFrame.from_dict(users, orient="index").sort_values(by=0, ascending=False)
        # users.to_csv(self.output_file[:-4] + "_tmp.csv")
        self.user_whitelist = set(users[(users[0] > self.min_samples)].index)

        # users = pd.read_csv(self.output_file[:-4] + "_tmp.csv", index_col="user")
        # self.user_whitelist = set(users[(users["num"] > self.min_samples)].index)
        
        logging.info(f"{len(self.user_whitelist)} users fit the specifications")
        
        
        if self.max_users and self.max_users < len(self.user_whitelist):
            logging.info(f"Randomly sampling to keep {self.max_users} users")
            self.user_whitelist = set(random.sample(self.user_whitelist, k=self.max_users))
        logging.info(f"Extracting data from  {len(self.user_whitelist)} users")
        
        df_user_whitelist = pd.DataFrame(self.user_whitelist, columns = ['user'])
        df_user_whitelist.to_csv('data/' + self.subreddit + '_user.csv')
        del users
        '''
        # Comment out to only generate user
        final_df = self.convert_to_df()
        final_df["date"] = pd.to_datetime(final_df.timestamp, unit="s")
        logging.info(f"Saving data to {self.output_file}.")
        final_df.to_csv(self.output_file)
        return final_df
        '''

        '''
        #This part is used to count total_user given different minimum sample size
        users = self.get_best_users() 
        
        users = pd.DataFrame.from_dict(users, orient="index").sort_values(by=0, ascending=False)
        
        res = []
        for i in [3, 5, 10, 20, 50, 100]:
             
            logging.info(f"Getting users with more than {i} samples") 
            
        # users.to_csv(self.output_file[:-4] + "_tmp.csv")
            self.user_whitelist = set(users[(users[0] > i)].index)

        # users = pd.read_csv(self.output_file[:-4] + "_tmp.csv", index_col="user")
        # self.user_whitelist = set(users[(users["num"] > self.min_samples)].index)
        
            logging.info(f"{len(self.user_whitelist)} users fit the specifications")
            res.append(len(self.user_whitelist))

        logging.info(f'{res}')
        return
        '''
if __name__ == "__main__":
    load_dotenv()

    downloader = ConvokitDownloader(path=os.path.join(download('subreddit-explainlikeimfive'), "utterances.jsonl"),
                                    output_file=os.getenv("DATA_PATH")+"comments_eli5_b.csv")
    downloader.construct_df()
