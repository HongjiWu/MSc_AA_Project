import os
import pandas as pd
from convokit import download
from download.convokit_downloader import ConvokitDownloader
from download.process_data import DataProcessor


from download.download_data_by_user_psaw import UserDownloader

#This script is designed for downloading the dataset

subreddit_name = 'gaming'
DATA_PATH = 'data/'
min_sample = 50



# In this part, we download all comments from the anchor subreddit and process those comments into auxiliary data samples
downloader = ConvokitDownloader(path = os.path.join(download('subreddit-' + subreddit_name), 'utterances.jsonl'),
       output_file = "data/comments_" + subreddit_name + ".csv",subreddit = subreddit_name)

downloader.construct_df()


DataProcessor(data_path = DATA_PATH + "comments_" + subreddit_name + ".csv", output_path = DATA_PATH + "processed_" + subreddit_name + "_100.csv", is_eli5 = True).process()

# In this part, we select qualified user who have at least n+1 anchor data samples

num_Of_Users = [0, 0, 0, 0, 0, 0]
user_set = set()


def process(x):
       global num_Of_Users
       if int(x['text']) >= 3:
              num_Of_Users[0] += 1

              if int(x['text']) >= 6:
                     num_Of_Users[1] += 1

                     if int(x['text']) >= 11:
                            num_Of_Users[2] += 1
                            if min_sample == 10:
                                   user_set.add(x['user'])
                            if int(x['text']) >= 21:
                                   num_Of_Users[3] += 1
                                   if min_sample == 20:
                                          user_set.add(x['user'])
                                   if int(x['text']) >= 51:
                                          num_Of_Users[4] += 1
                                          if min_sample == 50:
                                                 user_set.add(x['user'])
                                          if int(x['text']) >= 81:
                                                 num_Of_Users[5] += 1

                                                 if min_sample == 80:
                                                        user_set.add(x['user'])


df_og = pd.read_csv("data/processed_"+ subreddit_name + ".csv", usecols=["user",  "text" ],dtype={"user": "str", "text": "str"}).dropna()
merge = df_og.groupby('user', as_index = False).count()
merge.apply(process, axis = 1)
print(num_Of_Users)
# Number of user who have at least 3, 6, 11, 21, 51, 81  anchor samples will be printed
print(user_set)
# Name of users who have at least n+1 anchor samples will be printed
df_user = pd.DataFrame(user_set, columns = ['user'])
df_user.to_csv(DATA_PATH + subreddit_name + "_" + min_sample +"_user.csv")

# Record the first and last comment we collect in anchor data samples
# In order to control the time span of auxiliary data samples

df_og = pd.read_csv(DATA_PATH + "processed_" + subreddit_name + ".csv", usecols=["id", "user", "timestamp_min", "timestamp_max"],dtype={"id": "str", "user": "str"}).dropna().drop_duplicates()
min_time = df_og["timestamp_min"].min()
max_time = df_og["timestamp_max"].max()


# In this part, we retrieve comments for each qualified users from other subreddits to form auxiliary data sample

UserDownloader(DATA_PATH + subreddit_name + "_" + min_sample + "_user.csv", DATA_PATH + subreddit_name + "_" + str(min_sample) + "_comments_subs_spe.csv", timestamp_min = min_time, timestamp_max = max_time, subreddit = subreddit_name)

DataProcessor(DATA_PATH + subreddit_name + "_" + min_sample + "_comments_subs_spe.csv", DATA_PATH + subreddit_name + "_" + min_sample + "_processed_subs_spe.csv", is_eli5 = False).process()



