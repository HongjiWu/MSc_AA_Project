import logging
import numpy as np
import pandas as pd

from tqdm.auto import tqdm

tqdm.pandas()


#This script is designed for collecting mean comment's length for different subreddits


data_path = "data/comments_gadgets.csv"
df_og = pd.read_csv(data_path, usecols=["id", "text"], dtype = {"id" : "str", "text" : "str"}).dropna().drop_duplicates()

df_og["words"] = df_og["text"].apply(lambda x: len(x.split()))

print(df_og["words"].mean())
