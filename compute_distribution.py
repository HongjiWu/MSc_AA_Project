import numpy as np
import pandas as pd
from datetime import datetime
from subreddit_list import get_subreddits

# This script is designed for computing the distribution of comments in each subreddit.

subreddit_list = get_subreddits()
output_list = []
def process_date(x):
    global num_Of_Users
    year = datetime.fromtimestamp(x['timestamp_min']).year
    num_Of_Users[year] += 1

    return
index = 0
for subreddit in subreddit_list:
    print(subreddit)
    
    index += 1 
    #if i#ndex  == 5:
        #break
    data_path = "data/processed_" + subreddit + ".csv"

    num_Of_Users = {"name" : subreddit, 2005 : 0, 2006 : 0, 2007 : 0, 2008 : 0, 2009 : 0, 2010 : 0, 2011 : 0, 2012 : 0, 2013 : 0, 2014 : 0, 2015 : 0, 2016 : 0, 2017 : 0, 2018 : 0}
    df_og = pd.read_csv(data_path, usecols=["id", "user", "timestamp_min", "timestamp_max"],dtype={"id": "str", "user": "str"}).dropna().drop_duplicates()
    min_date = datetime.fromtimestamp(df_og["timestamp_min"].min())
    num_Of_Users["timestamp_min"] = df_og["timestamp_min"].min()
    num_Of_Users["date_min"] = min_date
    print(min_date)
    max_date = datetime.fromtimestamp(df_og["timestamp_max"].max())
    print(max_date)
    num_Of_Users["timestamp_max"] = df_og["timestamp_max"].max()
    num_Of_Users["date_max"] = max_date

    df_og.apply(process_date, axis = 1)
    print(num_Of_Users)
    output_list.append(num_Of_Users)

out_df = pd.DataFrame.from_dict(output_list, orient = "columns")
out_df.to_csv("data/time_distribution.csv")
