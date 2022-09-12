import os
from convokit import download
from download.convokit_downloader import ConvokitDownloader
from download.process_data import DataProcessor

# This script is designed for generating dataset with different sample length
# Which is required for Exp.3

subreddit_name = 'eli5'
DATA_PATH = 'data/'
min_sample = '50'
sample_length = 250


DataProcessor(data_path = DATA_PATH + "comments_" + subreddit_name + ".csv", output_path = DATA_PATH + "processed_" + subreddit_name + "_" + str (sample_length) + ".csv", is_eli5 = True, sl= sample_length).process()

DataProcessor(DATA_PATH + subreddit_name + "_" + min_sample + "_comments_subs.csv", DATA_PATH + subreddit_name + "_" + min_sample + "_processed_subs" + str (sample_length) + ".csv", is_eli5 = False, sl= sample_length).process()
