import os
from convokit import download
from download.convokit_downloader import ConvokitDownloader
from download.process_data import DataProcessor

#This script is designed for downloading and processing data from anchor subreddit

subreddit_name = 'eli5'
DATA_PATH = 'data/'
sample_length = 250


DataProcessor(data_path = DATA_PATH + "comments_" + subreddit_name + ".csv", output_path = DATA_PATH + "processed_" + subreddit_name + "_100.csv", is_eli5 = True, sl= sample_length).process()

