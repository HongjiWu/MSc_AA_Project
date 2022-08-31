import os
import logging
from dotenv import load_dotenv
from convokit import download
from tqdm.auto import tqdm

from download.download_data_by_id import IdDownloader
from download.convokit_downloader import ConvokitDownloader
from download.process_data import DataProcessor
from download.construct_contexts import ContextConstructor


tqdm.pandas()
load_dotenv()
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    logging.info("Running the Download and Process pipeline")
    if not os.path.exists(os.getenv("DATA_PATH") + "comments_subs.csv"):
        logging.info("Downloading comments through the PushShift API. Runtime: 3 hours")
        IdDownloader(os.getenv("DATA_PATH") + "ids_subs.csv",
                     output_path=os.getenv("DATA_PATH") + "comments_subs.csv")
        logging.info("Download complete")

    if not os.path.exists(os.getenv("DATA_PATH") + "comments_eli5.csv"):
        logging.info("Downloading ELI5 comments through Convokit. Runtime: 3 minutes")
        downloader = ConvokitDownloader(path=os.path.join(download('subreddit-explainlikeimfive'), "utterances.jsonl"),
                                        output_file=os.getenv("DATA_PATH") + "comments_eli5.csv")
        downloader.construct_df()
        logging.info("Download complete")

    if not os.path.exists(os.getenv("DATA_PATH") + "processed_eli5.csv"):
        logging.info("Processing ELI5 data. Runtime: 5 minutes")
        DataProcessor(data_path=os.getenv("DATA_PATH") + "comments_eli5.csv",
                      output_path=os.getenv("DATA_PATH") + "processed_eli5.csv", is_eli5=True).process()

    if not os.path.exists(os.getenv("DATA_PATH") + "processed_subs.csv"):
        logging.info("Processing data from subreddits (excluding ELI5). Runtime: 5 minutes")
        DataProcessor(data_path=os.getenv("DATA_PATH") + "comments_subs.csv",
                      output_path=os.getenv("DATA_PATH") + "processed_subs.csv", is_eli5=False).process()

    if not os.path.exists(os.getenv("DATA_PATH") + "processed_subs_lowsim.csv"):
        logging.info("Constructing sub-dataframes based on time and similarity")
        constructor = ContextConstructor(source_data_path=os.getenv("DATA_PATH") + "processed_eli5.csv",
                                         target_data_path=os.getenv("DATA_PATH") + "processed_subs.csv")
        constructor.split_by_time()
        constructor.split_by_similarity(sim_scores_path=os.getenv("DATA_PATH") + "subreddit_sim_score.csv")
