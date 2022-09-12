import os
from dotenv import load_dotenv
import pandas as pd
from authorship_attribution.experiments import BaseExperiment, ExperimentParams, CrossRedditExperiment, BaselineExperiment
from authorship_attribution.methods.sari.model_training import TrainingArgs

from scripts.method_list import get_methods
from pathlib import Path
import logging
from datetime import datetime
load_dotenv()

# This script is design for running Exp.2
# The experiment on varying number of candidate authors
# You could choose which dataset you want to use, and specify the parameter setting of aa_methods here.

data_path = 'data/'
anchor = 'eli5'
logging.basicConfig(level = logging.INFO)

args = TrainingArgs(train_epochs=20,
                    batch_size=10,
                    # from_pretrained="pretrained_embeddings.pth",
                    embedding_output_path=None,     # "pretrained_embeddings.pth",
                    )

params = ExperimentParams(eli5_path=data_path + "processed_" + anchor + ".csv",
                          subreddits_path=data_path + anchor + "_50_processed_subs.csv",
                          min_samples_per_author=50,
                          max_samples_per_author=None,
                          force_single_topic=False,
                          set_deterministic=True,
                          use_gap_statistic=False,
                          domains="tr",
                          open_world=False,
                          output_file=data_path + "experiments_author_num_" + anchor + "_tr_samples.csv")

# The method list is written in /script/method_list.py
# You could choose which method you like to test in that script
method_list = get_methods(args)
data = pd.read_csv(params.subreddits_path, usecols = ["user"])
users = list(data.user.unique())

for method in method_list:
    logging.info(method.name)
    # You could specify how many candidate authors you want to have in each iteration here
    for num in [2, 5, 10, 15, 25, 50, 100, 200, 500]:
        try:
            logging.info(datetime.now())
            for i in range(1):
                exp_intra = BaselineExperiment(params, method, author_pool = users)
                exp_intra.experiment_run(repeats = 5, num_authors = num)

                
        except Exception as exception:
            logging.info(exception)
        
        
    
        

        
