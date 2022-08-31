import os
from dotenv import load_dotenv
import pandas as pd
from authorship_attribution.experiments import BaseExperiment, ExperimentParams, CrossRedditExperiment, BaselineExperiment
from authorship_attribution.methods.sari.model_training import TrainingArgs

from scripts.method_list import get_methods
from pathlib import Path
import logging
from datetime import datetime

import tracemalloc
load_dotenv()
import resource
#data_path = str(Path(__file__).rewolver().parent[1]) + '/data'
#resource.setrlimit(resource.RLIMIT_AS, (10000000000,resource.RLIM_INFINITY))
data_path = 'data/'
anchor = 'todayilearned'
logging.basicConfig(level = logging.INFO)
args = TrainingArgs(train_epochs=100,
                    batch_size=5,
                    # from_pretrained="pretrained_embeddings.pth",
                    embedding_output_path=None,     # "pretrained_embeddings.pth",
                    )

params = ExperimentParams(eli5_path=data_path + "processed_" + anchor + ".csv",
                          subreddits_path=data_path + anchor + "_50_processed_subs.csv",
                          max_samples_per_author=None,
                          force_single_topic=False,
                          set_deterministic=True,
                          use_gap_statistic=False,
                          domains="tr",
                          open_world=False,
                          compute_train_accuracy=True,
                          output_file=data_path + "experiments_tr_sample_num_" + anchor + "_tr_samples.csv")
                          #output_file = data_path + "test_new_bert" + anchor + "_tr_samples.csv")


method_list = get_methods(args)
data = pd.read_csv(params.subreddits_path, usecols = ["user"])
users = list(data.user.unique())
last_user_pool = []
print(users)
print(len(users))
for method in method_list:
    logging.info(method.name)
    #for num in [2, 5, 10, 15, 20, 30, 40, 50, 65, 80, 100]:
    for num in [65, 80, 100] :
        try:
            logging.info(datetime.now())
            for i in range(1):
                
                #tracemalloc.start()
                

                params.min_samples_per_author = num
                exp_intra = BaselineExperiment(params, method, author_pool = users)
                exp_intra.experiment_run(repeats= 5 , num_authors = 100)
                
                #snapshot = tracemalloc.take_snapshot()
                #top_stats = snapshot.statistics("lineno")

                #for stat in top_stats[:10]:
                    #logging.info(str(stat))

                '''
                author_pool = exp_intra.author_pool
                
                logging.info(author_pool)
                logging.info("difference in author with last exp:")
                logging.info(list(set(author_pool)- set(last_user_pool)))


                exp_cross = CrossRedditExperiment(params, method, author_pool = author_pool)
                exp_cross.experiment_run(repeats=1 , num_authors = 100)
                logging.info(exp_cross.author_pool)
                logging.info("difference in author")
                logging.info(list(set(author_pool)- set(exp_cross.author_pool)))
                last_user_pool = exp_cross.author_pool
                '''
        except Exception as exception:
            logging.info(exception)
        
        
    
        

        
