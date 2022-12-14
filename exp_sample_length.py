from dotenv import load_dotenv

from itertools import product
import csv



from authorship_attribution.experiments import BaselineExperiment, ExperimentParams
from authorship_attribution.methods.sari import SariWordpieceMethod
from authorship_attribution.methods.sari.model_training import TrainingArgs

import logging


logging.basicConfig(level = logging.INFO)
load_dotenv()


# This script is design for running Exp.3
# The experiment on Varying Length of Data Samples
# You could choose which dataset you want to use, and specify the parameter setting of aa_methods here.
# For running this experiment, you have to generate dataset with different sample length by using modify_sample_length.py

data_path = 'data/'
anchor = 'eli5'

args = TrainingArgs(train_epochs = 150,
                    batch_size = 5,
                    embedding_output_path=None,


                    )

params = ExperimentParams(eli5_path=data_path + "processed_" + anchor + ".csv",
                         subreddits_path=data_path + anchor + "_50_processed_subs.csv",
                         min_samples_per_author = 50,
                         max_samples_per_author=None,
                         force_single_topic=False,
                         set_deterministic=True,
                         use_gap_statistic=False,
                         domains="tr",
                         open_world=False,
                         compute_train_accuracy = False,

                         output_file = data_path + "test_exp_sl" + anchor + ".csv")

#method = BertHybridMethod(pad_length = 512, training_args = args)
#method = NaiveBertHybridMethod(pad_length = 512, training_args = args)
#method = BertMethod(pad_length = 512, training_args = args)
#method = KopMethod(k1 = 50, k2 = 0.3)
#method = TripletSaediMethod(pad_length = 1000, training_args = args, learning_rate = 0.0001)
method = SariWordpieceMethod(pad_length = 1000, embedding_dim = 518, training_args = args, learning_rate = 0.001)
#method = ShresthaMethod(pad_length=2048, training_args=args,learning_rate = 0.0005)
#method = SariMethod(pad_length=1000, training_args=args, split_words=True)
#method = NarMethod()
sl_list = [50, 100, 250, 500]
min_sample_list = [250, 125, 50, 25]

for n in range(4):

    params.eli5_path = data_path + "processed_" + anchor + "_" + str(sl_list[n]) + ".csv"
    params.subreddits_path = data_path + anchor+ "_50_processed_subs_" + str(sl_list[n]) + ".csv" 
    params.min_samples_per_author = min_sample_list[n]
    
    exp = BaselineExperiment(params, method)
    exp.experiment_run(repeats = 3, num_authors = 100)





