from dotenv import load_dotenv

from itertools import product
import csv

#from authorship_attribution.methods.bert_hybrid.bert_method import BertHybridMethod
#from authorship_attribution.methods.bert.bert_method import BertMethod
from authorship_attribution.data_loader.tune_reader import DataReader
from authorship_attribution.experiments import BaselineExperiment, ExperimentParams, SimExperiment
from authorship_attribution.methods.sari.model_training import TrainingArgs
from authorship_attribution.methods import *
from authorship_attribution.methods.koppel.kop_method import KopMethod
from authorship_attribution.methods.saedi_triplet.triplet_saedi_method import TripletSaediMethod
import logging

# This script is designed for running Exp.4
# The experiment on Varying Content Divergence

logging.basicConfig(level = logging.INFO)
load_dotenv()

# This script is design for running Exp.4
# The experiment on Varying Content Divergence between Referencing and Targetting Data Samples
# You could choose which dataset you want to use, and specify the parameter setting of aa_methods here.
# For running this experiment, you have to generate dataset with similarity metrics by using compute_sim.py


data_path = 'data/'
anchor = 'eli5'

args_SAR = TrainingArgs(train_epochs = 150,
                    batch_size = 5,
                    embedding_output_path=None,


                    )
args_SAR2 = TrainingArgs(train_epochs = 100,
                    batch_size = 5,
                    embedding_output_path=None,


                    )

args_SHR = TrainingArgs(train_epochs = 150,
                    batch_size = 32,
                    embedding_output_path=None,


                    )

args_BERT = TrainingArgs(train_epochs = 20,
                    batch_size = 10,
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
                         #output_file=data_path + "experiments_tr_sample_num" + anchor + "_tr_samples.csv")
                         #output_file = data_path + "test_" + anchor + "_tr_samples.csv")
                         output_file = data_path + "test_sim_" + anchor + ".csv")
                         #output_file = data_path + "test_vocab_size" + anchor + ".csv")


logging.info("start")

methods = []    
methods.append(SariWordpieceMethod(pad_length = 1000, training_args = args_SAR2))
methods.append(KopMethod())
methods.append(ShresthaMethod(pad_length=2048, training_args=args_SHR,learning_rate = 0.0005))
methods.append(SariMethod(pad_length=1000, training_args=args_SAR, split_words=True))
methods.append(BertMethod(pad_length=512, training_args=args_BERT))
methods.append(NarMethod())

for method in methods:
    exp = SimExperiment(params, method)




    for sim_metric in [ 'bert']:
        for sim_level in ['low', 'middle', 'high']:

            
            exp.experiment_run(repeats = 5, num_authors = 100, sim_metric = sim_metric, sim_level = sim_level)




