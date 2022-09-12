from dotenv import load_dotenv
from statistics import mean
from itertools import product
import csv

#from authorship_attribution.methods.sari.sari_wordpiece_method import SariWordpieceMethod
from authorship_attribution.methods import *
from authorship_attribution.experiments import BaseExperiment, ExperimentParams, CrossRedditExperiment, TuneExperiment
from authorship_attribution.methods.sari.model_training import TrainingArgs


import logging

logging.basicConfig(level = logging.INFO)
load_dotenv()

# This script is used for running hyperparameter tuning
# You should do the hyperparameter tuning on the LifeProTips dataset, since we already block all comments involved in this dataset
# away from processing dataset with other anchor_subreddit

anchor = "LifeProTips"
data_path = 'data/'

args = TrainingArgs(train_epochs = 50,
                    batch_size = 32,
                    embedding_output_path=None,
        
        
                    )   


params = ExperimentParams(eli5_path=data_path + "processed_" + anchor + ".csv",
                          subreddits_path=data_path + anchor + "_20_processed_subs.csv",
                          max_samples_per_author=None,
                          min_samples_per_author=20,
                          force_single_topic=False,
                          set_deterministic=False,
                          use_gap_statistic=False,
                          domains="tr",
                          open_world=False,
                          compute_train_accuracy = True,
                          output_file = data_path + "hyperparam_tune_" +  anchor + "_KOP.csv")


# You could modify your hyperparameter set there


batch_size = [5]


train_epochs = [25, 50, 109]


embedding_dim = [518]

drop_out = [0.25]

learning_rate = [0.3, 0.4, 0.5, 0.7]
pad_length = [1000]

for bs, te, ed, lr, pl, do in product(batch_size, train_epochs, embedding_dim, learning_rate, pad_length, drop_out):
    logging.info(str([bs, te, ed, lr, pl, do]))
    args.batch_size = bs
    args.train_epochs = te
    args.drop_out = do
    #method = SariWordpieceMethod(embedding_dim = ed, pad_length = pl, training_args = args, learning_rate = lr)
    #method = SariMethod(embedding_dim = ed, pad_length = pl, training_args = args, learning_rate = lr, split_words = True)
    #method = TripletSaediMethod(pad_length = pl, training_args = args, learning_rate = lr) 
    #method = TripletSariMethod(pad_length= pl, training_args=args, learning_rate = lr)
   # method = BertMethod(pad_length = pl, training_args = args, learning_rate = lr)
    method = KopMethod(k1 = te, k2 = lr)
    #method = ShresthaMethod(pad_length=pl, training_args=args, learning_rate = lr)
    try:
        for i in range(1):
            exp_intra = TuneExperiment(params, method)
            acc_intra, acc_cross, acc_train = exp_intra.experiment_run(repeats= 2 , num_authors = 100)
            acc = acc_intra + acc_cross

            with open(params.output_file[:-4] + "_grid.csv", "a") as file:
                csv.writer(file).writerow([method.name, 'all', bs, te, ed, lr, pl, do, mean(acc), acc_intra, acc_cross, acc_train])
            logging.info(str([method.name, 'all', bs, te, ed, lr, pl, do, acc_intra, mean(acc), acc_cross, acc_train]))

    except Exception as exception:
        logging.info(exception)
            
    



