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
                          #output_file=data_path + "experiments_tr_sample_num" + anchor + "_tr_samples.csv")
                          #output_file = data_path + "test_" + anchor + "_tr_samples.csv")
                          output_file = data_path + "hyperparam_tune_" +  anchor + "_KOP.csv")

#batch_size = [16, 25, 64]
batch_size = [5]

#batch_size = [100]
train_epochs = [25, 50, 109] #for kop
#train_epochs = [100, 150, 200]
#train_epochs = [100, 150]
#embedding_dim = [100,256, 518] #for SAR
embedding_dim = [518]
#embedding_dim = [300] # for SHR
#embedding_dim = [128]
#drop_out = [0.25, 0.5, 0.75]
drop_out = [0.25]
#learning_rate = [0.001, 0.01, 0.005] #this is for SAR, SAR2
#learning_rate = [0.001]
#learning_rate = [0.0005, 0.0001, 0.001]
learning_rate = [0.3, 0.4, 0.5, 0.7] #this is for kop
pad_length = [1000]
#pad_length = [140, 512, 1024, 2048] # for SHR
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
            
    



