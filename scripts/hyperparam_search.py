import os
from dotenv import load_dotenv

from itertools import product
import csv

from authorship_attribution.methods import *
from authorship_attribution.experiments import BaseExperiment, ExperimentParams
from authorship_attribution.methods.sari.model_training import TrainingArgs


load_dotenv()


def get_methods(args=None):
    method_list = [
                   SariWordpieceMethod(pad_length=1000, training_args=args),
                   SariMethod(pad_length=1000, training_args=args, split_words=True),
                   ShresthaMethod(pad_length=1000, training_args=args),
                   TripletSaediMethod(pad_length=1000, training_args=args),
                   TripletSariMethod(pad_length=1000, training_args=args),
                   ]
    return method_list

args = TrainingArgs(train_epochs=50,
                    batch_size=32,
                    # from_pretrained="pretrained_embeddings.pth",
                    embedding_output_path=None,     # "pretrained_embeddings.pth",
                    )

params = ExperimentParams(eli5_path=os.getenv("DATA_PATH") + "500char_filtered_eli5.csv",
                          min_samples_per_author=8,
                          max_samples_per_author=10,
                          force_single_topic=False,
                          set_deterministic=False,
                          use_gap_statistic=False,
                          domains="tr",
                          open_world=False,
                          compute_train_accuracy=True,
                          output_file=os.getenv("DATA_PATH") + "test2.csv")


batch_size = [10, 32, 50, 100]
train_epochs = [10, 30, 50, 100, 200]
num = 100

for ba, tr in product(batch_size, train_epochs):
    args.batch_size = ba
    args.train_epochs = tr

    for method in get_methods(args):
        try:
            exp1 = BaseExperiment(params, method)
            acc, acc_train = exp1.experiment_run(repeats=1, num_authors=num)
            with open(params.output_file[:-4] + "_grid.csv", "a") as file:
                csv.writer(file).writerow([method.name, ba, tr, acc, acc_train])
            print(method.name, ba, tr, acc, acc_train)

        except Exception as exception:
            print(exception)

