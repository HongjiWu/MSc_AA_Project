import os
from dotenv import load_dotenv

from authorship_attribution.experiments import ExperimentParams, CrossRedditExperiment
from authorship_attribution.methods.sari.model_training import TrainingArgs
from authorship_attribution.methods import SariWordpieceMethod

load_dotenv()


args = TrainingArgs(train_epochs=50,
                    batch_size=32,
                    # from_pretrained="pretrained_embeddings.pth",
                    embedding_output_path=None,     # "pretrained_embeddings.pth",
                    )

params = ExperimentParams(
                          eli5_path=os.getenv("DATA_PATH") + "500char_filtered_eli5.csv",
                          subreddits_path=os.getenv("DATA_PATH") + "500char_filtered_subreddits_large.csv",
                          min_samples_per_author=10,
                          max_samples_per_author=None,
                          force_single_topic=False,
                          set_deterministic=True,
                          use_gap_statistic=False,
                          domains="tr",
                          open_world=True,
                          output_file=os.getenv("DATA_PATH") + "experiments_2003_gpu.csv"
                          )


method_list = [
                SariWordpieceMethod(pad_length=1000, training_args=args, num_linear=1),
                SariWordpieceMethod(pad_length=1000, training_args=args, num_linear=2),
                SariWordpieceMethod(pad_length=1000, training_args=args, num_linear=3),
                SariWordpieceMethod(pad_length=1000, training_args=args, num_linear=4),
                SariWordpieceMethod(pad_length=1000, training_args=args, num_linear=5),
               ]

for num_train in [2, 5, 10, 15, 20, 30, 40, 60, 80, 100]:
    params.min_samples_per_author = num_train
    for method in method_list:
        try:
            exp1 = CrossRedditExperiment(params, method)
            exp1.experiment_run(repeats=5, num_authors=100)
        except Exception as exception:
            print(exception)
