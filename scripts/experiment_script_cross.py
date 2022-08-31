import os
from dotenv import load_dotenv

from authorship_attribution.experiments import ExperimentParams, CrossRedditExperiment
from authorship_attribution.methods.sari.model_training import TrainingArgs
from method_list import get_methods


load_dotenv()

args = TrainingArgs(train_epochs=50,
                    batch_size=32,
                    # from_pretrained="pretrained_embeddings.pth",
                    embedding_output_path=None,     # "pretrained_embeddings.pth",
                    )

params = ExperimentParams(eli5_path=os.getenv("DATA_PATH") + "500char_filtered_eli5.csv",
                          subreddits_path=None,
                          min_samples_per_author=10,
                          max_samples_per_author=None,
                          force_single_topic=False,
                          set_deterministic=True,
                          use_gap_statistic=False,
                          domains="tr",
                          open_world=True,
                          output_file=os.getenv("DATA_PATH") + "experiments_2804_gpu.csv")


method_list = get_methods(args)

base_path = os.getenv("DATA_PATH") + "500char_filtered_subreddits_large"
for suffix in ["_lowsim3.csv", "_medsim3.csv", "_highsim3.csv"]:
    params.subreddits_path = base_path + suffix
    for method in method_list:
        for num in [100]:
            try:
                exp1 = CrossRedditExperiment(params, method)
                exp1.experiment_run(repeats=5, num_authors=num)
            except Exception as exception:
                print(exception)
