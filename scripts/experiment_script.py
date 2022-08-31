import os
from dotenv import load_dotenv

from authorship_attribution.experiments import BaseExperiment, ExperimentParams, CrossRedditExperiment
from authorship_attribution.methods.sari.model_training import TrainingArgs

from method_list import get_methods


load_dotenv()


args = TrainingArgs(train_epochs=50,
                    batch_size=32,
                    # from_pretrained="pretrained_embeddings.pth",
                    embedding_output_path=None,     # "pretrained_embeddings.pth",
                    )

params = ExperimentParams(eli5_path=os.getenv("DATA_PATH") + "500char_filtered_eli5.csv",
                          subreddits_path=os.getenv("DATA_PATH") + "500char_filtered_subreddits_large.csv",
                          min_samples_per_author=10,
                          max_samples_per_author=None,
                          force_single_topic=False,
                          set_deterministic=True,
                          use_gap_statistic=False,
                          domains="tr",
                          open_world=True,
                          output_file=os.getenv("DATA_PATH") + "experiments_2804_gpu_test.csv")


method_list = get_methods(args)

for method in method_list:
    for num in [2, 5, 10, 25, 50, 100, 200, 500, 1000, 2000]:
        try:
            exp1 = BaseExperiment(params, method)
            exp1.experiment_run(repeats=20 if num < 200 else 4, num_authors=num)

            exp1 = CrossRedditExperiment(params, method)
            exp1.experiment_run(repeats=20 if num < 200 else 4, num_authors=num)
        except Exception as exception:
            print(exception)
