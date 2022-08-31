import os
from dotenv import load_dotenv

from authorship_attribution.experiments import ExperimentParams, CrossRedditExperiment
from authorship_attribution.methods import NarMethod


load_dotenv()


params = ExperimentParams(
                eli5_path=os.getenv("DATA_PATH") + "500char_filtered_eli5.csv",
                subreddits_path=os.getenv("DATA_PATH") + "500char_filtered_subreddits_large.csv",
                min_samples_per_author=10,
                use_gap_statistic=False,
                open_world=False,
                force_single_topic=False,
                set_deterministic=False,
                source_data_path=os.getenv("DATA_PATH") + "500char_filtered_eli5.csv",
                target_data_path=os.getenv("DATA_PATH") + "500char_filtered_subreddits_large.csv",
                domains="tt",
                output_file=os.getenv("DATA_PATH") + "test.csv"
                )


exp1 = CrossRedditExperiment(params, NarMethod())

exp1.experiment_run(repeats=10, num_authors=2000)
