import os
from dotenv import load_dotenv

from authorship_attribution.experiments import ExperimentParams, CrossRedditExperiment
from authorship_attribution.methods.sari.model_training import TrainingArgs
from authorship_attribution.methods import SariMethod


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

args = TrainingArgs(train_epochs=1,
                    batch_size=100,
                    # from_pretrained="pretrained_embeddings.pth",
                    # embedding_output_path="pretrained_embeddings.pth",
                    print_progress=True
                    )

exp1 = CrossRedditExperiment(params, SariMethod(pad_length=1000, training_args=args))

exp1.experiment_run(repeats=1, num_authors=100)
