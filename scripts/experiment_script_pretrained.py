import os
from dotenv import load_dotenv


from authorship_attribution.experiments import ExperimentParams, CrossRedditExperiment
from authorship_attribution.methods.sari.model_training import TrainingArgs
from authorship_attribution.methods import TripletSaediMethod


load_dotenv()

args = TrainingArgs(train_epochs=50,
                    batch_size=100,
                    from_pretrained=None,
                    embedding_output_path="pretrained.pth",
                    )

params = ExperimentParams(eli5_path=os.getenv("DATA_PATH") + "500char_filtered_eli5.csv",
                          subreddits_path=os.getenv("DATA_PATH") + "500char_filtered_subreddits_large.csv",
                          min_samples_per_author=50,
                          max_samples_per_author=None,
                          force_single_topic=False,
                          set_deterministic=True,
                          use_gap_statistic=False,
                          domains="tr",
                          open_world=True,
                          output_file=os.getenv("DATA_PATH") + "experiments_2802_gpu.csv")

# Create pretrained embeddings
exp1 = CrossRedditExperiment(params, TripletSaediMethod(pad_length=1000, training_args=args))
exp1.experiment_run(repeats=1, num_authors=200)

args = TrainingArgs(train_epochs=0,
                    batch_size=32,
                    from_pretrained="pretrained.pth",
                    embedding_output_path=None,
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
                          output_file=os.getenv("DATA_PATH") + "experiments_2003_pre_gpu.csv"
                          )

for num in [2, 5, 25, 100, 500, 2000]:
    try:
        exp1 = CrossRedditExperiment(params, TripletSaediMethod(pad_length=1000, training_args=args))
        exp1.experiment_run(repeats=5, num_authors=num)
    except Exception as exception:
        print(exception)
