import csv
import os
from dotenv import load_dotenv


from authorship_attribution.helpers.overdorf_distortion import OverdorfDistortion
from authorship_attribution.experiments.base_experiment import ExperimentParams


load_dotenv()


def write_data(params, distortions):
    with open(os.getenv("DATA_PATH") + "distortions.csv", "a") as file:
        csv.writer(file).writerow([params.eli5_path,
                                   params.subreddits_path,
                                   params.min_samples_per_author,
                                   distortions[0],
                                   distortions[1],
                                   distortions[2],
                                   distortions[3]
                                   ])


params = ExperimentParams(eli5_path=os.getenv("DATA_PATH") + "500char_filtered_redditxtwitter.csv",
                          subreddits_path=os.getenv("DATA_PATH") + "500char_filtered_twitterxreddit.csv",
                          min_samples_per_author=6,
                          max_samples_per_author=None
                          )

distortions = OverdorfDistortion(params).compute_distortion()
write_data(params, distortions)


params = ExperimentParams(eli5_path=os.getenv("DATA_PATH") + "500char_filtered_eli5.csv",
                          subreddits_path=os.getenv("DATA_PATH") + "500char_filtered_subreddits_large_lowsim.csv",
                          min_samples_per_author=6,
                          max_samples_per_author=None
                          )

base_path = os.getenv("DATA_PATH") + "500char_filtered_subreddits_large"
for suffix in [".csv", "_lowsim.csv", "_medsim.csv", "_highsim.csv"]:
    params.subreddits_path = base_path + suffix
    distortions = OverdorfDistortion(params).compute_distortion()
    write_data(params, distortions)

params = ExperimentParams(eli5_path=os.getenv("DATA_PATH") + "500char_filtered_eli5.csv",
                          subreddits_path=os.getenv("DATA_PATH") + "500char_filtered_subreddits_large_time_cut.csv",
                          min_samples_per_author=6,
                          max_samples_per_author=None,
                          )

base_eli5_path = os.getenv("DATA_PATH") + "500char_filtered_eli5_time"
for suffix in ["_low.csv", "_med.csv", "_high.csv"]:
    params.eli5_path = base_eli5_path + suffix
    distortions = OverdorfDistortion(params).compute_distortion()
    write_data(params, distortions)
