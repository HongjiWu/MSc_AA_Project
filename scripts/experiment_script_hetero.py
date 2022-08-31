import os
from dotenv import load_dotenv


from authorship_attribution.experiments import ExperimentParams, CrossDomainExperiment
from authorship_attribution.methods.sari.model_training import TrainingArgs
from method_list import get_methods


load_dotenv()

args = TrainingArgs(train_epochs=50,
                    batch_size=32,
                    # from_pretrained="pretrained_embeddings.pth",
                    embedding_output_path=None,     # "pretrained_embeddings.pth",
                    )

params = ExperimentParams(eli5_path=os.getenv("DATA_PATH") + "500char_filtered_eli5.csv",
                          source_data_path=os.getenv("DATA_PATH") + "500char_filtered_eli5.csv",
                          target_data_path=os.getenv("DATA_PATH") + "500char_filtered_subreddits_large.csv",
                          min_samples_per_author=10,
                          max_samples_per_author=None,
                          force_single_topic=False,
                          set_deterministic=True,
                          use_gap_statistic=False,
                          domains="tr",
                          open_world=False,
                          output_file=os.getenv("DATA_PATH") + "experiments_2804_gpu.csv")

method_list = get_methods(args)

for method in method_list:
    for domain in ["tt", "tr"]:
        try:
            params.force_single_topic = False
            params.domains = domain
            exp1 = CrossDomainExperiment(params, method)
            exp1.experiment_run(repeats=5, num_authors=100)
        except Exception as exception:
            print(exception)

    try:
        params.domains = "tt"
        params.force_single_topic = True
        exp1 = CrossDomainExperiment(params, method)
        exp1.experiment_run(repeats=5, num_authors=100)
    except Exception as exception:
        print(exception)
