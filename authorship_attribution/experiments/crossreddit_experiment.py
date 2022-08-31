from authorship_attribution.data_loader.crossreddit_reader import CrossRedditReader
from authorship_attribution.experiments.base_experiment import BaseExperiment


class CrossRedditExperiment(BaseExperiment):
    def set_up(self):
        self.reader = CrossRedditReader(self.params.eli5_path,
                                        self.params.subreddits_path,
                                        self.params.min_samples_per_author,
                                        self.params.max_samples_per_author,
                                        self.params.force_single_topic)

        self.max_author = len(self.reader.users)

    def get_filenames(self):
        return self.params.eli5_path, self.params.subreddits_path
