from authorship_attribution.data_loader.crossdomain_reader import CrossDomainReader
from authorship_attribution.experiments.base_experiment import BaseExperiment


class CrossDomainExperiment(BaseExperiment):
    def set_up(self):
        self.reader = CrossDomainReader(self.params.source_data_path,
                                        self.params.target_data_path,
                                        self.params.min_samples_per_author,
                                        self.params.max_samples_per_author,
                                        self.params.force_single_topic)

        self.max_author = len(self.reader.users)

    def get_filenames(self):
        if self.params.domains == "rr":
            return self.params.source_data_path, self.params.source_data_path
        if self.params.domains == "tt":
            return self.params.target_data_path, self.params.target_data_path
        if self.params.domains == "tr":
            return self.params.target_data_path, self.params.source_data_path
        if self.params.domains == "rt":
            return self.params.source_data_path, self.params.target_data_path
        return "unknown", "unknown"
