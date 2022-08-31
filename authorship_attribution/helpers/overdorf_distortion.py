from scipy.spatial.distance import euclidean
import numpy as np
from tqdm.auto import tqdm

from authorship_attribution.data_loader.crossdomain_reader import CrossDomainReader
from authorship_attribution.feature_extractors.writeprints_extractor import WriteprintsExtractor


class OverdorfDistortion:
    def __init__(self, params):
        self.params = params
        self.reader = CrossDomainReader(self.params.eli5_path,
                                        self.params.subreddits_path,
                                        self.params.min_samples_per_author,
                                        self.params.max_samples_per_author)

        self.extractor = WriteprintsExtractor()

    @staticmethod
    def distance(features_a, features_b):
        return euclidean(features_a, features_b)

    def compute_distortion(self):

        distortions = []
        for author in tqdm(self.reader.users):
            source_df = self.reader.source_data[self.reader.source_data.user == author]
            target_df = self.reader.target_data[self.reader.target_data.user == author]

            target_df = target_df.sample(len(target_df))
            source_df = source_df.sample(len(source_df))

            if len(target_df) < 4:
                continue

            target_features = self.extractor.transform(list(target_df.text))
            source_features = self.extractor.transform(list(source_df.text))

            distance_1 = self.distance(target_features.mean(axis=0), source_features.mean(axis=0))
            distance_2 = distance_1
            # distance_2 = self.distance(source_features.mean(axis=0), target_features.mean(axis=0))

            split = int(len(source_features)/2)
            distance_3 = self.distance(source_features[:split].mean(axis=0), source_features[split:].mean(axis=0))

            split = int(len(target_features)/2)
            distance_4 = self.distance(target_features[:split].mean(axis=0), target_features[split:].mean(axis=0))

            distortions.append([distance_1, distance_2, distance_3, distance_4])

        return np.array(distortions).mean(axis=0)
