import logging
from uuid import uuid4
from typing import Optional
from dataclasses import dataclass
import csv
from datetime import datetime
import numpy as np

from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.interpolate import interp1d

from authorship_attribution.methods.base_aa_method import BaseAAMethod
from authorship_attribution.methods.bert_hybrid import BertHybridMethod, NaiveBertHybridMethod
from authorship_attribution.data_loader.tune_reader import DataReader


@dataclass
class ExperimentParams:
    """
    Parameter dataclass for the experiments.

    :param output_file: output file path
    :param domains: Parameter to choose what the source and target data is: Choose between ["rt", "tr", "tt", "rr"]
    :param force_single_topic: This parameter is useful for the domain heterogeneity experiments. Setting it to true
        guarantees every author in the data has data from a unique subreddit.
    :param min_samples_per_author: minimum number of samples for authors in the experiments
    :param max_samples_per_author: maximum number of samples for authors in the experiments
    :param store_result: Boolean value - Decides whether to write the prediction to the output file
    :param use_gap_statistic: Boolean value - Decides whether to use the gap statistic to compute precision/recall
        tradeoff metrics
    :param set_deterministic: Boolean value - Run experiments with no randomness
    :param open_world: Add authors not in the reference set to the target dataset to simulate open-world scenarios
    :param compute_train_accuracy: Compute performance on training set (for hyperparam tuning)
    """

    eli5_path: str = None
    subreddits_path: str = None
    source_data_path: str = None
    target_data_path: str = None
    output_file: str = None

    domains: Optional[str] = None  # "rt"
    force_single_topic: bool = False

    min_samples_per_author: int = 10
    max_samples_per_author: Optional[int] = None
    logging_level = logging.INFO
    store_result: bool = False

    use_gap_statistic: bool = False
    set_deterministic: bool = True
    open_world: bool = False
    compute_train_accuracy: bool = True


class BaselineExperiment:
    def __init__(self, params: ExperimentParams, aa_method: BaseAAMethod, author_pool = None):
        self.params = params
        logging.basicConfig(level=params.logging_level)
        logging.info(f"Reading data from {params.eli5_path}")

        self.reader = None
        self.max_author = None
        self.aa_method = aa_method
        self.set_up()
        self.author_pool = author_pool

    def set_up(self):
        self.reader = DataReader(self.params.eli5_path,
                                 self.params.subreddits_path,
                                 self.params.min_samples_per_author,
                                 self.params.max_samples_per_author,
                                 compensate_test_split=False)

        self.max_author = len(self.reader.users)
        # self.reader = TwitterRedditReader(self.params.path, self.params.min_samples_per_author)

    def compute_accuracies(self, predictions, test_labels):
        if self.params.open_world:
            predictions = predictions[test_labels != -1]
            test_labels = test_labels[test_labels != -1]
        if isinstance(self.aa_method, BertHybridMethod) or isinstance(self.aa_method, NaiveBertHybridMethod):
            res = [predictions[i] == test_labels[i] for i in range(len(predictions))]
            top_1 = res.count(1)/len(res)
            return top_1, -1, -1, -1
        top_1 = np.mean([(predictions[i, 0:1] == test_labels[i]).any() for i in range(len(predictions))])
        top_5 = np.mean([(predictions[i, 0:5] == test_labels[i]).any() for i in range(len(predictions))])
        top_10 = np.mean([(predictions[i, 0:10] == test_labels[i]).any() for i in range(len(predictions))])
        top_100 = np.mean([(predictions[i, 0:100] == test_labels[i]).any() for i in range(len(predictions))])
        logging.info(f"Closed Acc - Top 1: {top_1}, Top 5: {top_5}, Top 10: {top_10}, Top 100: {top_100}")

        return top_1, top_5, top_10, top_100

    def compute_unk_metrics(self,
                            predictions: np.ndarray,
                            scores: np.ndarray,
                            labels: np.ndarray,
                            fpr_val: float = 0.2,
                            tpr_val: float = 0.8):

        """
        :param predictions: array (num_labels, num candidates) with the sorted predicted labels for each sample
        :param scores: array (num_labels, num candidates) with the score for each pair of test sample/candidate
        in predictions
        :param labels: true labels of the test samples
        :param fpr_val: Desired False Positive Rate to compute True Positive Rate at
        :param tpr_val: Desired True Positive Rate to compute False Positive Rate at
        :return: auroc_1, auroc_2, auroc_3, tpr_rate, fpr_rate

        AUROC 1:
        TP: u in A - model predicts
        FP: u not in A - model predicts
        TN: u in A - model does not predict
        FN: u not in A - model does not predict

        AUROC 2:
        TP: (u in A) and T1 correct - model predicts
        FP: (u not in A) or T1 incorrect - model predicts
        TN: (u in A) and T1 correct - model does not predict
        FN: (u not in A) or T1 incorrect - model does not predict

        AUROC 3 (closed-world):
        TP: (u in A) and T1 correct - model predicts
        FP: (u in A) or T1 incorrect - model predicts
        TN: (u in A) and T1 correct - model does not predict
        FN: (u in A) or T1 incorrect - model does not predict
        """

        gap_statistic = (scores[:, 0] - scores[:, 1]) / scores[:, 0]
        correct_predictions = (predictions[:, 0] == labels)

        # auroc_1
        auroc_1 = roc_auc_score(labels != -1, gap_statistic) if self.params.open_world else 1

        # auroc_2
        mask_predictions = (labels != -1) & correct_predictions
        try:
            auroc_2 = roc_auc_score(mask_predictions, gap_statistic)
            fpr, tpr, thr = roc_curve(mask_predictions, gap_statistic, drop_intermediate=True)
            value_at_chosen_fpr = interp1d(fpr, thr, kind="linear")(fpr_val)
            tpr_rate = interp1d(thr, tpr, kind="linear")(value_at_chosen_fpr).item()

            value_at_chosen_tpr = interp1d(tpr, thr, kind="linear")(tpr_val)
            fpr_rate = interp1d(thr, fpr, kind="linear")(value_at_chosen_tpr).item()

            logging.info(f"TPR for FPR {fpr_val}: {tpr_rate} / FPR for TPR {tpr_val}: {fpr_rate} ")

        except Exception as e:
            logging.exception(e)
            auroc_2, fpr_rate, tpr_rate = 1, 0, 1

        # auroc_3 (closed_world)
        correct_predictions_closed = correct_predictions[labels != -1]
        gap_statistic_closed = gap_statistic[labels != -1]
        try:
            auroc_3 = roc_auc_score(correct_predictions_closed, gap_statistic_closed)
        except Exception as e:
            logging.exception(e)
            auroc_3 = 1

        logging.info(f"AUC (is inside) {auroc_1}, AUC (is correct) {auroc_2}, , AUC (is correct, closed) {auroc_3}")

        return auroc_1, auroc_2, auroc_3, tpr_rate, fpr_rate

    def fit_model(self, model, train_ids, test_features, test_labels, test_ids, acc, acc_train, exp_type, num_authors):
                    # Make predictions on the test data
        predictions, scores = self.aa_method.infer(model, test_features)

        assert len(predictions) == len(scores)

            # Compute stats
        closed_accuracies = self.compute_accuracies(predictions, test_labels)
        #auroc_1, auroc_2, auroc_3, tpr_rate, fpr_rate = self.compute_unk_metrics(predictions, scores,
        #                                                                                 test_labels)

        if self.params.use_gap_statistic:
            predictions = self.gap_statistic(predictions, scores, thresh=0.05)
        else:
            if not (isinstance(self.aa_method, BertHybridMethod) or isinstance(self.aa_method, NaiveBertHybridMethod)):

                predictions = predictions[:, 0]

                
        # Store results
        if self.params.output_file:
            run_id = str(uuid4())

            if self.params.store_result:
                #Store sample results
                with open(self.params.output_file, "a") as file:
                    file.writelines(
                                [f"{id_}, {pred}, {succ}, {run_id}\n" for id_, pred, succ in
                                    zip(test_ids, predictions, (predictions == test_labels[:len(predictions)]).astype(int))])

                    
            else:
                # Store sample results
                with open(self.params.output_file, "a") as file:
                    file.writelines(
                                [f"{id_}, {pred}, {run_id}\n" for id_, pred in
                                    zip(test_ids, (predictions == test_labels[: len(predictions)]).astype(int))])

            # Store training information
            with open(self.params.output_file[:-4] + "_training_info.csv", "a") as file:
                csv.writer(file).writerow([run_id,
                                                   num_authors,
                                                   #self.__class__.__name__ + f"{'_single' if self.params.force_single_topic else ''}",
                                                   exp_type,
                                                   self.aa_method.name,
                                                   self.reader.min_samples_per_author,
                                                   train_ids])

            # Store results
            with open(self.params.output_file[:-4] + "_metrics.csv", "a") as file:
                csv.writer(file).writerow([run_id,
                                                   num_authors,
                                                   #self.__class__.__name__ + f"{'_single' if self.params.force_single_topic else ''}",
                                                   exp_type,
                                                   self.aa_method.name,
                                                   self.reader.min_samples_per_author,
                                                   *self.get_filenames(),
                                                   *closed_accuracies,
                                                   #auroc_1,
                                                   #auroc_2,
                                                   #auroc_3,
                                                   #tpr_rate,
                                                   #fpr_rate
                                                   ])

        acc.append(closed_accuracies[0])



    def experiment_run(self, repeats=1, num_authors=50):
        """
        Main pipeline to run experiments. This method can be modified to compute new metrics or store other types of
        experimental outputs.

        :param repeats: Number of repetitions of the experiment to run (with different seeds).
        :param num_authors: Number of authors
        :return: accuracy: List (and (accuracy: List, training_accuracy: List) if compute_train_accuracy is True)
        """

        # If not enough authors, test with max
        num_authors = min(num_authors, self.max_author)
        

        logging.info(datetime.now())
        
        acc_intra = []
        acc_cross = []
        acc_train = []
        if self.params.compute_train_accuracy:
            acc_train = []

        for iter_num in tqdm(range(repeats)):
            try:

                # Generate train/test features
                if self.author_pool is not None:
                    training_data, testing_data, aux_data, self.author_pool =  self.reader.subsample_split_df(num_authors=num_authors,
                            domain=self.params.domains,
                            random_seed=iter_num if self.params.set_deterministic else None,
                            authors = self.author_pool,
                            open_world=self.params.open_world)
                    
                
                else:
                     training_data, testing_data, aux_data, self.author_pool =  self.reader.subsample_split_df(num_authors=num_authors,
                            domain=self.params.domains,
                            random_seed=iter_num if self.params.set_deterministic else None,
                            open_world=self.params.open_world)
                    
                logging.info(len(self.author_pool))
                processed_data = self.aa_method.data_processing(training_data, testing_data)
                processed_aux_data = self.aa_method.data_processing(training_data, aux_data)
                
                train_features, train_labels, train_ids, test_features, test_labels, test_ids = processed_data

                logging.info("2: " + str(datetime.now()))

                model = self.aa_method.fit_model(train_features, train_labels)
                if self.params.compute_train_accuracy:
                    train_predictions, _ = self.aa_method.infer(model, train_features)
                    top_1_acc_train, _, _, _ = self.compute_accuracies(train_predictions, train_labels)
                    acc_train.append(top_1_acc_train)

                del train_labels
                del train_features


                
                logging.info("after_train: " +  str(datetime.now()))
                self.fit_model( model, train_ids, test_features, test_labels, test_ids, acc_intra, acc_train, 'intra_context', num_authors)
                train_features, train_labels, train_ids, aux_test_features, aux_test_labels, aux_test_ids = processed_aux_data
                self.fit_model( model, train_ids, aux_test_features,  aux_test_labels, aux_test_ids, acc_cross, acc_train,'cross_context', num_authors)

                # Fit model

                del model
            except Exception as e:
                logging.exception(e)

        if self.params.compute_train_accuracy:
            return acc_intra, acc_cross, acc_train

        return acc_intra, acc_cross





    def get_filenames(self):
        return self.params.eli5_path, self.params.eli5_path

    @staticmethod
    def gap_statistic(predictions, scores, thresh=0.05):
        """General purpose gap statistic
        Scores and predictions should be sorted in descending order
        Adds an unknown class prediction"""

        relative_gap = (scores[:, 0] - scores[:, 1]) / scores[:, 0]
        predictions = predictions[:, 0]
        predictions[relative_gap < thresh] = -1

        return predictions
