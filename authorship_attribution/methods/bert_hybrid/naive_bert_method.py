import numpy as np
import pandas as pd

from simpletransformers.classification import ClassificationModel
from sklearn.linear_model import LogisticRegression

from torch.optim.adam import Adam
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score

from authorship_attribution.helpers.tokenizer_dataset import TokenizerDataset
from authorship_attribution.methods.bert.bert_model import BertNet

from authorship_attribution.methods.base_aa_method import BaseAAMethod
from authorship_attribution.methods.bert.model_training import Trainer, TrainingArgs

import logging
import tracemalloc
import resource

class NaiveBertHybridMethod(BaseAAMethod):

    # This method is based on the 
    def __init__(self,
                 embedding_dim: int = 768,
                 pad_length: int = 500,
                 tokenizer_name: str = "allenai/longformer-base-4096",
                 learning_rate = 0.01,
                 training_args: TrainingArgs = TrainingArgs()):
        super().__init__()

        self.extractor = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.embedding_dim = embedding_dim
        self.pad_length = pad_length
        self.training_args = training_args
        self.test_dataset = None
        self.lr = learning_rate
        self.raw_out_train = []
        self.y_proba_train = []
        self.clf = None
        self.train_labels = None


    def data_processing(self, train, test):
        #train_encodings = self.extractor(list(train.text), truncation=True, padding="max_length",
        #                                 max_length=self.pad_length)
        train_encodings = list(train.text)
        train_labels = np.array(train.dummy)
        train_ids = np.array(train.id)

        del train
        #test_encodings = self.extractor(list(test.text), truncation=True, padding="max_length",
        #                                max_length=self.pad_length)
        test_encodings = list(test.text)
        test_labels = np.array(test.dummy)
        test_ids = np.array(test.id)

        self.test_dataset = TokenizerDataset(test_encodings, test_labels)
        return train_encodings, train_labels, train_ids, test_encodings, test_labels, test_ids

    def extract_style(self, text):
        """
        Extracting stylometric features of a text
        """

        text = str(text)
        len_text = len(text)
        len_words = len(text.split())
        avg_len = np.mean([len(t) for t in text.split()])
        num_short_w = len([t for t in text.split() if len(t) < 3])
        per_digit = sum(t.isdigit() for t in text)/len(text)
        per_cap = sum(1 for t in text if t.isupper())/len(text)
        f_a = sum(1 for t in text if t.lower() == "a")/len(text)
        f_b = sum(1 for t in text if t.lower() == "b")/len(text)
        f_c = sum(1 for t in text if t.lower() == "c")/len(text)
        f_d = sum(1 for t in text if t.lower() == "d")/len(text)
        f_e = sum(1 for t in text if t.lower() == "e")/len(text)
        f_f = sum(1 for t in text if t.lower() == "f")/len(text)
        f_g = sum(1 for t in text if t.lower() == "g")/len(text)
        f_h = sum(1 for t in text if t.lower() == "h")/len(text)
        f_i = sum(1 for t in text if t.lower() == "i")/len(text)
        f_j = sum(1 for t in text if t.lower() == "j")/len(text)
        f_k = sum(1 for t in text if t.lower() == "k")/len(text)
        f_l = sum(1 for t in text if t.lower() == "l")/len(text)
        f_m = sum(1 for t in text if t.lower() == "m")/len(text)
        f_n = sum(1 for t in text if t.lower() == "n")/len(text)
        f_o = sum(1 for t in text if t.lower() == "o")/len(text)
        f_p = sum(1 for t in text if t.lower() == "p")/len(text)
        f_q = sum(1 for t in text if t.lower() == "q")/len(text)
        f_r = sum(1 for t in text if t.lower() == "r")/len(text)
        f_s = sum(1 for t in text if t.lower() == "s")/len(text)
        f_t = sum(1 for t in text if t.lower() == "t")/len(text)
        f_u = sum(1 for t in text if t.lower() == "u")/len(text)
        f_v = sum(1 for t in text if t.lower() == "v")/len(text)
        f_w = sum(1 for t in text if t.lower() == "w")/len(text)
        f_x = sum(1 for t in text if t.lower() == "x")/len(text)
        f_y = sum(1 for t in text if t.lower() == "y")/len(text)
        f_z = sum(1 for t in text if t.lower() == "z")/len(text)
        f_1 = sum(1 for t in text if t.lower() == "1")/len(text)
        f_2 = sum(1 for t in text if t.lower() == "2")/len(text)
        f_3 = sum(1 for t in text if t.lower() == "3")/len(text)
        f_4 = sum(1 for t in text if t.lower() == "4")/len(text)
        f_5 = sum(1 for t in text if t.lower() == "5")/len(text)
        f_6 = sum(1 for t in text if t.lower() == "6")/len(text)
        f_7 = sum(1 for t in text if t.lower() == "7")/len(text)
        f_8 = sum(1 for t in text if t.lower() == "8")/len(text)
        f_9 = sum(1 for t in text if t.lower() == "9")/len(text)
        f_0 = sum(1 for t in text if t.lower() == "0")/len(text)
        f_e_0 = sum(1 for t in text if t.lower() == "!")/len(text)
        f_e_1 = sum(1 for t in text if t.lower() == "-")/len(text)
        f_e_2 = sum(1 for t in text if t.lower() == ":")/len(text)
        f_e_3 = sum(1 for t in text if t.lower() == "?")/len(text)
        f_e_4 = sum(1 for t in text if t.lower() == ".")/len(text)
        f_e_5 = sum(1 for t in text if t.lower() == ",")/len(text)
        f_e_6 = sum(1 for t in text if t.lower() == ";")/len(text)
        f_e_7 = sum(1 for t in text if t.lower() == "'")/len(text)
        f_e_8 = sum(1 for t in text if t.lower() == "/")/len(text)
        f_e_9 = sum(1 for t in text if t.lower() == "(")/len(text)
        f_e_10 = sum(1 for t in text if t.lower() == ")")/len(text)
        f_e_11 = sum(1 for t in text if t.lower() == "&")/len(text)
        richness = len(list(set(text.split())))/len(text.split())

        return pd.Series([avg_len, len_text, len_words, num_short_w, per_digit, per_cap, f_a, f_b, f_c, f_d, f_e, f_f, f_g, f_h, f_i, f_j, f_k, f_l, f_m, f_n, f_o, f_p, f_q, f_r, f_s, f_t, f_u, f_v, f_w, f_x, f_y, f_z, f_0, f_1, f_2, f_3, f_4, f_5, f_6, f_7, f_8, f_9, f_e_0, f_e_1, f_e_2, f_e_3, f_e_4, f_e_5, f_e_6, f_e_7, f_e_8, f_e_9, f_e_10, f_e_11, richness])
    
    
    def fit_model(self, train_features, train_labels):
        #train_dataset = TokenizerDataset(train_features, train_labels)
        self.train_labels = train_labels
        logging.info(str(len(set(train_labels))))
        #model = ClassificationModel('bert', 'bert-base-cased', num_labels=len(set(train_labels)), args={'reprocess_input_data': True, 'overwrite_output_dir': True,  'num_train_epochs' : 15}, use_cuda=True)
        tracemalloc.start()
        model = ClassificationModel('bert', 'bert-base-cased', num_labels=len(set(train_labels)), args={ 'overwrite_output_dir': True, 'use_early_stopping' : True,  'use_multiprocessing' : False, 'use_multiprocessing_for_evaluation' : False, 'num_train_epochs' : 2, 'train_batch_size' : 32}, use_cuda=True)
        
        train_df = []
        print("1")
        
        for i in range(len(train_features)):
            train_df.append([train_features[i], train_labels[i]])
        print("2")
        print('len_of_training_size' + str(len(train_df)))
        print(train_df[0])
        model.train_model(pd.DataFrame(train_df))
        '''
        #build the style-based classifier

        predictions, self.raw_out_train = model.predict(list(train_features))
        
        X_style_train = []

        for text in train_features:
            X_style_train.append(self.extract_style(text))

        #logging.info(str(X_style_train))

        self.clf = LogisticRegression(random_state=0).fit(X_style_train, train_labels)
        self.y_proba_train = self.clf.predict_proba(X_style_train)
        
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics("lineno")

        for stat in top_stats[:10]:
            logging.info(str(stat))


        del train_df
        '''
        return model
    def infer(self, model, test_features) -> (np.array, np.array):
        '''
        trainer = Trainer(model,
                          None,
                          None,
                          self.training_args)

        test_dataset = TokenizerDataset(test_features, [0] * (len(test_features.encodings)))
        
        predictions, scores = trainer.infer(test_dataset=test_dataset)
        '''
        predictions, raw_outputs = model.predict(list(test_features))
        

        # For the feature_based techniques
        '''
        X_style_test = []

        for text in list(test_features):
            X_style_test.append(self.extract_style(text))

        y_proba = self.clf.predict_proba(X_style_test)
        
        
        feat_for_BERT_LR_train = np.concatenate([self.raw_out_train, self.y_proba_train], axis=1)
        feat_for_BERT_LR_test = np.concatenate([raw_outputs, y_proba], axis=1)

        clf_sum = LogisticRegression(random_state=0).fit(feat_for_BERT_LR_train, self.train_labels)
        y_pred = clf_sum.predict(feat_for_BERT_LR_test)
        
        '''
        
        scores = np.array([0] * (len(test_features)))
        return np.array(predictions), scores

