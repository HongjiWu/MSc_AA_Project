import pandas as pd
import numpy as np
from numpy.linalg import norm
from authorship_attribution.feature_extractors.writeprints_extractor import WriteprintsExtractor
import matplotlib.pyplot as plt
from sklearn import preprocessing

from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')


# Running this script will compute the similarity metrics between training and testing sample,
# Which is required for running Exp.4
# The only thing you need to modify is the subreddit name
# And the distribution of similarity scores will be plotted in the end

subreddit = 'AskReddit'
data_path = 'data/'


anchor_data = pd.read_csv(data_path + 'processed_' + subreddit + '.csv', usecols = ['user', 'text'], lineterminator= '\n')
aux_data = pd.read_csv(data_path + subreddit + "_50_processed_subs.csv", lineterminator= '\n')

aux_data['feature_sim'] = 0
aux_data['bert_sim'] = 0
aux_data['sim_metric_feature'] = 'middle'
aux_data['sim_metric_bert'] = 'middle'



users = list(aux_data.user.unique())

for user in users:
    anchor_vec = []

    cur_df = anchor_data[anchor_data['user'] == user]
    texts = list(cur_df['text'])
    anchor_emb_list = model.encode(texts)

    anchor_emb = [ np.mean([anchor_emb_list[j][i] for j in range(len(anchor_emb_list))]) for i in range(len(anchor_emb_list[0]))]





    for i, row in aux_data[aux_data['user'] == user].iterrows():
        if row['user'] == user:
            text = [row['text']]



            aux_emb = model.encode(text)

            cos_sim_bert = util.cos_sim(anchor_emb, aux_emb[0])
            aux_data.loc[i, 'bert_sim'] = cos_sim_bert.tolist()[0][0]








top =int(aux_data.shape[0] // 3) if int(aux_data.shape[0] // 3) != 0 else 1

aux_data = aux_data.sort_values('bert_sim', ascending=False)
aux_data.loc[aux_data.head(top).index, 'sim_metric_bert'] = 'high'
aux_data.loc[aux_data.tail(top).index, 'sim_metric_bert'] = 'low'


aux_data.to_csv(data_path + subreddit + "_50_processed_subs.csv")

data_high = list(aux_data[aux_data['sim_metric_bert'] == 'high']['bert_sim'])
data_middle = list(aux_data[aux_data['sim_metric_bert'] == 'middle']['bert_sim'])
data_lwo = list(aux_data[aux_data['sim_metric_bert'] == 'low']['bert_sim'])

plt.hist([data_lwo, data_middle, data_high], bins = 50, label= ['low', 'middle', 'high'], stacked= True)
plt.title('Similarity_Metric_r/gaming')
plt.legend(loc = 'upper left')
plt.ylabel("Num_of_Samples")
plt.xlabel("Similarity Score")
plt.show()



