import pandas as pd
import numpy as np
from numpy.linalg import norm
from authorship_attribution.feature_extractors.writeprints_extractor import WriteprintsExtractor
import matplotlib.pyplot as plt
from sklearn import preprocessing

from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')


subreddit = 'AskReddit'
data_path = 'data/'


anchor_data = pd.read_csv(data_path + 'processed_' + subreddit + '.csv', usecols = ['user', 'text'], lineterminator= '\n')
aux_data = pd.read_csv(data_path + subreddit + "_50_processed_subs.csv", lineterminator= '\n')

aux_data['feature_sim'] = 0
aux_data['bert_sim'] = 0
aux_data['sim_metric_feature'] = 'middle'
aux_data['sim_metric_bert'] = 'middle'

print(aux_data.head())

print(aux_data.columns)
#extractor = WriteprintsExtractor()

users = list(aux_data.user.unique())
print(len(users))
for user in users:
    anchor_vec = []
    print(user)
    cur_df = anchor_data[anchor_data['user'] == user]
    texts = list(cur_df['text'])
    anchor_emb_list = model.encode(texts)
    #print(anchor_emb_list)
    anchor_emb = [ np.mean([anchor_emb_list[j][i] for j in range(len(anchor_emb_list))]) for i in range(len(anchor_emb_list[0]))]
    #print(anchor_emb)


    #text_transform = extractor.transform(texts)
    #print(text_transform)
    #anchor_vec = [ np.mean([text_transform[j][i] for j in range(len(text_transform))]) for i in range(len(text_transform[0]))]

    #print(anchor_vec)

    #cur_aux_df = aux_data[aux_data['user'] == user]


    #aux_data.apply(process_data, axis = 1, raw = True, result_type= 'broadcast')

    for i, row in aux_data[aux_data['user'] == user].iterrows():
        if row['user'] == user:
            text = [row['text']]

            #aux_vec = extractor.transform(text)[0]
            #cos_sim_fea = np.dot(anchor_vec, aux_vec) / (norm(aux_vec) * norm(aux_vec))
            #print(cos_sim_fea)
            #aux_data.loc[i, 'feature_sim'] = cos_sim_fea

            aux_emb = model.encode(text)
            #print(aux_emb)
            cos_sim_bert = util.cos_sim(anchor_emb, aux_emb[0])
            aux_data.loc[i, 'bert_sim'] = cos_sim_bert.tolist()[0][0]
    '''
    cur_aux_df = aux_data[aux_data['user'] == user]
    top_20 = int(cur_aux_df.shape[0] * 0.2) if int(cur_aux_df.shape[0] * 0.2) != 0 else 1
    fea_high_bar = cur_aux_df.nlargest(top_20, 'feature_sim').iloc[-1]['feature_sim']
    fea_low_bar = cur_aux_df.nsmallest(top_20, 'feature_sim').iloc[-1]['feature_sim']
    bert_high_bar = cur_aux_df.nlargest(top_20, 'bert_sim').iloc[-1]['bert_sim']
    bert_low_bar = cur_aux_df.nsmallest(top_20, 'bert_sim').iloc[-1]['bert_sim']

    
    for i, row in aux_data.iterrows():
        if row['user'] == user:
            if row['feature_sim'] >= fea_high_bar:
                aux_data.loc[i, 'sim_metric_feature'] = 'high'
            elif row['feature_sim'] <= fea_low_bar:
                aux_data.loc[i, 'sim_metric_feature'] = 'low'

            if row['bert_sim'] >= bert_high_bar:
                aux_data.loc[i, 'sim_metric_bert'] = 'high'
            elif row['bert_sim'] <= bert_low_bar:
                aux_data.loc[i, 'sim_metric_bert'] = 'low'
    '''






    '''
    for text in cur_aux_df['text']:
        #print(text)
        aux_vec = extractor.transform([text])[0]
        #print(aux_vec)

        cos_sim = np.dot(anchor_vec,aux_vec)/(norm(aux_vec)*norm(aux_vec))
        print(cos_sim)
        
    '''
top =int(aux_data.shape[0] // 3) if int(aux_data.shape[0] // 3) != 0 else 1

aux_data = aux_data.sort_values('bert_sim', ascending=False)
aux_data.loc[aux_data.head(top).index, 'sim_metric_bert'] = 'high'
aux_data.loc[aux_data.tail(top).index, 'sim_metric_bert'] = 'low'
print(aux_data.head())
print(aux_data.tail())

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



