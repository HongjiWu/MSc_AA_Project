import numpy as np
import pandas as pd

#This script is designed for computing qualified user for each anchor subreddit

num_Of_Users = [0, 0, 0, 0, 0, 0]
user_set_80 = set()
user_set_50 = set()
user_set_10 = set()
user_set_20 = set()
subreddit = 'eli5'
print(num_Of_Users)
def process(x):
    global num_Of_Users
    if int(x['text']) >= 3:
        num_Of_Users[0] += 1
        
        if int(x['text']) >= 6:
            num_Of_Users[1] += 1
            
            if int(x['text']) >= 11:
                num_Of_Users[2] += 1
                user_set_10.add(x['user'])
                if int(x['text']) >= 21:
                    num_Of_Users[3] += 1
                    user_set_20.add(x['user'])
                    if int(x['text']) >= 51:
                        num_Of_Users[4] += 1
                        user_set_50.add(x['user'])
                        if int(x['text']) >= 81:
                            num_Of_Users[5] += 1
                            #print(x)
                            #print(x.columns)
                            user_set_80.add(x['user'])




df_og = pd.read_csv("data/processed_"+ subreddit + ".csv", usecols=["user",  "text" ],dtype={"user": "str", "text": "str"}).dropna()
merge = df_og.groupby('user', as_index = False).count()
#merge.columns = ['user', 'text']
print(merge)
#filter(lambda x: x['text'] >= 3)

merge.apply(process, axis = 1)
print(num_Of_Users)
print(user_set_80)
df_user = pd.DataFrame(user_set_50, columns = ['user'])
df_user.to_csv('data/' + subreddit + "_50_user.csv")
df_user = pd.DataFrame(user_set_80, columns = ['user'])
df_user.to_csv('data/' + subreddit + "_80_user.csv")
df_user = pd.DataFrame(user_set_10, columns = ['user'])
df_user.to_csv('data/' + subreddit + "_10_user.csv")
df_user = pd.DataFrame(user_set_20, columns = ['user'])
df_user.to_csv('data/' + subreddit + "_20_user.csv")


#print(type(merge))

