from download.download_data_by_user_psaw import UserDownloader
#from download.download_data_by_user import UserDownloader
from download.process_data import DataProcessor
data_path = "data/"
subreddit_name = "eli5"
min_sample = "50"
print('start')

UserDownloader("data/" + subreddit_name + "_" + min_sample + "_user.csv", "data/" + subreddit_name + "_" + min_sample + "_comments_subs_spe.csv", timestamp_min = 1311869756, timestamp_max = 1519551442, subreddit = subreddit_name)


#UserDownloader("data/" + subreddit_name + "_" + min_sample + "_user.csv", "data/" + subreddit_name + "_" + min_sample + "_comments_subs.csv")


DataProcessor(data_path + subreddit_name + "_" + min_sample + "_comments_subs_spe.csv", data_path + subreddit_name + "_" + min_sample + "_processed_subs_spe.csv", is_eli5 = False).process()
 
print("Finish")
