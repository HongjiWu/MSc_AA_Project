from authorship_attribution.methods import *


def get_methods(args=None):
    method_list = [
                   ##NaiveKopMethod(),
                   #NarMethod(),
                   #KopMethod(),
                   SariWordpieceMethod(pad_length=1000, embedding_dim = 518, training_args=args, learning_rate = 0.001),
                   #SariMethod(pad_length=1000, training_args=args, split_words=False),
                   #SariMethod(pad_length=1000, training_args=args, split_words=True),
                   #ShresthaMethod(pad_length=2048, training_args=args,learning_rate = 0.0005),
                   #ShresthaWordpieceMethod(pad_length=1000, training_args=args),
                   #TripletSariMethod(pad_length=1000, training_args=args),
                   #TripletSaediMethod(pad_length=1000, training_args=args),
                   #BertMethod(pad_length=512, training_args=args),
                   #BertHybridMethod(pad_length  = 512, training_args = args),

                   ]
    return method_list
