import pandas as pd
from ast import literal_eval
import glob

# root_path = '/Users/junho/Desktop/server/data/'
root_path = './data/'
mask_path = root_path+'mask/mask.png'
image_path = sorted(glob.glob(f'{root_path}image/*'))

class PreprocessData():
    def __init__(self):
        self.tokens_path = root_path+'tokens.csv'
        self.lines_token_path = root_path+'lines_token.csv'
        self.top10_path = root_path+'top10.csv'
        self.top10_token_path = root_path+'top10_token.csv'
        self.top140_token_path = root_path+'top140_token.csv'
        self.tf_sentences_path = root_path+'tf_sentences.csv'
        self.promise140_path = root_path+'promise140.csv'
        self.stopwords_path = root_path+'stopwords_all.csv'
        self.word_idx_path = root_path+'word_idx.csv'

    def word_idx_converted(self):
        word_idx_df = pd.read_csv(self.word_idx_path)
        return {word_idx_df.iloc[i,1]:i+1 for i in range(len(word_idx_df))}

    def stop_words_converted(self):
        stop_words_df = pd.read_csv(self.stopwords_path)
        return set(stop_words_df.iloc[:,0])

    def tokens_converted(self):
        tokens_df = pd.read_csv(self.tokens_path, converters={'0': literal_eval})
        tokens = tokens_df['0'].values.tolist()

        return tokens

    def lines_token_converted(self):
        lines_token_df = pd.read_csv(self.lines_token_path, converters={'0': literal_eval})
        lines_token = lines_token_df['0'].values.tolist()

        return lines_token

    def top10_converted(self):
        top10 = pd.read_csv(self.top10_path).values.tolist()

        return top10

    def top10_token_converted(self):
        top10_token = pd.read_csv(self.top10_token_path).values.tolist()

        return top10_token

    def top140_token_converted(self):
        top140_token = pd.read_csv(self.top140_token_path).values.tolist()

        return top140_token

    def tf_sentences_converted(self):
        tf_sentences_list = pd.read_csv(self.tf_sentences_path).values.tolist()
        tf_sentences = sum(tf_sentences_list, [])

        return tf_sentences

    def promise140_converted(self):
        promise140 = pd.read_csv(self.promise140_path)

        return promise140

pre = PreprocessData()

tokens = pre.tokens_converted()
lines_token = pre.lines_token_converted()
top10 = pre.top10_converted()
top10_token = pre.top10_token_converted()
top140_token = pre.top140_token_converted()
tf_sentences = pre.tf_sentences_converted()
promise140 = pre.promise140_converted()
stopwords = pre.stop_words_converted()
word_idx = pre.word_idx_converted()
