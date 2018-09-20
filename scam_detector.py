import re
from numpy import max
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score,f1_score, recall_score
from IPython.display import clear_output
import pickle
import json

class ScamDetector():

    def __init__(self,config_path):
        config = json.load(open(config_path))
        with open(config['model_path'],'rb') as f:
            self.model = pickle.load(f)
        self.decode_dict = {
            1.0: 'scam',
            0.0: 'no scam'
        }

    def __clean_tweet(self, text):
        text = str(text)
        return re.sub(pattern='https.+|\n|\\\\n|\"|^\d.\/|@\w+|\W', repl=' ', string=text).strip()

    def run(self,text):
        text = self.__clean_tweet(text)
        proba = self.model.predict_proba([text])[0]
        pred = self.model.predict([text])[0]
        return {
            'decision':self.decode_dict[pred],
            'confidence': max(proba)
        }
if __name__ == '__main__':
    sd = ScamDetector('config.json')
    while True:
        mes = input()
        if mes == 'q':
            break
        print(sd.run(mes))