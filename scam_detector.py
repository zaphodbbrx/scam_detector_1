import re
from numpy import max
import pickle
import json

class ScamDetector():

    def __init__(self,config_path):
        config = json.load(open(config_path))
        with open(config['model_path'],'rb') as f:
            self.model = pickle.load(f)
        self.decode_dict = {
            1.0: 'present',
            0.0: 'absent'
        }

    def __clean_tweet(self, text):
        text = str(text)
        return re.sub(pattern='https.+|\n|\\\\n|\"|^\d.\/|@\w+|\W', repl=' ', string=text).strip()

    def run(self,text):
        text = [self.__clean_tweet(t) for t in text]
        proba = self.model.predict_proba(text)
        pred = self.model.predict(text)
        return {
            'decision':[self.decode_dict[p] for p in pred.tolist()],
            'confidence': [max(proba[i,:]) for i in range(proba.shape[0])]
        }
if __name__ == '__main__':
    sd = ScamDetector('config_scam.json')
    test_mes = [
        'whats app mah niggas',
        'we are starting a free btc giveaway huge profits guaranteed'
    ]
    print(sd.run(test_mes))
    while True:
        mes = input()
        if mes == 'q':
            break
        print(sd.run(mes))