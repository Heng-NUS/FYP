import argparse
import sys

import gensim
import torch
from gensim.corpora import Dictionary
from gensim.models.phrases import Phrases
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from wordcloud import STOPWORDS
from model import *
from config import SUconfig, UNconfig

parser = argparse.ArgumentParser(description='Filtering Data')
parser.add_argument('--fname', type=str, help='file needed to be filtered')
parser.add_argument('--dataset', type=str, help=('name of dataset'), default='healthnews')

suconfig = SUconfig()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

STOP_WORDS = stopwords.words('english')

def filter(fname, vocab, model):
    '''
    filter out irrelevant data
    '''
    docs = []
    data = []
    model = model.to(device)
    model.eval()
    if not os.path.exists("./temp/"):
        os.makedirs('./temp/')
    out_file = os.path.join('./temp', args.dataset+'.txt')
    with open(fname, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line == '\n': 
                continue
            tokens = line.split()
            docs.append(line)
            padded = doc_padding(tokens, suconfig.max_sen_len)
            data.append(padded)
    with torch.no_grad():
        x = torch.LongTensor([vocab.doc2idx(doc, 1) for doc in data]).to(device)
        y_pred = model(x)
        predicted = torch.max(y_pred.cpu().data, 1)[1].numpy().tolist()
    count = 0
    with open(out_file, 'w', encoding='utf-8') as f:
        for doc, label in zip(docs, predicted):
            if label == 1:
                f.write(doc)
                count += 1
    print("Get {} positive prediction".format(count))

if __name__ == "__main__":
    for word in STOP_WORDS:
        STOPWORDS.add(word)
    modelfile = args.dataset + '_model.pkl'
    vocabfile = args.dataset + '_vocab.txt'
    supervised_model = torch.load(modelfile)
    supervised_vocab = Dictionary.load_from_text(vocabfile)
    filtered = filter(args.fname, supervised_vocab, supervised_model)
