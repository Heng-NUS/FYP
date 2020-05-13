import argparse
import sys
sys.path.append('./preprocess/')

import gensim
from gensim.corpora import Dictionary
from gensim.models.phrases import Phrases
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from wordcloud import STOPWORDS

from config import SUconfig
from model import *
from utils import *

STOP_WORDS = stopwords.words('english')

parser = argparse.ArgumentParser(description='Preprocess a single data file')
parser.add_argument('--fname', type=str, help='file needed to be processed')
parser.add_argument('--outpath', type=str,
                    help='file path to save the processed file')
parser.add_argument('--labeled', type=int, default=0, help='wether the first character of each line is its label')

args = parser.parse_args()
suconfig = SUconfig()

if __name__ == "__main__":
    for word in STOP_WORDS:
        STOPWORDS.add(word)

    docs = []
    labels = []
    lemmatizer = WordNetLemmatizer()
    with open(args.fname, encoding="utf-8") as f, open(args.outpath, 'w', encoding='utf-8') as out_f:
        for i, line in enumerate(f):
            if line == '\n':
                continue
            label = None
            if args.labeled == 1:
                label = line[0]
                line = text_pro.regularize(line[2:])
            tokens = gensim.utils.simple_preprocess(line)
            tokens = [lemmatizer.lemmatize(token) for token in tokens]
            tokens = [token for token in tokens if len(
                token) > 1 and token not in STOPWORDS and not token.isnumeric()]
            if len(tokens) < 4:
                continue
            else:
                docs.append(tokens)
                if label is not None:
                    labels.append(label)

        bigram = Phrases(docs, min_count=20)
        for idx in range(len(docs)):
            for token in bigram[docs[idx]]:
                if '_' in token:
                    # Token is a bigram, add to document.
                    docs[idx].append(token)

        dictionary = Dictionary(docs)
        dictionary.filter_extremes(no_below=5, no_above=0.4)
        dictionary.compactify()
        '''encode'''
        docs = [[token for token in doc if token in dictionary.token2id]
                for doc in docs]

        docs = [doc for doc in docs if len(set(doc)) > 1 and len(doc) > 3]

        if args.labeled:
            for label, doc in zip(labels, docs):
                out_f.write(label + ' ' + ' '.join(doc) + '\n')
        else:
            for doc in docs:
                out_f.write(' '.join(doc) + '\n')
