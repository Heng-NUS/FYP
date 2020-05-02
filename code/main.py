import json
import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gensim.models.phrases import Phrases
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from config import SUconfig, UNconfig
from modeling import *
from preprocess import *
from utils import *


def load_word_list(word_list_path):
    try:
        with open(word_list_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except IOError:
        print("Can't find file:", word_list_path)
        exit(0)


word_list_path = './Filter_rule/word_list.json'
WORD_LIST = load_word_list(word_list_path)
STOP_WORDS = stopwords.words('english')
MIN_FREQUENCY = 10
suconfig = SUconfig()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def filter(fname, vocab, model):
    '''
    filter out irrelevant data
    '''
    docs = []
    lemmatizer = WordNetLemmatizer()
    model = model.to(device)
    model.eval()
    if not os.path.exists("./temp/"):
        os.makedirs('./temp/')
    out_file = './temp/health_tweets.txt'
    with open(fname, encoding="utf-8") as f, open(out_file, 'w', encoding='utf-8') as out_f:
        for i, line in enumerate(f):
            line = text_pro.regularize(line)
            tokens = gensim.utils.simple_preprocess(line)
            tokens = [lemmatizer.lemmatize(token) for token in tokens]
            padded = doc_padding(tokens, suconfig.max_sen_len)
            x = torch.LongTensor([vocab.doc2idx(padded, 1)]).to(device)
            y_pred = model(x)
            predicted = torch.max(y_pred.cpu().data, 1)[1]
            if predicted == 1:
                docs.append(tokens)

        docs = [[token for token in doc if not token.isnumeric()]
                for doc in docs]
        docs = [[
            token for token in doc if len(token) > 1 and token not in STOP_WORDS
        ] for doc in docs]

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

        # # Remove docs that contains less than 3 words
        docs = [doc for doc in docs if len(set(doc)) > 1]
    return docs

if __name__ == "__main__":
    '''training supervised model'''
    # suconfig = SUconfig()
    # dataset = Dataset(suconfig)
    # dataset.load_data("./glove.twitter.27B.100d.txt", "./train.txt",
    #                   "./test.txt")
    # model = TextCNN(suconfig, len(dataset.vocab), dataset.word_embeddings)

    # if torch.cuda.is_available():
    #     model.cuda()
    # model.train()
    # optimizer = optim.Adam(model.parameters(), lr=SUconfig().lr)
    # loss_function = nn.CrossEntropyLoss()
    # model.add_optimizer(optimizer)
    # model.add_loss_op(loss_function)
    # ##############################################################

    # train_losses = []
    # val_accuracies = []
    # val_f1s = []

    # for i in range(config.max_epochs):
    #     print("Epoch: {}".format(i))
    #     train_loss, val_accuracy, val_f1 = model.run_epoch(dataset.train_data,
    #                                             dataset.val_data, i)
    #     train_losses.append(train_loss)
    #     val_accuracies.append(val_accuracy)
    #     val_f1s.append(val_f1)

    # train_acc, trian_f1 = evaluate_model(model, dataset.train_iterator)
    # val_acc, val_f1 = evaluate_model(model, dataset.val_iterator)
    # test_acc, test_f1 = evaluate_model(model, dataset.test_iterator)

    # print('Final Training Accuracy: {:.4f}, Training F1: {:.4f}'.format(train_acc, trian_f1))
    # print('Final Validation Accuracy: {:.4f}, Validation F1: {:.4f}'.format(val_acc, val_f1))
    # print('Final Test Accuracy: {:.4f}, Test F1: {:.4f}'.format(test_acc, test_f1))
    '''filtering'''
    test_file = './positive_news.txt'
    supervised_model = torch.load('./textcnn.pkl')
    supervised_vocab = Dictionary.load_from_text('./vocab.txt')

    filtered = filter(test_file, supervised_vocab, supervised_model)
    filtered_file = './temp/health_tweets.txt'
    '''detecting'''
    # os.system("./runBTM.sh")
