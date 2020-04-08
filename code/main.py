import json
import logging
import multiprocessing
import os
import pickle

import bcolz
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from nltk.corpus import stopwords

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
unconfig = UNconfig()

vectors_path = './27B.100d.dat/'
words_path = './words.pkl'
word2idx_path = './idx.pkl'
# vectors = bcolz.open(vectors_path)[:]
# words = pickle.load(open(words_path, 'rb'))
# word2idx = pickle.load(open(word2idx_path, 'rb'))

# glove = {w: vectors[word2idx[w]] for w in words}


def collect_all(dir):
    file_list = find_suffix('json', dir)
    out_file = os.path.join(dir, "training.txt")
    out_f = open(out_file, 'w', encoding='utf-8')
    for file in file_list:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                text = None
                try:
                    data = json.loads(line)
                    text = data['text']
                except:
                    continue
                tokens = text_pro.exclude_stop_word(text, STOP_WORDS)
                if tokens:
                    text = ' '.join(tokens)
                    out_f.write(text + "\n")

def negative(file, out_file):
    with open(file, 'r', encoding='utf-8') as f, open(out_file, 'w', encoding='utf-8') as out_f:
        for line in f:
            tokens = line.split()
            tokens = [x for x in tokens if len(x) > 2 and "\'" not in x]
            Found = False
            if len(tokens) < 4:
                continue
            for w in WORD_LIST['include_words']:
                if w.lower() in tokens:
                    Found = True
                    break
            if not Found:
                for phrase in WORD_LIST['phrases']:
                    if phrase.lower() in line:
                        Found = True
                        break
            if Found:
                continue
            else:
                out_f.write(" ".join(tokens) + '\n')



if __name__ == "__main__":
    # input_path = "F:\Twitter\\2019\\01"
    # out_dir = 'F:\\temp'
    # acv = Archive(WORD_LIST)
    # acv.filter_dirs(input_path, out_dir, ret=True, keywords=False)

    # data_count(input_path,func = 'line')

    # unzip_tree("/Volumes/NonBee5/Twitter/2018")

    '''save preprocessed cdc'''
    # cdc_in_path = '/Users/NonBee/Desktop/FYP/code/Data/cdc/StateDatabySeason59_58,57.csv'
    # cdc_out_path = './Data/dataset/'
    # cdc = CDC_preprocessor()
    # cdc.get_information(cdc_in_path, cdc_out_path)

    '''unzip files recursively'''
    # tar_path = "/Volumes/Data/Twitter/2019/01"
    # unzip_tree(tar_path, "bz2", tar_path)

    '''labeling'''
    # data_path = './Data/dataset/2018/10/2018_10_05.json'
    # out_name = './Data/twitter.json'
    # text_pro.label_file(data_path, out_name)

    '''extracting corpus'''
    # out_dir = '/Volumes/White/text'
    # in_dir = '/Volumes/White/Data/filtered/2018/01/2018_01_04.json'
    # text_pro.text_only_dir(in_dir, out_dir, stop_words=STOP_WORDS, rm_low_frequency=True,
    #             min_frequency=MIN_FREQUENCY)

    '''count line'''
    # dirc = "/Volumes/White/Data/filtered/2018"
    # dicts = [os.path.join(dirc, x) for x in os.listdir(dirc)]
    # for path in dicts:
    #      line_count(path, "json")

    # lable_news("H:\\training")
    # collect_all("H:\Data\\tweets\\2019\\2019\\01")

    '''supervised model'''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    suconfig = SUconfig()
    dataset = Dataset(suconfig)
    dataset.load_data("./glove.twitter.27B.100d.txt", "./train.txt",
                      "./test.txt")
    model = TextCNN(suconfig, len(dataset.vocab), dataset.word_embeddings)

    if torch.cuda.is_available():
        model.cuda()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=SUconfig().lr)
    loss_function = nn.CrossEntropyLoss()
    model.add_optimizer(optimizer)
    model.add_loss_op(loss_function)
    ##############################################################

    train_losses = []
    val_accuracies = []
    val_f1s = []

    for i in range(config.max_epochs):
        print("Epoch: {}".format(i))
        train_loss, val_accuracy, val_f1 = model.run_epoch(dataset.train_data,
                                                dataset.val_data, i)
        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)
        val_f1s.append(val_f1)

    train_acc, trian_f1 = evaluate_model(model, dataset.train_iterator)
    val_acc, val_f1 = evaluate_model(model, dataset.val_iterator)
    test_acc, test_f1 = evaluate_model(model, dataset.test_iterator)

    print('Final Training Accuracy: {:.4f}, Training F1: {:.4f}'.format(train_acc, trian_f1))
    print('Final Validation Accuracy: {:.4f}, Validation F1: {:.4f}'.format(val_acc, val_f1))
    print('Final Test Accuracy: {:.4f}, Test F1: {:.4f}'.format(test_acc, test_f1))
