#modeling.py
import json
import os
import random
from collections import defaultdict

import gensim
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gensim import corpora, matutils, models, similarities
from gensim.corpora import Dictionary
from gensim.utils import deaccent, simple_preprocess
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, f1_score
from smart_open import open
from torch.autograd import Variable
from torchtext import data
from torchtext.vocab import Vectors

from utils import *

dtype = torch.FloatTensor
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def read_corpus(fname, tokens_only=True):
    with open(fname, encoding="utf-8") as f:
        for i, line in enumerate(f):
            tokens = gensim.utils.simple_preprocess(line)
            if tokens_only:
                yield tokens
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

def read_labeled(fname):
    examples = []
    with open(fname, encoding="utf-8") as f:
        for line in f:
            if line == '\n':
                continue
            tokens = gensim.utils.simple_preprocess(line[2:], deacc=True)
            try:
                examples.append((tokens, int(line[0])))
            except:
                print(line[0])
    random.shuffle(examples)
    return [x[0] for x in examples], [x[1] for x in examples]


class Dataset(object):
    '''load data'''
    def __init__(self, config):
        self.config = config
        self.train_data = None
        self.test_data = None
        self.val_data = None
        self.vocab = None
        self.word_embeddings = None

    def get_pandas_df(self, filename):
        '''
        Load the data into Pandas.DataFrame object
        This will be used to convert data to torchtext object
        '''
        with open(filename, 'r', encoding='utf-8') as datafile:
            data = [line.strip().split(' ', maxsplit=1) for line in datafile]
            data_text = list(map(lambda x: x[1], data))
            data_label = list(map(lambda x: x[0], data))

        full_df = pd.DataFrame({"text": data_text, "label": data_label})
        return full_df

    def load_data(self,
                  train_file,
                  test_file,
                  embed_file=None,
                  val_file=None,
                  voc_file='vocab.txt',
                  new_embed='word_embeddings.pkl'):
        '''
        Loads the data from files
        Sets up iterators for training, validation and test data
        Also create vocabulary and word embeddings based on the data
        
        Inputs:
            embed_file (String): absolute path to file containing word embeddings (GloVe/Word2Vec)
            train_file (String): absolute path to training file
            test_file (String): absolute path to test file
            val_file (String): absolute path to validation file
        '''
        #load embeddings

        train_X, train_Y = read_labeled(train_file)
        test_X, test_Y = read_labeled(test_file)
        val_X = None
        val_Y = None
        if val_file:
            val_X, val_Y = read_labeled(val_file)
        else:
            sp = int(len(train_X) * 0.8)
            train_X, val_X = (train_X[:sp], train_X[sp:])
            train_Y, val_Y = (train_Y[:sp], train_Y[sp:])
        train_X = [doc_padding(x, self.config.max_sen_len) for x in train_X]
        test_X = [doc_padding(x, self.config.max_sen_len) for x in test_X]
        val_X = [doc_padding(x, self.config.max_sen_len) for x in val_X]

        if os.path.isfile(voc_file):
            self.vocab = Dictionary.load_from_text(voc_file)
        else:
            self.vocab = Dictionary(train_X)
            special_tokens = {'<pad>': 0, '<unk>': 1}
            self.vocab.patch_with_special_tokens(special_tokens)
            self.vocab.save_as_text('vocab.txt')
        #build vocab
        train_X = [self.vocab.doc2idx(x, 1) for x in train_X]
        test_X = [self.vocab.doc2idx(x, 1) for x in test_X]
        val_X = [self.vocab.doc2idx(x, 1) for x in val_X]
        #transform words to index
        if os.path.isfile(new_embed):
            self.word_embeddings = torch.load(new_embed)
        else:
            if not embeds:
                print("need a word embedings")
                exit(0)
            embeds = Vectors(embed_file,
                             unk_init=lambda x: torch.Tensor(
                                 np.random.normal(scale=0.6, size=(x.size()))))
            self.word_embeddings = weight_matrix(self.config.embed_size, self.vocab, embeds)
            torch.save(self.word_embeddings, "word_embeddings.pkl")
        self.train_data = (train_X, train_Y)
        self.test_data = (test_X, test_Y)
        self.val_data = (val_X, val_Y)

        print("Loaded {} training examples".format(len(train_X)))
        print("Loaded {} test examples".format(len(test_X)))
        print("Loaded {} validation examples".format(len(val_X)))


    def train_iterator(self):
        return batch_iter(*self.train_data, self.config.batch_size)
    def test_iterator(self):
        return batch_iter(*self.test_data, self.config.batch_size, False)
    def val_iterator(self):
        return batch_iter(*self.val_data, self.config.batch_size, False)


class Unsupervised(nn.Module):
    '''Unsupervised clustering model'''
    def __init__(self):
        super(Unsupervised, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv1d(embedding_size, topics, filter_size, bias=True, padding=padding, dilation=dilation, stride=stride),
                                   nn.ReLU())
        l_out = (sequence_length + 2 * padding - dilation * (filter_size-1) - 1) / stride + 1
        self.out = nn.Linear(topics * l_out, embedding_size)


    def forward(self, x):
        x = x.permute(0, 2, 1)  # x (num_doc, embedding_size, sequence_length) transpose
        x = self.layer1(x)
        x = x.view(x.size()[0], -1)  # flat
        out = self.out(x)
        return out


def batch_iter(x, y, batch_size=64, shuffle=True):
    """return batches"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    x_shuffle = x
    y_shuffle = y
    if shuffle:
        examples = list(zip(x, y))
        random.shuffle(examples)
        # indices = np.random.permutation(np.arange(data_len))
        # x_shuffle = x[indices]
        # y_shuffle = y[indices]
        x_shuffle = [x[0] for x in examples]
        y_shuffle = [x[1] for x in examples]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield torch.LongTensor(x_shuffle[start_id:end_id]), torch.LongTensor(y_shuffle[start_id:end_id])


def doc_padding(tokens, sen_length, padding_with = '<pad>'):
    '''pad a list to a fixed length with certain word'''
    if len(tokens) >= sen_length:
        return tokens[:sen_length]
    else:
        return tokens + [padding_with for i in range(sen_length - len(tokens))]

class TextCNN(nn.Module):
    def __init__(self, config, vocab_size, word_embeddings):
        super(TextCNN, self).__init__()
        self.config = config

        # Embedding Layer
        #self.embeddings = create_emb_layer(word_embeddings)
        self.embeddings = nn.Embedding(vocab_size, self.config.embed_size)
        self.embeddings.weight = nn.Parameter(word_embeddings,
                                              requires_grad=True)
        self.embeddings.padding_idx = 1

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=self.config.embed_size,
                      out_channels=self.config.num_channels,
                      kernel_size=self.config.kernel_size[0]), nn.ReLU(),
            nn.MaxPool1d(self.config.max_sen_len - self.config.kernel_size[0] +
                         1))
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=self.config.embed_size,
                      out_channels=self.config.num_channels,
                      kernel_size=self.config.kernel_size[1]), nn.ReLU(),
            nn.MaxPool1d(self.config.max_sen_len - self.config.kernel_size[1] +
                         1))
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=self.config.embed_size,
                      out_channels=self.config.num_channels,
                      kernel_size=self.config.kernel_size[2]), nn.ReLU(),
            nn.MaxPool1d(self.config.max_sen_len - self.config.kernel_size[2] +
                         1))

        self.dropout = nn.Dropout(self.config.dropout_keep)

        # Fully-Connected Layer
        self.fc = nn.Linear(
            self.config.num_channels * len(self.config.kernel_size),
            self.config.output_size)

        # Softmax non-linearity
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x.shape = (max_sen_len, batch_size)
        embedded_sent = self.embeddings(x)
        embedded_sent = torch.transpose(embedded_sent,1,2)

        conv_out1 = self.conv1(embedded_sent).squeeze(2)
        conv_out2 = self.conv2(embedded_sent).squeeze(2)
        conv_out3 = self.conv3(embedded_sent).squeeze(2)

        all_out = torch.cat((conv_out1, conv_out2, conv_out3), 1)
        final_feature_map = self.dropout(all_out)
        final_out = self.fc(final_feature_map)
        # return self.softmax(final_out)
        return final_out

    def add_optimizer(self, optimizer):
        self.optimizer = optimizer

    def add_loss_op(self, loss_op):
        self.loss_op = loss_op

    def reduce_lr(self):
        print("Reducing LR")
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] / 2

    def run_epoch(self, train_data, val_data, epoch):
        losses = []
        self.train()

        # # Reduce learning rate as number of epochs increase
        # if (epoch == int(self.config.max_epochs / 3)) or (epoch == int(
        #         2 * self.config.max_epochs / 3)):
        #     self.reduce_lr()

        train_iterator = batch_iter(*train_data, self.config.batch_size)

        for i, batch in enumerate(train_iterator):
            self.optimizer.zero_grad()
            x = batch[0].to(device)
            y = batch[1].to(device)
            y_pred = self.__call__(x)
            loss = self.loss_op(y_pred, y)
            loss.backward()
            losses.append(loss.data.cpu().numpy())
            self.optimizer.step()

        val_iterator = batch_iter(*val_data, self.config.batch_size,
                                    False)
        print("Iter: {}".format(i + 1))
        avg_train_loss = np.mean(losses)
        print("\tAverage training loss: {:.5f}".format(avg_train_loss))

        # Evalute Accuracy on validation set
        val_accuracy, val_f1 = evaluate_model(self, val_iterator)
        print("\tVal Accuracy: {:.4f}, Val F1: {:.4f}".format(val_accuracy, val_f1))

        return avg_train_loss, val_accuracy, val_f1


def evaluate_model(model, iterator):
    '''batch[0]: data, batch[1]: label'''
    model.eval()
    all_preds = []
    all_y = []
    for idx, batch in enumerate(iterator):
        x = batch[0].to(device)
        y_pred = model(x)
        predicted = torch.max(y_pred.cpu().data, 1)[1]
        all_preds.extend(predicted.numpy())
        all_y.extend(batch[1].numpy())
    accuracy = accuracy_score(all_y, np.array(all_preds).flatten())
    f1 = f1_score(all_y, np.array(all_preds).flatten())
    return accuracy,f1


def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer#, num_embeddings, embedding_dim


def weight_matrix(embedding_size, target_vocab, word_embeddings):
    '''creat a look-up table'''
    matrix_len = len(target_vocab)

    matrix = torch.zeros((matrix_len, embedding_size))
    #matrix[0] = torch.Tensor(np.random.normal(scale=0.6, size=(embedding_size, )))
    #vector of padding word

    for i in range(2,len(target_vocab)):
        try:
            matrix[i] = word_embeddings[target_vocab[i]]
        except KeyError:
            matrix[i] = torch.Tensor(np.random.normal(scale=0.6,
                                                 size=(embedding_size, )))
    return matrix

def predict(vocab, tokens, label, max_sen_len, model):
    '''predict the class of a single text'''
    model = model.to(device)
    model.eval()
    padded = doc_padding(tokens, max_sen_len)
    x = torch.LongTensor([vocab.doc2idx(padded, 1)]).to(device)
    y_pred = model(x)
    predicted = torch.max(y_pred.cpu().data, 1)[1]
    print(' '.join(tokens) + ' predict class:', predicted, "True class:", label)
