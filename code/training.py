import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from config import SUconfig, UNconfig
from model import *
from preprocess import *
from utils import *

parser = argparse.ArgumentParser(description='Train supervised model')
parser.add_argument('--embed', type=str, help='the path of embeddings')
parser.add_argument('--train', type=str,
                    help='the path of training set', default='./su_data/train.txt')
parser.add_argument('--test', type=str,
                    help='the path of test set', default='./su_data/test.txt')
parser.add_argument('--dataset', type=str,
                    help=('name of dataset'), default='healthnews')
parser.add_argument('--epoch', type=int, help='maximum training epoch', default=15)
parser.add_argument('--savestep', type=int, help='when to save the model', default=10)
parser.add_argument('--restart', type=int, help='whether to continue training', default=0)

args = parser.parse_args()


if __name__ == "__main__":
    '''supervised model'''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    suconfig = SUconfig()
    dataset = Dataset(suconfig)
    dataset.load_data(args.train, args.test, args.dataset, args.embed)
    model = TextCNN(suconfig, dataset.word_embeddings)
    if args.restart == 1:
        if os.path.isfile(args.dataset + '_model.pkl'):
            model = torch.load(args.dataset + '_model.pkl')
    model.to(device)
    '''training'''
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=SUconfig().lr)
    loss_function = nn.CrossEntropyLoss()
    model.add_optimizer(optimizer)
    model.add_loss_op(loss_function)
    #############################################################

    train_losses = []
    val_accuracies = []
    val_f1s = []

    for i in range(args.epoch):
        print ("Epoch: {}".format(i))
        train_loss,val_accuracy, val_f1 = model.run_epoch(dataset.train_data, dataset.val_data, i)
        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)
        val_f1s.append(val_f1)
        if i % args.savestep == 0:
            torch.save(model, args.dataset + '_model.pkl')

    train_acc, train_f1 = evaluate_su_model(model, dataset.train_iterator())
    val_acc, val_f1 = evaluate_su_model(model, dataset.val_iterator())
    test_acc, test_f1 = evaluate_su_model(model, dataset.test_iterator())

    print ('Final Training Accuracy: {:.4f}'.format(train_acc))
    print ('Final Validation Accuracy: {:.4f}'.format(val_acc))
    print ('Final Test Accuracy: {:.4f}'.format(test_acc))
    torch.save(model, args.dataset + '_model.pkl')
    # torch.save(model.state_dict(), args.dataset + '_parameter.pkl')
