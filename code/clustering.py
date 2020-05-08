import argparse
import multiprocessing
import sys

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import requests
import torch
from bokeh.io import output_notebook
from bokeh.models import Label
from bokeh.plotting import figure, output_file, show
from gensim.models.phrases import Phrases
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from nltk.corpus import stopwords
from sklearn import datasets
from sklearn.manifold import TSNE
from wordcloud import STOPWORDS, WordCloud

from model import *
from palmettopy.palmetto import Palmetto

stop_words = stopwords.words('english')


parser = argparse.ArgumentParser(description='Topic modeling')
parser.add_argument('--fname', type=str, help='file needed to be clustered',
                    default='./temp/health_tweets.txt')
parser.add_argument('--model', type=str, help='ebtm, btm, lda', default='btm')
parser.add_argument('--iter', type=int,
                    help='maximum training iteration', default=100)
parser.add_argument('--K', type=int, help='number of topics', default=8)
parser.add_argument('--maxwords', type=int,
                    help='maximum words of a topic', default=10)
parser.add_argument('--dataset', type=str,
                    help=('name of dataset'), default='healthnews')

palmetto = Palmetto()
args = parser.parse_args()
topics = None
p_z_ds = None
voca = None

w2id = {}


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.1f}'.format(height * 100),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def indexFile(pt, res_pt):
    wf = open(res_pt, 'w')
    for l in open(pt):
        ws = l.strip().split()
        for w in ws:
            if w not in w2id:
                w2id[w] = len(w2id)

        wids = [w2id[w] for w in ws]
        print(' '.join(map(str, wids)), file=wf)


def write_w2id(res_pt):
    wf = open(res_pt, 'w')
    for w, wid in sorted(w2id.items(), key=lambda d: d[1]):
        print('%d\t%s' % (wid, w), file=wf)


def read_voca(pt):
    voca = {}
    for l in open(pt):
        wid, w = l.strip().split('\t')[:2]
        voca[int(wid)] = w
    return voca


def read_pz(pt):
    return [float(p) for p in open(pt).readline().split()]


def read_pzd(pt):
    probabilities = []
    with open(pt) as f:
        for line in f:
            probabilities.append([float(x) for x in line.split()])
    return probabilities


def BTMTopics(pt, vocab, pz, max_words=args.maxwords):
    k = 0
    topics = []
    for l in open(pt):
        vs = [float(v) for v in l.split()]
        wvs = zip(range(len(vs)), vs)
        top_words = sorted(wvs, key=lambda d: d[1], reverse=True)
        top_words = [(vocab[w], v) for w, v in top_words[:max_words]]
        topics.append((pz[k], top_words))
        k += 1
    # topics = sorted(topics, key=lambda x: x[0], reverse=True)
    return topics


def get_request(url):
    for _ in range(5):
        try:
            return float(requests.get(url).text)
        except:
            pass
    return None


def get_coherence(topic):
    try:
        cp = palmetto.get_coherence(topic, coherence_type="cp")
        ca = palmetto.get_coherence(topic, coherence_type="ca")
        return cp+ca
    except:
        return -1


if __name__ == "__main__":
    vis_dir = './visual/'
    if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
    for word in stop_words:
        STOPWORDS.add(word)

    if args.model == 'btm':
        doc_pt = args.fname
        dwid_pt = './temp/doc_wids.txt'
        voca_pt = './temp/voca.txt'
        model_dir = './temp/model/'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        indexFile(doc_pt, dwid_pt)
        write_w2id(voca_pt)
        vocab_size = len(w2id)
        # encode documents and build vocab

        alpha = 50 / args.K
        beta = 0.005
        save = 501

        os.system('make -C ./btm')
        os.system('./btm/btm est {} {} {} {} {} {} {} {}'.format(args.K, vocab_size, alpha, beta, args.iter, save, dwid_pt, model_dir))
        # train btm
        os.system('./btm/btm inf sum_b {} {} {}'.format(args.K, dwid_pt, model_dir))
        # infer

        voca = read_voca(voca_pt)
        pz_pt = model_dir + 'pz'
        pzd_pt = model_dir + 'pz_d'
        pz = read_pz(pz_pt)
        p_z_ds = read_pzd(pzd_pt)
        zw_pt = model_dir + 'pw_z'
        topics = BTMTopics(zw_pt, voca, pz)
        # return topics

    elif args.model == 'ebtm':
        unconfig = UNconfig()
        undataset = UnDataset(unconfig)
        undataset.load_data(filtered_file)
        ebtm = EBTM(args.K, len(dataset.vocab), unconfig.hidden, embeddings=torch.Tensor(
            undataset.embeddings), batch_size=unconfig.batch_size, theta_act=unconfig.act, enc_drop=unconfig.dropout_keep)
        optimizer = optim.AdamW(ebtm.parameters(), lr=0.01)
        ebtm.add_optimizer(optimizer)
        losses = []
        com_times = []
        for i in range(args.iter):
            avg_kl, avg_rec, avg_loss, compute_time = nbtm.run_epoch(
                dataset.biterms, i)
            losses.append(avg_loss)
            com_times.append(compute_time)
            if avg_kl < 0.005:
                break
        topics = ebtm.topics(undataset.vocab, 10)

    elif args.model == 'lda':
        pass
    else:
        exit(0)

    '''visualization'''
    p_zs = [x[0] for x in topics]
    p_w_zs = [[x[1] for x in data[1]] for data in topics]
    top_words = [[x[0] for x in data[1]] for data in topics]

    # url = "https://palmetto.demos.dice-research.org/service/{}?words={}"
    # reqs = [url.format(s, '%20'.join(topic))
    #         for s in ['ca', 'cp'] for topic in topics]

    # pool = multiprocessing.Pool()
    # coherences = pool.map(get_request, reqs)
    # pool.close()
    # pool.terminate()
    # pool.join()
    # del pool
    # for i, topic in enumerate(topics):
    #     try:
    #         cp = palmetto.get_coherence(topic, coherence_type="cp")
    #         ca = palmetto.get_coherence(topic, coherence_type="ca")
    #         sum_cohe = cp+ca
    #         sum_cohes.append(sum_cohe)
    #     except:
    #         continue

    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
    cloud = WordCloud(stopwords=STOPWORDS,
                      background_color='white',
                      width=2500,
                      height=1800,
                      max_words=10,
                      colormap='tab10',
                      color_func=lambda *args, **kwargs: cols[i],
                      prefer_horizontal=1.0)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        cloud.generate_from_frequencies(dict(topics[i][1]), max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i+1), fontdict=dict(size=16))
        plt.gca().axis('off')


    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, args.dataset + '_word_cloud.jpg'))


    
    if p_z_ds is not None:
        # docs = []
        # representives = []
        # representives_num = 1
        # with open('./temp/doc_wids.txt') as f:
        #     for line in f:
        #         docs.append([int(x) for x in line.split()])
        # p_z_ds = np.array(p_z_ds)
        # p_d_zs = p_z_ds.T
        # for i in range(args.K):
        #     representives.append(p_d_zs[i].argsort()[-representives_num:][::-1])
        # representives = [[[voca[w] for w in docs[idx]] for idx in representive] for representive in representives]
        # print(representives)
        p_z_ds = np.array(p_z_ds)
        prediction = np.argmax(p_z_ds, 1)
        count = {}
        for i in prediction:
            if i in count:
                count[i] += 1
            else:
                count[i] = 1

        normalised = np.array(list(count.values())) / sum(list(count.values()))
        labels = ['Topic' + str(i+1) for i in range(len(normalised))]

        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots(figsize=(10,4))
        rects1 = ax.bar(x - width/2, normalised, width, label='True')
        rects2 = ax.bar(x + width/2, pz, width, label='Theoretical')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Percentage')
        ax.set_title('True topic distribution VS Theoretical topic distribution')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        autolabel(rects1)
        autolabel(rects2)

        fig.tight_layout()
        plt.savefig(os.path.join(vis_dir, args.dataset + '_distribution.jpg'))
