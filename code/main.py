import logging
import multiprocessing

from nltk.corpus import stopwords

from modeling import *
from preprocess import *
from utility import *


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
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if __name__ == "__main__":
    # input_path = "/Volumes/Data/Twitter/2019/01"
    # out_dir = '/Volumes/White/Data/tweets'
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

    '''other data'''
    # input_dir = "/Volumes/White/Health-Tweets"
    # out_dir = "/Volumes/White/training"
    # otherset = Other_dataset()
    # otherset.extract_news_dir(input_dir, out_dir)
