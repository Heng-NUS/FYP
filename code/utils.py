import bz2
import gc
import html
import json
import os
import re
import tarfile
import time
import zlib
from collections import defaultdict
from pprint import pprint

import emoji
import nltk
import numpy as np
from gensim.utils import deaccent
from nltk.tokenize import word_tokenize
from smart_open import open


def find_suffix(suffix, dir_path):
    '''find all files ending with suffix in directory'''
    if not os.path.exists(dir_path):
        print("Can't find", dir_path)
        exit(0)
    name_buffer = []
    for file in os.listdir(dir_path):
        if file[0] == ".":
            continue
        subpath = os.path.join(dir_path, file)
        if os.path.isdir(subpath):
            name_buffer += find_suffix(suffix, subpath)
        else:
            if file.endswith(suffix):
                name_buffer.append(subpath)
    return name_buffer


def bz2_unzip(file_path, out_path=None):
    '''unzip a single bz2 file'''
    # default unzip to its original directory
    file_name = file_path.split('/')[-1]
    if file_path.endswith('.bz2') and file_name[0] != '.':
        try:
            print('Unzipping:', file_path)
            zipfile = bz2.BZ2File(file_path)
            data = zipfile.read()
        except IOError:
            print("can't read", file_path)
            # get the decompressed data
        else:
            newfilepath = file_path[:-4]
            # assuming the filepath ends with .bz2
            if out_path is not None:
                newfilepath = os.path.join(out_path, newfilepath)
            try:
                with open(newfilepath, 'wb') as file:
                    file.write(data)
                    del data
                    print('Unzipped:', file_path)
                    # write a uncompressed file
            except IOError:
                print("can't unzip", file_path)
                with open('unzip_error.log', 'a') as log:
                    log.write(file_path)
                    # if can't unzip, record it
            else:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    # remove this file
    else:
        print("not a bz2 file:", file_path)


def tar_unzip(file_path, out_path=None):
    '''unzip a single bz2 file'''
    file_name = file_path.split('/')[-1]
    if file_path.endswith('.tar') and file_name[0] != '.':
        try:
            file = tarfile.open(file_path)
        except IOError:
            print("Can't unzip:", file_path)
        else:
            print('Unzipping:', file_path)
            newfilepath = file_path[:-4]
            if out_path:
                file.extractall(out_path)
            else:
                file.extractall(newfilepath)
            file.close()
            print('Unzipped:', file_path)
            if os.path.exists(file_path):
                os.remove(file_path)
                # remove this file


def unzip_tree(dir_path, zip_type, out_dir=None):
    '''unzip all files in this dir recursively'''
    if not os.path.exists(dir_path):
        print("Can't find", dir_path)
        exit(0)
    else:
        path_list = find_suffix(zip_type, dir_path)
        if zip_type == "bz2":
            for file_path in path_list:
                if out_dir:
                    bz2_unzip(file_path, out_dir)
                else:
                    bz2_unzip(file_path)
                gc.collect()
        elif zip_type == "tar":
            for file_path in path_list:
                if out_dir:
                    tar_unzip(file_path, out_dir)
                else:
                    tar_unzip(file_path)
                gc.collect()


def line_count(path, suffix=None):
    if os.path.isfile(path):
        count = 0
        with open(path, 'r', encoding='utf-8') as file:
            for f in file:
                count += 1
        print(path, count)
        return count
    elif os.path.isdir(path):
        path_list = find_suffix(suffix, path)
        all_count = 0
        for file in path_list:
            count = 0
            with open(file, 'r', encoding='utf-8') as file:
                for f in file:
                    count += 1
            all_count += count
        print(path, all_count, "average:", int(all_count/len(path_list)))
        return all_count


def pattern_count(path, pattern='^RT @'):
    count = 0
    with open(path, 'r', encoding='utf-8') as file:
        for f in file:
            data = json.loads(f)
            if 'text' in data and re.search(pattern, data['text']):
                count += 1
    return count


def data_count(dir_path, func, suffix='json'):
    '''count line_numbers of all files with suffix in directroy'''
    path_list = find_suffix(suffix, dir_path)
    ori_count = 0
    for file_path in path_list:
        if func == 'line':
            ori = line_count(file_path)
        elif func == 'pattern':
            ori = pattern_count(file_path)
        else:
            break
        ori_count += ori
    print("ori:", ori_count)


def fail_record(file_path):
    log_file = time.strftime("%Y_%m_%d %H_%M", time.localtime())
    error_path = './log/' + log_file + '.log'
    with open(error_path, 'a+')as file:
        file.write(file_path + '\n')


class mapper(object):
    """map twitter text to ILI level"""

    def __init__(self, tweet_count, user_count):
        super(mapper, self).__init__()
        self.tweet_count = tweet_count
        self.user_count = user_count
        self.coefficient = 0.15
        # default coefficient is 0.15
        self.normalised = False

    def normalise(self):
        # normalise the regional data based on its number of users
        mean = np.mean(list(self.user_count.values()))
        if not self.normalised:
            try:
                for state in self.tweet_count.keys():
                    if state in self.user_count:
                        self.tweet_count[state] /= self.user_count[state] / 100
                    else:
                        self.tweet_count[state] /= mean / 100
                    # T(s,n) = T(s,o) * 100/ user_number(s)
            except IOError:
                pass
            self.normalised = True

    def map_level(self):
        # map the number of tweets to ILI level
        self.normalise()
        mean = np.mean(list(self.tweet_count.values()))
        std = np.std(list(self.tweet_count.values()))
        level_num = 10
        self.level = dict()
        # initialize the level array to the maximum level

        for key in self.tweet_count.keys():
            for i in range(1, level_num):
                if self.tweet_count[key] <= mean + (i-2) * std * self.coefficient:
                    self.level[key] = i
                    break

        return self.level

    def update_tweet_count(self, tweet_count, user_count=None):
        self.tweet_count = tweet_count
        if user_count:
            self.user_count = user_count
        self.normalised = False


class Diffusion(object):
    """predict the diffusion of diseases"""

    def __init__(self, tweet_count, map_path='./Data/usa_adjacency.json'):
        super(Diffusion, self).__init__()
        try:
            with open("./Data/usa_adjacency.json", 'r', encoding='utf-8') as file:
                self.adjacency_mat = json.load(file)
        except IOError:
            print("Can't find file:", map_path)


class text_pro(object):
    """Some functions for processing text"""
    def __init__(self):
        super(text_pro, self).__init__()

    def regularize(text):
        '''return the regularized text'''
        r1 = u'^RT @.*?: |@+[^\s]*|^RT\s'  # exclude RT @... \n
        # http(s)
        r2 = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        r3 = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'  # e-mail
        r4 = '\s+'  # multiple empty chars
        r5 = 'http[s]?:.*? '
        # r6 = "[^A-Za-z0-9_']"  # not alphabet number and _
        r7 = '\*.+?\*'
        sub_rule = r1 + '|' + r2 + '|' + r3 + '|' + r5 + '|' + r7
        text = html.unescape(text)
        text = deaccent(text)
        text = re.sub(sub_rule, " ", text)
        text = emoji.demojize(text, delimiters=('emo_', ' '))
        # text = re.sub(r6, ' ', text)
        text = re.sub(r4, ' ', text)
        return text.lower()

    def regularize_json(in_path, out_path):
        '''regularize all tweets in a json file and write the processed text'''
        with open(in_path, 'r', encoding='utf-8') as in_file, open(out_path, 'w', encoding='utf-8') as out_file:
            for line in in_file:
                if line != "\n":
                    try:
                        data = json.loads(line)
                    except IOError:
                        print("Can't read ", in_path)
                    else:
                        if 'lang' not in data or data['lang'] != 'en':
                            continue
                        data['text'] = text_pro.regularize(data['text'])
                        json.dump(data, out_file)
                        out_file.write("\n")

    def geo_convert(data):
        if data['geo']:
            pass
        elif data['location']:
            pass

    def exclude_same_text(in_path, out_path):
        with open(in_path, 'r', encoding='utf-8') as in_file, open(out_path, 'w', encoding='utf-8') as out_file:
            exclude_buffer = set()
            for line in in_file:
                data = json.loads(line)
                if data['text'] not in exclude_buffer:
                    exclude_buffer.add(data['text'])
                    json.dump(data, out_file)
                    out_file.write('\n')

    def label_data(data):
        '''label single data'''
        print(data['text'] + "\n")
        label = input("Relevant to health? ")
        while label != '' and label != ' ':
            label = input("Enter: 0 Space: 1 ")
        if label == '':
            data['label'] = 0
        elif label == ' ':
            data['label'] = 1

    def label_file(in_path, out_path):
        file_name = in_path.split('/')[-1]
        log_file = './log/label_log.json'
        if not os.path.exists(log_file):
            with open(log_file, 'w', encoding='utf-8') as log:
                data = {'data_count': 0, 'positive': 0, file_name: 0}
                json.dump(data, log)
                log.flush
        rule = ""
        ex_rule = ""
        for x in WORD_LIST['include_words']:
            new_words = "(?:^|\W)" + x + "(?:$|\W)|"
            rule += new_words
        rule = rule[:-1]

        for x in WORD_LIST['exclude_words']:
            new_words = "(?:^|\W)" + x + "(?:$|\W)|"
            ex_rule += new_words
        ex_rule = ex_rule[:-1]

        check_list = set()
        with open(in_path, 'r') as in_file, open(out_path, 'a+', encoding='utf-8') as out_file:
            count = 0
            log_count = 0
            data_count = 0
            positive = 0
            read_log = {}
            with open(log_file, 'r', encoding='utf-8') as log:
                try:
                    read_log = json.load(log)
                except:
                    print("Created a new log file")
                else:
                    data_count = read_log['data_count']
                    positive = read_log['positive']
                    if file_name in read_log:
                        log_count = read_log[file_name]

            for line in in_file:
                if log_count != 0:
                    log_count -= 1
                    count += 1
                    continue

                if line != "\n":
                    try:
                        data = json.loads(line)
                    except IOError:
                        print("Can't read ", in_path)
                    else:
                        text = data['text']
                        if text in check_list:
                            count += 1
                            continue

                        check_list.add(text)
                        new_data = dict(text=text)

                        if not re.search(rule, text, flags=re.IGNORECASE) or re.search(ex_rule, text, flags=re.IGNORECASE):
                            new_data['label'] = 0
                        else:
                            print(text + "\n")
                            label = input(
                                "Relevant to ful? (Enter q to quit) ")
                            while label != '' and label != ' ' and label != 'q' and label != 's':
                                label = input("Enter: 0 Space: 1 Q:quit")
                            if label == '':
                                new_data['label'] = 0
                                print("\n")
                            elif label == ' ':
                                new_data['label'] = 1
                                positive += 1
                            elif label == 's':
                                count += 1
                                read_log[file_name] = count
                                read_log['data_count'] = data_count
                                read_log['positive'] = positive
                                with open(log_file, 'w', encoding='utf-8') as log:
                                    json.dump(read_log, log, indent=4)
                                continue
                            elif label == 'q':
                                break

                        json.dump(new_data, out_file)
                        out_file.write("\n")
                        out_file.flush()
                        data_count += 1
                        print("data_number:", data_count,
                              'positive:', positive, '\n')

                count += 1
                read_log[file_name] = count
                read_log['data_count'] = data_count
                read_log['positive'] = positive
                with open(log_file, 'w', encoding='utf-8') as log:
                    json.dump(read_log, log, indent=4)

            read_log[file_name] = count
            read_log['data_count'] = data_count
            read_log['positive'] = positive
            with open(log_file, 'w', encoding='utf-8') as log:
                json.dump(read_log, log, indent=4)

    def process_tweets(path):
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line, encoding='utf-8')
                text = data['text']
                text_pro.regularize(text)

    def exclude_stop_word(text, stop_words):
        '''exclude the stop words from text'''
        filtered = [
            word for word in word_tokenize(text) if word not in stop_words and len(word) > 1]
        temp = []
        r1 = "[0-9]|__+|^_"
        r2 = '[A-Za-z]'
        for token in filtered:
            if not re.search(r2, token) or re.search(r1, token):
                continue
            else:
                temp.append(token)

        if len(temp) < 4:
            return 0
        return temp

    def word_frequency_json(file):
        '''return the word frequency of a twitter file'''
        frequency = defaultdict(int)
        with open(file, 'r', encoding='utf-8') as f:
            line_count = 0
            for line in f:
                line_count += 1
                if line != "\n":
                    try:
                        data = json.loads(line)
                    except:
                        print("Error with:", in_file, "Line", line_count)
                        continue
                    else:
                        text = data['text']
                        token = word_tokenize(text)
                        for word in token:
                            frequency[word] += 1
        return frequency

    def text_only(in_file, out_file, stop_words=None, frequency=None, min_frequency=3):
        '''save extracted text from a single file to out_file'''
        with open(in_file, 'r', encoding='utf-8') as in_f, open(out_file, 'a', encoding='utf-8') as out_f:
            line_count = 0
            print("processing: ", in_file)
            start_t = time.clock()
            for line in in_f:
                line_count += 1
                if line != "\n":
                    try:
                        data = json.loads(line)
                    except:
                        print("Error with:", in_file, "Line", line_count)
                        continue
                    else:
                        text = data['text']
                        if stop_words:
                            text = text_pro.exclude_stop_word(text, stop_words)
                            if text == 0:
                                continue
                        else:
                            text = text.split()
                            if len(text) < 4:
                                continue
                        if frequency:
                            text = [
                                token for token in text if frequency[token] >= min_frequency]
                        text = " ".join(text)
                        out_text = (text + "\n")
                        #out_text = data['text']
                        out_f.write(out_text)
            end_t = time.clock()
            print("Finished:", end_t - start_t)

    def text_only_dir(in_dir, out_dir, stop_words=None, suffix="json", rm_low_frequency=True, min_frequency=3):
        '''extract corpus from a directory'''
        start_t = time.clock()
        path_list = []
        if os.path.isfile(in_dir):
            path_list = [in_dir]
        elif os.path.exists(in_dir):
            path_list = find_suffix(suffix, in_dir)
        else:
            print("Can't find", in_dir)
            exit(0)

        frequency = defaultdict(int)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        if rm_low_frequency:
            for in_file in path_list:
                temp = text_pro.word_frequency_json(in_file)
                for word, count in temp.items():
                    frequency[word] += count
            with open(os.path.join(out_dir, 'frequency.json'), 'w', encoding='utf-8') as file:
                json.dump(obj=frequency, fp=file)
        # frequency list

        for in_file in path_list:
            file_name = os.path.basename(in_file)
            file_name = file_name.split('.')[0]

            out_file = os.path.join(out_dir, file_name + ".txt")
            if rm_low_frequency:
                text_pro.text_only(in_file, out_file, frequency=frequency,
                                   min_frequency=min_frequency, stop_words=stop_words)
            else:
                text_pro.text_only(in_file, out_file, stop_words=stop_words)
        end_t = time.clock()
        print("Extracting Finished:", end_t-start_t)


class Visualization(object):
    '''some toolkit for analysis and visualization'''

    def __init__(self):
        super(Visualization, self).__init__()

    def rank_retweets(in_dir):
        '''return the top retweets'''
        record = {}
        file_list = find_suffix('json', in_dir)
        for file in file_list:
            with open(file, 'r', encoding='utf-8') as f:
                temp = ""
                for line in f:
                    if line == '\n':
                        continue
                    else:
                        temp += line

                    if '}\n' in line:
                        data = json.loads(temp, encoding='utf-8')
                        for text, value in data.items():
                            if text not in record:
                                record[text] = value
                            else:
                                record[text] += value
                        temp = ""

        rank_result = sorted(record.items(), key=lambda x: x[1], reverse=True)
        return rank_result
# tweet_count = dict()
# user_count = dict()
# with open("./Data/test_result.json", 'r', encoding='utf-8') as file:
# 	tweet_count = json.load(file)
# with open("./Data/test_users.json", 'r', encoding='utf-8') as file:
# 	user_count = json.load(file)

# a = mapper(tweet_count, user_count)
# a.map_level()
# print(a.level)
# b = Diffusion(tweet_count)
