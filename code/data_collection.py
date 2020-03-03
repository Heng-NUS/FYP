import bz2
import json
import os
import re
import zlib
import tarfile
import time
import gc
import numpy as np
import pandas as pd
import emoji
import html
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

global WORD_LIST


def load_word_list(word_list_path):
    try:
        with open(word_list_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except IOError:
        print("Can't find file:", word_list_path)
        exit(0)


word_list_path = './Filter_rule/word_list.json'
WORD_LIST = load_word_list(word_list_path)


def find_suffix(suffix, dir_path):
    '''find all files ending with suffix in directory'''
    if not os.path.exists(dir_path):
        print('no such file')
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
        print('no such file')
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


def line_count(path):
    count = 0
    with open(path, 'r', encoding='utf-8') as file:
        for f in file:
            count += 1
    return count-1


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


class archive(object):
    """preprocess archive twitter data"""

    def __init__(self):
        super(archive, self).__init__()
        self.include_list = WORD_LIST['include_words']
        self.seclude_list = WORD_LIST['exclude_words']
        self.in_rule = ""
        self.ex_rule = ""
        self.stop_rule = ""
        self.out_dir = os.path.join(os.getcwd(), '/Data/dataset/')

        for x in WORD_LIST['include_words']:
            new_words = x + "|"
            self.in_rule += new_words  # loose inclusion rule
        self.in_rule = self.in_rule[:-1]

        for x in WORD_LIST['exclude_words']:
            new_words = "(?:^|\W)" + x + "(?:$|\W)|"  # strict exclusion rule
            self.ex_rule += new_words
        self.ex_rule = self.ex_rule[:-1]

        stop_words = stopwords.words('english')
        for x in stop_words:
            new_words = "(?:^|\W)" + x + "(?:$|\W)|"
            self.stop_rule += new_words  # stop words exclusion
        self.stop_rule = self.stop_rule[:-1]

    def buffer_to_json(self, out_dir, out_buffer):
        '''save buffer to json file'''
        for date, content in out_buffer.items():
            file_name = date.replace('/', '_') + '.json'
            date_split = date.split('/')
            date_dir = os.path.join(out_dir, date_split[0], date_split[1])
            if not os.path.exists(date_dir):
                os.makedirs(date_dir)
            out_path = os.path.join(date_dir, file_name)

            with open(out_path, 'a', encoding='utf-8') as out_file:
                for data in content:
                    out_file.write(json.dumps(data) + '\n')

            del content

    def ret_buffer_to_json(self, out_dir, out_buffer):
        '''save retweets to json file'''
        for date, content in out_buffer.items():
            file_name = date.replace('/', '_') + '.json'
            date_split = date.split('/')
            date_dir = os.path.join(out_dir, date_split[0], date_split[1])
            if not os.path.exists(date_dir):
                os.makedirs(date_dir)
            out_path = os.path.join(date_dir, file_name)

            with open(out_path, 'a', encoding='utf-8') as out_file:            
                out_file.write(json.dumps(content, indent=4) + '\n')

            del content

    def exclude_stop_word(self, text):
        '''exclude words in stop word list'''
        text = re.sub(self.stop_rule, text, ' ', flags=re.IGNORECASE)
        text = re.sub(self.stop_rule, text, ' ', flags=re.IGNORECASE)
        # excluding twice is required
        return text

    def regularize(self, text):
        '''return the regularized text'''

        return text_pro.regularize(text)

    def filter_tweets(self, path, out_dir, ignore_geo=True, language='en', ret=True, keywords=False):
        '''select out all useful tweets from a single json file and save them to a file'''
        print("Processing", path)
        start = time.clock()
        name = path.split('/')[-1]
        months = dict(Jan='01', Feb='02', Mar='03', Apr='04', May='05',
                      Jun='06', Jul='07', Sept='09', Oct='10', Nov='11', Dec='12')
        retweet = 0
        ori_count = 0  # record the original number
        extracted = 0  # record the extracted number
        retweet_dir = os.path.join(out_dir, 'Retweets')

        if name[0] != ".":
            with open(path, 'r', encoding='utf-8') as file:
                max_buffer_size = 5000
                out_buffer = {}
                size_count = 0

                max_ret_size = 1000
                ret_buffer = {}
                ret_size = 0

                for f in file:
                    data = json.loads(f)
                    ori_count += 1
                    if 'lang' not in data or data['lang'] != language:
                        continue

                    if 'text' in data and (ignore_geo or data['user']['geo_enabled']):
                        text = data['text']

                        time_split = data['created_at'].split()
                        date = time_split[-1] + '/' + \
                            months[time_split[1]] + '/' + time_split[2]
                        # creation date

                        if "RT @" in text:
                            retweet += 1
                            if not ret:
                                continue  # exclude Retweets if needed
                            else:
                                if date not in ret_buffer:
                                    ret_buffer[date] = {}
                                else:
                                    r1 = u'^RT @.*?: '
                                    text = re.sub(r1, '', text)
                                    if text not in ret_buffer[date]:
                                        ret_buffer[date][text] = 1
                                        ret_size += 1
                                    else:
                                        ret_buffer[date][text] += 1
                                        # set counters of retweets
                                if ret_size >= max_ret_size:
                                    self.ret_buffer_to_json(
                                        retweet_dir, ret_buffer)
                                    ret_size = 0
                                    ret_buffer.clear()

                        #text = self.exclude_stop_word(text)
                        # exclude stop words
                        text = self.regularize(text)
                        # regularize text

                        if keywords and not re.search(self.in_rule, text, flags=re.IGNORECASE):
                            continue
                            # if not in inclusion list or in exclusion list, skip it

                        location = data['user']['location']
                        coordinates = data['coordinates']
                        place = data['place']
                        created_at = date
                        selected_data = dict(created_at=created_at, text=text,
                                             location=location, coordinates=coordinates, place=place)
                        if date not in out_buffer:
                            out_buffer[date] = []
                        out_buffer[date].append(selected_data)
                        extracted += 1
                        size_count += 1
                        if size_count >= max_buffer_size:
                            # write the buffer
                            self.buffer_to_json(out_dir, out_buffer)
                            size_count = 0
                            out_buffer.clear()

                        del data

                    else:
                        del data
                        continue

                self.buffer_to_json(out_dir, out_buffer)
                del out_buffer

                if ret:
                    self.ret_buffer_to_json(retweet_dir, ret_buffer)
                    del ret_buffer
                    # save retweets if needed
        endt = time.clock()

        print("ori:", ori_count, "ret:",
              retweet, "extracted", extracted)
        print("Processing time: %s s"%(endt - start) )
        return (ori_count - 1, extracted, retweet)

    def filter_dirs(self, dir_path, out_dir, ignore_geo=True, language='en', ret=True, keywords=False):
        '''process all json files in directroy'''
        path_list = find_suffix('json', dir_path)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        ori_count = 0
        extracted = 0
        retweet = 0
        for file_path in path_list:
            (ori, ext, ret) = self.filter_tweets(
                file_path, out_dir, ignore_geo, language, ret, keywords)
            ori_count += ori
            extracted += ext
            retweet += ret
        print(dir_path, "ori:", ori_count, "ret:",
              retweet, "extracted", extracted)


class text_pro(object):
    """procoss textual data"""

    def __init__(self):
        super(text_pro, self).__init__()
        self.stop_rule = ""
        stop_words = stopwords.words('english')
        for x in stop_words:
            new_words = "(?:^|\W)" + x + "(?:$|\W)|"
            self.stop_rule += new_words  # stop words exclusion
        self.stop_rule = self.stop_rule[:-1]

    def regularize(text):
        '''return the regularized text'''
        r1 = u'^RT @.*?: |@+[^\s]*|^RT\s'  # exclude RT @... \n
        # http(s)
        r2 = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        r3 = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'  # e-mail
        r4 = '\s+'  # multiple empty chars
        r5 = 'http[s]?:.*? '
        r6 = '[^A-Za-z0-9_]'  # not alphabet number and _
        sub_rule = r1 + '|' + r2 + '|' + r3 + '|' + r5
        text = html.unescape(text)
        text = re.sub(sub_rule, "", text)
        text = emoji.demojize(text, delimiters=('emo_', ' '))
        text = re.sub(r6, ' ', text)
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
        label = input("Relevant to ful? ")
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

    def tokenize(text):
        '''tokenize a text'''
        return word_tokenize(text)



def text_only(in_file, out_file):
    '''save extracted text to out_file'''
    with open(in_file, 'r', encoding='utf-8') as in_f, open(out_file, 'a', encoding='utf-8') as out_f:
        for line in in_f:
            if line != "\n":
                try:
                    data = json.loads(line)
                except IOError:
                    print("Can't read ", in_file)
                else:
                    text = text_pro.regularize(data['text'])
                    out_text = (text + "\n")
                    #out_text = data['text']
                    out_f.write(out_text)


def text_only_tree(in_dir, out_dir):
    if not os.path.exists(in_dir):
        print('no such file')
        exit(0)
    else:
        path_list = find_suffix("json", in_dir)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        for in_file in path_list:
            file_name = in_file.split('_')[-1]
            out_file = os.path.join(out_dir, file_name[:2] + ".txt")
            text_only(in_file, out_file)

   


class CDC_preprocessor(object):
    """preprocess CDC data"""

    def __init__(self):
        super(CDC_preprocessor, self).__init__()

    def get_information(self, path, out_path=None, season="2018-19"):
        try:
            data = pd.read_csv(path)
        except IOError:
            print("Can't read ", path)
        else:
            columns = ['STATENAME', 'ACTIVITY LEVEL', 'ACTIVITY LEVEL LABEL',
                       'WEEKEND', 'WEEK', 'SEASON']
            data = data[columns]
            data = data[data['SEASON'] == season].reset_index(drop=True)
            data.to_csv(os.path.join(out_path, season + ".csv"), index=0)

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
                for line in f:
                    data = json.loads(line, encoding='utf-8')
                    for text, value in data.items():
                        if text not in record:
                            record[text] = value
                        else:
                            record[text] += value
        rank_result = sorted(record.items(), lambda x: x[1], reverse=True)
        return rank_result
        
input_path = "/Volumes/Data/Twitter/2018/01/01"
out_dir = './Data/topic'
acv = archive()
acv.filter_dirs(input_path, out_dir)

# data_count(input_path,func = 'line')

# unzip_tree("/Volumes/NonBee5/Twitter/2018")

'''save preprocessed cdc'''
# cdc_in_path = '/Users/NonBee/Desktop/FYP/code/Data/cdc/StateDatabySeason59_58,57.csv'
# cdc_out_path = './Data/dataset/'
# cdc = CDC_preprocessor()
# cdc.get_information(cdc_in_path, cdc_out_path)

'''unzip files recursively'''
# tar_path = "/Volumes/NonBee5/Twitter/2018/01/"
# unzip_tree(tar_path, "bz2", tar_path)

'''labeling'''
# data_path = './Data/dataset/2018/10/2018_10_05.json'
# out_name = './Data/twitter.json'
# text_pro.label_file(data_path, out_name)

# input_file = './Data/dataset/2018/01/2018_01_12.json'
# out_dir = 'testData'

# text_only(input_file, out_file)
# text_only_tree('./Data/dataset/2018/01', out_dir)
