from utility import *
import pandas as pd


class Archive(object):
    """preprocess archive twitter data"""

    def __init__(self, word_list):
        super(Archive, self).__init__()
        self.include_list = word_list['include_words']
        self.seclude_list = word_list['exclude_words']
        self.phrases = word_list['phrases']
        self.in_rule = ""
        self.ex_rule = ""
        self.stop_rule = ""
        self.out_dir = os.path.join(os.getcwd(), '/Data/dataset/')
        self.hashtag_re = re.compile("(?:^|\s)[＃#]{1}(\w+)", re.UNICODE)

        for x in word_list['include_words']:
            new_words = x + "|"
            self.in_rule += new_words  # loose inclusion rule
        self.in_rule = self.in_rule[:-1]

        for x in word_list['exclude_words']:
            new_words = "(?:^|\W)" + x + "(?:$|\W)|"  # strict exclusion rule
            self.ex_rule += new_words
        self.ex_rule = self.ex_rule[:-1]

        # stop_words = stopwords.words('english')
        # for x in stop_words:
        #     new_words = "(?:^|\W)" + x + "(?:$|\W)|"
        #     self.stop_rule += new_words  # stop words exclusion
        # self.stop_rule = self.stop_rule[:-1]

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

    def regularize(self, text):
        '''return the regularized text'''

        return text_pro.regularize(text)

    def keywords_search(self, text):
        temp = text.split()
        Found = False
        for w in self.include_list:
            if w.lower() in temp:
                Found = True
                break
        if not Found:
            for phrase in self.phrases:
                if phrase.lower() in text:
                    Found = True
                    break
        return Found

    def keywords_flitering(self, path, out_dir):
        '''select out all useful tweets from a single json file and save them to a file'''
        print("Keywords flitering:", path)
        start = time.clock()
        name = path.split('/')[-1]

        ori_count = 0  # record the original number
        extracted = 0  # record the extracted number

        if name[0] != ".":
            with open(path, 'r', encoding='utf-8') as file:
                max_buffer_size = 5000
                out_buffer = {}
                size_count = 0

                for f in file:
                    data = {}
                    try:
                        data = json.loads(f)
                    except:
                        continue
                    ori_count += 1

                    if 'text' in data:
                        text = data['text']
                        date = data['created_at']

                        temp = text.split()
                        if len(temp) < 3:
                            continue
                        # exclude text that has less than three words
                        # if keywords and not re.search(self.in_rule, text, flags=re.IGNORECASE):
                        #     continue
                        if not self.keywords_search(text):
                            continue
                        # if not in inclusion list or in phrases list, skip it

                        if date not in out_buffer:
                            out_buffer[date] = []
                        out_buffer[date].append(data)
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
                # save retweets if needed
        endt = time.clock()

        print("ori:", ori_count, "extracted:", extracted)
        print("Processing time: %s s" % (endt - start))
        return (ori_count - 1, extracted)

    def keywords_dirs(self, dir_path, out_dir):
        '''process all json files in directroy'''
        path_list = find_suffix('json', dir_path)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        ori_count = 0
        extracted = 0
        start_time = time.clock()
        for file_path in path_list:
            (ori, ext) = self.keywords_flitering(file_path, out_dir)
            ori_count += ori
            extracted += ext
        end_time = time.clock()

        log = dir_path + " ori: " + \
            str(ori_count) + " extracted: " + str(extracted) + '\n'
        with open(os.path.join(out_dir, 'logs.txt'), 'a', encoding='utf-8') as f:
            f.write(log)
        print(log)
        print('Processing time:', end_time-start_time)

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

                lang_count = 0

                for f in file:
                    data = {}
                    try:
                        data = json.loads(f)
                    except:
                        continue
                    ori_count += 1
                    if 'lang' not in data or data['lang'] != language:
                        continue
                    else:
                        lang_count += 1

                    if 'text' in data and (ignore_geo or data['user']['geo_enabled']):
                        text = data['text']
                        hashtags = self.hashtag_re.findall(text)

                        time_split = data['created_at'].split()
                        date = time_split[-1] + '/' + \
                            months[time_split[1]] + '/' + time_split[2]
                        # creation date

                        if "RT @" in text:
                            retweet += 1
                            if ret:
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
                            # save retweets to a single file if needed

                        else:  # not retweet
                            text = self.regularize(text)
                            # regularize text
                            temp = text.split()
                            if len(temp) < 3:
                                continue
                            # exclude text that has less than three words
                            # if keywords and not re.search(self.in_rule, text, flags=re.IGNORECASE):
                            #     continue
                            if keywords and not self.keywords_search(text):
                                continue
                            # if not in inclusion list or in phrases list, skip it

                            location = data['user']['location']
                            coordinates = data['coordinates']
                            place = data['place']
                            created_at = date
                            selected_data = dict(created_at=created_at, text=text,
                                                 location=location, coordinates=coordinates, place=place, hashtags=hashtags)
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
              retweet, "extracted:", extracted, "language:", lang_count)
        print("Processing time: %s s" % (endt - start))
        return (ori_count - 1, extracted, retweet, lang_count)

    def filter_dirs(self, dir_path, out_dir, ignore_geo=True, language='en', ret=False, keywords=False):
        '''process all json files in directroy'''
        path_list = find_suffix('json', dir_path)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        ori_count = 0
        extracted = 0
        retweet = 0
        lang_count = 0
        start_time = time.clock()
        for file_path in path_list:
            (ori, ext, ret, lang) = self.filter_tweets(
                file_path, out_dir, ignore_geo=ignore_geo, language=language, ret=ret, keywords=keywords)
            ori_count += ori
            extracted += ext
            retweet += ret
            lang_count += lang
        end_time = time.clock()

        log = dir_path + " ori: " + str(ori_count) + " ret: " + str(
            retweet) + " language: " + str(lang_count) + " extracted: " + str(extracted) + '\n'
        with open('logs.txt', 'a', encoding='utf-8') as f:
            f.write(log)
        print(log)
        print('Processing time:', end_time-start_time)


class Other_dataset(object):
    '''extract covid-19 data'''

    def extract_covid(self, in_file, out_file):
        data = pd.read_csv(in_file)
        for index, row in tweets.iterrows():
            if row["Tweet Language"] != ["English"]:
                continue
                pass
            
    def extract_news_dir(self, in_dir, out_dir, suffix='txt'):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if os.path.isfile(in_dir):
            file_name = os.path.basename(in_dir)
            out_file = os.path.join(out_dir, file_name)
            self.extract_news(file, out_file)
        elif os.path.isdir(in_dir):
            file_list = find_suffix(suffix, in_dir)
            for file in file_list:
                file_name = os.path.basename(file)
                out_file = os.path.join(out_dir, file_name)
                self.extract_news(file, out_file)

    def extract_news(self, in_file, out_file):
        '''extract text from healthcare tweets'''
        print("Processing", in_file)
        with open(in_file, 'r') as in_f, open(out_file, 'a', encoding="utf-8") as out_f:
            for line in in_f:
                if len(line) > 1:
                    text = line.split("|")[-1]
                    text = text_pro.regularize(text)
                    out_f.write(text + '\n')

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
