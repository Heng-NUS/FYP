import json
import logging
from collections import defaultdict
from pprint import pprint

import numpy as np
import tqdm
from gensim import corpora, matutils, models, similarities
from gensim.utils import deaccent, simple_preprocess
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from smart_open import open

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class Topic_mode(object):
    pass
