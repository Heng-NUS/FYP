from utils import *
import json
from collections import defaultdict
from pprint import pprint

import numpy as np
import tqdm
from gensim import corpora, matutils, models, similarities
from gensim.utils import deaccent, simple_preprocess
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from smart_open import open

class Topic_mode(object):
    pass
