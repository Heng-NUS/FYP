# config.py

class SUconfig(object):
    '''config of supervised model'''
    num_channels = 100
    kernel_size = [3,4,5]
    output_size = 2
    lr = 0.01
    batch_size = 256
    max_sen_len = 15
    dropout_keep = 0.5

class UNconfig(object):
    '''config of unsupervised model'''
    embedding_size = 300
    hidden = 800
    min_count = 5
    max_percent = 0.4
    batch_size = 4096
    dropout_keep = 0.5
    act = 'tanh'
    max_epoch = 100
