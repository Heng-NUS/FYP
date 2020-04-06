# config.py

class SUconfig(object):
    '''config of supervised model'''
    embed_size = 100
    num_channels = 100
    kernel_size = [3,4,5]
    output_size = 2
    max_epochs = 15
    lr = 0.01
    batch_size = 64
    max_sen_len = 15
    dropout_keep = 0.5

class UNconfig(object):
    '''config of unsupervised model'''
    embedding_size = 100
    sequence_length = 15
    topics = 16
    filter_size = 2  # n-gram window
    stride = 1
    padding = 0
    dilation = 1