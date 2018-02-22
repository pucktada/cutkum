#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Pucktada Treeratpituk (https://pucktada.github.io/)
# License: MIT
# 2017-05-21
from __future__ import unicode_literals
from os import listdir
from os.path import isfile, isdir, join
import re
import tempfile
import numpy as np
import tensorflow as tf

from char_dictionary import CharDictionary
from sequence_handler import save_tfrecord, read_and_decode_single_example, get_num_records

# 100 - 1000+
# [71826 43962 14179 12159  3280  1916   950   412   168    67    76]
# 40 - 400+
# [33404 25192 26999 19487 10706  6448  5108  5375  5738  3669  6869]
# total = 132081 sentences
def lines_from_file(filepath):
    '''
        read in a file as a list of words, ex:
        [['กฎหมาย', 'กับ', 'การ', 'เบียดบัง', 'คน', 'จน'], 
         ['จาก', 'ต้นฉบับ', 'เรื่อง', 'คน', 'จน', 'ภาย', 'ใต้', 'กฎหมาย'], 
         ['ไพสิฐ พาณิชย์กุล']]
    '''
    lines = []
    with open(filepath, 'r') as f:
        for s in f: # s is the line string
            if s and (len(s) > 0):
                tokens = re.split('[|]', s.strip())[:-1]
                words = [] # words in a line
                for t in tokens:
                    t = re.sub('<[^>]*>', '', t)
                    words += [t]
                lines += [words]
    return lines

def convert_data_to_tfrecord(dirpaths, outdir, num_look_ahead=6):
    '''
       take in the list of directory path, convert them into tf file 
       of sequence of int64 of cids, and labels
    '''
    files = []
    for d in dirpaths:
        files += [(f.replace('.txt', '.tf'), join(d, f)) 
            for f in listdir(d) if isfile(join(d, f))]
    
    char_dict = CharDictionary() # character dictionary 
    
    for fname, fpath in files:
        lines = lines_from_file(fpath)
        
        sequences = []
        label_sequences = []
        for words in lines: # for each line 
            cids, labels = char_dict.words2cids(words)

            # 11/02/2018 pad sequence (for look ahead)
            # pad space=' ' at the end of each line, and shift labels by the same amount
            cids   = cids + char_dict.padding_cids(num_look_ahead)
            labels = char_dict.padding_labels(num_look_ahead) + labels
            
            sequences += [cids]
            label_sequences += [labels]
        
        tf_filename = join(outdir, fname)
        writer = tf.python_io.TFRecordWriter(tf_filename)
        save_tfrecord(writer, sequences, label_sequences)
        writer.close()

def read_tfrecord_length_distribution(tf_filename):
    num_records = get_num_records(tf_filename)

    buckets = np.zeros(11, dtype=np.int32)
    
    graph = tf.Graph()
    with graph.as_default():
        key, context, features = read_and_decode_single_example([tf_filename], shuffle=True, num_epochs=1)
        cids   = features['source']
        labels = features['target']
    
    with tf.Session(graph=graph) as sess:
        init_op = tf.group(tf.global_variables_initializer(), 
                           tf.local_variables_initializer())
        sess.run(init_op)
    
        coord = tf.train.Coordinator()    
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(num_records):
            _cids, _labels = sess.run([cids, labels])
            length = len(_cids)
            if length >= 400:
                bid = 10
            else:
                bid = length // 40
            
            buckets[bid] = buckets[bid] + 1
            
        # stop our queue threads and properly close the session
        coord.request_stop()
        coord.join(threads)
    return buckets
        
def read_tfrecord(tf_filename):
    '''
       read in the 'tf_filename' and use CharDictionary to convert
       the sequeneces of cids, labels back to list of words
    '''
    char_dict = CharDictionary()
    
    graph = tf.Graph()
    with graph.as_default():
        key, context, features = read_and_decode_single_example([tf_filename], shuffle=True)
        cids   = features['source']
        labels = features['target']
    
    with tf.Session(graph=graph) as sess:
        init_op = tf.group(tf.global_variables_initializer(), 
                           tf.local_variables_initializer())
        sess.run(init_op)
    
        coord = tf.train.Coordinator()    
        threads = tf.train.start_queue_runners(coord=coord)

        print('\nread back the file...')
        for i in range(100):
            _cids, _labels = sess.run([cids, labels])
            _chars = char_dict.cids2chars(_cids)
            words = char_dict.chars2words(_chars, _labels)
            print(i, words[:10])
            
        # stop our queue threads and properly close the session
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    #dirpaths = ['data/5M.BEST2010/article', 
    #            'data/5M.BEST2010/encyclopedia',
    #            'data/5M.BEST2010/news', 
    #            'data/5M.BEST2010/novel']
    
    convert_data_to_tfrecord(['data/test_txt'], 'data/all_s2_c4', num_look_ahead=2)

    #dirname = 'data'
    #files = [join(dirname, f) for f in listdir(dirname) if isfile(join(dirname, f))]
    #files = [join(dirname, f) for f in listdir(dirpaths) if isfile(join(dirname, f))]

    #print(files)
    #buckets = np.zeros(11, dtype=np.int32)
    #for fname in files:
    #    print(fname)
        #b = read_tfrecord_length_distribution(fname)
    #    buckets += b
    #print(buckets)