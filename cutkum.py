#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Pucktada Treeratpituk (https://pucktada.github.io/)
# License: MIT
# 2017-05-01
#
# A command-line function to load a trained model and compute the word segmentation
from __future__ import unicode_literals
from os import listdir
from os.path import isfile, isdir, join
import argparse
import logging
import math
import numpy as np
import re
import sys
import tensorflow as tf
from ck_model.char_dictionary import CharDictionary
from ck_model.ck_model import CkModel, load_graph, load_settings
from ck_model import util #viterbi, load_validation_set

def process_sentence(sess, model_settings, model_vars, one_hot_by_t):
    """ run the inference to segment the given 'sentence' represented by the one_hot vector by time
    """
    feed = { model_vars['inputs']:  one_hot_by_t, 
        model_vars['seq_lengths']: [one_hot_by_t.shape[0]], 
        model_vars['keep_prob']: 1.0
    }
    probs = sess.run([model_vars['probs']], feed_dict=feed)

    return np.squeeze(probs)

def process_batch(sess, model_settings, model_vars, one_hot_by_t, seq_lengths):

    feed = { model_vars['inputs']: one_hot_by_t, 
        model_vars['seq_lengths']: seq_lengths, 
        model_vars['keep_prob']: 1.0
    }
    probs = sess.run(model_vars['probs'], feed_dict=feed)
    return probs
    
def process_input_sentence(sess, char_dict, model_settings, model_vars, sentence):
    """ run the inference to segment the given 'sentence' into words seperated by '|' 
    """
    
    uni_s = u"%s" % sentence.decode('utf-8') # only for 2.7 (for Python 3, no need to decode)
    #uni_s = u"%s" % sentence # only for 2.7 (for Python 3, no need to decode)

    chars = list(uni_s.strip())
    cids  = char_dict.chars2cids(chars)
        
    in_embedding = np.eye(model_settings['input_classes'])
    one_hot      = in_embedding[cids]
    one_hot_by_t = np.expand_dims(one_hot, 1)

    probs  = process_sentence(sess, model_settings, model_vars, one_hot_by_t)
    #labels = util.viterbi(probs)
    labels = util.max_prob(probs)
    words  = char_dict.chars2words(chars, labels)
    print('|'.join(words))

def process_input_file(sess, char_dict, model_settings, model_vars, input_file):
    one_hot_by_t, seq_lengths, chars_mat = util.load_files_into_matrix([input_file])
    probs = process_batch(sess, model_settings, model_vars, one_hot_by_t, seq_lengths)
    
    _, n_examples, _ = probs.shape
    for i in range(n_examples):
        if (seq_lengths[i] != 0):
            p = probs[0:seq_lengths[i], i, :]
            #labels = util.viterbi(p)
            labels = util.max_prob(p)
            words  = char_dict.chars2words(chars_mat[i], labels)
            print('|'.join(words))
        else:
            print('')
    
if __name__ == '__main__':
    
    p = argparse.ArgumentParser()
    p.add_argument('-v', '--verbose', help='verbose', action='store_true')
    p.add_argument('-m', '--model_file', required=True, help='model file to load')
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('-i', '--input_file', help='input file')
    group.add_argument('-s', '--sentence', help='sentence to parse')

    opts = vars(p.parse_known_args()[0])

    verbose = opts['verbose']
    if (verbose):
        log_level = logging.INFO
    else:
        log_level = logging.WARNING    
    logging.basicConfig(format='%(levelname)s:%(message)s', level=log_level)
          
    # OTHERS ARGS
    model_file      = opts['model_file']
    input_file      = opts['input_file']
    input_sentence  = opts['sentence']  
    
    graph = load_graph(model_file)

    with tf.Session(graph=graph) as sess:
        char_dict = CharDictionary()
        #model_settings, model_vars = load_model2(sess, meta_file, checkpoint_file)
        model_settings, model_vars = load_settings(sess)
        
        if input_file is not None:
            process_input_file(sess, char_dict, model_settings, model_vars, input_file)        
        elif input_sentence is not None:
            process_input_sentence(sess, char_dict, model_settings, model_vars, input_sentence)

      