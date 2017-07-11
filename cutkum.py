#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Pucktada Treeratpituk (https://pucktada.github.io/)
# License: MIT
# 2017-05-01
#
# A command-line function to load a trained model and compute the word segmentation
from __future__ import unicode_literals
import math
from os import listdir
from os.path import isfile, isdir, join
import sys
import re
import configargparse
import logging
import tensorflow as tf
import numpy as np
from ck_model.char_dictionary import CharDictionary
from ck_model.ck_model import CkModel, load_model
from ck_model import util #viterbi, load_validation_set

def process_sentence(sess, model_settings, model_vars, one_hot_by_t):
    """ run the inference to segment the given 'sentence' represented by the one_hot vector by time
    """
    #fw_states = np.zeros((model_settings['num_layers'], 2, 1, model_settings['cell_size']))
    #bw_states = np.zeros((model_settings['num_layers'], 2, 1, model_settings['cell_size']))
    feed = { model_vars['inputs']:  one_hot_by_t, 
        model_vars['seq_lengths']: [one_hot_by_t.shape[0]], 
        #model_vars['fw_state']: fw_states, 
        #model_vars['bw_state']: bw_states, 
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
    chars = list(uni_s.strip())
    cids  = char_dict.chars2cids(chars)
        
    in_embedding = np.eye(model_settings['input_classes'])
    one_hot      = in_embedding[cids]
    one_hot_by_t = np.expand_dims(one_hot, 1)

    probs = process_sentence(sess, model_settings, model_vars, one_hot_by_t)
    labels = util.viterbi(probs)
    words   = char_dict.chars2words(chars, labels)
    print('|'.join(words))

def process_input_file(sess, char_dict, model_settings, model_vars, input_file):
    one_hot_by_t, seq_lengths, chars_mat = util.load_files_into_matrix([input_file])
    probs = process_batch(sess, model_settings, model_vars, one_hot_by_t, seq_lengths)
    
    _, n_examples, _ = probs.shape
    for i in range(n_examples):
        if (seq_lengths[i] != 0):
            p = probs[0:seq_lengths[i], i, :]
            labels = util.viterbi(p)
            words  = char_dict.chars2words(chars_mat[i], labels)
            print('|'.join(words))
        else:
            print('')


def process_input_file_prev(sess, char_dict, model_settings, model_vars, input_file):
    """ read the input_file line by line, and run the inference to segment each line 
        into words seperated by '|'
    """
    
    with open(input_file, 'r') as f:
        for s in f: # s is the line string
            if s and (len(s) > 0):
                uni_s = u"%s" % s.decode('utf-8') # only for 2.7 (for Python 3, no need to decode)
                chars = list(uni_s.strip())
                cids = char_dict.chars2cids(chars)
                
                in_embedding = np.eye(model_settings['input_classes'])
                one_hot      = in_embedding[cids]
                one_hot_by_t = np.expand_dims(one_hot, 1)

                # run session to retriev prob
                probs = process_sentence(sess, model_settings, model_vars, one_hot_by_t)                
                labels = viterbi(probs)
                words   = char_dict.chars2words(chars, labels)
                print('|'.join(words))
    
if __name__ == '__main__':
    
    p = configargparse.getArgParser()
    p.add('-v', '--verbose', help='verbose', action='store_true')
    p.add('-m', '--meta_file', required=True, help='meta file')
    p.add('-c', '--checkpoint_file', required=True, help='checkpoint file')
    group = p.add_mutually_exclusive_group(required=True)
    group.add('-i', '--input_file', help='input file')
    group.add('-s', '--sentence', help='sentence to parse')

    opts = vars(p.parse_known_args()[0])

    verbose = opts['verbose']
    if (verbose):
        log_level = logging.INFO
    else:
        log_level = logging.WARNING    
    logging.basicConfig(format='%(levelname)s:%(message)s', level=log_level)
          
    # OTHERS ARGS
    checkpoint_file = opts['checkpoint_file']
    meta_file       = opts['meta_file']

    input_file      = opts['input_file']
    input_sentence  = opts['sentence']    

    with tf.Session() as sess:
        char_dict = CharDictionary()
        model_settings, model_vars = load_model(sess, meta_file, checkpoint_file)
        
        #if input_file is not None:
        #    process_input_file(sess, char_dict, model_settings, model_vars, input_file)        
        #elif input_sentence is not None:
        #    process_input_sentence(sess, char_dict, model_settings, model_vars, input_sentence)

      