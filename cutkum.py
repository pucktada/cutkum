#!/usr/bin/env python3
#
# Pucktada Treeratpituk (https://pucktada.github.io/)
# License: MIT
# 2017-05-01
#
# A command-line function to load a trained model and compute the word segmentation

import math
from os import listdir
from os.path import isfile, isdir, join
import sys
import configargparse
import logging
import tensorflow as tf
import numpy as np
from char_dictionary import CharDictionary
from ck_model import CkModel

# shape(prob_matrix) = [#times, #classes]
def viterbi(prob_matrix):
    """ find the most likely sequence of labels using the viterbi algorithm on prob_matrix """
    TINY          = 1e-6    # to avoid NaNs in logs

    # if prob_matrix is 1D, make it 2D
    if len(np.shape(prob_matrix)) == 1:
        prob_matrix = [prob_matrix]
        
    length = len(prob_matrix)

    probs  = np.zeros_like(prob_matrix)
    backpt = np.ones_like(prob_matrix, dtype=np.int32) * -1
    
    for i in [0,1,2,3,4]:
        probs[0][i] = np.log(prob_matrix[0][i]+TINY)
        
    # {B, M, E, S} <=== 0:begin, 1:middle, 2:end, 3:single
    for t in range(1, length):
        # E, S -> B | B, M -> M | B, M -> E | E, S -> S
        previous_of = [[0,0], [3,4], [1,2], [1,2], [3,4]]
        for i in range(5):
            prevs = previous_of[i]
            max_id = prevs[np.argmax([probs[t-1][prevs[0]], probs[t-1][prevs[1]]])]
            backpt[t][i] = max_id
            probs[t][i]  = np.log(prob_matrix[t][i]+TINY) + probs[t-1][max_id]

    seq = np.ones(length, 'int32') * -1
    #print(probs[length-1])
    seq[length-1] = np.argmax(probs[length-1])
    #print(seq[length-1])
    max_prob = probs[length-1][seq[length-1]]
    for t in range(1, length):
        seq[length-1-t] = backpt[length-t][seq[length-t]]
    
    return seq

def process_sentence(sess, model_settings, model_vars, one_hot_by_t):
    """ run the inference to segment the given 'sentence' represented by the one_hot vector by time
    """
    states = np.zeros((model_settings['num_layers'], 2, 1, model_settings['lstm_size']))
    
    feed = { model_vars['inputs']:  one_hot_by_t, model_vars['init_state']: states }
    probs = sess.run([model_vars['probs']], feed_dict=feed)

    return np.squeeze(probs)

def process_input_sentence(sess, char_dict, model_settings, model_vars, sentence):
    """ run the inference to segment the given 'sentence' into words seperated by '|' 
    """
    chars = list(sentence.strip())
    cids = char_dict.chars2cids(chars)
    
    in_embedding = np.eye(model_settings['input_classes'])
    one_hot      = in_embedding[cids]
    one_hot_by_t = np.expand_dims(one_hot, 1)

    probs = process_sentence(sess, model_settings, model_vars, one_hot_by_t)
    labels = viterbi(probs)
    words   = char_dict.chars2words(chars, labels)
    print('|'.join(words))

def process_input_file(sess, char_dict, model_settings, model_vars, input_file):
    """ read the input_file line by line, and run the inference to segment each line 
        into words seperated by '|'
    """
    
    with open(input_file, 'r') as f:
        for s in f: # s is the line string
            if s and (len(s) > 0):
                chars = list(s.strip())
                cids = char_dict.chars2cids(chars)
                
                in_embedding = np.eye(model_settings['input_classes'])
                one_hot      = in_embedding[cids]
                one_hot_by_t = np.expand_dims(one_hot, 1)

                # run session to retriev prob
                probs = process_sentence(sess, model_settings, model_vars, one_hot_by_t)                
                labels = viterbi(probs)
                words   = char_dict.chars2words(chars, labels)
                print('|'.join(words))

def load_model(sess, meta_file, checkpoint_file):
    """ loading necessary configuration of the network from the meta file & 
        the checkpoint file together with variables that are needed for the inferences
    """
    saver = tf.train.import_meta_graph(meta_file)
    saver.restore(sess, checkpoint_file)
    
    configs = tf.get_collection('configs')
    pvars   = tf.get_collection('placeholders')
    
    model_settings = dict()
    for c in configs:
        name = c.name.split(':')[0]
        model_settings[name] = sess.run(c)
        
    model_vars = dict()
    for p in pvars:
        name = p.name.split(':')[0]
        model_vars[name] = p
    model_vars['probs'] = tf.get_collection('probs')[0]
    
    return model_settings, model_vars
    
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
        
        if input_file is not None:
            process_input_file(sess, char_dict, model_settings, model_vars, input_file)        
        elif input_sentence is not None:
            process_input_sentence(sess, char_dict, model_settings, model_vars, input_sentence)

      