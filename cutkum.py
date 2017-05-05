#!/usr/local/bin/python

import math
from os import listdir
from os.path import isfile, isdir, join
import sys
import logging

import tensorflow as tf
import numpy as np

import configargparse

from char_dictionary import CharDictionary
from ck_model import CkModel

# shape(prob_matrix) = [#times, #classes]
def viterbi(prob_matrix):
    TINY          = 1e-6    # to avoid NaNs in logs
    
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

def process_sentence(sess, model, one_hot_by_t):
    num_rolls = math.ceil(one_hot_by_t.shape[0] / model.num_unroll)

    probs  = np.zeros(shape=[0, 1, model.label_classes])
    states = np.zeros((model.num_layers, 2, 1, model.lstm_size))
    
    for i in range(num_rolls):
        start_t = i * model.num_unroll 
        end_t   = min((i + 1) * model.num_unroll, one_hot_by_t.shape[0])
        
        feed = { model.inputs:  one_hot_by_t[start_t:end_t,:,:],
                 model.init_state: states }
        _p, _s = sess.run([model.probs, model.states], feed_dict=feed)
        states = _s
        probs  = np.concatenate((probs, _p), axis=0)

    return np.squeeze(probs) 

def process_input_sentence(graph, char_dict, model, model_file, sentence):
    # start a session...
    with tf.Session(graph=graph) as sess:

        saver = tf.train.Saver()
        if isfile(model_file + '.meta'):
            print('loading model:', model_file)
            saver.restore(sess, model_file)
        
        chars = list(sentence.strip())
        cids = char_dict.chars2cids(chars)
        
        in_embedding = np.eye(model.input_classes)
        one_hot      = in_embedding[cids]
        one_hot_by_t = np.expand_dims(one_hot, 1)

        probs = process_sentence(sess, model, one_hot_by_t)
        labels = viterbi(probs)
        words   = char_dict.chars2words(chars, labels)
        print('|'.join(words))

def process_input_file(graph, char_dict, model, model_file, input_file):
    # start a session...
    with tf.Session(graph=graph) as sess:

        saver = tf.train.Saver()
        if isfile(model_file + '.meta'):
            logging.info('loading model: %s', model_file)
            saver.restore(sess, model_file)
        
        with open(input_file, 'r') as f:
            for s in f: # s is the line string
                if s and (len(s) > 0):
                    chars = list(s.strip())
                    cids = char_dict.chars2cids(chars)
                    
                    in_embedding = np.eye(model.input_classes)
                    one_hot      = in_embedding[cids]
                    one_hot_by_t = np.expand_dims(one_hot, 1)

                    probs = process_sentence(sess, model, one_hot_by_t)
                    labels = viterbi(probs)
                    words   = char_dict.chars2words(chars, labels)
                    print('|'.join(words))

def build_ckmodel(model_settings):
    char_dict = CharDictionary()
    model_settings['input_classes'] = char_dict.num_char_classes() + 1
    model_settings['label_classes'] = char_dict.num_label_classes() + 1
    
    # build a model graph...
    graph = tf.Graph()
    with graph.as_default():
        model = CkModel(model_settings)
        model.build_graph()
    return (graph, char_dict, model)
                
if __name__ == '__main__':
    
    p = configargparse.getArgParser()
    p.add('-c', '--config', required=True, is_config_file=True, help='config file path')
    p.add('-v', '--verbose', help='verbose', action='store_true')
    p.add('-m', '--model_file',  required=True, help='model file')
    p.add('--lstm_size',  required=True, help='number of lstm states', type=int)
    p.add('--num_layers', required=True, help='number of layers', type=int)
    p.add('--num_unroll', required=True, help='number of unroll for rnn', type=int)
    p.add('--learning_rate', required=True, help='initial learning rate', type=float)
    group = p.add_mutually_exclusive_group(required=True)
    group.add('-i', '--input_file', help='input file')
    group.add('-s', '--sentence', help='sentence to parse')

    opts = vars(p.parse_known_args()[0])

    verbose = opts['verbose']    
    # MODEL ARGS
    model_settings = dict()
    model_settings['num_unroll'] = opts['num_unroll']
    model_settings['num_layers'] = opts['num_layers']
    model_settings['lstm_size']  = opts['lstm_size']
    model_settings['learning_rate'] = opts['learning_rate'] #Initial learning rate
    
    # OTHERS ARGS
    model_file = opts['model_file']
    input_file = opts['input_file']
    input_sentence = opts['sentence']
    
    if (verbose):
        log_level = logging.INFO
    else:
        log_level = logging.WARNING    
    logging.basicConfig(format='%(levelname)s:%(message)s', level=log_level)
    logging.info('settings: %s', model_settings)
    logging.info('model_file: %s', model_file)

    graph, char_dict, model = build_ckmodel(model_settings)
    if input_file is not None:
        process_input_file(graph, char_dict, model, model_file, input_file)        
    elif input_sentence is not None:
        process_input_sentence(graph, char_dict, model, model_file, input_sentence)

      