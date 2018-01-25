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
from cutkum.char_dictionary import CharDictionary
from cutkum.ck_model import CkModel, load_graph, load_settings
from cutkum import util #viterbi, load_validation_set

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

    prob_list = []
    for i in range(len(seq_lengths)): # for each line
        if (i % 10) == 0:
            logging.info('sen: %d', i)
        if (seq_lengths[i] != 0):
            line_one_hot_by_t = one_hot_by_t[0:seq_lengths[i],i:(i+1),:]
            feed = { model_vars['inputs']: line_one_hot_by_t, 
                model_vars['seq_lengths']: [seq_lengths[i]], 
                model_vars['keep_prob']: 1.0
            }
            probs = sess.run(model_vars['probs'], feed_dict=feed)
            prob_list.append(probs)
        else:
            prob_list.append([])

    return prob_list
    
def process_input_sentence(sess, char_dict, model_settings, model_vars, sentence, max_flag):
    """ run the inference to segment the given 'sentence' into words seperated by '|' 
    """

    if (sys.version_info > (3, 0)):
        uni_s = u"%s" % sentence # only for 2.7 (for Python 3, no need to decode)
    else:
        uni_s = u"%s" % sentence.decode('utf-8') # only for 2.7 (for Python 3, no need to decode)

    chars = list(uni_s.strip())
    cids  = char_dict.chars2cids(chars)
        
    in_embedding = np.eye(model_settings['input_classes'])
    one_hot      = in_embedding[cids]
    one_hot_by_t = np.expand_dims(one_hot, 1)

    p  = process_sentence(sess, model_settings, model_vars, one_hot_by_t)
    if (max_flag == 'max'):
        labels = util.max_prob(p)
    else: # viterbi
        labels = util.viterbi(p)

    words  = char_dict.chars2words(chars, labels)
    print('|'.join(words))

def process_input_file(sess, char_dict, model_settings, model_vars, input_file, max_flag):
    """ run the inference to segment the given file into words seperated by '|' 
    """

    # load file into matrix
    one_hot_by_t, seq_lengths, chars_mat = util.load_files_into_matrix([input_file])

    probs = process_batch(sess, model_settings, model_vars, one_hot_by_t, seq_lengths)

    n_examples= len(probs)
    for i in range(n_examples):
        if (seq_lengths[i] != 0):
            p = probs[i][0:seq_lengths[i], 0, :]
            #p = probs[0:seq_lengths[i], i, :]

            if (max_flag == 'max'):
                labels = util.max_prob(p)
            else: # viterbi
                labels = util.viterbi(p)

            words  = char_dict.chars2words(chars_mat[i], labels)
            line_out = '|'.join(words)
            if (sys.version_info <= (3, 0)):
                line_out = line_out.encode('utf8')
            print(line_out)
        else:
            print('')

def process_input_directory(sess, char_dict, model_settings, model_vars, input_dir, output_dir, max_flag):
    """ run the inference to segment each file in the given directory into words seperated by '|' 
    """

    input_files = listdir(input_dir)

    for f in input_files:
        infile_path  = join(input_dir, f)
        outfile_path = join(output_dir, f)

        if not isfile(outfile_path):
            outfile = open(outfile_path, 'w')
            one_hot_by_t, seq_lengths, chars_mat = util.load_files_into_matrix([infile_path])
            probs = process_batch(sess, model_settings, model_vars, one_hot_by_t, seq_lengths)

            n_examples= len(probs)
            for i in range(n_examples):
                if (seq_lengths[i] != 0):
                    p = probs[i][0:seq_lengths[i], 0, :]
                    #p = probs[0:seq_lengths[i], i, :]

                    if (max_flag == 'max'):
                        labels = util.max_prob(p)
                    else: # viterbi
                        labels = util.viterbi(p)

                    words  = char_dict.chars2words(chars_mat[i], labels)

                    line_out = '|'.join(words)
                    if (sys.version_info <= (3, 0)):
                       line_out = line_out.encode('utf8')
                    outfile.write(line_out)
                    outfile.write('\n')
                else:
                    outfile.write('\n')
            outfile.close()
            logging.info('%s ... DONE', f)

if __name__ == '__main__':

    p = argparse.ArgumentParser()
    p.add_argument('-v', '--verbose', help='verbose', action='store_true')
    p.add_argument('-m', '--model_file', required=True, help='model file to load')
    g1 = p.add_mutually_exclusive_group(required=True)
    g1.add_argument('-d', '--directory', help='input directory')
    g1.add_argument('-i', '--input_file', help='input file')
    g1.add_argument('-s', '--sentence', help='sentence to parse')
    p.add_argument('-o', '--output_dir', help='output directory if --directory is given')

    g2 = p.add_mutually_exclusive_group(required=False)
    g2.add_argument('--max', action='store_const', dest='max_flag', const='max', help='output word boundary using maximum probabilities (default)')
    g2.add_argument('--viterbi', action='store_const', dest='max_flag', const='vit', help='output word boundary using viterbi')
    p.set_defaults(max_flag='max')

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
    input_dir       = opts['directory']
    output_dir      = opts['output_dir']
    max_flag        = opts['max_flag']

    if (input_dir is not None) & (output_dir is None):
        p.error('--directory and --output_dir must be given together')

    graph = load_graph(model_file)

    with tf.Session(graph=graph) as sess:
        char_dict = CharDictionary()
        model_settings, model_vars = load_settings(sess)
        
        if input_file is not None:
            process_input_file(sess, char_dict, model_settings, model_vars, input_file, max_flag)
        elif (input_dir is not None) & (output_dir is not None):
            process_input_directory(sess, char_dict, model_settings, model_vars, input_dir, output_dir, max_flag)
        elif input_sentence is not None:
            process_input_sentence(sess, char_dict, model_settings, model_vars, input_sentence, max_flag)

      