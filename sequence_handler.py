#!/usr/bin/env python3
#
# Pucktada Treeratpituk (https://pucktada.github.io/)
# License: MIT
# 2017-05-01
#
# A set of utility functions for reading and writing to TFRecord

import tensorflow as tf
import numpy as np
from os import listdir
from os.path import isfile, isdir, join

def get_num_records(filename):
    """ return the number of record within a given TFRecord file """
    return len([x for x in tf.python_io.tf_record_iterator(filename)])
    
# Write all examples into a TFRecords file
def save_tfrecord(writer, sources, targets):
    """ write the sources and targets sequences into a TFRecord file (given by writer) """
    for source, target in zip(sources, targets):
        ex = make_example(source, target)
        writer.write(ex.SerializeToString())

def make_example(source, target):
    """ create a SequenceExample out of a sequence of source and target """
    # The object we return
    ex = tf.train.SequenceExample()
    # A non-sequential feature of our example
    sequence_length = len(source)
    ex.context.feature["length"].int64_list.value.append(sequence_length)
    
    # Feature lists for the two sequential features of our example
    fl_source = ex.feature_lists.feature_list["source"]
    fl_target = ex.feature_lists.feature_list["target"]
    for src, tar in zip(source, target):
        #print(type(token))
        fl_source.feature.add().int64_list.value.append(src)
        fl_target.feature.add().int64_list.value.append(tar)
    return ex

def read_and_decode_single_example(filenames, shuffle=False, num_epochs=None):
    # first construct a queue containing a list of filenames.
    # this lets a user split up there dataset in multiple files to keep size down
    #filename_queue = tf.train.string_input_producer([filename], num_epochs=10)
    filename_queue = tf.train.string_input_producer(filenames, 
        shuffle=shuffle, num_epochs=num_epochs)
    
    reader = tf.TFRecordReader()
    # One can read a single serialized example from a filename
    # serialized_example is a Tensor of type string.
    key, serialized_ex = reader.read(filename_queue)
    context, sequences = tf.parse_single_sequence_example(serialized_ex,
            context_features = {
                "length": tf.FixedLenFeature([], dtype=tf.int64)
            },
            sequence_features={
                # We know the length of both fields. If not the
                # tf.VarLenFeature could be used
                "source": tf.FixedLenSequenceFeature([], dtype=tf.int64),
                "target": tf.FixedLenSequenceFeature([], dtype=tf.int64)
            })
    return (key, context, sequences)
        
def test_sequence_handler():
    print('testing sequence_handler')
    tmp_filename = 'tf.tmp'

    writer = tf.python_io.TFRecordWriter(tmp_filename)
    xlist = [[1, 2, 3], [1, 2], [3, 2, 1], [5,6,7,8,9,9,8], [7,6,6,7]]
    ylist = [[0, 1, 0], [2, 0], [2, 2, 2], [3,1,3,1,3,1,1], [4,1,4,1]]
    print(xlist)
    print(ylist)
    save_tfrecord(writer, xlist, ylist)
    writer.close()
    
    graph = tf.Graph()
    with graph.as_default():
        key, context, features = read_and_decode_single_example([tmp_filename])
        tokens = features['source']
        labels = features['target']
        
    with tf.Session(graph=graph) as sess:
        init_op = tf.group(tf.global_variables_initializer(), 
                           tf.local_variables_initializer())
        sess.run(init_op)
    
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        print('\nread back the file...')
        for i in range(len(xlist)):
            t, l = sess.run([tokens, labels])
            print('tokens:', t)
            print('labels:', l)
            print('')            

        # stop our queue threads and properly close the session
        coord.request_stop()
        coord.join(threads)
  
def count_sentence(dirname):
    files = [join(dirname, f) for f in listdir(dirname)]
    
    total = 0
    for f in files:
        num = get_num_records(f)
        total += num
        
    print('total', total)
        
if __name__ == '__main__':
    #test_sequence_handler()
    count_sentence('data/train')
    
    