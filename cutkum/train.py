#!/usr/bin/env python3
import math
from os.path import join
from tensorflow.python.lib.io import file_io
import sys
import re
from random import randint
import configargparse
import logging
import tensorflow as tf
import numpy as np
from char_dictionary import CharDictionary
from ck_model import CkModel
from sequence_handler import read_and_decode_single_example, get_num_records
import util as util

def get_number_of_records(dirpath):
    ''' get the number of records for all tf files inside a dirpath '''
    train_files = [join(dirpath, f) for f in file_io.list_directory(dirpath)]
    
    num_records = 0
    for filename in train_files:
        num_records += get_num_records(filename)
    return num_records

def validate(sess, model, char_dict, one_hot_by_t, seq_lengths, chars_mat, boundary_mat):
    #states = np.zeros((model.num_layers, 2, len(seq_lengths), model.cell_sizes[0]))
    batch_size = one_hot_by_t.shape[1] # one_hot_by_t:   [max_time, #examples, #dict]
    fw_state = model.init_fw_states(batch_size)

    feed = { model.inputs: one_hot_by_t, 
        model.seq_lengths: seq_lengths,  
        model.fw_state: fw_state,
        model.keep_prob: 1.0
    }

    probs  = sess.run(model.probs, feed_dict=feed)

    total_c = 0.0
    total_r = 0.0
    total_o = 0.0
    _, n_examples, _ = probs.shape

    for i in range(n_examples):
        p = probs[0:seq_lengths[i], i, :]

        #labels = util.max_prob(p)        
        labels = util.viterbi(p)

        padding = char_dict.padding_labels(model.look_ahead)
        labels = np.concatenate((labels[model.look_ahead:], padding))
        words  = char_dict.chars2words(chars_mat[i], labels)
        print('|'.join(words))
        x = boundary_mat[i]
        y = util.get_boundary_array(words)

        n_answers = util.word_count_from_boundary_array(x)
        n_words   = len(words)
        correct, n_refwords, n_outwords = util.count_correct_words(x, y, n_answers, n_words)

        total_c += correct
        total_r += n_refwords
        total_o += n_outwords

    r = (float)(total_c / total_r)
    p = (float)(total_c / total_o)
    if (r == 0) or (p == 0):
        f = 0
    else:
        f = 2 * p * r / (p + r)
    return r, p, f

# [article, encyclopedia, news, novel] ==> 148995 sentences
def train(model_settings, train_settings):
    train_files = [join(train_settings['input_dir'], f) for f in file_io.list_directory(train_settings['input_dir'])]
    logging.info('#training_files %d', len(train_files))
    test_files = [join(train_settings['eval_dir'], f) for f in file_io.list_directory(train_settings['eval_dir'])]
    logging.info('#testing_files %d', len(test_files))
    
    char_dict = CharDictionary()
    model_settings['input_classes'] = char_dict.num_char_classes() + 1
    model_settings['label_classes'] = char_dict.num_label_classes() + 1

    test_one_hot_by_t, test_seq_lengths, test_chars_mat, test_boundary_mat = util.load_validation_set(test_files, 
        num_look_ahead=model_settings['look_ahead'])

    bucket_boundaries = [20,40,80,120,160,200,240,280,320,360,400,500,600]
    graph = tf.Graph()
    recall, precision, fmeasure = 0, 0, 0
    with graph.as_default():
        #with tf.device('/gpu:0'):
        #    model = CkModel(model_settings)
        #    model.build_graph()
        model = CkModel(model_settings)
        model.build_graph()

        key, context, sequences = read_and_decode_single_example(train_files, shuffle=True)
        length = tf.cast(context["length"], tf.int32)
        src = sequences['source']
        tar = sequences['target']
        
        batch_lengths, [batch_src, batch_tar] = tf.contrib.training.bucket_by_sequence_length(
            input_length=length,
            tensors=[src, tar],
            batch_size=train_settings['batch_size'],
            bucket_boundaries=bucket_boundaries,
            dynamic_pad=True,
            name='batch'
        )

        in_embedding  = np.eye(model.input_classes)
        one_hot_src   = tf.nn.embedding_lookup(in_embedding, batch_src)
        
        out_embedding = np.eye(model.label_classes)
        one_hot_tar   = tf.nn.embedding_lookup(out_embedding, batch_tar)

        train_ops = [model.train_op, model.summary_op, model.mean_loss, model.num_entries, model.probs, model.states]
        
    with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        init_op = tf.group(tf.global_variables_initializer(), 
                            tf.local_variables_initializer())
        sess.run(init_op)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord=coord)

        writer = tf.summary.FileWriter(train_settings['job_dir'] + "/log/" + train_settings['job_name'])
        writer.add_graph(sess.graph)
        
        saver = tf.train.Saver(max_to_keep=3, keep_checkpoint_every_n_hours=1) 
        ckpt = tf.train.get_checkpoint_state(train_settings['job_dir'])
        if ckpt and ckpt.model_checkpoint_path:
            logging.info('loading checkpoint: %s', ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        
        step = sess.run(model.global_step) # load step from checkpoint
        
        #for step in range(train_settings['num_step']):
        while step < train_settings['num_step']:
            _len, _in, _out, _src, _tar = sess.run([batch_lengths, batch_src, batch_tar, 
                one_hot_src, one_hot_tar])
            src_by_times = np.stack(_src, axis=1)
            tar_by_times = np.stack(_tar, axis=1)

            offset = randint(0, min(model_settings['num_unroll'], max(_len), 5)-1)
            #offset = 0

            max_length = src_by_times.shape[0]
            num_rolls = (int)(math.ceil((max_length - offset) / float(model_settings['num_unroll'])))
            total_loss    = 0
            total_entries = 0

            probs  = np.zeros(shape=[0, train_settings['batch_size'], model.label_classes])
            fw_state = model.init_fw_states(train_settings['batch_size'])

            len_left   = _len - offset

            padded_len = np.full(src_by_times.shape[1], model_settings['num_unroll'])

            #for i in range(1):
            for i in range(num_rolls):
                start_t = i * model_settings['num_unroll'] + offset
                end_t   = min((i + 1) * model_settings['num_unroll'] + offset, max_length)

                roll_len = np.minimum.reduce([len_left, padded_len])
                roll_len[roll_len < 0] = 0
                len_left = len_left - roll_len
                feed = { model.inputs:  src_by_times[start_t:end_t,:,:],
                         model.outputs: tar_by_times[start_t:end_t,:,:],
                         model.fw_state: fw_state,
                         model.seq_lengths: roll_len,
                         model.keep_prob: model_settings['keep_prob'] }
                _, _summary, _mean_loss, _num_entries, _p, _s = sess.run(train_ops, feed_dict=feed)
                
                fw_state = model.flatten_fw_states(_s)
                probs  = np.concatenate((probs, _p), axis=0)
                total_loss += (_mean_loss * _num_entries)
                total_entries += _num_entries
                
            #break

            step, _ = sess.run([model.global_step, model.increment_global_step_op])

            if (step % 20 == 0) and (total_entries != 0):
                #print('saving ckpoint...')
                avg_loss = total_loss / total_entries                
                writer.add_summary(_summary, global_step=step)
                print("step: %6d, loss: %f" % (step, avg_loss))

                if step % 100 == 0:
                    saver.save(sess, train_settings['job_dir'] + '/' + train_settings['job_name'], 
                        global_step=model.global_step)
                #if 1:

                    #mask = np.sign(_in)
                    #mask = mask[:, offset:]
                    #results = np.argmax(probs, axis=2)
                    #results = np.stack(results, axis=1)

                    #results = results * mask
                    
                    #_cids   = _in[0, offset:]
                    #_labels = _out[0, offset:]
                    #_pred   = results[0,:]
                    
                    #_chars  = char_dict.cids2chars(_cids)
                    #_words  = char_dict.chars2words(_chars, _labels)
                    #_pwords = char_dict.chars2words(_chars, _pred)
                    #print('|'.join(_words))
                    #print('')
                    #print('|'.join(_pwords))

                    # UNDO THIS LATER
                    recall, precision, f = validate(sess, model, char_dict, test_one_hot_by_t, 
                        test_seq_lengths, test_chars_mat, test_boundary_mat)

                    #pr_summary = tf.Summary(value=[
                    #    tf.Summary.Value(tag="recall", simple_value=recall),                    
                    #    tf.Summary.Value(tag="precision", simple_value=precision),
                    #    tf.Summary.Value(tag="F-measure", simple_value=f)
                    #])
                    #writer.add_summary(pr_summary, global_step=step)
        
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    logging.info('train word segmentor model')

    p = configargparse.getArgParser()
    p.add('-c', '--config', required=False, is_config_file=True, help='config file path')
    p.add('-v', '--verbose', help='verbose', action='store_true')
    p.add('-d', '--input-dir', required=True, help='input directory of tfrecord files for training')
    p.add('-e', '--eval-dir',  required=True, help='directory of text files for cross validation during training')
    p.add('-o', '--job-dir', required=True, help='output directory to save')
    p.add('-j', '--job-name', required=True, help='job name (used in saving model and logging)')
    
    #p.add('-l', '--logdir', required=True, help='log directory for tensorboard')
    p.add('--cell-sizes',  required=True, help='number of cell states', type=int, action='append')
    p.add('--num-layers', required=True, help='number of layers', type=int)
    p.add('--keep-prob', required=True, help='keep probability (for dropout)', type=float)
    
    #p.add('--num-unroll', required=True, help='number of unroll for rnn', type=int)
    p.add('--learning-rate', required=True, help='initial learning rate', type=float)
    p.add('--l2-regularization', required=True, help='l2 regularization factor', type=float)
    p.add('--num-step', required=True, help='number of steps to train', type=int)
    p.add('--batch-size', required=True, help='batch size', type=int)
    p.add('--look-ahead', required=True, help='number of look-ahead', type=int)

    cell_type = p.add_mutually_exclusive_group(required=True)
    cell_type.add('--rnn',  help='use RNN cell',  dest='cell', action='store_const', const='rnn')
    cell_type.add('--lstm', help='use LSTM cell', dest='cell', action='store_const', const='lstm')
    cell_type.add('--gru',  help='use GRU cell',  dest='cell', action='store_const', const='gru')
    #group.set_defaults(cell='rnn')
    
    #direction = p.add_mutually_exclusive_group(required=True)
    #direction.add('--fw', help='only forward layer',   dest='direction', action='store_const', const=1)
    #direction.add('--bi', help='bi-directional layer', dest='direction', action='store_const', const=2)

    opts = vars(p.parse_known_args()[0])
    verbose = opts['verbose']
    assert(len(opts['cell_sizes']) == opts['num_layers'])
    print(opts)

    # MODEL
    model_settings = dict()
    model_settings['num_layers'] = opts['num_layers']
    model_settings['cell_sizes']  = opts['cell_sizes']
    model_settings['keep_prob']  = opts['keep_prob']
    model_settings['look_ahead'] = opts['look_ahead']
    model_settings['num_unroll'] = 30 #opts['num_unroll']
    model_settings['learning_rate'] = opts['learning_rate'] #Initial learning rate
    model_settings['l2_regularization'] = opts['l2_regularization']
    model_settings['cell_type'] = opts['cell']
    #model_settings['direction'] = opts['direction']
    #print(model_settings)
    
    # TRAIN
    train_settings = dict()
    train_settings['num_step']   = opts['num_step']
    train_settings['batch_size'] = opts['batch_size']
    train_settings['input_dir']  = opts['input_dir']
    train_settings['eval_dir']   = opts['eval_dir']
    train_settings['job_dir']    = opts['job_dir']
    train_settings['job_name']   = opts['job_name']    
    #train_settings['logdir']     = opts['logdir']
    
    if (verbose):
        log_level = logging.INFO
    else:
        log_level = logging.WARNING    
    logging.basicConfig(format='%(levelname)s:%(message)s', level=log_level)
    logging.info('model settings: %s', model_settings)
    logging.info('train settings: %s', train_settings)
    #logging.info('check point dir: %s', ckpoint_dir)
    
    train(model_settings, train_settings)