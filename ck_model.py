#!/usr/bin/env python3
#
# Pucktada Treeratpituk (https://pucktada.github.io/)
# License: MIT
# 2017-05-01
#
# A recurrent neural network model (LSTM) for thai word segmentation

import logging
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

from char_dictionary import CharDictionary

class CkModel:
    """ utkum model: LSTM recurrent neural network model """
    
    def __init__(self, model_settings):
        logging.info('...init WordSegmentor')
                
        self.lstm_size  = model_settings["lstm_size"]
        self.num_unroll = model_settings["num_unroll"]
        self.num_layers = model_settings["num_layers"]
        self.input_classes = model_settings['input_classes']
        self.label_classes = model_settings['label_classes']
        self.learning_rate = model_settings['learning_rate']
        
        tf.add_to_collection('configs', tf.constant(self.lstm_size, name="lstm_size"))
        tf.add_to_collection('configs', tf.constant(self.num_unroll, name="num_unroll"))
        tf.add_to_collection('configs', tf.constant(self.num_layers, name="num_layers"))
        tf.add_to_collection('configs', tf.constant(self.input_classes, name="input_classes"))
        tf.add_to_collection('configs', tf.constant(self.label_classes, name="label_classes"))
        tf.add_to_collection('configs', tf.constant(self.learning_rate, name="learning_rate"))           
        
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        
    def _create_placeholders(self):
        logging.info('...create placeholder')
        # (time, batch, in)        
        self.inputs  = tf.placeholder(tf.float32, (None, None, self.input_classes), name="inputs")
        # (time, batch, out)
        self.outputs = tf.placeholder(tf.float32, (None, None, self.label_classes), name="outputs")
        self.init_state = tf.placeholder(tf.float32, [self.num_layers, 2, None, self.lstm_size], name="init_state")
        
        tf.add_to_collection('placeholders', self.inputs)
        tf.add_to_collection('placeholders', self.outputs)
        tf.add_to_collection('placeholders', self.init_state)
    
    def _inference(self):
        logging.info('...create inference')
        
        init_state_list = tf.unstack(self.init_state, axis=0)
        init_state_tuple = tuple(
            [tf.contrib.rnn.LSTMStateTuple(init_state_list[idx][0], 
                                           init_state_list[idx][1])
             for idx in range(self.num_layers)])

        cells = list()
        for i in range(0, self.num_layers):
            cell = tf.contrib.rnn.LSTMCell(num_units=self.lstm_size, state_is_tuple=True)
            cells.append(cell)
        self.cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
        
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(self.cell, 
            self.inputs, initial_state=init_state_tuple, time_major=True)

        # project output from rnn output size to OUTPUT_SIZE. Sometimes it is worth adding
        # an extra layer here.
        self.projection = lambda x: layers.linear(x, 
            num_outputs=self.label_classes, activation_fn=tf.nn.sigmoid)

        self.logits = tf.map_fn(self.projection, rnn_outputs, name="logits")
        self.probs  = tf.nn.softmax(self.logits, name="probs")
        self.states = rnn_states
        
        tf.add_to_collection('probs',  self.probs)
    
    def _create_loss(self):
        logging.info('...create loss')

        # shape=[Time * Batch, label_classes]   
        outputs_flat = tf.reshape(self.outputs, [-1, self.label_classes])
        logits_flat  = tf.reshape(self.logits,  [-1, self.label_classes])

        # calculate the losses shape=[Time * Batch]
        losses = tf.nn.softmax_cross_entropy_with_logits(
            labels=outputs_flat, logits=logits_flat)
            
         # create mask [Time * Batch] where 0: padded, 1: not-padded
        mask = outputs_flat[:,0]
        mask = tf.abs(tf.subtract(mask, tf.ones_like(mask)))
        # mask the losses
        masked_losses = mask * losses

        self.num_entries = tf.reduce_sum(mask)
        self.mean_loss = tf.reduce_sum(masked_losses) / self.num_entries        
        
    def _create_optimizer(self):
        logging.info('...create optimizer')
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.mean_loss, global_step=self.global_step)
    
    def build_graph(self):
        self._create_placeholders()
        self._inference()
        self._create_loss()
        self._create_optimizer()

if __name__ == '__main__':
    print('create word segmentor model')
    char_dict = CharDictionary()
    
    # MODEL
    model_settings = dict()
    #model_settings["l2_regularisation"] = 0.0 # not usring right now
    model_settings['num_unroll'] = 12
    model_settings['num_layers'] = 3
    model_settings['lstm_size'] = 64
    model_settings['input_classes'] = char_dict.num_char_classes() + 1
    model_settings['label_classes'] = char_dict.num_label_classes() + 1
    model_settings['learning_rate'] = 0.001 # Initial learning rate
    
    model = CkModel(model_settings)
    model.build_graph()
