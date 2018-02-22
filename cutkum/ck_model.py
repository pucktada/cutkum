#!/usr/bin/env python
#
# Pucktada Treeratpituk (https://pucktada.github.io/)
# License: MIT
# 2017-05-01
#
# A recurrent neural network model (LSTM) for thai word segmentation
import logging
import re
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.rnn as rnn

#from . import char_dictionary

def load_settings(sess):

    model_settings = dict()        
    model_vars = dict()

    graph = tf.get_default_graph()
    #for v in sess.graph.get_operations():
    #    print('P:', v.name)

    configs = ['cell_sizes', 'look_ahead', 'num_layers', 'input_classes', 'label_classes', 'learning_rate', 'l2_regularization', 'cell_type'] #, 'direction']
    for c in configs:
        name = 'prefix/%s:0' % c
        model_settings[c] = sess.run(graph.get_tensor_by_name(name))

    model_vars['inputs']  = graph.get_tensor_by_name('prefix/placeholder/inputs:0')
    model_vars['fw_state'] = graph.get_tensor_by_name('prefix/placeholder/fw_state:0')    
    model_vars['seq_lengths'] = graph.get_tensor_by_name('prefix/placeholder/seq_lengths:0')
    model_vars['keep_prob'] = graph.get_tensor_by_name('prefix/placeholder/keep_prob:0')

    model_vars['probs'] = graph.get_tensor_by_name('prefix/probs:0')

    return model_settings, model_vars

def load_graph(model_file):
    """ loading necessary configuration of the network from the meta file & 
        the checkpoint file together with variables that are needed for the inferences
    """

    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(model_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the 
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None, 
            return_elements=None, 
            name='prefix',
            op_dict=None,
            producer_op_list=None)

    return graph

def load_model2(sess, meta_file, checkpoint_file):
    """ loading necessary configuration of the network from the meta file & 
        the checkpoint file together with variables that are needed for the inferences
    """
    saver = tf.train.import_meta_graph(meta_file, clear_devices=True)
    saver.restore(sess, checkpoint_file)    

    configs = tf.get_collection('configs')
    pvars   = tf.get_collection('placeholders')

    model_settings = dict()
    for c in configs:
        name = c.name.split(':')[0]
        model_settings[name] = sess.run(c)
        
    model_vars = dict()
    for p in pvars:
        scope, name, _ = re.split('[:/]', p.name)
        model_vars[name] = p
    model_vars['probs'] = tf.get_collection('probs')[0]

    return model_settings, model_vars

class CkModel:
    """ cutkum model: LSTM recurrent neural network model """
    
    def __init__(self, model_settings):
        logging.info('...init WordSegmentor')

        self.num_layers = model_settings["num_layers"]        
      
        self.cell_sizes  = model_settings["cell_sizes"] # list of cell_size, same length as num_layers
        self.total_cells = sum(self.cell_sizes)
        self.cell_start  = [sum(self.cell_sizes [:i]) for i in range(self.num_layers)]

        # keep number of look_ahead (not used in the training, but so that people know how to use the model)
        self.look_ahead = model_settings['look_ahead']

        #self.num_unroll = model_settings["num_unroll"]
        self.input_classes = model_settings['input_classes']
        self.label_classes = model_settings['label_classes']
        self.learning_rate = model_settings['learning_rate']
        self.l2_regularization = model_settings['l2_regularization'] # 0.1
        self.cell_type = model_settings['cell_type']
        #self.direction = model_settings['direction']

        #self.states = None

        tf.add_to_collection('configs', tf.constant(self.cell_sizes, name="cell_sizes"))
        tf.add_to_collection('configs', tf.constant(self.look_ahead, name="look_ahead"))  
        #tf.add_to_collection('configs', tf.constant(self.num_unroll, name="num_unroll"))
        tf.add_to_collection('configs', tf.constant(self.num_layers, name="num_layers"))
        tf.add_to_collection('configs', tf.constant(self.input_classes, name="input_classes"))
        tf.add_to_collection('configs', tf.constant(self.label_classes, name="label_classes"))
        tf.add_to_collection('configs', tf.constant(self.learning_rate, name="learning_rate"))           
        tf.add_to_collection('configs', tf.constant(self.l2_regularization, name="l2_regularization"))
        tf.add_to_collection('configs', tf.constant(self.cell_type, name="cell_type"))
        #tf.add_to_collection('configs', tf.constant(self.direction, name="direction"))
        
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.increment_global_step_op = tf.assign(self.global_step, self.global_step+1)
        
    def _create_placeholders(self):
        logging.info('...create placeholder')
        with tf.name_scope("placeholder"):       
            # (time, batch, in)        
            self.inputs  = tf.placeholder(tf.float32, (None, None, self.input_classes), name="inputs")
            # (time, batch, out)
            self.outputs = tf.placeholder(tf.float32, (None, None, self.label_classes), name="outputs")
            # [batch]
            self.seq_lengths = tf.placeholder(tf.int32, [None], name="seq_lengths")

            # LSTM - [2, None, sum(cell_sizes)]
            # GRU, RNN - [1, None, sum(cell_sizes)]
            if (self.cell_type == 'lstm'):
                self.fw_state = tf.placeholder(tf.float32, [2, None, self.total_cells], name="fw_state")
            else: # gru, rnn
                self.fw_state = tf.placeholder(tf.float32, [1, None, self.total_cells], name="fw_state")

            self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
            
            tf.add_to_collection('placeholders', self.inputs)
            tf.add_to_collection('placeholders', self.outputs)
            tf.add_to_collection('placeholders', self.seq_lengths)
            tf.add_to_collection('placeholders', self.keep_prob)         
    
    # 
    def init_fw_states(self, batch_size):
        if (self.cell_type == 'lstm'):
            return np.zeros(shape=[2, batch_size, self.total_cells])
        else: # GRU, RNN
            return np.zeros(shape=[1, batch_size, self.total_cells])

    # state tuple to tensor
    def flatten_fw_states(self, fw_state_tuple):
        if (self.cell_type == 'lstm'):
            # fw_state_tuple is tuple of LSTMStateTuple of lenghts 'num_layers'
            # states = [2, batch_size, self.total_cells]
            c_tensor = np.concatenate([fw_state_tuple[i].c for i in range(self.num_layers)], axis=1)
            h_tensor = np.concatenate([fw_state_tuple[i].h for i in range(self.num_layers)], axis=1)
            state = np.stack([c_tensor, h_tensor])
        else: # GRU, RNN
            # fw_state_tuple is tuple of ndarray of lenghts 'num_layers'
            c_tensor = np.concatenate([fw_state_tuple[i] for i in range(self.num_layers)], axis=1)
            state = np.expand_dims(c_tensor, axis=0)
        return state #.eval()

    # state tensor to tuple
    def unstack_fw_states(self, fw_state):
        if (self.cell_type == 'lstm'):
            # states = [2, batch_size, self.total_cells]
            fw_state_tuple = tuple(
                [tf.contrib.rnn.LSTMStateTuple(
                    fw_state[0, :, self.cell_start[i]:self.cell_start[i]+self.cell_sizes[i]], 
                    fw_state[1, :, self.cell_start[i]:self.cell_start[i]+self.cell_sizes[i]]) 
                for i in range(self.num_layers)])
        else: # GRU, RNN
            # states = [1, batch_size, self.total_cells]
            fw_state_tuple = tuple(
                [fw_state[0, :, self.cell_start[i]:self.cell_start[i]+self.cell_sizes[i]] 
                    for i in range(self.num_layers)])
        return fw_state_tuple

    def _inference(self):
        logging.info('...create inference')

        fw_state_tuple = self.unstack_fw_states(self.fw_state)

        fw_cells   = list()
        for i in range(0, self.num_layers):
            if (self.cell_type == 'lstm'):
                cell = rnn.LSTMCell(num_units=self.cell_sizes[i], state_is_tuple=True)
            elif (self.cell_type == 'gru'):
                # change to GRU
                cell = rnn.GRUCell(num_units=self.cell_sizes[i])
            else:
                cell = rnn.BasicRNNCell(num_units=self.cell_sizes[i])

            cell = rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
            fw_cells.append(cell)
        self.fw_cells = rnn.MultiRNNCell(fw_cells, state_is_tuple=True)

        rnn_outputs, states = tf.nn.dynamic_rnn(
            self.fw_cells, 
            self.inputs, 
            initial_state=fw_state_tuple,
            sequence_length=self.seq_lengths,
            dtype=tf.float32, time_major=True)

        # project output from rnn output size to OUTPUT_SIZE. Sometimes it is worth adding
        # an extra layer here.
        self.projection = lambda x: layers.linear(x, 
            num_outputs=self.label_classes, activation_fn=tf.nn.sigmoid)

        self.logits = tf.map_fn(self.projection, rnn_outputs, name="logits")
        self.probs  = tf.nn.softmax(self.logits, name="probs")
        self.states = states

        tf.add_to_collection('probs',  self.probs)
    
    def _create_loss(self):
        logging.info('...create loss')
        with tf.name_scope("loss"):
            # shape=[Time * Batch, label_classes]   
            outputs_flat = tf.reshape(self.outputs, [-1, self.label_classes])
            logits_flat  = tf.reshape(self.logits,  [-1, self.label_classes])

            # calculate the losses shape=[Time * Batch]
            # pre-tensorflow 1.5
            #losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=outputs_flat, logits=logits_flat)            
            losses = tf.nn.softmax_cross_entropy_with_logits(labels=outputs_flat, logits=logits_flat)
            
             # create mask [Time * Batch] where 0: padded, 1: not-padded
            mask = outputs_flat[:,0]
            mask = tf.abs(tf.subtract(mask, tf.ones_like(mask)))
            # mask the losses
            masked_losses = mask * losses

            l2_reg = self.l2_regularization
            l2 = l2_reg * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables() 
                if not ("noreg" in tf_var.name or "Bias" in tf_var.name))
            self.losses = masked_losses + l2

            self.num_entries = tf.reduce_sum(mask)
            self.mean_loss = tf.reduce_sum(masked_losses) / self.num_entries
            
            # accuracy
            correct_pred = tf.cast(tf.equal(tf.argmax(outputs_flat, 1), tf.argmax(logits_flat, 1)), tf.float32)
            mask_correct_pred = mask * correct_pred
            self.accuracy = tf.reduce_sum(mask_correct_pred) / self.num_entries
        
    def _create_optimizer(self):
        logging.info('...create optimizer')
        with tf.name_scope("train"):        
            #self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.mean_loss, global_step=self.global_step)
            max_gradient_norm = 1.0
            params    = tf.trainable_variables()
            gradients = tf.gradients(self.mean_loss, params)
            clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
            #self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate)\
            #    .apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate)\
                .apply_gradients(zip(clipped_gradients, params))
    
    def _create_summary(self):
        logging.info('...create summary')
        tf.summary.scalar("mean_loss", self.mean_loss)
        tf.summary.scalar("accuracy",  self.accuracy)
        self.summary_op = tf.summary.merge_all()
    
    def build_graph(self):
        self._create_placeholders()
        self._inference()
        self._create_loss()
        self._create_optimizer()
        self._create_summary()

if __name__ == '__main__':
    print('create word segmentor model')
    char_dict = CharDictionary()
    
    # MODEL
    model_settings = dict()
    #model_settings["l2_regularisation"] = 0.0 # not usring right now
    model_settings['num_unroll'] = 12
    model_settings['num_layers'] = 3
    model_settings['cell_size'] = 64
    model_settings['input_classes'] = char_dict.num_char_classes() + 1
    model_settings['label_classes'] = char_dict.num_label_classes() + 1
    model_settings['learning_rate'] = 0.001 # Initial learning rate
    
    model = CkModel(model_settings)
    model.build_graph()
