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
import logging
import math
import numpy as np
import re
import sys
import tensorflow as tf
from .char_dictionary import CharDictionary
from .ck_model import CkModel, load_graph, load_settings
from .util import max_prob, viterbi, load_files_into_matrix, load_validation_set

class Cutkum:
	""" cutkum model: LSTM recurrent neural network model """

	def __init__(self, model_file):
		logging.info('...init Cutkum')

		self.char_dict = CharDictionary()
		#self.model_file = 'model/lstm.l6.d2.pb'
		self.graph = load_graph(model_file)

		with tf.Session(graph=self.graph) as sess:
			model_settings, model_vars = load_settings(sess)
			self.model_settings = model_settings
			self.model_vars = model_vars

		logging.info('...done init')

	def _process_sentence(self, one_hot_by_t):
		""" run the inference to segment the given 'sentence' represented by the one_hot vector by time
		"""
		probs = None
		with tf.Session(graph=self.graph) as sess:
			feed = { 
				self.model_vars['inputs']:  one_hot_by_t, 
				self.model_vars['seq_lengths']: [one_hot_by_t.shape[0]], 
				self.model_vars['keep_prob']: 1.0
			}
			probs = sess.run([self.model_vars['probs']], feed_dict=feed)

		return np.squeeze(probs)

	def _process_batch(self, one_hot_by_t, seq_lengths):

		prob_list = []
		with tf.Session(graph=self.graph) as sess:
			for i in range(len(seq_lengths)): # for each line
				if (i % 10) == 0:
					logging.info('sen: %d', i)
				if (seq_lengths[i] != 0):
					line_one_hot_by_t = one_hot_by_t[0:seq_lengths[i],i:(i+1),:]
					feed = { self.model_vars['inputs']: line_one_hot_by_t, 
						self.model_vars['seq_lengths']: [seq_lengths[i]], 
						self.model_vars['keep_prob']: 1.0
					}
					probs = sess.run(self.model_vars['probs'], feed_dict=feed)
					prob_list.append(probs)
				else:
					prob_list.append([])
		return prob_list

	def tokenize(self, sentence, use_viterbi=False):
		""" run the inference to segment the given 'sentence' into words seperated by '|' 
		"""

		if (sys.version_info > (3, 0)):
			uni_s = u"%s" % sentence # only for 2.7 (for Python 3, no need to decode)
		else:
			uni_s = u"%s" % sentence.decode('utf-8') # only for 2.7 (for Python 3, no need to decode)

		chars = list(uni_s.strip())
		cids  = self.char_dict.chars2cids(chars)

		in_embedding = np.eye(self.model_settings['input_classes'])
		one_hot      = in_embedding[cids]
		one_hot_by_t = np.expand_dims(one_hot, 1)

		p  = self._process_sentence(one_hot_by_t)
		if (use_viterbi): # viterbi
			labels = viterbi(p)
		else:
			labels = max_prob(p)

		words  = self.char_dict.chars2words(chars, labels)
		return words

	def tokenize_file(self, input_file, output_file, use_viterbi=False):
		""" run the inference to segment the given file into words seperated by '|' 
		"""

		# load file into matrix
		one_hot_by_t, seq_lengths, chars_mat = load_files_into_matrix([input_file])
		probs = self._process_batch(one_hot_by_t, seq_lengths)

		with open(output_file, 'w') as outfile:

			n_examples= len(probs)
			for i in range(n_examples):
				if (seq_lengths[i] != 0):
					p = probs[i][0:seq_lengths[i], 0, :]

					if (use_viterbi): # viterbi
						labels = viterbi(p)
					else:
						labels = max_prob(p)

					words  = self.char_dict.chars2words(chars_mat[i], labels)
					line_out = '|'.join(words)
					if (sys.version_info <= (3, 0)):
						line_out = line_out.encode('utf8')
					outfile.write(line_out)
					outfile.write('\n')
				else:
					outfile.write('\n')

	def tokenize_directory(self, input_dir, output_dir, use_viterbi=False):
		""" run the inference to segment each file in the given directory into words seperated by '|' 
		"""
		input_files = listdir(input_dir)

		for f in input_files:
			infile_path  = join(input_dir, f)
			outfile_path = join(output_dir, f)

			if isfile(outfile_path):
				logging.info('%s ... SKIP (already exists in the output directory)', f)
			else:
				outfile = open(outfile_path, 'w')
				one_hot_by_t, seq_lengths, chars_mat = load_files_into_matrix([infile_path])
				probs = self._process_batch(one_hot_by_t, seq_lengths)

				n_examples= len(probs)
				for i in range(n_examples):
					if (seq_lengths[i] != 0):
						p = probs[i][0:seq_lengths[i], 0, :]

						if (use_viterbi): # viterbi
							labels = viterbi(p)
						else:
							labels = max_prob(p)
						words  = self.char_dict.chars2words(chars_mat[i], labels)

						line_out = '|'.join(words)
						if (sys.version_info <= (3, 0)):
							line_out = line_out.encode('utf8')
						outfile.write(line_out)
						outfile.write('\n')
					else:
						outfile.write('\n')
				outfile.close()
				logging.info('%s ... DONE', f)


