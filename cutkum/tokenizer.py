# -*- coding: utf-8 -*-
#
# Pucktada Treeratpituk (https://pucktada.github.io/)
# License: MIT
# 2017-05-01
#
# A command-line function to load a trained model and compute the word segmentation
from __future__ import unicode_literals
import pkg_resources
from os import listdir
from os.path import isfile, isdir, join
import logging
import math
import numpy as np
import re
import sys
import tensorflow as tf
from char_dictionary import CharDictionary
from ck_model import CkModel, load_graph, load_settings
from util import max_prob, viterbi, load_files_into_matrix, load_validation_set, get_boundary_array, count_correct_chars, count_correct_words, word_count_from_boundary_array

class Cutkum:
	""" cutkum model: LSTM recurrent neural network model """

	def __init__(self, model_file=None):
		logging.info('...init Cutkum')

		self.char_dict = CharDictionary()
		if (model_file is None):
			model_file = pkg_resources.resource_filename(__name__, 'frozen_model.pb')
		#self.model_file = 'model/lstm.l6.d2.pb'
		self.graph = load_graph(model_file)
		
		with tf.Session(graph=self.graph) as sess:
			model_settings, model_vars = load_settings(sess)
			self.model_settings = model_settings
			self.model_vars = model_vars

		self.model_settings['total_cells'] = sum(self.model_settings['cell_sizes'])
		logging.info('...done init')

	def _init_fw_states(self, batch_size):
		if (self.model_settings['cell_type'] == 'lstm'):
			return np.zeros(shape=[2, batch_size, self.model_settings['total_cells']])
		else: # GRU, RNN
			return np.zeros(shape=[1, batch_size, self.model_settings['total_cells']])

	def _process_sentence(self, one_hot_by_t):
		""" run the inference to segment the given 'sentence' represented by the one_hot vector by time
		"""
		probs = None
		fw_state = self._init_fw_states(batch_size=1)

		with tf.Session(graph=self.graph) as sess:
			feed = { 
				self.model_vars['inputs']:  one_hot_by_t, 
				self.model_vars['fw_state']: fw_state,
				self.model_vars['seq_lengths']: [one_hot_by_t.shape[0]], 
				self.model_vars['keep_prob']: 1.0
			}
			probs = sess.run([self.model_vars['probs']], feed_dict=feed)

		return np.squeeze(probs)

	def _process_batch(self, one_hot_by_t, seq_lengths):

		prob_list = []
		#fw_state = self._init_fw_states(batch_size=len(seq_lengths))
		fw_state = self._init_fw_states(batch_size=1)

		with tf.Session(graph=self.graph) as sess:
			for i in range(len(seq_lengths)): # for each line
				if (i % 10) == 0:
					logging.info('sen: %d', i)
				if (seq_lengths[i] != 0):
					line_one_hot_by_t = one_hot_by_t[0:seq_lengths[i],i:(i+1),:]
					feed = { self.model_vars['inputs']: line_one_hot_by_t, 
						self.model_vars['fw_state']: fw_state,					
						self.model_vars['seq_lengths']: [seq_lengths[i]], 
						self.model_vars['keep_prob']: 1.0
					}
					probs = sess.run(self.model_vars['probs'], feed_dict=feed)
					prob_list.append(probs)
				else:
					prob_list.append([])
		return prob_list

	def tokenize(self, sentence, use_viterbi=True):
		""" run the inference to segment the given 'sentence' into words seperated by '|' 
		"""

		if (sys.version_info > (3, 0)):
			uni_s = u"%s" % sentence # only for 2.7 (for Python 3, no need to decode)
		else:
			uni_s = u"%s" % sentence.decode('utf-8') # only for 2.7 (for Python 3, no need to decode)

		# add padding
		cids_padding = self.char_dict.padding_cids(self.model_settings['look_ahead'])

		chars = list(uni_s.strip())
		cids  = self.char_dict.chars2cids(chars) + cids_padding

		in_embedding = np.eye(self.model_settings['input_classes'])
		one_hot      = in_embedding[cids]
		one_hot_by_t = np.expand_dims(one_hot, 1)

		# p.shape = (len(s) + #look_ahead, #classes)
		p  = self._process_sentence(one_hot_by_t)
		if (use_viterbi): # viterbi
			labels = viterbi(p)
		else:
			labels = max_prob(p)

		words  = self.char_dict.chars2words(chars, labels[self.model_settings['look_ahead']:])
		return words

	def tokenize_file(self, input_file, output_file, use_viterbi=True):
		""" run the inference to segment the given file into words seperated by '|' 
		"""
		# load file into matrix
		one_hot_by_t, seq_lengths, chars_mat = load_files_into_matrix([input_file], self.model_settings['look_ahead'])
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

					words  = self.char_dict.chars2words(chars_mat[i], labels[self.model_settings['look_ahead']:])
					line_out = '|'.join(words)
					if (sys.version_info <= (3, 0)):
						line_out = line_out.encode('utf8')
					outfile.write(line_out)
					outfile.write('\n')
				else:
					outfile.write('\n')

	def tokenize_directory(self, input_dir, output_dir, use_viterbi=True):
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
				one_hot_by_t, seq_lengths, chars_mat = load_files_into_matrix([infile_path], self.model_settings['look_ahead'])
				probs = self._process_batch(one_hot_by_t, seq_lengths)

				n_examples= len(probs)
				for i in range(n_examples):
					if (seq_lengths[i] != 0):
						p = probs[i][0:seq_lengths[i], 0, :]

						if (use_viterbi): # viterbi
							labels = viterbi(p)
						else:
							labels = max_prob(p)
						words  = self.char_dict.chars2words(chars_mat[i], labels[self.model_settings['look_ahead']:])

						line_out = '|'.join(words)
						if (sys.version_info <= (3, 0)):
							line_out = line_out.encode('utf8')
						outfile.write(line_out)
						outfile.write('\n')
					else:
						outfile.write('\n')
				outfile.close()
				logging.info('%s ... DONE', f)

	def validate(self, eval_dir):
		test_files = listdir(eval_dir)
		with tf.Session(graph=self.graph) as sess:

			all_correct = all_n_refwords = all_n_outwords = 0
			all_correct_c = all_n_refwords_c = all_n_outwords_c = 0

			# one file at a time
			for test_file in test_files:
				file_path = join(eval_dir, test_file)
				one_hot_by_t, seq_lengths, chars_mat, boundary_mat = load_validation_set([file_path], 
					num_look_ahead=self.model_settings['look_ahead'])

				batch_size = one_hot_by_t.shape[1] # one_hot_by_t:   [max_time, #examples, #dict]
				fw_state = self._init_fw_states(batch_size=batch_size)

				feed = { 
					self.model_vars['inputs']:  one_hot_by_t, 
					self.model_vars['fw_state']: fw_state,
					self.model_vars['seq_lengths']: seq_lengths, 
					self.model_vars['keep_prob']: 1.0
				}

				probs  = sess.run(self.model_vars['probs'], feed_dict=feed)

				total_c  = total_r  = total_o  = 0.0
				total_cc = total_rc = total_oc = 0.0

				_, n_examples, _ = probs.shape

				for i in range(n_examples):
					p = probs[0:seq_lengths[i], i, :]

					#labels = util.max_prob(p)
					labels = viterbi(p)

					padding = self.char_dict.padding_labels(self.model_settings['look_ahead'])
					labels = np.concatenate((labels[self.model_settings['look_ahead']:], padding))
					words  = self.char_dict.chars2words(chars_mat[i], labels)

					x = boundary_mat[i]
					y = get_boundary_array(words)

					n_answers = word_count_from_boundary_array(x)
					n_words   = len(words)
					correct, n_refwords, n_outwords = count_correct_words(x, y, n_answers, n_words)

					correct_c, n_refwords_c, n_outwords_c = count_correct_chars(x, y)

					total_c += correct
					total_r += n_refwords
					total_o += n_outwords

					total_cc += correct_c
					total_rc += n_refwords_c
					total_oc += n_outwords_c

				all_correct += total_c
				all_n_refwords += total_r
				all_n_outwords += total_o

				all_correct_c += total_cc
				all_n_refwords_c += total_rc
				all_n_outwords_c += total_oc			

				r = (float)(total_c / total_r)
				p = (float)(total_c / total_o)
				if (r == 0) or (p == 0):
					f = 0
				else:
					f = 2 * p * r / (p + r)

				rc = (float)(total_cc / total_rc)
				pc = (float)(total_cc / total_oc)
				if (rc == 0) or (pc == 0):
					fc = 0
				else:
					fc = 2 * pc * rc / (pc + rc)

				print('[%s]  R: %.3f,  P: %.3f,  F: %.3f' % (test_file, r, p, f))
				print('[%s] RC: %.3f, PC: %.3f, FC: %.3f' % (test_file, rc, pc, fc))
			
			all_r = (float)(all_correct / all_n_refwords)
			all_p = (float)(all_correct / all_n_outwords)
			if (all_r == 0) or (all_p == 0):
				all_f = 0
			else:
				all_f = 2 * all_p * all_r / (all_p + all_r)

			all_rc = (float)(all_correct_c / all_n_refwords_c)
			all_pc = (float)(all_correct_c / all_n_outwords_c)
			if (all_rc == 0) or (all_pc == 0):
				all_fc = 0
			else:
				all_fc = 2 * all_pc * all_rc / (all_pc + all_rc)

			print('[%s]  R: %.3f,  P: %.3f,  F: %.3f' % ('TOTAL', all_r, all_p, all_f))
			print('[%s] RC: %.3f, PC: %.3f, FC: %.3f' % ('TOTAL', all_rc, all_pc, all_fc))


