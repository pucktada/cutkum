#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import argparse
import logging
from .tokenizer import Cutkum

def main():
	p = argparse.ArgumentParser()
	p.add_argument('-v', '--verbose', help='verbose', action='store_true')
	#p.add_argument('-m', '--model_file', required=True, help='model file to load')
	g1 = p.add_mutually_exclusive_group(required=True)
	g1.add_argument('-s', '--sentence', help='sentence to parse')
	g1.add_argument('-i', '--input_file', help='input file')
	g1.add_argument('-id', '--input_dir', help='input directory')

	g2 = p.add_mutually_exclusive_group(required=False)
	g2.add_argument('-o', '--output_file', help='output file if --input_file is given')	
	g2.add_argument('-od', '--output_dir', help='output directory if --input_dir is given')

	g3 = p.add_mutually_exclusive_group(required=False)
	g3.add_argument('--viterbi', action='store_const', dest='max_flag', const='vit', help='output word boundary using viterbi (default)')	
	g3.add_argument('--max', action='store_const', dest='max_flag', const='max', help='output word boundary using maximum probabilities')
	p.set_defaults(max_flag='max')

	opts = vars(p.parse_known_args()[0])

	verbose = opts['verbose']
	if (verbose):
		log_level = logging.INFO
	else:
		log_level = logging.WARNING    
	logging.basicConfig(format='%(levelname)s:%(message)s', level=log_level)

	# OTHERS ARGS
	#model_file		= opts['model_file']
	input_sentence  = opts['sentence']  
	input_file      = opts['input_file']
	output_file     = opts['output_file']
	input_dir       = opts['input_dir']
	output_dir      = opts['output_dir']
	max_flag        = opts['max_flag']

	if (input_file is not None) & (output_file is None):
		p.error('--input_file and --output_file must be given together')
	if (input_dir is not None) & (output_dir is None):
		p.error('--input_dir and --output_dir must be given together')				
	use_viterbi = True
	if (max_flag == 'max'):
		use_viterbi = False

	ck = Cutkum()
	#ck = Cutkum(model_file)
	if input_file is not None:
		ck.tokenize_file(input_file, output_file, use_viterbi)
	elif (input_dir is not None) & (output_dir is not None):
		ck.tokenize_directory(input_dir, output_dir, use_viterbi)
	elif input_sentence is not None:
		words = ck.tokenize(input_sentence, use_viterbi)
		print("|".join(words))

if __name__ == '__main__':
	main()