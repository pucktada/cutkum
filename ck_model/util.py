#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import sys
from tensorflow.python.lib.io import file_io
#from .char_dictionary import char_dictionary
from .char_dictionary import CharDictionary
import numpy as np
import re

def fix_space_and_dot(prob_matrix, cids):
    # if prob_matrix is 1D, make it 2D
    if len(np.shape(prob_matrix)) == 1:
        prob_matrix = [prob_matrix]
        
    length = len(prob_matrix)
    assert(length == len(cids))
    for t in range(length):
        # ' '=> 75, '.'=>76
        if (cids[t] == 75): #or (cids[t] == 76):
            for i in range(5):
                prob_matrix[t][i] = 0.0
            prob_matrix[t][4] = 1.0
    return prob_matrix

# shape(prob_matrix) = [#times, #classes]
def max_prob(prob_matrix):
    if len(np.shape(prob_matrix)) == 1:
        prob_matrix = [prob_matrix]
        
    length = len(prob_matrix)

    seq = np.ones(length, 'int32') * 2
    # {B, M, E, S} <=== 1:begin, 2:middle, 3:end, 4:single

    seq[0] = 1
    for t in range(1, length):
        # find where 1, 4 is max
        max_id = np.argmax(prob_matrix[t])
        if (max_id == 1) or (max_id == 4):
            seq[t] = 1

    for t in range(length):
        if seq[t] == 1:
            if (t != length-1) and (seq[t+1] == 1):
                seq[t] = 4
        if seq[t] == 2:
            if (t != length-1) and (seq[t+1] == 1):
                seq[t] = 3
    return seq

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
        
    # {B, M, E, S} <=== 1:begin, 2:middle, 3:end, 4:single
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

def word_count_from_boundary_array(x):
    n_edges = 0
    for i in x:
        if (i != 0):
            n_edges += 1
    return n_edges-1
    
'''
    words: array of words ... (can have tags and not have tags)
    words = เขา|ร้อง|บท|<POEM>เย็นย่ำ จะค่ำอยู่แล้วลงรอนรอน</POEM>|
    words = เขา|ร้อง|บท|เย็นย่ำ จะค่ำอยู่แล้วลงรอนรอน|
    
    return word boundary array of length+1
    1 is the start of a word
    0 is the middle of a word
    2 is the start of a entity word
'''
def get_boundary_array(words):
    length = 0
    for w in words:
        if w.startswith('<NE>') or w.startswith('<AB>'):
            length += (len(w) - (2*4 + 1))
        elif w.startswith('<NER>'):
            length += (len(w) - (2*5 + 1))
        elif w.startswith('<POEM>'):
            length += (len(w) - (2*6 + 1))
        else:
            length += len(w)
    
    x = np.zeros(length+1, dtype=np.int32)
    x[0] = 1
    
    cur = 0
    for w in words:
        # NE, AB, NER and POEM 
        if w.startswith('<NE>') or w.startswith('<AB>') or w.startswith('<POEM>') or w.startswith('<NER>'):
            x[cur] = 2
            t = re.sub('<[^>]*>', '', w)
            cur += len(t)
            x[cur] = 1
        else:
            cur += len(w)            
            x[cur] = 1

    return x
    
'''
    NE, AB and POEM

                    x= เขา|ร้อง|บท|<POEM>เย็นย่ำ จะค่ำอยู่แล้วลงรอนรอน</POEM>|
    R=4/4, P=4/4    b= เขา|ร้อง|บท|เย็นย่ำ จะค่ำอยู่แล้วลงรอนรอน|
    R=4/4, P=4/4    c= เขา|ร้อง|บท|เย็น|ย่ำ| |จะ|ค่ำ|อยู่แล้ว|ลง|รอนรอน| 11-7=4
    R=2/4, P=2/3    d= เขา|ร้อง|บทเย็น|ย่ำ จะค่ำอยู่แล้วลงรอนรอน|        4-1

       F-Measure = 2 x Precision x Recall / (Precision + Recall)

    Corr       = จำนวนคำที่ผู้เข้าแข่งขันแบ่งได้อย่างถูกต้อง
    RefWord    = จำนวนคำที่แบ่งโดยคณะกรรมการ
    OutputWord = จำนวนคำที่ผู้เข้าแข่งขันแบ่งมาทั้งหมด
    
    x: answer boundary array 
    y: output boundary array
    n_answers: number of words in the answers (before discounting)
    n_outputs: number of words in the outputs (before discounting)
'''
def count_correct_words(answer_boundary, output_boundary, n_answers, n_outputs):
    x = answer_boundary
    y = output_boundary
    
    length = len(x) - 1
    n_outwords = n_outputs
    n_refwords = n_answers
    correct = 0
    
    # counting 'n_outwords'
    #discount = 0
    cur = 0
    while cur <= length:
        if (x[cur] > 1): # start of <NE>, <AB>, <POEM>
            end = cur+1
            while x[end] == 0:
                end += 1
            _c = 0
            # j = [cur,...,end]
            for j in range(cur+1, end+1):
                if y[j] > 0:
                    _c += 1
            n_outwords = n_outwords - (_c - 1)
            if (y[cur] > 1) and (y[end] > 1):
                n_outwords += 1
            #n_outwords 
            cur = end
        else:
            cur += 1
    
    # counting 'correct'
    cur = 0
    while cur <= length:
        # found 
        if (x[cur] > 0) and (y[cur] > 0): # start match
            if x[cur] > 1: # start of <NE>, <AB>, <POEM> & match...
                cur += 1
                # find the end of NE,AB,POEM
                while cur <= length:
                    if x[cur] > 0:
                        break
                    cur += 1
                if y[cur] > 0: # ... also match the ending
                    correct += 1
            else:
                cur += 1            
                while cur <= length:
                    if (x[cur] > 0) and (y[cur] > 0): # end match...
                        correct += 1
                        #print(cur)
                        break
                    elif (x[cur] > 0) or (y[cur] > 0): # not match..
                        break
                    cur += 1
        else:
            cur += 1
    
    return (correct, n_refwords, n_outwords)

'''
    test_files: list of file to load

    return
    
    one_hot_by_t:   [max_time, #examples, #dict]
    seq_lengths:    lengths of each example [#examples], used to feed the model
    chars_mat:      list of list of characters ... can be used for reconstruction of sentences
'''
def load_files_into_matrix(test_files):
    char_dict = CharDictionary()
    input_classes = char_dict.num_char_classes() + 1

    cids_mat  = []
    chars_mat = []
    seq_lengths = []
    for input_file in test_files:
        with file_io.FileIO(input_file, 'r') as f:
            for s in f: # s is the line string
                if (sys.version_info <= (3, 0)):
                    s = s.decode('utf8')
                if s and (len(s) > 0):                    
                    chars = list(s.strip())
                    cids = char_dict.chars2cids(chars)
                    seq_lengths += [len(cids)]
                    cids_mat.append(cids)
                    chars_mat.append(chars)

    n_examples = len(cids_mat)
    
    max_len = max(seq_lengths)
    batch_labels = np.zeros((n_examples, max_len), dtype=np.int32)
    for i in range(n_examples):
        batch_labels[i,0:len(cids_mat[i])] = cids_mat[i]

    in_embedding = np.eye(input_classes) # 2
    one_hot_flat = in_embedding[list(batch_labels.flatten())]
    one_hot      = np.reshape(one_hot_flat, (n_examples, max_len, input_classes))
    
    #(time, batch, in)
    one_hot_by_t = np.transpose(one_hot, (1,0,2))
    return one_hot_by_t, seq_lengths, chars_mat
'''
    test_files: list of file to load

    return
    
    one_hot_by_t:   [max_time, #examples, #dict]
    seq_lengths:    lengths of each example [#examples], used to feed the model
    chars_mat:      list of list of characters ... can be used for reconstruction of sentences
    boundary_mat:   list of boundary array (each is of length #chars+1)
'''
def load_validation_set(test_files):
    char_dict = CharDictionary()
    input_classes = char_dict.num_char_classes() + 1

    cids_mat  = []
    chars_mat = []
    boundary_mat = []
    seq_lengths = []
    for input_file in test_files:
        with file_io.FileIO(input_file, 'r') as f:
            for s in f: # s is the line string
                if (sys.version_info <= (3, 0)):
                    s = s.decode('utf8')
                if s and (len(s) > 0):
                    t = re.sub("<[^>]*>", "", s.strip())
                    t = re.sub("[|]", "", t)

                    # [:-1] remove the ending empty word (resulting from "|" at the end of line in BEST2010)
                    answers = re.split('[|]', s.strip())[:-1]
                    boundary = get_boundary_array(answers)
                    boundary_mat.append(boundary)
                    
                    chars = list(t)
                    cids = char_dict.chars2cids(chars)
                    seq_lengths += [len(cids)]
                    cids_mat.append(cids)
                    chars_mat.append(chars)

    n_examples = len(cids_mat)
    
    max_len = max(seq_lengths)
    batch_labels = np.zeros((n_examples, max_len), dtype=np.int32)
    for i in range(n_examples):
        batch_labels[i,0:len(cids_mat[i])] = cids_mat[i]

    in_embedding = np.eye(input_classes) # 2
    one_hot_flat = in_embedding[list(batch_labels.flatten())]
    one_hot      = np.reshape(one_hot_flat, (n_examples, max_len, input_classes))
    
    #(time, batch, in)
    one_hot_by_t = np.transpose(one_hot, (1,0,2))
    
    return one_hot_by_t, seq_lengths, chars_mat, boundary_mat

if __name__ == '__main__':
    

    #test_files = ['data/test_txt/article_00179.txt', 
    #    'data/test_txt/encyclopedia_00099.txt', 
    #    'data/test_txt/news_00088.txt', 
    #    'data/test_txt/novel_00098.txt']
    test_files = ['data/test_txt/article_00179.txt']
        
    one_hot_by_t, seq_lengths, chars_mat, boundary_mat = load_validation_set(test_files)
    print(one_hot_by_t.shape)
    #print(len(seq_lengths))
    #print(len(chars_mat))
    #print(len(boundary_mat))
    #labels = viterbi(probs)
    #words  = char_dict.chars2words(chars, labels)
    #print('|'.join(words))
    
    answers = u"เขา|ร้อง|บท|<POEM>เย็นย่ำ จะค่ำอยู่แล้วลงรอนรอน</POEM>|".split('|')[:-1]
    print('|'.join(answers))

    words = u"เขา|ร้อง|บท|เย็น|ย่ำ| |จะ|ค่ำ|อยู่แล้ว|ลง|รอนรอน".split('|')
    print('|'.join(words))

    x = get_boundary_array(answers)
    y = get_boundary_array(words)
    print(x)
    print(y)
    print(len(answers))
    print(len(words))
    print('len x:', word_count_from_boundary_array(x))
    print('len y:', word_count_from_boundary_array(y))    
    correct, n_refwords, n_outwords = count_correct_words(x, y, len(answers), len(words))
    
    r = float(correct) / float(n_refwords)
    p = float(correct) / float(n_outwords)
    f = 2 * p * r / (p + r)
    print("recall = %.3f, precision = %.3f, F = %.3f" % (r, p, f))
