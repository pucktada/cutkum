#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Pucktada Treeratpituk (https://pucktada.github.io/)
# License: MIT
# 2017-05-21
from __future__ import unicode_literals
from __future__ import print_function
import numpy as np
import sys

class CharDictionary:
    """ a utility class for converting characters into character ids. 
        the id 0 is reserved for padding (both for character ids and label ids)
    """
    
    def __init__(self):
        thai_chars = 'กขฃคฅฆงจฉชซฌญฐฎฏฐฑฒณดตถทธนบปผฝพฟภมยรฤลฦวศษสหฬอฮฯะัาำิีึืุูเแโใไๅๆ็่้๊๋์ํ' # 1..73
        numbers = '๐๑๒๓๔๕๖๗๘๙0123456789' # => 74 ' '=> 75, '.'=>76
        symbols1 = ':;^+-*/_=#!~%\\/\'"`?' # 77
        symbols2 = '()[]<>{}' # 78
        eng_chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ' # 79

        self._class_for = {'B':1, 'M':2, 'E':3, 'S':4}
        #self._class_for = {'M':1, 'E':2}

        self._cid2char = ['', '_'] # ''=padding, _=_UNK
        self._char2cid = dict()
        
        next_idx = 2
        # 1 ... len(thai_chars)
        for c in thai_chars:
            self._char2cid[c] = next_idx
            self._cid2char.append(c)
            next_idx += 1
                    
        last_char_id = len(thai_chars) + 1

        for d in numbers:
            self._char2cid[d] = last_char_id + 1
        self._cid2char.append('$')
        
        self._char2cid[' '] = last_char_id + 2
        self._cid2char.append(' ')
        
        self._char2cid['.'] = last_char_id + 3
        self._cid2char.append('.')
        
        for s in symbols1:
            self._char2cid[s] = last_char_id + 4
        self._cid2char.append('#')  

        for s in symbols2:
            self._char2cid[s] = last_char_id + 5
        self._cid2char.append(':')

        for s in eng_chars:
            self._char2cid[s] = last_char_id + 6
        self._cid2char.append('A')

    # generate padding ' ' of "length"
    def padding_cids(self, length):
        return [self._char2cid[' ']] * length

    # generate padding label of "length"
    def padding_labels(self, length):
        if (length == 1):
            return [self._class_for['S']]
        elif (length == 2):
            return [self._class_for['B']] + [self._class_for['E']]
        return [self._class_for['B']] + [self._class_for['M']] * (length-2) + [self._class_for['E']]

    def num_char_classes(self):
        """ return the number of character classes """
        # minus 1... cause cid 0 is not used... reserved for padding
        return len(self._cid2char) - 1 # 
    
    def num_label_classes(self):
        """ return the number of label classes """        
        return len(self._class_for) # B, M, E, S

    def words2cids(self, words):
        """ return the arrays of character ids and labels for a given array of words        
        """
        chars, labels = self.words2chars(words)
        cids = self.chars2cids(chars)
        return cids, labels

   # labels = {B, M, E, S} <=== 0:begin, 1:middle, 2:end, 3:single
    # convert words array to character array, and create a labels
    def words2chars(self, words):
        """ return the arrays of characters and labels for a given array of words        
        """
        chars, labels = [], []

        for w in words:
            if len(w) == 1: # single char
                chars.append(w[0])
                labels.append(self._class_for['S']) # 'S'
            elif len(w) > 0:
                chars.append(w[0])
                labels.append(self._class_for['B']) # 'B'
                for i in range(len(w)-2):
                    chars.append(w[i+1])
                    labels.append(self._class_for['M']) # 'M'
                chars.append(w[-1])
                labels.append(self._class_for['E']) # 'E'
        return chars, labels

    def chars2words(self, chars, labels):
        """ return array of words for the given arrays of characters and labels
            (the reverse of 'words2chars')
        """

        words = []
        word = ''
        class_for = {'B':1, 'M':2, 'E':3, 'S':4}
        for i in range(len(chars)):
            c = chars[i] # reverse_dict[data[i]]
            l = labels[i] # 0:begin, 1:middle, 2:end, 3:single
            if l == self._class_for['B']: # begin
                if len(word) != 0: # somehow, still have left-over word
                    words += [word]
                word = c
            elif l == self._class_for['M']: # middle
                word += c
            elif l == self._class_for['E']: # end
                word += c
                words += [word]
                word = ''
            elif l == self._class_for['S']: # single
                if (len(word) != 0):
                    words += [word]
                words += [c]
                word = ''
        if (len(word) != 0):
            words += [word]
        return words

    # labels = {M, E} <=== 0:begin, 1:middle, 2:end, 3:single
    # convert words array to character array, and create a labels
    def words2chars_2class(self, words):
        """ return the arrays of characters and labels for a given array of words        
        """
        chars, labels = [], []

        for w in words:
            if len(w) == 1: # single char
                chars.append(w[0])
                labels.append(self._class_for['E']) # 'S'
            elif len(w) > 0:
                chars.append(w[0])
                labels.append(self._class_for['M']) # 'B'
                for i in range(len(w)-2):
                    chars.append(w[i+1])
                    labels.append(self._class_for['M']) # 'M'
                chars.append(w[-1])
                labels.append(self._class_for['E']) # 'E'
        return chars, labels

    def chars2words_2class(self, chars, labels):
        """ return array of words for the given arrays of characters and labels
            (the reverse of 'words2chars')
        """
        words = []
        word = ''
        for i in range(len(chars)):
            c = chars[i] # reverse_dict[data[i]]
            l = labels[i] # 0:begin, 1:middle, 2:end, 3:single

            if l == self._class_for['M']: # middle
                word += c
            elif l == self._class_for['E']: # end
                word += c
                words += [word]
                word = ''

        if (len(word) != 0):
            words += [word]
        return words
    
    def cidOf(self, c):
        """ return the character id of the given character 'c' """
        if c in self._char2cid:
            return self._char2cid[c]
        else:
            return 1 # UNK_ID
    
    def chars2cids(self, chars):
        """ return the array of character ids for the given arrays of characters """        
        return [self.cidOf(c) for c in chars]
        
    def cids2chars(self, cids):
        """ return the array of characters for the given arrays of character ids 
            (the reverse of chars2cids)
        """
        return [self._cid2char[i] for i in cids]

def test_char_dict():
    print('testing character dictionary')
    words = ['กฎหมาย', 'กับ', '1', 'ก าร', '(', 'เบียดบัง', 'คน', ')', 'A', 'CD']
    
    dic = CharDictionary()
    
    #sen = list('สารานุกรมไทยสำหรับเยาวชนฯ')
    #chars = dic.words2chars(sen)
    #print(chars)
    
    chars, labels = dic.words2chars(words)
    cids = dic.chars2cids(chars)
    
    cids   += dic.padding_cids(6)
    labels = dic.padding_labels(6) + labels

    print(' '.join(chars))
    #print(dic.chars2cids(chars))
    #print(' '.join(dic.cids2chars(cids)))
    print("[%s]" % '|'.join(words))

    print(len(cids), cids)
    print(len(labels), labels)

    pchars = dic.cids2chars(cids)
    pwords = dic.chars2words(pchars[0:-6], labels[6:])
    print("[%s]" % '|'.join(pwords))

if __name__ == '__main__':
    test_char_dict()

