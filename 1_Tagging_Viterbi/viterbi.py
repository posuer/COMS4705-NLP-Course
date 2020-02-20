#!/usr/bin/env python

import re
from math import log
import os

from count_freqs3 import Hmm
from get_rare_words import get_rare_words

def replace_word(word):
    if re.match(r'[A-Z]', word):
        word = '_CAP_'
    elif re.match(r'\d', word):
        word = '_NUM_' 
    else: word = '_RARE_'
    return word

'''
---- build ner train data with replaced pattern ---
'''

rare_words = get_rare_words()

writer = open('ner_train_pattern.dat','w')
reader = open('ner_train.dat','r')

for line in reader.readlines():
    line_list = line.strip().split(' ')
    if len(line_list) < 2:
        writer.write('\n')
        continue
    if line_list[0] in rare_words:
        line_list[0] = replace_word(line_list[0])
    #print(line_list)
    writer.write(line_list[0]+' '+line_list[1]+'\n')

writer.close()
reader.close()


'''
----- Build new ner counts -----
'''
os.system('python count_freqs3.py ner_train_pattern.dat > ner_pattern.counts')

'''
----- Predict -----
'''
tags = ["I-PER", "I-LOC", "I-ORG", "I-MISC", "O","B-LOC", "B-ORG", "B-MISC","B-PER"]
S = list()
for tag1 in tags:
    for tag2 in tags:
        S.append((tag1, tag2))

#Read emission and ngram counts
readfile = open("ner_pattern.counts",'r')
readlines = readfile.readlines()
counter = Hmm()
counter.read_counts(readlines)
d = counter.emission_counts
ngram_counts = counter.ngram_counts

#calculate emission
y_count = dict()
for tag in tags:
    y_count[tag] = 0

for key, value in d.items():
    y_count[key[1]] += value

emission = dict()
vocabulary = set()
for key, value in d.items():
    emission[key] = float(value) / float(y_count[key[1]])
    vocabulary.add(key[0])

def get_q(trigrams):
    if (trigrams[0],trigrams[1],trigrams[2]) not in ngram_counts[2].keys():
        q = 1.0 / float(len(ngram_counts[1]))
    else:
        q = float(ngram_counts[2][(trigrams[0],trigrams[1],trigrams[2])]) / float(ngram_counts[1][(trigrams[0],trigrams[1])])
    return log(q)

def viterbi(sentence):
    pi = {(-1,'*','*'):0}
    backpoints = {}
    #Apply viterbi on this sentence
    for k in range(len(sentence)):
        predict_word = sentence[k]
        
        #build S for all situation
        if k == 0:
            w_tags = ['*']
            thisS = [('*', tag) for tag in tags]
        elif k == 1:
            w_tags = ['*']
            thisS = list(S)
        else:
            w_tags = list(tags)
            thisS = list(S)

        for tag in thisS: #loop u, v in S_k-1, S_k
            pi_list = []
            max_pi = float("-inf")
            max_w = ''
            for w in w_tags: #loop w in S_k-1
                #get emission value
                if (predict_word, tag[1]) in emission.keys() and (w, tag[0], tag[1]) in ngram_counts[2].keys():
                    emission_value = emission[(predict_word, tag[1])] 
                    q = get_q((w, tag[0], tag[1]))
                    pi_value = pi[(k-1,w,tag[0])] + q + log(emission_value) 
                    if pi_value > max_pi:
                        max_pi = pi_value
                        max_w = w
            pi[(k, tag[0],tag[1])] = max_pi
            backpoints[(k, tag[0], tag[1])] =  [max_w, max_pi]

    predict_tags = [' '] * len(sentence)
    #last 2 tag
    max_value = float("-inf")
    for tag in thisS :
        value = pi[(len(sentence)-1,tag[0],tag[1])] + get_q((tag[0],tag[1],'STOP')) #if (len(sentence)-1,tag[0],tag[1]) in pi.keys()  if (len(sentence)-1,tag[0],tag[1]) in pi.keys()  and (tag[0],tag[1],'STOP') in ngram_counts[2].keys()
        if value > max_value:
            max_value = value
            if len(sentence) >= 2:
                predict_tags[-2], predict_tags[-1] = ([tag[0],max_value], [tag[1],max_value])
            else:
                predict_tags[-1] = [tag[1],max_value]
            
    #get predict tags
    for i in reversed(range(len(sentence)-2)):
        predict_tags[i] = backpoints[(i+2,predict_tags[i+1][0],predict_tags[i+2][0])]
    
    return predict_tags

def run(output_file, dev_file):
    writer = open(output_file,'w')

    with open(dev_file,'r') as f:
        sentence = []
        for line in f.readlines():
            word = line.strip()
            if word == '': #Got a whole sentence

                sentence_pattern = []
                for word in sentence:
                    if word not in vocabulary:
                        sentence_pattern.append(replace_word(word))   
                    else: sentence_pattern.append(word)

                predict_tags = viterbi(sentence_pattern)
                #write result to file
                for i in range(len(predict_tags)):
                    writer.write(sentence[i]+' '+predict_tags[i][0]+' '+str(predict_tags[i][1])+'\n')

                #initial a new sentence
                writer.flush()
                writer.write('\n')
                sentence = []
            else:  
                sentence.append(word)
    writer.close()

run('predict_result.txt', 'ner_dev.dat')
