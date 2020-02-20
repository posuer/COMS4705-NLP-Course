#!/usr/bin/python3
import pickle
import sys

from count_freqs3 import Hmm

def get_rare_words():
	
	readfile = open("ner.counts",'r')
	readlines = readfile.readlines()
	counter = Hmm()
	counter.read_counts(readlines)	
	d = counter.emission_counts
	#O_words, GENE_words = set(), set()
	rare_words = set()
	total_count = dict()
	for key, value in d.items():
	    if key[0] in total_count.keys():
	        total_count[key[0]] += value
	    else:
	        total_count[key[0]] = value
	for key, value in total_count.items():
	    if value < 5:
	        rare_words.add(key)

	return rare_words#(O_words, GENE_words)


if __name__ == "__main__":

	rare_words = get_rare_words()

	with open("rare_words.pickle", "wb") as f:
		pickle.dump(rare_words, f)