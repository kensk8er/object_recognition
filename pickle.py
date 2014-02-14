#!/usr/local/bin/python

def pickle(data, file):
	import cPickle
	fo = open(file, 'w')
	cPickle.dump(data, fo, protocol=1)
	fo.close()
