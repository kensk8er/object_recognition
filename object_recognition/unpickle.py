#!/usr/local/bin/python

from pprint import pprint
import gzip, numpy, var_dump


def unpickle(file):
	import cPickle
	fo = open(file, 'rb')
	dict = cPickle.load(fo)
	fo.close()
	return dict
