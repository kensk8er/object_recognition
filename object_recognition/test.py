#!/usr/local/bin/python

from pprint import pprint
import gzip, numpy, var_dump


def unpickle(file):
	import cPickle
	fo = open(file, 'rb')
	dict = cPickle.load(fo)
	fo.close()
	return dict



if __name__ == '__main__':

	'''
	train = []
	for i in range(1,6):
		data = unpickle('cifar-10-batches-py/data_batch_' + str(i))
		data_set = (data['data'], data['labels'])
		#var_dump.var_dump(train)
		train.extend(data_set)

	var_dump.var_dump(train)
	'''

	'''
	var_dump.var_dump(train_set['data'][0])
	print len(train_set['data'][0])
	var_dump.var_dump(train_set['labels'][0])
	print len(train_set['labels'][0])
	'''

	test_set = unpickle('data/test_set_gray.pkl')
	var_dump.var_dump(test_set)
	
